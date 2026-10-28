"""Benchmark Lightweight YOLOX Detectors (YOLOX-Nano & YOLOX-Tiny) with ByteTrack and Fuzzy-Adaptive ByteTrack.

This script benchmarks lightweight YOLOX models on MOT17 to compare against
modern lightweight detectors (YOLOv8n, YOLO11n) and the original ByteTrack detector (YOLOX-X).
- YOLOX-Nano: 0.91M parameters, 1.08 GFLOPs (Depthwise Conv)
- YOLOX-Tiny: 5.06M parameters, 6.45 GFLOPs
"""

from __future__ import annotations

import argparse
import configparser
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess
from src.bytetrack_adapter import AdaptiveByteTracker, DetectionResults, xyxy_to_xywh
from src.fuzzy_controller import FuzzyByteTrackController
from src.metrics import evaluate_with_trackeval, tracking_csv_to_mot, validate_mot_file
from src.tracking_logger import write_csv


def extract_yolox_detections(
    exp_name: str,
    weight_path: Path,
    img_files: list[Path],
    input_size: tuple[int, int] = (416, 416),
    conf_thresh: float = 0.01,
    nms_thresh: float = 0.65,
    device: str = "cuda:0",
) -> tuple[dict[int, np.ndarray], float]:
    """Chạy trích xuất bounding box người (class 0) từ YOLOX."""
    print(f"\n[YOLOX DETECTOR] Loading {exp_name} from {weight_path} (size={input_size}, device={device})...")
    exp = get_exp(None, exp_name)
    model = exp.get_model()
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    dets_by_frame: dict[int, np.ndarray] = {}
    det_times: list[float] = []

    # Warmup
    dummy = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
    dummy_pre, _ = preproc(dummy, input_size)
    dummy_t = torch.from_numpy(dummy_pre).unsqueeze(0).float().to(device)
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy_t)

    t0_all = time.perf_counter()
    for idx, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        t0 = time.perf_counter()
        img_pre, ratio = preproc(img, input_size)
        img_tensor = torch.from_numpy(img_pre).unsqueeze(0).float().to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            outputs = postprocess(outputs, num_classes=80, conf_thre=conf_thresh, nms_thre=nms_thresh, class_agnostic=True)

        dt = (time.perf_counter() - t0) * 1000.0
        det_times.append(dt)

        fid = idx + 1
        if outputs[0] is not None:
            raw = outputs[0].cpu().numpy()
            person_mask = (raw[:, 6] == 0)
            person_dets = raw[person_mask]
            if len(person_dets) > 0:
                xyxy = person_dets[:, :4] / ratio
                confs = person_dets[:, 4] * person_dets[:, 5]
                clss = person_dets[:, 6].astype(int)
                arr = np.column_stack([xyxy, confs, clss]).astype(np.float32)
            else:
                arr = np.empty((0, 6), dtype=np.float32)
        else:
            arr = np.empty((0, 6), dtype=np.float32)

        dets_by_frame[fid] = arr

        if (idx + 1) % 100 == 0 or (idx + 1) == len(img_files):
            print(f"  Processed {idx + 1}/{len(img_files)} frames | Avg det latency: {np.mean(det_times):.2f} ms")

    total_time = time.perf_counter() - t0_all
    avg_det_ms = float(np.mean(det_times))
    print(f"[OK] Hoàn tất {exp_name} ({input_size}) trong {total_time:.2f}s (Det latency: {avg_det_ms:.2f}ms)")
    return dets_by_frame, avg_det_ms


def run_tracker(
    dets_by_frame: dict[int, np.ndarray],
    tracker_config: dict[str, Any],
    mode: str,
) -> tuple[list[dict[str, Any]], float]:
    """Chạy tracker trên detections đã có."""
    tracker = AdaptiveByteTracker(tracker_config)
    tracker.reset_state()

    fuzzy_controller = FuzzyByteTrackController() if mode == "fuzzy" else None
    frame_rows: list[dict[str, Any]] = []
    trk_times: list[float] = []

    for fid in sorted(dets_by_frame.keys()):
        raw_dets = dets_by_frame[fid]
        if len(raw_dets) > 0:
            xyxy = raw_dets[:, :4]
            xywh = xyxy_to_xywh(xyxy)
            confs = raw_dets[:, 4]
            clss = raw_dets[:, 5].astype(int)
        else:
            xywh = np.empty((0, 4), dtype=np.float32)
            confs = np.empty((0,), dtype=np.float32)
            clss = np.empty((0,), dtype=np.int32)

        det_results = DetectionResults(xywh, confs, clss)

        dynamic_high_thresh = None
        if mode == "fuzzy" and fuzzy_controller is not None:
            high_base = float(tracker_config["track_high_thresh"])
            low_ratio = float(np.sum(confs < high_base) / len(confs)) if len(confs) > 0 else 0.0
            spread = float(np.std(confs)) if len(confs) > 0 else 0.0
            prior_unmatched = tracker.prior_unmatched_ratio

            adj = fuzzy_controller.adjust(
                low_conf_ratio=low_ratio,
                unmatched_ratio=prior_unmatched,
                confidence_spread=spread,
                track_high_thresh=high_base,
                track_low_thresh=float(tracker_config["track_low_thresh"]),
                new_track_thresh=float(tracker_config["new_track_thresh"]),
                track_buffer=int(tracker_config["track_buffer"]),
            )
            dynamic_high_thresh = adj.track_high_thresh

        t0_trk = time.perf_counter()
        tracks, _, _ = tracker.update_with_threshold(det_results, dynamic_high_thresh=dynamic_high_thresh)
        trk_ms = (time.perf_counter() - t0_trk) * 1000.0
        trk_times.append(trk_ms)

        if len(tracks) > 0:
            for trk in tracks:
                frame_rows.append({
                    "frame_id": fid,
                    "track_id": int(trk[4]),
                    "x1": float(trk[0]),
                    "y1": float(trk[1]),
                    "x2": float(trk[2]),
                    "y2": float(trk[3]),
                    "confidence": float(trk[5]),
                    "class_id": int(trk[6]),
                })

    avg_trk_ms = float(np.mean(trk_times)) if trk_times else 0.0
    return frame_rows, avg_trk_ms


def main():
    parser = argparse.ArgumentParser(description="Lightweight YOLOX Benchmark (Nano vs Tiny)")
    parser.add_argument("--sequence", default="MOT17-09-FRCNN", help="Tên sequence MOT17")
    parser.add_argument("--device", default="cuda:0", help="Thiết bị chạy")
    parser.add_argument("--test-size", type=int, default=416, choices=[416, 640], help="Kích thước ảnh inference")
    args = parser.parse_args()

    seq_name = args.sequence
    seq_dir = PROJECT_ROOT / "data" / "MOT17" / "train" / seq_name
    gt_path = seq_dir / "gt" / "gt.txt"
    ini_path = seq_dir / "seqinfo.ini"

    cp = configparser.ConfigParser()
    cp.read(ini_path)
    seq_fps = float(cp["Sequence"].get("frameRate", 30.0))
    seq_width = int(cp["Sequence"].get("imWidth", 1920))
    seq_height = int(cp["Sequence"].get("imHeight", 1080))

    output_dir = PROJECT_ROOT / "outputs" / "experiments" / f"yolox_lightweight_{args.test_size}" / seq_name
    output_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted((seq_dir / "img1").glob("*.jpg"))

    print(f"\n{'='*75}")
    print(f"BENCHMARK: Lightweight YOLOX (Nano vs Tiny) trên {seq_name} (Input size: {args.test_size}x{args.test_size})")
    print(f"Frames: {len(img_files)} | Device: {args.device} | Resolution: {seq_width}x{seq_height}")
    print(f"{'='*75}")

    baseline_cfg = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.60,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.70,
        "track_buffer": 30,
        "match_thresh": 0.80,
        "fuse_score": True,
    }

    ga_v8_yaml = PROJECT_ROOT / "configs" / "ga_bytetrack.yaml"
    if ga_v8_yaml.is_file():
        with ga_v8_yaml.open(encoding="utf-8") as f:
            ga_cfg = yaml.safe_load(f)
    else:
        ga_cfg = baseline_cfg.copy()

    fuzzy_cfg = baseline_cfg.copy()

    yolox_models = [
        {
            "name": "YOLOX-Nano",
            "exp_name": "yolox-nano",
            "weights": PROJECT_ROOT / "weights" / "yolox_nano.pth",
            "params": "0.91M",
            "gflops": "1.08" if args.test_size == 416 else "2.56",
        },
        {
            "name": "YOLOX-Tiny",
            "exp_name": "yolox-tiny",
            "weights": PROJECT_ROOT / "weights" / "yolox_tiny.pth",
            "params": "5.06M",
            "gflops": "6.45" if args.test_size == 416 else "15.25",
        },
    ]

    summary_rows = []

    for model_info in yolox_models:
        m_name = model_info["name"]
        exp_name = model_info["exp_name"]
        w_path = model_info["weights"]

        dets_by_frame, avg_det_ms = extract_yolox_detections(
            exp_name=exp_name,
            weight_path=w_path,
            img_files=img_files,
            input_size=(args.test_size, args.test_size),
            conf_thresh=0.01,
            nms_thresh=0.65,
            device=args.device,
        )

        total_raw_boxes = sum(len(d) for d in dets_by_frame.values())

        tracker_modes = {
            "Baseline": (baseline_cfg, "baseline"),
            "GA-ByteTrack": (ga_cfg, "ga_bytetrack"),
            "Fuzzy-Adaptive": (fuzzy_cfg, "fuzzy"),
        }

        for trk_name, (trk_cfg, mode_type) in tracker_modes.items():
            print(f"\n--- Đang chạy {m_name} ({args.test_size}p) + {trk_name} ---")
            frame_rows, trk_ms = run_tracker(dets_by_frame, trk_cfg, mode_type)

            run_id = f"{exp_name}_{args.test_size}_{mode_type}"
            run_out_dir = output_dir / run_id
            run_out_dir.mkdir(parents=True, exist_ok=True)

            csv_path = run_out_dir / "track.csv"
            write_csv(frame_rows, csv_path)
            mot_path = run_out_dir / "pred_mot.txt"
            tracking_csv_to_mot(csv_path, mot_path)
            validate_mot_file(mot_path)

            unique_tracks = len(set(r["track_id"] for r in frame_rows)) if frame_rows else 0
            hota, idf1, mota, idsw, assa, deta = 0.0, 0.0, 0.0, 0, 0.0, 0.0

            if gt_path.is_file():
                try:
                    eval_res = evaluate_with_trackeval(
                        prediction_file=mot_path,
                        gt_file=gt_path,
                        sequence_name=seq_name,
                        work_dir=run_out_dir / "eval_tmp",
                        frame_count=len(img_files),
                        width=seq_width,
                        height=seq_height,
                        fps=seq_fps,
                    )
                    hota = eval_res.get("HOTA", 0.0) * 100.0
                    idf1 = eval_res.get("IDF1", 0.0) * 100.0
                    mota = eval_res.get("MOTA", 0.0) * 100.0
                    idsw = int(eval_res.get("ID_switches", 0))
                    assa = eval_res.get("AssA", 0.0) * 100.0
                    deta = eval_res.get("DetA", 0.0) * 100.0
                except Exception as e:
                    print(f"TrackEval exception: {e}")

            pipeline_ms = avg_det_ms + trk_ms
            e2e_fps = 1000.0 / pipeline_ms if pipeline_ms > 0 else 0.0
            tracker_fps = 1000.0 / trk_ms if trk_ms > 0 else 0.0

            res_row = {
                "Detector": m_name,
                "Input_Size": f"{args.test_size}x{args.test_size}",
                "Model_Params": model_info["params"],
                "Model_GFLOPs": model_info["gflops"],
                "Tracker": trk_name,
                "Total_Raw_Dets": total_raw_boxes,
                "Pred_Boxes": len(frame_rows),
                "Pred_Tracks": unique_tracks,
                "HOTA (%)": round(hota, 2),
                "IDF1 (%)": round(idf1, 2),
                "MOTA (%)": round(mota, 2),
                "AssA (%)": round(assa, 2),
                "DetA (%)": round(deta, 2),
                "IDSW": idsw,
                "Det_Latency (ms)": round(avg_det_ms, 2),
                "Tracker_FPS": round(tracker_fps, 1),
                "End_to_End_FPS": round(e2e_fps, 1),
            }
            summary_rows.append(res_row)
            print(f"  -> HOTA: {hota:.2f}% | IDF1: {idf1:.2f}% | MOTA: {mota:.2f}% | IDSW: {idsw} | E2E FPS: {e2e_fps:.1f}")

    df_summary = pd.DataFrame(summary_rows)
    csv_out = output_dir / "comparison_summary.csv"
    df_summary.to_csv(csv_out, index=False)

    print(f"\n{'='*75}")
    print("BẢNG TỔNG HỢP SO SÁNH LIGHTWEIGHT YOLOX (NANO & TINY):")
    print(f"{'='*75}")
    print(df_summary.to_string(index=False))
    print(f"\n[OK] Đã lưu kết quả tại: {csv_out}")


if __name__ == "__main__":
    main()
