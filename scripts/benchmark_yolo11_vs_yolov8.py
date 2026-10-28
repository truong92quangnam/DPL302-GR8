"""Benchmark & Detailed Comparison: YOLOv8n vs YOLO11n on MOT17 with ByteTrack, GA, and Fuzzy.

Chạy đối sánh toàn diện:
- Detector: YOLOv8n (3.2M params, 8.7 GFLOPs) vs YOLO11n (2.6M params, 6.7 GFLOPs, C3k2 + C2PSA)
- Tracker: Baseline ByteTrack, GA-ByteTrack, Fuzzy-Adaptive ByteTrack
- Metrics: HOTA, IDF1, MOTA, IDSW, AssA, DetA, FPS, Total Detections, Pred Tracks
"""

from __future__ import annotations

import argparse
import configparser
import json
import sys
import time
from pathlib import Path
from typing import Any

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from src.bytetrack_adapter import AdaptiveByteTracker, DetectionResults, xyxy_to_xywh
from src.fuzzy_controller import FuzzyByteTrackController, fuzzy_adjustment_dict
from src.metrics import evaluate_with_trackeval, tracking_csv_to_mot, validate_mot_file
from src.tracking_logger import write_csv


def extract_detections_for_model(
    model_name: str,
    img_files: list[Path],
    imgsz: int = 1280,
    conf_thresh: float = 0.01,
    iou_thresh: float = 0.7,
    device: str = "cpu",
) -> tuple[dict[int, np.ndarray], float]:
    """Chạy detector trên toàn bộ danh sách ảnh và trả về dictionary detections theo frame."""
    print(f"\n[DETECTOR] Đang chạy trích xuất detections cho {model_name} (device={device})...")
    model = YOLO(model_name)
    dets_by_frame: dict[int, np.ndarray] = {}
    det_times: list[float] = []

    # Warmup
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=imgsz, device=device, verbose=False)

    t0_all = time.perf_counter()
    for idx, img_path in enumerate(img_files):
        t0 = time.perf_counter()
        pred = model.predict(
            str(img_path),
            imgsz=imgsz,
            conf=conf_thresh,
            iou=iou_thresh,
            classes=[0],  # Person
            device=device,
            verbose=False,
        )[0]
        dt = (time.perf_counter() - t0) * 1000.0
        det_times.append(dt)

        boxes = pred.boxes
        fid = idx + 1
        if len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            arr = np.column_stack([xyxy, confs, clss]).astype(np.float32)
        else:
            arr = np.empty((0, 6), dtype=np.float32)
        dets_by_frame[fid] = arr

        if (idx + 1) % 100 == 0 or (idx + 1) == len(img_files):
            print(f"  Processed {idx + 1}/{len(img_files)} frames | Latency: {np.mean(det_times):.1f} ms")

    total_time = time.perf_counter() - t0_all
    avg_det_ms = float(np.mean(det_times))
    print(f"[OK] Hoàn tất {model_name} trong {total_time:.2f}s (Avg det latency: {avg_det_ms:.1f}ms)")
    return dets_by_frame, avg_det_ms


def run_tracker_on_cached_dets(
    dets_by_frame: dict[int, np.ndarray],
    tracker_config: dict[str, Any],
    mode: str,  # "baseline", "ga_bytetrack", "fuzzy"
) -> tuple[list[dict[str, Any]], float]:
    """Chạy tracker trên detections đã nạp sẵn."""
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
    parser = argparse.ArgumentParser(description="YOLOv8n vs YOLO11n Benchmark")
    parser.add_argument("--sequence", default="MOT17-09-FRCNN", help="Tên sequence MOT17")
    parser.add_argument("--max-frames", type=int, default=None, help="Giới hạn số frame để test nhanh")
    parser.add_argument("--device", default="cpu", help="Thiết bị chạy: cpu hoặc cuda/0")
    args = parser.parse_args()

    seq_name = args.sequence
    seq_dir = Path(f"data/MOT17/train/{seq_name}")
    gt_path = seq_dir / "gt" / "gt.txt"
    ini_path = seq_dir / "seqinfo.ini"

    cp = configparser.ConfigParser()
    cp.read(ini_path)
    seq_fps = float(cp["Sequence"].get("frameRate", 30.0))
    seq_width = int(cp["Sequence"].get("imWidth", 1920))
    seq_height = int(cp["Sequence"].get("imHeight", 1080))

    output_dir = Path("outputs/experiments/yolo11_vs_yolov8") / seq_name
    output_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted((seq_dir / "img1").glob("*.jpg"))
    if args.max_frames:
        img_files = img_files[: args.max_frames]
        # Tạo file GT tương ứng với số frame được test
        gt_df = pd.read_csv(gt_path, header=None)
        gt_subset_path = output_dir / f"gt_subset_{args.max_frames}.txt"
        gt_df[gt_df[0] <= args.max_frames].to_csv(gt_subset_path, header=False, index=False)
        eval_gt_file = gt_subset_path
    else:
        eval_gt_file = gt_path

    print(f"\n{'='*70}")
    print(f"BENCHMARK: YOLOv8n vs YOLO11n trên {seq_name}")
    print(f"Frames: {len(img_files)} | Device: {args.device} | Resolution: {seq_width}x{seq_height}")
    print(f"{'='*70}")

    # 1. Định nghĩa cấu hình Tracker
    baseline_cfg = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.60,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.70,
        "track_buffer": 30,
        "match_thresh": 0.80,
        "fuse_score": True,
    }

    # Đọc GA config của YOLOv8n
    ga_v8_yaml = Path("configs/ga_bytetrack.yaml")
    if ga_v8_yaml.is_file():
        with ga_v8_yaml.open(encoding="utf-8") as f:
            ga_v8_cfg = yaml.safe_load(f)
    else:
        ga_v8_cfg = baseline_cfg.copy()

    # Đọc GA config của YOLO11n
    ga_11_yaml = Path("configs/yolo11_ga_bytetrack.yaml")
    if ga_11_yaml.is_file():
        with ga_11_yaml.open(encoding="utf-8") as f:
            ga_11_cfg = yaml.safe_load(f)
    else:
        ga_11_cfg = baseline_cfg.copy()

    fuzzy_cfg = baseline_cfg.copy()

    # 2. Chạy detections cho cả 2 models
    detectors = {
        "YOLOv8n": "yolov8n.pt",
        "YOLO11n": "yolo11n.pt",
    }

    det_cache_all = {}
    det_latency_all = {}
    for det_label, model_file in detectors.items():
        dets_by_frame, avg_det_ms = extract_detections_for_model(
            model_file, img_files, imgsz=1280, conf_thresh=0.01, iou_thresh=0.7, device=args.device
        )
        det_cache_all[det_label] = dets_by_frame
        det_latency_all[det_label] = avg_det_ms

    # 3. Chạy từng tracker mode trên detections của mỗi model
    summary_rows = []

    for det_label in ["YOLOv8n", "YOLO11n"]:
        dets_by_frame = det_cache_all[det_label]
        det_ms = det_latency_all[det_label]
        total_raw_boxes = sum(len(d) for d in dets_by_frame.values())

        if det_label == "YOLOv8n":
            tracker_modes = {
                "Baseline": (baseline_cfg, "baseline"),
                "GA-ByteTrack": (ga_v8_cfg, "ga_bytetrack"),
                "Fuzzy-Adaptive": (fuzzy_cfg, "fuzzy"),
            }
        else:
            tracker_modes = {
                "Baseline": (baseline_cfg, "baseline"),
                "GA-ByteTrack (Cross-v8)": (ga_v8_cfg, "ga_bytetrack_cross"),
                "GA-ByteTrack (Tailored)": (ga_11_cfg, "ga_bytetrack"),
                "Fuzzy-Adaptive": (fuzzy_cfg, "fuzzy"),
            }

        for trk_name, (trk_cfg, mode_type) in tracker_modes.items():
            print(f"\n--- Đang chạy {det_label} + {trk_name} ---")
            frame_rows, trk_ms = run_tracker_on_cached_dets(dets_by_frame, trk_cfg, mode_type)

            run_id = f"{det_label.lower()}_{mode_type}"
            run_out_dir = output_dir / run_id
            run_out_dir.mkdir(parents=True, exist_ok=True)

            # Xuất tracking csv và mot file
            csv_path = run_out_dir / "track.csv"
            write_csv(frame_rows, csv_path)
            mot_path = run_out_dir / "pred_mot.txt"
            tracking_csv_to_mot(csv_path, mot_path)
            validate_mot_file(mot_path)

            # Tính TrackEval metrics
            unique_tracks = len(set(r["track_id"] for r in frame_rows)) if frame_rows else 0
            hota, idf1, mota, idsw, assa, deta = 0.0, 0.0, 0.0, 0, 0.0, 0.0
            if gt_path.is_file():
                try:
                    eval_res = evaluate_with_trackeval(
                        prediction_file=mot_path,
                        gt_file=eval_gt_file,
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

            pipeline_ms = det_ms + trk_ms
            e2e_fps = 1000.0 / pipeline_ms if pipeline_ms > 0 else 0.0
            tracker_fps = 1000.0 / trk_ms if trk_ms > 0 else 0.0

            res_row = {
                "Detector": det_label,
                "Model_Params": "3.2M" if det_label == "YOLOv8n" else "2.6M",
                "Model_GFLOPs": "8.7" if det_label == "YOLOv8n" else "6.7",
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
                "Det_Latency (ms)": round(det_ms, 1),
                "Tracker_FPS": round(tracker_fps, 1),
                "End_to_End_FPS": round(e2e_fps, 1),
            }
            summary_rows.append(res_row)
            print(f"  -> HOTA: {hota:.2f}% | IDF1: {idf1:.2f}% | MOTA: {mota:.2f}% | IDSW: {idsw} | Pred Tracks: {unique_tracks}")

    df_summary = pd.DataFrame(summary_rows)
    csv_out = output_dir / "comparison_summary.csv"
    df_summary.to_csv(csv_out, index=False)

    print(f"\n{'='*70}")
    print("BẢNG TỔNG HỢP SO SÁNH YOLOV8N VS YOLO11N:")
    print(f"{'='*70}")
    print(df_summary.to_string(index=False))
    print(f"\n[OK] Đã lưu kết quả tại: {csv_out}")


if __name__ == "__main__":
    main()
