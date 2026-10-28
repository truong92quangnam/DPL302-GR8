"""Benchmark Toàn Diện Toàn Bộ Dataset MOT17 (7 Sequences) cho YOLO11n.

Mô hình thực hiện:
1. Trích xuất & Cache Detections cho cả 7 sequence bằng yolo11n.pt (imgsz=1280, conf=0.01, iou=0.7).
2. Chạy 3 biến thể Tracker trên từng sequence:
   - Baseline ByteTrack (default static thresholds)
   - GA-ByteTrack (Tailored) - bộ gen tiến hóa riêng cho YOLO11n (configs/yolo11_ga_bytetrack.yaml)
   - Fuzzy-Adaptive ByteTrack (Mamdani dynamic controller)
3. Đánh giá chuẩn bằng TrackEval đối chiếu với ground truth chính thức của MOTChallenge.
4. Tính toán giá trị Trung bình Macro và Tổng toàn bộ dataset giống như bài báo gốc ByteTrack (ECCV 2022).
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.bytetrack_adapter import AdaptiveByteTracker, DetectionResults, xyxy_to_xywh
from src.fuzzy_controller import FuzzyByteTrackController, fuzzy_adjustment_dict
from src.metrics import evaluate_with_trackeval, tracking_csv_to_mot, validate_mot_file
from src.tracking_logger import write_csv


ALL_SEQUENCES = [
    ("MOT17-02-FRCNN", "development"),
    ("MOT17-04-FRCNN", "local_held_out"),
    ("MOT17-05-FRCNN", "development"),
    ("MOT17-09-FRCNN", "development"),
    ("MOT17-10-FRCNN", "development"),
    ("MOT17-11-FRCNN", "local_held_out"),
    ("MOT17-13-FRCNN", "local_held_out"),
]


def parse_seqinfo(seq_dir: Path) -> dict[str, Any]:
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"Thiếu seqinfo.ini: {ini_path}")
    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    sec = parser["Sequence"]
    return {
        "name": sec.get("name", seq_dir.name),
        "seq_length": int(sec.get("seqLength", "0")),
        "width": int(sec.get("imWidth", "1920")),
        "height": int(sec.get("imHeight", "1080")),
        "frame_rate": float(sec.get("frameRate", "30")),
    }


def cache_detections_for_sequence(
    seq_name: str,
    seq_dir: Path,
    cache_dir: Path,
    model_path: str = "yolo11n.pt",
    imgsz: int = 1280,
    conf_thresh: float = 0.01,
    iou_thresh: float = 0.7,
    device: str = "cpu",
) -> tuple[dict[int, np.ndarray], float]:
    """Trích xuất và lưu cache detections dạng npz + json."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_file = cache_dir / f"{seq_name}_detections.npz"
    meta_file = cache_dir / f"{seq_name}_meta.json"

    # Nếu cache đã tồn tại thì nạp ngay
    if npz_file.is_file() and meta_file.is_file():
        try:
            with meta_file.open(encoding="utf-8") as f:
                meta = json.load(f)
            loaded = np.load(npz_file)
            dets_by_frame = {int(k.replace("frame_", "")): loaded[k] for k in loaded.files}
            avg_lat = meta.get("avg_detector_latency_ms", 110.0)
            print(f"[{seq_name}] Đã nạp từ cache có sẵn ({len(dets_by_frame)} frames, avg det: {avg_lat:.1f}ms)")
            return dets_by_frame, avg_lat
        except Exception as e:
            print(f"[{seq_name}] Lỗi đọc cache: {e}, tiến hành trích xuất lại...")

    img_dir = seq_dir / "img1"
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    total_frames = len(img_files)
    print(f"\n[{seq_name}] Bắt đầu trích xuất detections YOLO11n ({total_frames} frames)...")

    model = YOLO(model_path)
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

        fid = idx + 1
        boxes = pred.boxes
        if len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            arr = np.column_stack([xyxy, confs, clss]).astype(np.float32)
        else:
            arr = np.empty((0, 6), dtype=np.float32)
        dets_by_frame[fid] = arr

        if (idx + 1) % 150 == 0 or (idx + 1) == total_frames:
            print(f"  [{seq_name}] Frame {idx + 1}/{total_frames} | Latency: {np.mean(det_times):.1f} ms/frame")

    total_time = time.perf_counter() - t0_all
    avg_det_ms = float(np.mean(det_times))

    # Lưu npz và meta
    save_dict = {f"frame_{fid}": arr for fid, arr in dets_by_frame.items()}
    np.savez_compressed(npz_file, **save_dict)
    meta = {
        "sequence": seq_name,
        "model": model_path,
        "imgsz": imgsz,
        "conf": conf_thresh,
        "iou": iou_thresh,
        "total_frames": total_frames,
        "avg_detector_latency_ms": avg_det_ms,
        "total_time_sec": total_time,
    }
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[{seq_name}] Hoàn tất trích xuất và lưu cache tại {npz_file} ({total_time:.1f}s)")
    return dets_by_frame, avg_det_ms


def run_tracker_on_dets(
    dets_by_frame: dict[int, np.ndarray],
    tracker_config: dict[str, Any],
    mode: str,
) -> tuple[list[dict[str, Any]], float]:
    """Chạy tracker trên danh sách detections."""
    tracker = AdaptiveByteTracker(tracker_config)
    tracker.reset_state()
    fuzzy_controller = FuzzyByteTrackController() if mode == "fuzzy" else None

    frame_rows: list[dict[str, Any]] = []
    trk_times: list[float] = []

    sorted_frames = sorted(dets_by_frame.keys())
    for fid in sorted_frames:
        det_arr = dets_by_frame[fid]
        if len(det_arr) > 0:
            bboxes_xyxy = det_arr[:, :4]
            confidences = det_arr[:, 4]
            class_ids = det_arr[:, 5].astype(int)
            bboxes_xywh = xyxy_to_xywh(bboxes_xyxy)
        else:
            bboxes_xywh = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=int)

        dynamic_high_thresh = None
        if mode == "fuzzy" and fuzzy_controller is not None:
            prior_unmatched = tracker.prior_unmatched_ratio
            if len(confidences) > 0:
                low_ratio = float(np.mean(confidences < float(tracker_config["track_high_thresh"])))
                spread = float(np.std(confidences))
            else:
                low_ratio = 0.0
                spread = 0.0
            high_base = float(tracker_config["track_high_thresh"])
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

        det_results = DetectionResults(
            xywh=bboxes_xywh,
            conf=confidences,
            cls=class_ids,
        )

        t0 = time.perf_counter()
        tracks, actual_thresh, _ = tracker.update_with_threshold(det_results, dynamic_high_thresh=dynamic_high_thresh)
        dt = (time.perf_counter() - t0) * 1000.0
        trk_times.append(dt)

        if len(tracks) > 0:
            for trk in tracks:
                frame_rows.append({
                    "frame_id": int(fid),
                    "track_id": int(trk[4]),
                    "x1": float(trk[0]),
                    "y1": float(trk[1]),
                    "x2": float(trk[2]),
                    "y2": float(trk[3]),
                    "confidence": float(trk[5]),
                    "class_id": int(trk[6]),
                    "active_high_thresh": float(actual_thresh),
                })

    avg_trk_ms = float(np.mean(trk_times)) if trk_times else 0.5
    return frame_rows, avg_trk_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="Thiết bị chạy detector (cpu hoặc 0)")
    parser.add_argument("--data-root", default="data/MOT17/train", help="Thư mục chứa MOT17 train")
    parser.add_argument("--output-dir", default="outputs/experiments/yolo11_full_benchmark", help="Thư mục xuất kết quả")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path("data/cache/detections_yolo11n")

    # 1. Cấu hình các biến thể Tracker
    baseline_cfg = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.60,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.70,
        "track_buffer": 30,
        "match_thresh": 0.80,
        "fuse_score": True,
    }

    ga_yaml = Path("configs/yolo11_ga_bytetrack.yaml")
    if ga_yaml.is_file():
        with ga_yaml.open(encoding="utf-8") as f:
            ga_cfg = yaml.safe_load(f)
        print(f"[Cấu hình GA YOLO11n]: {ga_cfg}")
    else:
        ga_cfg = baseline_cfg.copy()

    fuzzy_cfg = baseline_cfg.copy()

    tracker_modes = {
        "baseline": (baseline_cfg, "1. Baseline ByteTrack"),
        "ga_bytetrack": (ga_cfg, "2. GA-ByteTrack (Tailored)"),
        "fuzzy": (fuzzy_cfg, "3. Fuzzy-Adaptive"),
    }

    print("\n" + "=" * 80)
    print("BENCHMARK TOÀN DIỆN YOLO11n TRÊN TOÀN BỘ 7 SEQUENCES MOT17")
    print(f"Tổng số chuỗi: {len(ALL_SEQUENCES)} | Detector: yolo11n.pt | Device: {args.device}")
    print("=" * 80)

    all_results: list[dict[str, Any]] = []

    # 2. Vòng lặp qua từng Sequence
    for seq_name, split_type in ALL_SEQUENCES:
        seq_dir = data_root / seq_name
        if not seq_dir.is_dir():
            print(f"[CẢNH BÁO] Không tìm thấy {seq_dir}, bỏ qua!")
            continue

        seq_info = parse_seqinfo(seq_dir)
        gt_path = seq_dir / "gt" / "gt.txt"
        total_frames = seq_info["seq_length"]
        width = seq_info["width"]
        height = seq_info["height"]
        fps = seq_info["frame_rate"]

        # Trích xuất / nạp detections
        dets_by_frame, avg_det_ms = cache_detections_for_sequence(
            seq_name=seq_name,
            seq_dir=seq_dir,
            cache_dir=cache_dir,
            model_path="yolo11n.pt",
            imgsz=1280,
            conf_thresh=0.01,
            iou_thresh=0.7,
            device=args.device,
        )

        # Đọc thông số Ground Truth
        gt_boxes, gt_tracks = 0, 0
        if gt_path.is_file():
            gt_df = pd.read_csv(gt_path, header=None)
            gt_boxes = len(gt_df)
            gt_tracks = gt_df[1].nunique()

        # Chạy từng biến thể Tracker
        for mode_key, (trk_cfg, mode_title) in tracker_modes.items():
            print(f"  -> Đang chạy {seq_name} | {mode_title}...")
            frame_rows, avg_trk_ms = run_tracker_on_dets(dets_by_frame, trk_cfg, mode_key)

            run_id = f"{seq_name}_{mode_key}"
            run_out = output_dir / run_id
            run_out.mkdir(parents=True, exist_ok=True)

            csv_path = run_out / "track.csv"
            write_csv(frame_rows, csv_path)
            mot_path = run_out / "pred_mot.txt"
            tracking_csv_to_mot(csv_path, mot_path)
            validate_mot_file(mot_path, allow_empty=True)

            # Đánh giá bằng TrackEval
            hota, idf1, mota, idsw = 0.0, 0.0, 0.0, 0
            if gt_path.is_file():
                try:
                    eval_res = evaluate_with_trackeval(
                        prediction_file=mot_path,
                        gt_file=gt_path,
                        sequence_name=seq_name,
                        work_dir=run_out / "trackeval_tmp",
                        frame_count=total_frames,
                        width=width,
                        height=height,
                        fps=fps,
                    )
                    hota = float(eval_res.get("HOTA", 0.0)) * 100.0
                    idf1 = float(eval_res.get("IDF1", 0.0)) * 100.0
                    mota = float(eval_res.get("MOTA", 0.0)) * 100.0
                    idsw = int(eval_res.get("ID_switches", 0))
                except Exception as e:
                    print(f"     [Lỗi TrackEval {seq_name}]: {e}")

            pred_boxes = len(frame_rows)
            pred_tracks = len(set(r["track_id"] for r in frame_rows)) if frame_rows else 0
            pipe_ms = avg_det_ms + avg_trk_ms
            e2e_fps = 1000.0 / pipe_ms if pipe_ms > 0 else 0.0
            trk_fps = 1000.0 / avg_trk_ms if avg_trk_ms > 0 else 0.0

            row = {
                "mode": mode_key,
                "sequence": seq_name,
                "split": split_type,
                "pred_boxes": pred_boxes,
                "pred_tracks": pred_tracks,
                "gt_boxes": gt_boxes,
                "gt_tracks": gt_tracks,
                "HOTA": round(hota, 2),
                "IDF1": round(idf1, 2),
                "MOTA": round(mota, 2),
                "IDSW": idsw,
                "tracker_only_fps": round(trk_fps, 1),
                "end_to_end_fps": round(e2e_fps, 1),
            }
            all_results.append(row)
            print(f"     [OK] HOTA: {hota:.2f}% | IDF1: {idf1:.2f}% | MOTA: {mota:.2f}% | IDSW: {idsw} | Tracks: {pred_tracks}")

    # 3. Xuất bảng chi tiết CSV
    df_all = pd.DataFrame(all_results)
    detail_csv = output_dir / "comparison_table_yolo11n.csv"
    df_all.to_csv(detail_csv, index=False)
    print(f"\n[OK] Đã lưu bảng chi tiết 21 lượt chạy YOLO11n tại: {detail_csv}")

    # 4. Tính toán Macro-Average và Grand Total cho cả dataset
    summary_rows = []
    mode_names = {
        "baseline": "Baseline ByteTrack",
        "ga_bytetrack": "GA-ByteTrack (Tailored)",
        "fuzzy": "Fuzzy-Adaptive",
    }

    print("\n" + "=" * 90)
    print("TỔNG HỢP TRUNG BÌNH TOÀN BỘ DATASET MOT17 (7 SEQUENCES) CHO YOLO11n:")
    print("=" * 90)

    for m_key in ["baseline", "ga_bytetrack", "fuzzy"]:
        sub = df_all[df_all["mode"] == m_key]
        if sub.empty:
            continue
        hota_mean = float(sub["HOTA"].mean())
        idf1_mean = float(sub["IDF1"].mean())
        mota_mean = float(sub["MOTA"].mean())
        idsw_sum = int(sub["IDSW"].sum())
        pred_tracks_sum = int(sub["pred_tracks"].sum())
        gt_tracks_sum = int(sub["gt_tracks"].sum())
        trk_fps_mean = float(sub["tracker_only_fps"].mean())
        e2e_fps_mean = float(sub["end_to_end_fps"].mean())

        s_row = {
            "Tracker": mode_names[m_key],
            "Mean_HOTA (%)": round(hota_mean, 2),
            "Mean_IDF1 (%)": round(idf1_mean, 2),
            "Mean_MOTA (%)": round(mota_mean, 2),
            "Total_IDSW": idsw_sum,
            "Total_Pred_Tracks": pred_tracks_sum,
            "Total_GT_Tracks": gt_tracks_sum,
            "Mean_Tracker_FPS": round(trk_fps_mean, 1),
            "Mean_Pipeline_FPS": round(e2e_fps_mean, 1),
        }
        summary_rows.append(s_row)
        print(f"[*] {mode_names[m_key]}:")
        print(f"    MOTA: {mota_mean:.2f}% | IDF1: {idf1_mean:.2f}% | HOTA: {hota_mean:.2f}% | Total IDSW: {idsw_sum} | Pred Tracks: {pred_tracks_sum}")

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "dataset_macro_average_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n[OK] Đã lưu bảng tổng kết dataset tại: {summary_csv}")


if __name__ == "__main__":
    main()

