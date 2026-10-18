"""Benchmark GPU Stress Test: So sánh YOLOv8n, YOLOv8s, YOLOv8m End-to-End trên RTX 3050.

Đo lường:
- Peak VRAM allocated / reserved (MB)
- Detector latency (ms), Tracker latency (ms)
- End-to-End FPS
- Tracking metrics: HOTA, MOTA, IDF1, IDSW
"""

from __future__ import annotations

import argparse
import configparser
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from src.bytetrack_adapter import AdaptiveByteTracker, DetectionResults, xyxy_to_xywh
from src.metrics import evaluate_with_trackeval, tracking_csv_to_mot, validate_mot_file
from src.tracking_logger import write_csv


def run_gpu_stress_test(
    sequence_dir: Path,
    models: list[str],
    imgsz: int = 1280,
    conf_thresh: float = 0.01,
    iou_thresh: float = 0.7,
    output_dir: Path = Path("outputs/experiments/gpu_stress"),
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Đọc thông số sequence
    ini_path = sequence_dir / "seqinfo.ini"
    cp = configparser.ConfigParser()
    cp.read(ini_path)
    fps = float(cp["Sequence"].get("frameRate", 30.0))
    width = int(cp["Sequence"].get("imWidth", 1920))
    height = int(cp["Sequence"].get("imHeight", 1080))
    seq_name = sequence_dir.name
    gt_path = sequence_dir / "gt" / "gt.txt"

    # Lấy danh sách ảnh
    img_dir = sequence_dir / "img1"
    img_files = sorted(img_dir.glob("*.jpg"))
    total_frames = len(img_files)
    print(f"\n[GPU STRESS TEST] Sequence: {seq_name} ({total_frames} frames, {width}x{height} @ {fps} FPS)")

    tracker_config = {
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.80,
        "fuse_score": True,
    }

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"ÉP TẢI GPU: Model = {model_name} | ImgSz = {imgsz} | Device = CUDA")
        print(f"{'='*60}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t_model_start = time.perf_counter()
        model = YOLO(model_name)
        model.to("cuda")

        tracker = AdaptiveByteTracker(tracker_config)
        tracker.reset_state()

        detector_times = []
        tracker_times = []
        frame_rows = []

        # Warmup GPU
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model.predict(dummy, device=0, verbose=False)
        torch.cuda.synchronize()

        t_loop_start = time.perf_counter()

        for idx, img_file in enumerate(img_files):
            # 1. Đo Detector trên GPU với CUDA synchronize
            t_det0 = time.perf_counter()
            preds = model.predict(
                str(img_file),
                imgsz=imgsz,
                conf=conf_thresh,
                iou=iou_thresh,
                classes=[0],  # Person
                device=0,
                verbose=False,
            )[0]
            torch.cuda.synchronize()
            t_det_ms = (time.perf_counter() - t_det0) * 1000.0
            if idx >= 5:  # Bỏ warmup 5 frames
                detector_times.append(t_det_ms)

            # Trích xuất detections
            boxes = preds.boxes
            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                xywh = xyxy_to_xywh(xyxy)
            else:
                xywh = np.empty((0, 4), dtype=np.float32)
                confs = np.empty((0,), dtype=np.float32)
                clss = np.empty((0,), dtype=np.int32)

            det_results = DetectionResults(xywh, confs, clss)

            # 2. Đo Tracker
            t_trk0 = time.perf_counter()
            tracks, actual_thresh, _ = tracker.update_with_threshold(det_results)
            t_trk_ms = (time.perf_counter() - t_trk0) * 1000.0
            if idx >= 5:
                tracker_times.append(t_trk_ms)

            fid = idx + 1
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

            if (idx + 1) % 150 == 0 or (idx + 1) == total_frames:
                vram_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
                print(f"  Frame {idx+1}/{total_frames} | Det: {t_det_ms:.1f} ms | Trk: {t_trk_ms:.2f} ms | Max VRAM: {vram_alloc:.1f} MB")

        total_loop_time = time.perf_counter() - t_loop_start
        e2e_fps = total_frames / total_loop_time

        peak_vram_alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_vram_res_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

        # Lưu predictions và đánh giá TrackEval
        model_out = output_dir / model_name.replace(".pt", "")
        model_out.mkdir(parents=True, exist_ok=True)
        tracking_csv = model_out / "track.csv"
        mot_txt = model_out / "pred_mot.txt"
        write_csv(frame_rows, tracking_csv)
        tracking_csv_to_mot(tracking_csv, mot_txt)

        te_metrics = {"HOTA": 0.0, "IDF1": 0.0, "MOTA": 0.0, "ID_switches": 0}
        if gt_path.is_file():
            try:
                te_metrics = evaluate_with_trackeval(
                    prediction_file=mot_txt,
                    gt_file=gt_path,
                    sequence_name=seq_name,
                    work_dir=model_out / "eval_tmp",
                    frame_count=total_frames,
                    width=width,
                    height=height,
                    fps=fps,
                )
            except Exception as e:
                print(f"TrackEval error: {e}")

        det_mean_ms = float(np.mean(detector_times))
        trk_mean_ms = float(np.mean(tracker_times))

        row = {
            "model": model_name,
            "imgsz": imgsz,
            "total_frames": total_frames,
            "peak_vram_allocated_mb": round(peak_vram_alloc_mb, 1),
            "peak_vram_reserved_mb": round(peak_vram_res_mb, 1),
            "vram_utilization_pct": round(peak_vram_res_mb / 4096.0 * 100, 1),
            "detector_mean_ms": round(det_mean_ms, 2),
            "tracker_mean_ms": round(trk_mean_ms, 2),
            "end_to_end_fps": round(e2e_fps, 1),
            "total_detections": len(frame_rows),
            "HOTA": round(te_metrics["HOTA"] * 100, 2),
            "IDF1": round(te_metrics["IDF1"] * 100, 2),
            "MOTA": round(te_metrics["MOTA"] * 100, 2),
            "IDSW": te_metrics["ID_switches"],
        }
        results.append(row)
        print(f"\n[KẾT QUẢ {model_name}]: Max VRAM: {peak_vram_res_mb:.1f} MB ({row['vram_utilization_pct']}%) | "
              f"Det Latency: {det_mean_ms:.1f} ms | E2E FPS: {e2e_fps:.1f} | "
              f"HOTA: {row['HOTA']}% | MOTA: {row['MOTA']}% | IDF1: {row['IDF1']}% | IDSW: {row['IDSW']}")

    df_stress = pd.DataFrame(results)
    csv_out = output_dir / "gpu_stress_benchmark.csv"
    df_stress.to_csv(csv_out, index=False)
    print(f"\n[OK] Đã xuất bảng benchmark GPU: {csv_out}")
    print(df_stress.to_string())
    return df_stress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Stress Benchmark")
    parser.add_argument("--seq-dir", default="data/MOT17/train/MOT17-02-FRCNN")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    seq_path = Path(args.seq_dir).resolve()
    models_to_test = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    run_gpu_stress_test(seq_path, models_to_test, imgsz=args.imgsz)

