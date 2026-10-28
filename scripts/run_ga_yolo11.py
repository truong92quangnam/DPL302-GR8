"""Chạy Genetic Algorithm để tối ưu hóa siêu tham số ByteTrack RIÊNG cho YOLO11n.

Khắc phục vấn đề "cross-model transfer":
- YOLO11n có kiến trúc C2PSA Attention và độ nhạy cao hơn ở dải confidence thấp.
- GA sẽ tự động tìm ra bộ gen tối ưu riêng cho đặc tính phân phối của YOLO11n:
  (track_high_thresh, track_low_thresh, new_track_thresh, track_buffer, match_thresh)
"""

from __future__ import annotations

import argparse
import configparser
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bytetrack_adapter import AdaptiveByteTracker, DetectionResults, xyxy_to_xywh
from src.genetic_optimizer import Candidate, GeneticOptimizer
from src.metrics import evaluate_with_trackeval, tracking_csv_to_mot, validate_mot_file
from src.tracking_logger import write_csv


def optimize_ga_for_yolo11(
    sequence: str = "MOT17-09-FRCNN",
    max_frames: int = 150,
    population_size: int = 12,
    generations: int = 8,
    seed: int = 42,
) -> tuple[Candidate, pd.DataFrame]:
    seq_dir = Path(f"data/MOT17/train/{sequence}")
    gt_path = seq_dir / "gt" / "gt.txt"
    ini_path = seq_dir / "seqinfo.ini"

    cp = configparser.ConfigParser()
    cp.read(ini_path)
    seq_fps = float(cp["Sequence"].get("frameRate", 30.0))
    seq_width = int(cp["Sequence"].get("imWidth", 1920))
    seq_height = int(cp["Sequence"].get("imHeight", 1080))

    output_dir = Path("outputs/experiments/yolo11_ga") / sequence
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nạp detections của YOLO11n đã cache hoặc trích xuất
    cache_csv = Path(f"outputs/experiments/yolo11_vs_yolov8/{sequence}/yolo11n_baseline/track.csv")
    img_files = sorted((seq_dir / "img1").glob("*.jpg"))[:max_frames]

    print(f"\n[GA for YOLO11n] Bắt đầu trích xuất detections bằng yolo11n.pt...")
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    
    dets_by_frame = {}
    for idx, img_p in enumerate(img_files, start=1):
        pred = model.predict(str(img_p), imgsz=1280, conf=0.01, iou=0.7, classes=[0], device="cpu", verbose=False)[0]
        boxes = pred.boxes
        if len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            dets_by_frame[idx] = np.column_stack([xyxy, confs, clss]).astype(np.float32)
        else:
            dets_by_frame[idx] = np.empty((0, 6), dtype=np.float32)

    # Tạo GT subset
    gt_df = pd.read_csv(gt_path, header=None)
    eval_gt_file = output_dir / f"gt_subset_{max_frames}.txt"
    gt_df[gt_df[0] <= max_frames].to_csv(eval_gt_file, header=False, index=False)

    print(f"[OK] Đã sẵn sàng {len(dets_by_frame)} frames detections của YOLO11n!")

    # 2. Định nghĩa hàm evaluator cho GA
    eval_tmp_dir = output_dir / "tmp_eval"

    def evaluate_chromosome(params: dict[str, float | int], eval_id: str) -> dict[str, float]:
        cfg = {
            "tracker_type": "bytetrack",
            "fuse_score": True,
            **params,
        }
        tracker = AdaptiveByteTracker(cfg)
        tracker.reset_state()

        frame_rows = []
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
            tracks, _, _ = tracker.update_with_threshold(det_results)

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

        run_dir = eval_tmp_dir / eval_id
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_p = run_dir / "track.csv"
        write_csv(frame_rows, csv_p)
        mot_p = run_dir / "pred_mot.txt"
        tracking_csv_to_mot(csv_p, mot_p)

        try:
            eval_res = evaluate_with_trackeval(
                prediction_file=mot_p,
                gt_file=eval_gt_file,
                sequence_name=sequence,
                work_dir=run_dir / "trackeval_tmp",
                frame_count=len(img_files),
                width=seq_width,
                height=seq_height,
                fps=seq_fps,
            )
            hota = float(eval_res.get("HOTA", 0.0) * 100.0)
            idf1 = float(eval_res.get("IDF1", 0.0) * 100.0)
            mota = float(eval_res.get("MOTA", 0.0) * 100.0)
            idsw = float(eval_res.get("ID_switches", 0))
        except Exception:
            hota, idf1, mota, idsw = 0.0, 0.0, 0.0, 999.0

        return {
            "HOTA": hota,
            "IDF1": idf1,
            "MOTA": mota,
            "IDSW": idsw,
            "FPS": 60.0,
        }

    # 3. Khởi tạo và chạy GA
    print(f"\n[TIẾN HÓA GA] Population: {population_size} | Generations: {generations} (Tổng {population_size * generations} evaluations)...")
    optimizer = GeneticOptimizer(
        evaluator=evaluate_chromosome,
        population_size=population_size,
        generations=generations,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elitism=2,
        seed=seed,
    )

    base_config = {"tracker_type": "bytetrack", "fuse_score": True}
    config_out = Path("configs/yolo11_ga_bytetrack.yaml")
    best_candidate = optimizer.optimize(output_dir, config_out, base_config)

    print(f"\n{'='*70}")
    print(f"🎉 GA HOÀN TẤT TỐI ƯU HÓA CHO YOLO11n!")
    print(f"{'='*70}")
    print(f"Bộ gen tối ưu: {json.dumps(best_candidate.parameters, indent=2)}")
    print(f"Metrics tối ưu: {best_candidate.metrics}")
    print(f"Đã lưu cấu hình tại: {config_out}")

    return best_candidate, pd.DataFrame(optimizer.history)


if __name__ == "__main__":
    optimize_ga_for_yolo11()

