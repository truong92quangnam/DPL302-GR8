"""Deep GA Evolution: Tối ưu hóa Genetic Algorithm quy mô sâu và vẽ biểu đồ hội tụ.

Mở rộng:
- Quần thể: 15 cá thể
- Thế hệ: 12 thế hệ (180 lượt đánh giá đa mục tiêu)
- Xuất biểu đồ hội tụ: deep_ga_convergence.png
- Lưu cấu hình tối ưu: configs/deep_ga_bytetrack.yaml
"""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.bytetrack_adapter import AdaptiveByteTracker, DetectionResults, xyxy_to_xywh
from src.detection_cache import load_cached_detections
from src.genetic_optimizer import Candidate, GeneticOptimizer
from src.metrics import evaluate_with_trackeval, tracking_csv_to_mot
from src.tracking_logger import write_csv


def run_deep_ga_evolution(
    config_path: Path = Path("configs/experiment.yaml"),
    splits_path: Path = Path("configs/splits.yaml"),
    population_size: int = 15,
    generations: int = 12,
    seed: int = 100,
    output_dir: Path = Path("outputs/experiments/deep_ga"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with splits_path.open("r", encoding="utf-8") as f:
        splits_config = yaml.safe_load(f)

    dataset_root = Path(config["dataset"]["root_dir"]).resolve()
    cache_dir = Path(config["cache"]["cache_dir"]).resolve()
    dev_names = splits_config["splits"]["development"]["sequences"]

    dev_sequences = []
    for name in dev_names:
        seq_path = dataset_root / "train" / name
        gt_path = seq_path / "gt" / "gt.txt"
        dev_sequences.append((name, seq_path, gt_path))

    print(f"\n{'='*60}")
    print(f"DEEP GA EVOLUTION: Population = {population_size} | Generations = {generations}")
    print(f"Không gian đánh giá: Development Split ({len(dev_sequences)} sequences)")
    print(f"{'='*60}")

    def deep_evaluator(params: dict[str, Any], eval_id: str) -> dict[str, float]:
        candidate_cfg = {**config["bytetrack"], **params}
        tracker = AdaptiveByteTracker(candidate_cfg)
        total_hota, total_idf1, total_mota = 0.0, 0.0, 0.0
        total_frames = 0
        t0 = time.perf_counter()

        for seq_name, seq_path, gt_path in dev_sequences:
            try:
                dets_by_frame, _, _ = load_cached_detections(cache_dir, seq_name)
            except FileNotFoundError:
                continue

            tracker.reset_state()
            fids = sorted(dets_by_frame.keys())
            rows = []
            for fid in fids:
                raw_dets = dets_by_frame[fid]
                if len(raw_dets) > 0:
                    xywh = xyxy_to_xywh(raw_dets[:, :4])
                    confs = raw_dets[:, 4]
                    clss = raw_dets[:, 5].astype(int)
                else:
                    xywh = np.empty((0, 4), dtype=np.float32)
                    confs = np.empty((0,), dtype=np.float32)
                    clss = np.empty((0,), dtype=np.int32)

                dets = DetectionResults(xywh, confs, clss)
                tracks, _, _ = tracker.update_with_threshold(dets)
                for trk in tracks:
                    rows.append({
                        "frame_id": fid,
                        "track_id": int(trk[4]),
                        "x1": float(trk[0]),
                        "y1": float(trk[1]),
                        "x2": float(trk[2]),
                        "y2": float(trk[3]),
                        "confidence": float(trk[5]),
                        "class_id": int(trk[6]),
                    })

            total_frames += len(fids)
            cand_dir = output_dir / "tmp_eval" / eval_id / seq_name
            mot_txt = cand_dir / "pred.txt"
            tracking_csv = cand_dir / "track.csv"
            write_csv(rows, tracking_csv)
            tracking_csv_to_mot(tracking_csv, mot_txt)

            ini_path = seq_path / "seqinfo.ini"
            fps = 30.0
            width, height = 1920, 1080
            if ini_path.is_file():
                cp = configparser.ConfigParser()
                cp.read(ini_path)
                if "Sequence" in cp:
                    fps = float(cp["Sequence"].get("frameRate", 30.0))
                    width = int(cp["Sequence"].get("imWidth", 1920))
                    height = int(cp["Sequence"].get("imHeight", 1080))

            if gt_path.is_file():
                try:
                    te = evaluate_with_trackeval(
                        prediction_file=mot_txt,
                        gt_file=gt_path,
                        sequence_name=seq_name,
                        work_dir=cand_dir,
                        frame_count=len(fids),
                        width=width,
                        height=height,
                        fps=fps,
                    )
                    total_hota += te["HOTA"]
                    total_idf1 += te["IDF1"]
                    total_mota += te["MOTA"]
                except Exception:
                    pass

        elapsed = max(time.perf_counter() - t0, 1e-4)
        n_seq = max(len(dev_sequences), 1)
        shutil.rmtree(output_dir / "tmp_eval" / eval_id, ignore_errors=True)

        return {
            "HOTA": total_hota / n_seq,
            "IDF1": total_idf1 / n_seq,
            "MOTA": total_mota / n_seq,
            "FPS": total_frames / elapsed,
        }

    optimizer = GeneticOptimizer(
        evaluator=deep_evaluator,
        population_size=population_size,
        generations=generations,
        crossover_rate=0.85,
        mutation_rate=0.25,
        elitism=2,
        seed=seed,
    )

    ga_config_output = Path("configs/deep_ga_bytetrack.yaml")
    best = optimizer.optimize(output_dir=output_dir, config_output=ga_config_output, base_config=config["bytetrack"])

    # Phân tích đường cong hội tụ
    history_df = pd.DataFrame(optimizer.history)
    history_csv = output_dir / f"deep_ga_history_seed{seed}.csv"
    history_df.to_csv(history_csv, index=False)

    gen_stats = history_df.groupby("generation")["fitness"].agg(["max", "mean", "std"]).reset_index()

    plt.figure(figsize=(9, 5))
    plt.plot(gen_stats["generation"], gen_stats["max"], "o-", color="#d95f02", label="Best Fitness (Elitism)", linewidth=2.5)
    plt.plot(gen_stats["generation"], gen_stats["mean"], "s--", color="#1b9e77", label="Population Mean Fitness", linewidth=2.0)
    plt.fill_between(
        gen_stats["generation"],
        gen_stats["mean"] - gen_stats["std"],
        gen_stats["mean"] + gen_stats["std"],
        color="#1b9e77",
        alpha=0.2,
        label="Mean ± Std",
    )
    plt.xlabel("Generation (Epoch)", fontsize=12)
    plt.ylabel("Multi-Objective Fitness", fontsize=12)
    plt.title("Deep Genetic Algorithm Convergence Curve (MOT17 Development Split)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plot_path = output_dir / "deep_ga_convergence.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"\n[OK] Đã vẽ biểu đồ hội tụ: {plot_path}")
    print(f"Cấu hình tối ưu lưu tại: {ga_config_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=15)
    parser.add_argument("--gen", type=int, default=12)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    run_deep_ga_evolution(population_size=args.pop, generations=args.gen, seed=args.seed)

