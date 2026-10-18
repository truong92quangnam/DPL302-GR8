"""Tổng hợp kết quả thực nghiệm thành bảng so sánh và biểu đồ khoa học.

Đầu ra:
- comparison_table.csv: Bảng tổng hợp HOTA, IDF1, MOTA, IDSW, Tracker FPS, Pipeline FPS.
- hota_comparison.png: Biểu đồ cột so sánh HOTA.
- mota_idf1_comparison.png: Biểu đồ cột so sánh MOTA và IDF1.
- fps_comparison.png: Biểu đồ so sánh tốc độ (Tracker-only FPS vs End-to-End FPS).
- tracking_duration_distribution.png: Phân bố vòng đời track (Predicted Track Lifespan).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def collect_experiment_data(results_dir: Path) -> pd.DataFrame:
    """Thu thập kết quả từ evaluation_summary.csv và các summary.json."""
    eval_csv = results_dir / "evaluation_summary.csv"
    if not eval_csv.is_file():
        raise FileNotFoundError(f"Không tìm thấy file đánh giá: {eval_csv}. Vui lòng chạy evaluate trước.")

    df_eval = pd.read_csv(eval_csv)

    # Thu thập thêm thông số timing từ summary.json
    timing_map: dict[tuple[str, str], dict[str, float]] = {}
    for summary_path in results_dir.rglob("summary.json"):
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            mode = data.get("mode")
            seq = data.get("sequence")
            timing = data.get("timing", {})
            trk_fps = timing.get("tracker", {}).get("tracker_only_fps", 0.0)
            e2e_fps = timing.get("end_to_end_pipeline", {}).get("end_to_end_fps", 0.0)
            timing_map[(mode, seq)] = {
                "tracker_fps": trk_fps,
                "pipeline_fps": e2e_fps,
            }
        except Exception:
            continue

    tracker_fps_list = []
    pipeline_fps_list = []
    for _, row in df_eval.iterrows():
        key = (row["mode"], row["sequence"])
        t = timing_map.get(key, {"tracker_fps": np.nan, "pipeline_fps": np.nan})
        tracker_fps_list.append(t["tracker_fps"])
        pipeline_fps_list.append(t["pipeline_fps"])

    df_eval["tracker_only_fps"] = tracker_fps_list
    df_eval["end_to_end_fps"] = pipeline_fps_list
    return df_eval


def create_comparison(results_dir: Path) -> Path:
    """Tạo bảng so sánh và các biểu đồ phân tích."""
    table = collect_experiment_data(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Lưu bảng so sánh tổng hợp
    comp_csv = results_dir / "comparison_table.csv"
    table.to_csv(comp_csv, index=False)
    print(f"\n[OK] Đã lưu bảng so sánh: {comp_csv}")

    # Vẽ biểu đồ HOTA
    if "HOTA" in table.columns:
        _plot_grouped_bar(table, "HOTA", "HOTA Score (%)", results_dir / "hota_comparison.png")

    # Vẽ biểu đồ MOTA & IDF1
    if "MOTA" in table.columns and "IDF1" in table.columns:
        _plot_mota_idf1(table, results_dir / "mota_idf1_comparison.png")

    # Vẽ biểu đồ FPS
    if "tracker_only_fps" in table.columns:
        _plot_fps(table, results_dir / "fps_comparison.png")

    # Vẽ phân bố vòng đời track (Predicted Track Lifespan)
    _plot_lifespan_distribution(results_dir, results_dir / "tracking_duration_distribution.png")

    print(f"[OK] Toàn bộ biểu đồ so sánh đã xuất ra: {results_dir}")
    return comp_csv


def _plot_grouped_bar(df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    """Vẽ bar chart theo mode và sequence."""
    sequences = df["sequence"].unique()
    modes = df["mode"].unique()

    x = np.arange(len(sequences))
    width = 0.35

    plt.figure(figsize=(max(8, len(sequences) * 2), 5))
    for i, mode in enumerate(modes):
        mode_vals = []
        for seq in sequences:
            sub = df[(df["mode"] == mode) & (df["sequence"] == seq)]
            val = sub[metric].values[0] if len(sub) > 0 and pd.notna(sub[metric].values[0]) else 0.0
            mode_vals.append(val)
        plt.bar(x + i * width, mode_vals, width, label=mode.capitalize())

    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"Comparison of {metric} across Sequences", fontsize=14)
    plt.xticks(x + width / 2, sequences, rotation=15)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_mota_idf1(df: pd.DataFrame, output_path: Path) -> None:
    """So sánh MOTA và IDF1 cho các sequence."""
    modes = df["mode"].unique()
    plt.figure(figsize=(10, 5))
    for mode in modes:
        sub = df[df["mode"] == mode]
        plt.scatter(sub["MOTA"], sub["IDF1"], label=mode.capitalize(), s=100, alpha=0.8)
        for _, r in sub.iterrows():
            plt.annotate(r["sequence"].replace("MOT17-", ""), (r["MOTA"], r["IDF1"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    plt.xlabel("MOTA (%)", fontsize=12)
    plt.ylabel("IDF1 (%)", fontsize=12)
    plt.title("MOTA vs. IDF1 Trade-off", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_fps(df: pd.DataFrame, output_path: Path) -> None:
    """Vẽ biểu đồ tốc độ FPS (Tracker-only)."""
    sequences = df["sequence"].unique()
    modes = df["mode"].unique()

    x = np.arange(len(sequences))
    width = 0.35

    plt.figure(figsize=(max(8, len(sequences) * 2), 5))
    for i, mode in enumerate(modes):
        fps_vals = []
        for seq in sequences:
            sub = df[(df["mode"] == mode) & (df["sequence"] == seq)]
            val = sub["tracker_only_fps"].values[0] if len(sub) > 0 and pd.notna(sub["tracker_only_fps"].values[0]) else 0.0
            fps_vals.append(val)
        plt.bar(x + i * width, fps_vals, width, label=f"{mode.capitalize()} (Tracker FPS)")

    plt.ylabel("Tracker Processing Speed (FPS)", fontsize=12)
    plt.title("Tracker Processing Speed (cached_tracking)", fontsize=14)
    plt.xticks(x + width / 2, sequences, rotation=15)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_lifespan_distribution(results_dir: Path, output_path: Path) -> None:
    """Vẽ boxplot phân bố vòng đời track (Predicted Track Lifespan)."""
    lifespan_files = list(results_dir.rglob("predicted_track_lifespan.csv"))
    if not lifespan_files:
        return

    data = []
    labels = []
    for lf in sorted(lifespan_files):
        mode = lf.parent.parent.name
        seq = lf.parent.name
        try:
            df_life = pd.read_csv(lf)
            if "span_seconds" in df_life and len(df_life) > 0:
                data.append(df_life["span_seconds"].values)
                labels.append(f"{mode[:4]}_{seq[-8:]}")
        except Exception:
            continue

    if data:
        plt.figure(figsize=(max(10, len(data) * 1.2), 5))
        plt.boxplot(data, tick_labels=labels)
        plt.ylabel("Span Duration (seconds)", fontsize=12)
        plt.title("Predicted Track Lifespan Distribution", fontsize=14)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
