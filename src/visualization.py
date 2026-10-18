"""Tổng hợp summary JSON thành bảng và biểu đồ so sánh."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def collect_summaries(results_dir: Path) -> pd.DataFrame:
    """Đọc JSON kết quả có schema run, bỏ qua artifact không phải lần đánh giá."""
    summaries: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("*.json")):
        # Tuning trials thuộc validation, không phải run đánh giá độc lập để so sánh.
        if "ga_candidates" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"[warning] Bỏ qua {path}: {error}")
            continue
        if isinstance(payload, dict) and "run_name" in payload and any(
            metric in payload for metric in ("HOTA", "IDF1", "MOTA", "fps")
        ):
            summaries.append({**payload, "summary_file": str(path)})
    return pd.DataFrame(summaries)


def create_comparison(results_dir: Path) -> Path:
    """Ghi CSV so sánh và bốn biểu đồ yêu cầu."""
    table = collect_summaries(results_dir)
    if table.empty:
        raise ValueError(f"Không có summary.json hợp lệ trong {results_dir}; chưa có kết quả để so sánh.")
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "comparison_table.csv", index=False)
    _bar_plot(table, "HOTA", "HOTA", results_dir / "hota_comparison.png")
    _bar_plot(table, "ID_switches", "ID switches", results_dir / "id_switch_comparison.png")
    _bar_plot(table, "fps", "FPS", results_dir / "fps_comparison.png")

    duration_rows: list[dict[str, Any]] = []
    for _, run in table.iterrows():
        duration_file = run.get("track_durations")
        if isinstance(duration_file, str) and Path(duration_file).is_file():
            durations = pd.read_csv(duration_file)
            if "duration_seconds" in durations:
                duration_rows.extend(
                    {"run_name": run["run_name"], "duration_seconds": value}
                    for value in durations["duration_seconds"].dropna().tolist()
                )
    duration_plot = results_dir / "tracking_duration_distribution.png"
    if duration_rows:
        duration_frame = pd.DataFrame(duration_rows)
        runs = duration_frame["run_name"].unique()
        plt.figure(figsize=(max(8, len(runs) * 1.5), 5))
        plt.boxplot(
            [duration_frame.loc[duration_frame["run_name"] == run, "duration_seconds"] for run in runs],
        )
        plt.ylabel("Track duration (seconds)")
        plt.xticks(range(1, len(runs) + 1), runs, rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(duration_plot, dpi=160)
        plt.close()
    else:
        _empty_plot(duration_plot, "No track duration data")
    print(f"[compare] Bảng và biểu đồ đã ghi vào {results_dir}")
    return results_dir / "comparison_table.csv"


def _bar_plot(table: pd.DataFrame, column: str, ylabel: str, output: Path) -> None:
    """Vẽ metric nếu có dữ liệu; NaN được giữ thành missing."""
    if column not in table:
        _empty_plot(output, f"No {column} data")
        return
    values = pd.to_numeric(table[column], errors="coerce")
    plt.figure(figsize=(max(8, len(table) * 1.2), 5))
    plt.bar(table["run_name"].astype(str), values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _empty_plot(path: Path, message: str) -> None:
    """Tạo ảnh có thông báo thay vì giả lập metric."""
    plt.figure(figsize=(7, 3))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
