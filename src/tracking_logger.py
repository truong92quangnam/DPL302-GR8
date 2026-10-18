"""Ghi log tracking, xử lý processing time, predicted track lifespan và identity continuity.

Tuân thủ nghiêm ngặt 3 định nghĩa:
A. Processing time:
   - Tách riêng detector, tracker, fuzzy controller, I/O.
   - Đơn vị milliseconds.
   - Báo cáo mean, median, p95, FPS riêng biệt cho tracker-only và end-to-end.
   - Tách thời gian warm-up khỏi số liệu chính.

B. Predicted track lifespan:
   - sequence, track_id, first_frame, last_frame
   - span_frames = last_frame - first_frame + 1
   - span_seconds = span_frames / sequence_fps
   - observed_frames, missing_frames_within_span
   - longest_consecutive_observed_frames, longest_consecutive_observed_seconds
   - mean_confidence
   (Không gọi span_seconds là thời gian tracking đúng).

C. Correct identity continuity:
   - Phân tích chuỗi liên tục ghép GT với predicted ID (khi có GT).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def write_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Ghi CSV kể cả khi rỗng để schema đầu ra luôn hiện diện."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def compute_longest_consecutive(frames: list[int]) -> int:
    """Tính chuỗi frame quan sát liên tiếp dài nhất."""
    if not frames:
        return 0
    unique_frames = sorted(set(frames))
    max_streak = 1
    current_streak = 1
    for i in range(1, len(unique_frames)):
        if unique_frames[i] == unique_frames[i - 1] + 1:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 1
    return max_streak


def summarize_predicted_track_lifespan(
    frame_rows: list[dict[str, Any]],
    sequence_name: str,
    sequence_fps: float,
) -> list[dict[str, Any]]:
    """Tổng hợp chi tiết Predicted track lifespan theo chuẩn Mục 6B.

    LƯU Ý: Không gọi span_seconds là thời gian tracking đúng; track tồn tại lâu
    chưa chắc bám đúng đối tượng.
    """
    observations: dict[int, list[dict[str, Any]]] = {}
    for row in frame_rows:
        tid = row.get("track_id")
        if tid is not None and pd.notna(tid) and int(tid) > 0:
            observations.setdefault(int(tid), []).append(row)

    summaries: list[dict[str, Any]] = []
    fps = float(sequence_fps) if sequence_fps > 0 else 30.0

    for track_id, rows in observations.items():
        frames = [int(r["frame_id"]) for r in rows]
        first_frame = min(frames)
        last_frame = max(frames)
        span_frames = last_frame - first_frame + 1
        span_seconds = span_frames / fps
        observed_frames = len(set(frames))
        missing_frames = max(span_frames - observed_frames, 0)
        longest_streak = compute_longest_consecutive(frames)
        longest_seconds = longest_streak / fps

        confs = [float(r["confidence"]) for r in rows if pd.notna(r.get("confidence"))]
        mean_conf = float(np.mean(confs)) if confs else 0.0

        summaries.append({
            "sequence": sequence_name,
            "track_id": track_id,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "span_frames": span_frames,
            "span_seconds": round(span_seconds, 4),
            "observed_frames": observed_frames,
            "missing_frames_within_span": missing_frames,
            "longest_consecutive_observed_frames": longest_streak,
            "longest_consecutive_observed_seconds": round(longest_seconds, 4),
            "mean_confidence": round(mean_conf, 4),
        })

    return sorted(summaries, key=lambda item: item["track_id"])


class ProcessingTimeTracker:
    """Bộ theo dõi thời gian thực thi (Processing time - Mục 6A)."""

    def __init__(self, warmup_frames: int = 5):
        self.warmup_frames = warmup_frames
        self.detector_times_ms: list[float] = []
        self.tracker_times_ms: list[float] = []
        self.fuzzy_times_ms: list[float] = []
        self.total_pipeline_times_ms: list[float] = []
        self.frame_count = 0

    def add_frame_time(
        self,
        detector_ms: float,
        tracker_ms: float,
        fuzzy_ms: float = 0.0,
        total_ms: float | None = None,
    ) -> None:
        self.frame_count += 1
        # Tách warm-up khỏi số liệu chính
        if self.frame_count <= self.warmup_frames:
            return

        self.detector_times_ms.append(float(detector_ms))
        self.tracker_times_ms.append(float(tracker_ms))
        self.fuzzy_times_ms.append(float(fuzzy_ms))
        tot = total_ms if total_ms is not None else (detector_ms + tracker_ms + fuzzy_ms)
        self.total_pipeline_times_ms.append(float(tot))

    def get_summary(self, mode: str = "cached_tracking") -> dict[str, Any]:
        """Báo cáo mean, median, p95 và FPS cho scope đo lường cụ thể."""
        def stats(arr: list[float]) -> dict[str, float]:
            if not arr:
                return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
            return {
                "mean_ms": round(float(np.mean(arr)), 3),
                "median_ms": round(float(np.median(arr)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
            }

        tracker_stats = stats(self.tracker_times_ms)
        tracker_fps = round(1000.0 / tracker_stats["mean_ms"], 1) if tracker_stats["mean_ms"] > 0 else 0.0

        pipeline_stats = stats(self.total_pipeline_times_ms)
        pipeline_fps = round(1000.0 / pipeline_stats["mean_ms"], 1) if pipeline_stats["mean_ms"] > 0 else 0.0

        return {
            "measurement_mode": mode,
            "warmup_frames_excluded": self.warmup_frames,
            "measured_frames": len(self.tracker_times_ms),
            "tracker": {
                **tracker_stats,
                "tracker_only_fps": tracker_fps,
            },
            "detector": stats(self.detector_times_ms),
            "fuzzy_controller": stats(self.fuzzy_times_ms),
            "end_to_end_pipeline": {
                **pipeline_stats,
                "end_to_end_fps": pipeline_fps,
            },
        }


def save_run_summary(path: Path, summary: dict[str, Any]) -> None:
    """Ghi JSON UTF-8 với thời điểm UTC nếu chưa có."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Không thể serialize kiểu {type(value).__name__} sang JSON")
