"""Adapter và Subclass mở rộng cho BYTETracker của Ultralytics.

Đáp ứng yêu cầu Mục 7:
- Không tạo lại tracker mỗi frame làm mất trạng thái Kalman.
- Áp dụng ngưỡng động track_high_thresh thực sự vào args.track_high_thresh trước khi update().
- Ghi nhận chính xác ngưỡng thực sự tracker sử dụng trong từng frame.
- Tính toán unmatched_ratio từ frame trước để tránh vòng lặp logic (circular dependency).
- Reset sạch trạng thái (Kalman filter, IDs, tracks) giữa hai sequence.
- Hỗ trợ cả hai chế độ:
  A. cached_tracking (đọc từ detection cache npz siêu tốc, không chạy YOLO).
  B. end_to_end (chạy YOLO detector -> tracker).
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker


class DetectionResults:
    """Wrapper gọn nhẹ chứa bboxes, scores, classes để cấp vào BYTETracker.update()."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> DetectionResults:
        return DetectionResults(self.xywh[mask], self.conf[mask], self.cls[mask])


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """Chuyển đổi [x1, y1, x2, y2] sang [center_x, center_y, width, height]."""
    if len(xyxy) == 0:
        return np.empty((0, 4), dtype=np.float32)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    return np.stack([cx, cy, w, h], axis=-1)


class AdaptiveByteTracker(BYTETracker):
    """Subclass kế thừa BYTETracker, hỗ trợ điều chỉnh track_high_thresh động trên từng frame."""

    def __init__(self, config: dict[str, Any]):
        # Tạo args chuẩn cho BYTETracker
        self.config_dict = dict(config)
        args = SimpleNamespace(
            track_high_thresh=float(config["track_high_thresh"]),
            track_low_thresh=float(config["track_low_thresh"]),
            new_track_thresh=float(config["new_track_thresh"]),
            track_buffer=int(config["track_buffer"]),
            match_thresh=float(config["match_thresh"]),
            fuse_score=bool(config.get("fuse_score", True)),
        )
        super().__init__(args)
        self.initial_high_thresh = float(args.track_high_thresh)
        self.current_applied_high_thresh = float(args.track_high_thresh)
        self.prior_active_ids: set[int] = set()
        self.prior_unmatched_ratio: float = 0.0

    def reset_state(self) -> None:
        """Reset toàn bộ trạng thái tracker giữa các sequence."""
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        self.args.track_high_thresh = self.initial_high_thresh
        self.current_applied_high_thresh = self.initial_high_thresh
        self.prior_active_ids.clear()
        self.prior_unmatched_ratio = 0.0
        self.reset_id()
        self.kalman_filter = self.get_kalmanfilter()

    def update_with_threshold(
        self,
        results: Any,
        dynamic_high_thresh: float | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Thực hiện update tracker với ngưỡng động được chỉ định.

        Returns:
            (tracks_output, actual_thresh_used, prior_unmatched_ratio_used)
        """
        # Áp dụng ngưỡng động cho lượt association thứ nhất của frame này
        if dynamic_high_thresh is not None:
            # Ràng buộc an toàn: không được nhỏ hơn track_low_thresh và không lớn hơn 1.0
            clipped = float(np.clip(dynamic_high_thresh, self.args.track_low_thresh, 1.0))
            self.args.track_high_thresh = clipped

        self.current_applied_high_thresh = float(self.args.track_high_thresh)
        ratio_used = self.prior_unmatched_ratio

        # Thực thi association của BYTETracker
        tracks = super().update(results)

        # Tính toán unmatched_ratio sau khi association để DÙNG CHO FRAME KẾ TIẾP
        # tracks output dạng: [x1, y1, x2, y2, track_id, conf, cls, det_idx]
        if len(tracks) > 0:
            current_active_ids = {int(t[4]) for t in tracks if len(t) > 4}
        else:
            current_active_ids = set()

        if self.prior_active_ids:
            unmatched_count = len(self.prior_active_ids - current_active_ids)
            self.prior_unmatched_ratio = float(unmatched_count / len(self.prior_active_ids))
        else:
            self.prior_unmatched_ratio = 0.0

        self.prior_active_ids = current_active_ids

        return tracks, self.current_applied_high_thresh, ratio_used

