"""Unit tests cho AdaptiveByteTracker và FuzzyByteTrackController.

Kiểm tra:
1. Giới hạn tham số (parameter bounds clipping).
2. Frame không có detection (empty frames / zero detections).
3. Reset sequence (toàn bộ Kalman, track lists, ID counter được reset sạch).
4. Tính toán unmatched_ratio không bị vòng lặp logic (no circular dependency).
5. Suy luận Fuzzy ổn định trên các giá trị biên (0.0, 1.0, NaN protection).
"""

from __future__ import annotations

import unittest
import numpy as np

from src.bytetrack_adapter import AdaptiveByteTracker, DetectionResults, xyxy_to_xywh
from src.fuzzy_controller import FuzzyByteTrackController


class TestAdaptiveByteTrack(unittest.TestCase):

    def setUp(self) -> None:
        self.config = {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.25,
            "track_buffer": 30,
            "match_thresh": 0.80,
            "fuse_score": True,
        }
        self.tracker = AdaptiveByteTracker(self.config)
        self.fuzzy = FuzzyByteTrackController()

    def test_parameter_bounds(self) -> None:
        """Kiểm tra clipping: dynamic_high_thresh không được < track_low_thresh."""
        # Thử ép ngưỡng thấp hơn track_low_thresh (0.10)
        empty_dets = DetectionResults(np.empty((0, 4)), np.empty((0,)), np.empty((0,)))
        _, actual_thresh, _ = self.tracker.update_with_threshold(empty_dets, dynamic_high_thresh=0.02)
        self.assertGreaterEqual(actual_thresh, self.config["track_low_thresh"])

        # Thử ép ngưỡng > 1.0
        _, actual_thresh, _ = self.tracker.update_with_threshold(empty_dets, dynamic_high_thresh=1.5)
        self.assertLessEqual(actual_thresh, 1.0)

    def test_empty_detection_frames(self) -> None:
        """Kiểm tra chuỗi frame liên tiếp không có detection."""
        empty_dets = DetectionResults(np.empty((0, 4)), np.empty((0,)), np.empty((0,)))
        for _ in range(10):
            tracks, actual_thresh, _ = self.tracker.update_with_threshold(empty_dets)
            self.assertEqual(len(tracks), 0)
            self.assertAlmostEqual(actual_thresh, self.config["track_high_thresh"], places=4)
        self.assertEqual(self.tracker.frame_id, 10)

    def test_sequence_reset(self) -> None:
        """Kiểm tra reset_state() xóa sạch toàn bộ trạng thái giữa hai sequence."""
        # Tạo 1 detection và cập nhật
        dets = DetectionResults(
            xywh=np.array([[100, 100, 50, 100]]),
            conf=np.array([0.9]),
            cls=np.array([0]),
        )
        tracks, _, _ = self.tracker.update_with_threshold(dets, dynamic_high_thresh=0.35)
        self.assertGreater(len(tracks), 0)
        self.assertEqual(self.tracker.frame_id, 1)
        self.assertAlmostEqual(self.tracker.current_applied_high_thresh, 0.35, places=4)

        # Thực hiện reset
        self.tracker.reset_state()
        self.assertEqual(self.tracker.frame_id, 0)
        self.assertEqual(len(self.tracker.tracked_stracks), 0)
        self.assertEqual(len(self.tracker.lost_stracks), 0)
        self.assertEqual(len(self.tracker.removed_stracks), 0)
        self.assertEqual(len(self.tracker.prior_active_ids), 0)
        self.assertAlmostEqual(self.tracker.current_applied_high_thresh, self.config["track_high_thresh"], places=4)

    def test_no_circular_dependency(self) -> None:
        """Kiểm tra unmatched_ratio được lấy từ frame trước và cập nhật tuần tự."""
        # Frame 1: Chưa có prior track -> prior_unmatched_ratio = 0.0
        dets1 = DetectionResults(
            xywh=np.array([[100, 100, 50, 100], [200, 200, 50, 100]]),
            conf=np.array([0.9, 0.85]),
            cls=np.array([0, 0]),
        )
        _, _, ratio_used_1 = self.tracker.update_with_threshold(dets1)
        self.assertEqual(ratio_used_1, 0.0)
        self.assertEqual(len(self.tracker.prior_active_ids), 2)

        # Frame 2: Chỉ còn 1 detection gần box 1 (box 2 biến mất -> 1 unmatched)
        dets2 = DetectionResults(
            xywh=np.array([[102, 101, 50, 100]]),
            conf=np.array([0.91]),
            cls=np.array([0]),
        )
        # Tại thời điểm gọi update cho Frame 2, tỷ lệ unmatched lấy từ sau Frame 1 (lúc đó = 0)
        _, _, ratio_used_2 = self.tracker.update_with_threshold(dets2)
        self.assertEqual(ratio_used_2, 0.0)

        # Frame 3: Lúc này prior_unmatched_ratio từ Frame 2 (mất 1 trong 2 track) đã được tính: 1/2 = 0.5
        dets3 = DetectionResults(
            xywh=np.array([[104, 102, 50, 100]]),
            conf=np.array([0.92]),
            cls=np.array([0]),
        )
        _, _, ratio_used_3 = self.tracker.update_with_threshold(dets3)
        self.assertAlmostEqual(ratio_used_3, 0.5, places=2)

    def test_fuzzy_boundary_values(self) -> None:
        """Kiểm tra suy luận Fuzzy không bị lỗi ở các giá trị biên (0.0, 1.0)."""
        for low_r in [0.0, 0.5, 1.0]:
            for unmatch_r in [0.0, 0.5, 1.0]:
                for spread in [0.0, 0.3, 1.0]:
                    adj = self.fuzzy.adjust(
                        low_conf_ratio=low_r,
                        unmatched_ratio=unmatch_r,
                        confidence_spread=spread,
                        track_high_thresh=0.25,
                        track_low_thresh=0.10,
                        new_track_thresh=0.25,
                        track_buffer=30,
                    )
                    self.assertFalse(np.isnan(adj.high_thresh_delta))
                    self.assertGreaterEqual(adj.track_high_thresh, 0.0)
                    self.assertLessEqual(adj.track_high_thresh, 1.0)


if __name__ == "__main__":
    unittest.main()

