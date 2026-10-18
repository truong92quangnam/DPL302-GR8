"""Chạy Ultralytics YOLO + ByteTrack theo stream, ghi CSV và video có annotation."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from fuzzy_controller import FuzzyByteTrackController, fuzzy_adjustment_dict
from tracking_logger import save_run_summary, summarize_tracks, write_csv


def load_tracker_config(path: Path) -> dict[str, Any]:
    """Đọc cấu hình YAML và kiểm tra các trường bắt buộc của ByteTrack."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tracker config: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "tracker_type", "track_high_thresh", "track_low_thresh", "new_track_thresh",
        "track_buffer", "match_thresh", "fuse_score",
    }
    if not isinstance(config, dict) or required.difference(config):
        missing = sorted(required.difference(config or {}))
        raise ValueError(f"Tracker config không hợp lệ; thiếu: {', '.join(missing)}")
    if config["tracker_type"] != "bytetrack":
        raise ValueError("Phiên bản này chỉ hỗ trợ tracker_type: bytetrack")
    for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
        value = float(config[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} phải nằm trong [0,1], nhận được {value}")
        config[key] = value
    buffer_value = float(config["track_buffer"])
    if not buffer_value.is_integer() or not 5 <= int(buffer_value) <= 90:
        raise ValueError("track_buffer phải nằm trong [5,90]")
    config["track_buffer"] = int(buffer_value)
    if not isinstance(config["fuse_score"], bool):
        raise ValueError("fuse_score phải là boolean true/false")
    if config["track_low_thresh"] > config["track_high_thresh"]:
        raise ValueError("track_low_thresh không được lớn hơn track_high_thresh")
    if config["track_high_thresh"] > config["new_track_thresh"]:
        raise ValueError("track_high_thresh không được lớn hơn new_track_thresh")
    return config


def resolve_device(requested: str) -> str:
    """Chọn thiết bị, fallback CPU khi CUDA không khả dụng."""
    if requested.lower() in {"auto", "cuda"}:
        requested = "0" if torch.cuda.is_available() else "cpu"
    elif requested.lower().startswith("cuda:"):
        requested = requested.split(":", maxsplit=1)[1]
    if requested != "cpu" and not torch.cuda.is_available():
        warnings.warn(f"Đã yêu cầu device={requested} nhưng CUDA không khả dụng; dùng CPU.", stacklevel=2)
        return "cpu"
    resolved = requested
    if resolved == "cpu":
        print("[device] CPU")
    else:
        indices = [int(index) for index in resolved.split(",")]
        if any(index < 0 or index >= torch.cuda.device_count() for index in indices):
            raise ValueError(f"CUDA device không tồn tại: {resolved}")
        print(f"[device] CUDA {', '.join(torch.cuda.get_device_name(index) for index in indices)}")
    return resolved


def run_tracking(
    *,
    video: Path,
    model_path: str,
    device: str,
    config: dict[str, Any],
    output_dir: Path,
    run_name: str,
    fuzzy: bool = False,
    save_video: bool = True,
) -> dict[str, Any]:
    """Chạy một video; fuzzy output áp dụng lên ByteTracker cho frame tiếp theo."""
    if not video.is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {video}")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"OpenCV không mở được video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Video không có kích thước frame hợp lệ: {video}")
    if fps <= 0:
        warnings.warn("Video không khai báo FPS hợp lệ; duration_seconds sẽ không có giá trị.", stacklevel=2)

    resolved_device = resolve_device(device)
    model = YOLO(model_path)
    controller = FuzzyByteTrackController() if fuzzy else None
    # Ultralytics nhận đường dẫn YAML cho tracker; ghi bản runtime để mọi mode dùng cùng schema.
    runtime_config = output_dir / "tracker_runtime.yaml"
    tracker_keys = (
        "tracker_type", "track_high_thresh", "track_low_thresh", "new_track_thresh",
        "track_buffer", "match_thresh", "fuse_score",
    )
    runtime_config.write_text(
        yaml.safe_dump({key: config[key] for key in tracker_keys}, sort_keys=False),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    fuzzy_rows: list[dict[str, Any]] = []
    prior_ids: set[int] = set()
    writer: cv2.VideoWriter | None = None
    if save_video and width > 0 and height > 0:
        writer = cv2.VideoWriter(
            str(output_dir / "tracked_video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 30.0,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise OSError(f"Không tạo được video output trong {output_dir}")

    iterator: Any = None
    frame_id = 0
    total_wall_seconds = 0.0
    inference_seconds_total = 0.0
    tracking_overhead_seconds_total = 0.0
    try:
        iterator = iter(model.track(
            source=str(video),
            stream=True,
            persist=True,
            tracker=str(runtime_config),
            device=resolved_device,
            conf=0.01,
            verbose=False,
        ))
        while True:
            frame_started = time.perf_counter()
            try:
                result = next(iterator)
            except StopIteration:
                break
            frame_id += 1
            frame = result.orig_img
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                xyxy = np.empty((0, 4), dtype=float)
                confidences = np.empty((0,), dtype=float)
                class_ids = np.empty((0,), dtype=int)
                track_ids: list[int | None] = []
            else:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confidences = boxes.conf.detach().cpu().numpy()
                class_ids = boxes.cls.detach().cpu().numpy().astype(int)
                raw_ids = boxes.id
                track_ids = (
                    raw_ids.detach().cpu().numpy().astype(int).tolist()
                    if raw_ids is not None
                    else [None] * len(xyxy)
                )

            speed = getattr(result, "speed", {}) or {}
            inference = max(float(speed.get("inference", 0.0)) / 1000.0, 0.0)
            inference_seconds_total += inference
            current_ids = {int(track_id) for track_id in track_ids if track_id is not None}
            if controller is not None:
                high_threshold = float(config["track_high_thresh"])
                low_ratio = (
                    sum(float(score) < high_threshold for score in confidences) / len(confidences)
                    if len(confidences) else 0.0
                )
                unmatched_ratio = len(prior_ids - current_ids) / len(prior_ids) if prior_ids else 0.0
                spread = float(np.std(confidences)) if len(confidences) else 0.0
                adjustment = controller.adjust(
                    low_conf_ratio=low_ratio,
                    unmatched_ratio=unmatched_ratio,
                    confidence_spread=spread,
                    track_high_thresh=float(config["track_high_thresh"]),
                    track_low_thresh=float(config["track_low_thresh"]),
                    new_track_thresh=float(config["new_track_thresh"]),
                    track_buffer=int(config["track_buffer"]),
                )
                config["track_high_thresh"] = adjustment.track_high_thresh
                config["track_low_thresh"] = adjustment.track_low_thresh
                config["track_buffer"] = adjustment.track_buffer
                _update_live_tracker(model, config)
                fuzzy_rows.append({"frame_id": frame_id, **fuzzy_adjustment_dict(adjustment)})

            if len(xyxy):
                for box, confidence, class_id, track_id in zip(xyxy, confidences, class_ids, track_ids):
                    row = _frame_row(frame_id, box, confidence, class_id, track_id)
                    row["inference_time_seconds"] = inference
                    row["tracking_time_seconds"] = None
                    row["total_frame_time_seconds"] = None
                    rows.append(row)
                    if frame is not None:
                        _draw_track(frame, box, confidence, track_id, class_id)
            else:
                rows.append({
                    "frame_id": frame_id, "track_id": None,
                    "x1": None, "y1": None, "x2": None, "y2": None,
                    "confidence": None, "class_id": None,
                    "inference_time_seconds": inference,
                    "tracking_time_seconds": None,
                    "total_frame_time_seconds": None,
                })

            # Đo tới khi đã chuẩn hóa/vẽ box, nhưng loại encode output để so sánh công bằng.
            frame_elapsed = max(time.perf_counter() - frame_started, 0.0)
            if writer is not None and frame is not None:
                writer.write(frame)
            total_wall_seconds += frame_elapsed
            # Phần dư gồm decode/pre/post/association; Ultralytics không public timer thuần ByteTrack.
            overhead = max(frame_elapsed - inference, 0.0)
            tracking_overhead_seconds_total += overhead
            for row in rows[-max(len(xyxy), 1):]:
                row["tracking_time_seconds"] = overhead
                row["total_frame_time_seconds"] = frame_elapsed
            prior_ids = current_ids
    finally:
        if writer is not None:
            writer.release()
        close = getattr(iterator, "close", None)
        if close is not None:
            close()

    if frame_id == 0:
        raise ValueError(f"Video không đọc được frame nào: {video}")
    track_summaries = summarize_tracks(rows, fps)
    write_csv(rows, output_dir / "tracking_log.csv")
    write_csv(track_summaries, output_dir / "track_durations.csv")
    write_csv(fuzzy_rows, output_dir / "fuzzy_adjustments.csv")
    run_summary: dict[str, Any] = {
        "run_name": run_name,
        "mode": "fuzzy" if fuzzy else "bytetrack",
        "device": resolved_device,
        "model": model_path,
        "video": str(video),
        "config": config.copy(),
        "frames": frame_id,
        "declared_video_frames": declared_frames,
        "fps": frame_id / total_wall_seconds if total_wall_seconds > 0 else 0.0,
        "tracking_time_seconds": tracking_overhead_seconds_total,
        "total_processing_time_seconds": total_wall_seconds,
        "inference_time_seconds": inference_seconds_total,
        "number_of_tracks": len(track_summaries),
        "ID_switches": None,
        "HOTA": None,
        "IDF1": None,
        "MOTA": None,
        "tracking_log": str(output_dir / "tracking_log.csv"),
        "track_durations": str(output_dir / "track_durations.csv"),
        "tracked_video": str(output_dir / "tracked_video.mp4") if save_video else None,
        "timing_note": "FPS và total_frame_time_seconds đo thời gian lấy/xử lý frame từ stream, không gồm vẽ/ghi MP4. tracking_time_seconds là tổng residual sau inference, còn gồm decode/pre/post; không phải timer thuần ByteTrack.",
    }
    save_run_summary(output_dir / "summary.json", run_summary)
    return run_summary


def _update_live_tracker(model: YOLO, config: dict[str, Any]) -> None:
    """Cập nhật args tracker sau frame để threshold mới có hiệu lực ở frame kế."""
    trackers = getattr(getattr(model, "predictor", None), "trackers", None)
    if not trackers:
        raise RuntimeError(
            "Không truy cập được tracker Ultralytics đang chạy; fuzzy mode cần phiên bản Ultralytics "
            "có predictor.trackers và tracker.args."
        )
    for tracker in trackers:
        args = getattr(tracker, "args", None)
        if args is None:
            raise RuntimeError("BYTETracker không cung cấp args để cập nhật threshold an toàn.")
        for key in ("track_high_thresh", "track_low_thresh", "track_buffer"):
            setattr(args, key, config[key])
        # ByteTracker cache buffer khi khởi tạo; cập nhật timeout hiện tại để fuzzy có hiệu lực.
        if hasattr(tracker, "max_frames_lost"):
            tracker.max_frames_lost = int(config["track_buffer"])
        elif hasattr(tracker, "max_time_lost"):
            frame_rate = float(getattr(tracker, "frame_rate", 30.0))
            tracker.max_time_lost = int(frame_rate / 30.0 * int(config["track_buffer"]))
        else:
            raise RuntimeError("BYTETracker không có thuộc tính timeout được hỗ trợ để cập nhật track_buffer.")


def _frame_row(
    frame_id: int,
    box: np.ndarray,
    confidence: float,
    class_id: int,
    track_id: int | None,
) -> dict[str, Any]:
    """Chuẩn hóa detection/track thành schema CSV."""
    return {
        "frame_id": frame_id,
        "track_id": track_id,
        "x1": float(box[0]),
        "y1": float(box[1]),
        "x2": float(box[2]),
        "y2": float(box[3]),
        "confidence": float(confidence),
        "class_id": class_id,
    }


def _draw_track(
    frame: np.ndarray,
    box: np.ndarray,
    confidence: float,
    track_id: int | None,
    class_id: int,
) -> None:
    """Vẽ bounding box và nhãn track/class/confidence."""
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    color = (40, 220, 40) if track_id is not None else (0, 165, 255)
    label = f"ID {track_id if track_id is not None else '-'} | C{class_id} | {confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
