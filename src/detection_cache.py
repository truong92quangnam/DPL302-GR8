"""Quản lý Detection Cache cho các sequence MOT17.

Mục tiêu:
- Chạy YOLO một lần duy nhất cho mỗi sequence và lưu cache (định dạng npz + json meta).
- Giữ detection confidence thấp (conf >= 0.01) để ByteTrack có thể association ở lượt 2.
- Lưu chi tiết metadata: model, imgsz, conf, nms iou, library versions, config hash.
- Cung cấp hàm đọc cache siêu tốc cho chế độ cached_tracking (dành cho GA và Fuzzy).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Tính hash SHA256 cho cấu hình detector để kiểm tra cache validity."""
    serialized = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def get_cache_paths(cache_dir: Path, sequence_name: str) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{sequence_name}_detections.npz"
    meta_path = cache_dir / f"{sequence_name}_meta.json"
    return npz_path, meta_path


def is_cache_valid(
    cache_dir: Path,
    sequence_name: str,
    detector_config: dict[str, Any],
    expected_frames: int | None = None,
) -> bool:
    """Kiểm tra cache đã tồn tại và có khớp cấu hình không."""
    npz_path, meta_path = get_cache_paths(cache_dir, sequence_name)
    if not npz_path.is_file() or not meta_path.is_file():
        return False

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        expected_hash = compute_config_hash(detector_config)
        if meta.get("config_hash") != expected_hash:
            return False

        if expected_frames is not None and meta.get("total_frames") != expected_frames:
            return False

        return True
    except Exception:
        return False


def cache_sequence_detections(
    sequence_dir: Path,
    cache_dir: Path,
    detector_config: dict[str, Any],
    device: str = "cpu",
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Chạy detector qua toàn bộ frame của sequence và lưu cache npz + json."""
    sequence_name = sequence_dir.name
    npz_path, meta_path = get_cache_paths(cache_dir, sequence_name)

    model_name = detector_config.get("model_name", "yolov8n.pt")
    imgsz = int(detector_config.get("imgsz", 1280))
    conf_thresh = float(detector_config.get("conf", 0.01))
    iou_thresh = float(detector_config.get("iou", 0.7))
    classes = detector_config.get("classes", [0])

    # Tìm danh sách ảnh
    img_dir = sequence_dir / "img1"
    if not img_dir.is_dir():
        img_dir = sequence_dir
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not images:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong {img_dir}")

    if max_frames is not None:
        images = images[:max_frames]

    total_frames = len(images)
    print(f"[{sequence_name}] Đang sinh detection cache cho {total_frames} frames...")

    model = YOLO(model_name)
    config_hash = compute_config_hash(detector_config)

    frames_detections: list[np.ndarray] = []
    inference_times_ms: list[float] = []
    total_detections = 0

    # Khởi động detector (warm-up)
    dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    _ = model.predict(dummy_img, imgsz=imgsz, device=device, verbose=False)

    for i, img_path in enumerate(images):
        frame_idx = i + 1  # 1-indexed MOT format
        t0 = time.perf_counter()

        # Detector inference với confidence cực thấp (conf=0.01) để giữ box cho lượt 2
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf_thresh,
            iou=iou_thresh,
            classes=classes,
            device=device,
            verbose=False,
        )
        t_ms = (time.perf_counter() - t0) * 1000.0
        inference_times_ms.append(t_ms)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
            confs = boxes.conf.cpu().numpy().reshape(-1, 1)  # (N, 1)
            clss = boxes.cls.cpu().numpy().reshape(-1, 1)  # (N, 1)
            frame_col = np.full((len(xyxy), 1), frame_idx, dtype=float)

            # Cột: [frame_idx, x1, y1, x2, y2, conf, cls]
            frame_dets = np.hstack([frame_col, xyxy, confs, clss])
            frames_detections.append(frame_dets)
            total_detections += len(xyxy)
        else:
            frames_detections.append(np.empty((0, 7), dtype=float))

        if (i + 1) % 100 == 0 or (i + 1) == total_frames:
            print(f"  Processed {i + 1}/{total_frames} frames ({total_detections} dets)...")

    # Ghép toàn bộ detections thành 1 ma trận duy nhất
    if frames_detections and any(len(d) > 0 for d in frames_detections):
        all_dets = np.vstack([d for d in frames_detections if len(d) > 0])
    else:
        all_dets = np.empty((0, 7), dtype=float)

    # Lưu file .npz nén
    np.savez_compressed(
        npz_path,
        detections=all_dets,
        inference_times_ms=np.array(inference_times_ms, dtype=np.float32),
        total_frames=total_frames,
    )

    # Lưu metadata
    metadata = {
        "sequence_name": sequence_name,
        "detector_config": detector_config,
        "config_hash": config_hash,
        "model_name": model_name,
        "imgsz": imgsz,
        "conf_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "classes": classes,
        "device": device,
        "total_frames": total_frames,
        "total_detections": total_detections,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "mean_inference_time_ms": float(np.mean(inference_times_ms)) if inference_times_ms else 0.0,
        "p95_inference_time_ms": float(np.percentile(inference_times_ms, 95)) if inference_times_ms else 0.0,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Cache đã lưu tại: {npz_path}")
    print(f"     Metadata: {meta_path} (Trung bình: {metadata['mean_inference_time_ms']:.2f} ms/frame)")
    return metadata


def load_cached_detections(
    cache_dir: Path,
    sequence_name: str,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    """Đọc detections từ cache theo dictionary {frame_id: (N, 6) [x1, y1, x2, y2, conf, cls]}."""
    npz_path, meta_path = get_cache_paths(cache_dir, sequence_name)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy detection cache: {npz_path}")

    data = np.load(npz_path)
    all_dets = data["detections"]  # [frame_idx, x1, y1, x2, y2, conf, cls]
    inference_times_ms = data["inference_times_ms"]
    total_frames = int(data["total_frames"])

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # Nhóm theo frame_id
    frame_dict: dict[int, np.ndarray] = {i: np.empty((0, 6), dtype=float) for i in range(1, total_frames + 1)}
    if len(all_dets) > 0:
        frames = all_dets[:, 0].astype(int)
        for fid in np.unique(frames):
            mask = frames == fid
            frame_dict[int(fid)] = all_dets[mask, 1:]  # [x1, y1, x2, y2, conf, cls]

    return frame_dict, inference_times_ms, meta

