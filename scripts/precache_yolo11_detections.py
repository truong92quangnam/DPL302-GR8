"""Pre-cache YOLO11n detections cho toàn bộ 7 sequences của MOT17 bằng ThreadPoolExecutor.

Chạy song song 3 workers trên 12 CPU cores để tăng tốc tối đa.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO

ALL_SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]


def cache_single_sequence(seq_name: str, data_root: Path, cache_dir: Path) -> str:
    npz_file = cache_dir / f"{seq_name}_detections.npz"
    meta_file = cache_dir / f"{seq_name}_meta.json"

    if npz_file.is_file() and meta_file.is_file():
        return f"[SKIP] {seq_name} đã có cache hợp lệ!"

    seq_dir = data_root / seq_name
    img_dir = seq_dir / "img1"
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    total_frames = len(img_files)

    print(f"[{seq_name}] Bắt đầu cache {total_frames} frames...")
    model = YOLO("yolo11n.pt")
    
    # Warmup
    dummy = np.zeros((1280, 1280, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=1280, device="cpu", verbose=False)

    dets_by_frame: dict[int, np.ndarray] = {}
    det_times: list[float] = []

    t0_all = time.perf_counter()
    for idx, img_path in enumerate(img_files):
        t0 = time.perf_counter()
        pred = model.predict(
            str(img_path),
            imgsz=1280,
            conf=0.01,
            iou=0.7,
            classes=[0],
            device="cpu",
            verbose=False,
        )[0]
        dt = (time.perf_counter() - t0) * 1000.0
        det_times.append(dt)

        fid = idx + 1
        boxes = pred.boxes
        if len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            arr = np.column_stack([xyxy, confs, clss]).astype(np.float32)
        else:
            arr = np.empty((0, 6), dtype=np.float32)
        dets_by_frame[fid] = arr

        if (idx + 1) % 200 == 0 or (idx + 1) == total_frames:
            print(f"  [{seq_name}] Frame {idx + 1}/{total_frames} ({np.mean(det_times):.1f} ms/frame)")

    total_time = time.perf_counter() - t0_all
    avg_det_ms = float(np.mean(det_times))

    save_dict = {f"frame_{fid}": arr for fid, arr in dets_by_frame.items()}
    np.savez_compressed(npz_file, **save_dict)
    meta = {
        "sequence": seq_name,
        "model": "yolo11n.pt",
        "imgsz": 1280,
        "conf": 0.01,
        "iou": 0.7,
        "total_frames": total_frames,
        "avg_detector_latency_ms": avg_det_ms,
        "total_time_sec": total_time,
    }
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return f"[DONE] {seq_name} ({total_frames} frames, {total_time:.1f}s, {avg_det_ms:.1f} ms/f)"


def main():
    data_root = Path("data/MOT17/train")
    cache_dir = Path("data/cache/detections_yolo11n")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BẮT ĐẦU PRE-CACHE DETECTIONS CHO 7 SEQUENCES MOT17 (YOLO11n)")
    print(f"CPU Threads / Workers: 3 | Resolution: 1280x1280")
    print("=" * 70)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(cache_single_sequence, s, data_root, cache_dir): s for s in ALL_SEQUENCES}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            print(res)

    print(f"\n[HOÀN TẤT] Toàn bộ 7 sequences đã được cache trong {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()

