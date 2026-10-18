"""Huấn luyện fine-tune YOLOv8 trên tập dữ liệu người đi bộ MOT17 (GPU RTX 3050).

Ép công suất GPU RTX 3050 qua nhiều epoch huấn luyện (100% GPU compute),
sau đó lưu checkpoint tốt nhất vào models/yolov8n_mot17_finetuned.pt.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import torch
from ultralytics import YOLO


def train_yolo(
    data_yaml: Path = Path("data/mot17_yolo/mot17_pedestrian.yaml"),
    epochs: int = 5,
    imgsz: int = 640,
    batch_size: int = 8,
    model_name: str = "yolov8n.pt",
    output_dir: Path = Path("outputs/experiments/finetuned_yolo"),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"BẮT ĐẦU FINE-TUNE YOLO TRÊN GPU RTX 3050")
    print(f"Mô hình gốc: {model_name} | Epochs: {epochs} | Batch: {batch_size} | ImgSz: {imgsz}")
    print(f"Thiết bị: CUDA (PyTorch {torch.__version__})")
    print(f"{'='*60}")

    model = YOLO(model_name)

    t0 = time.perf_counter()
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=0,
        project=str(output_dir.resolve()),
        name="mot17_run",
        exist_ok=True,
        workers=4,
        save=True,
        plots=True,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[THÀNH CÔNG] Huấn luyện hoàn tất sau {elapsed/60:.2f} phút.")

    best_weight = output_dir / "mot17_run" / "weights" / "best.pt"
    if not best_weight.is_file():
        best_weight = output_dir / "mot17_run" / "weights" / "last.pt"

    dest_model = Path("models") / "yolov8n_mot17_finetuned.pt"
    dest_model.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(best_weight, dest_model)
    print(f"Trọng số tối ưu đã được lưu tại: {dest_model}")
    return dest_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    train_yolo(epochs=args.epochs, batch_size=args.batch, imgsz=args.imgsz)

