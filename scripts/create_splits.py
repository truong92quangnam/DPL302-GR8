"""Script tạo và xác minh các split dữ liệu (Development vs Held-out) theo chuẩn chống rò rỉ dữ liệu.

Đảm bảo:
- Tách theo video scene gốc.
- Không trùng lặp scene giữa development và held-out.
- Toàn bộ các biến thể detector của cùng một video scene nằm trọn vẹn trong một split.
- Xuất kết quả tổng hợp ra outputs/splits_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_splits_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình split: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_splits(splits_data: dict[str, Any]) -> dict[str, Any]:
    splits = splits_data.get("splits", {})
    if "development" not in splits or "held_out" not in splits:
        raise ValueError("Cấu hình split bắt buộc phải có hai phần: 'development' và 'held_out'")

    dev_scenes = set(splits["development"].get("base_scenes", []))
    held_scenes = set(splits["held_out"].get("base_scenes", []))

    # Kiểm tra rò rỉ dữ liệu giữa 2 split
    overlap = dev_scenes.intersection(held_scenes)
    if overlap:
        raise ValueError(f"RÒ RỈ DỮ LIỆU PHÁT HIỆN: Các base scene sau xuất hiện ở cả 2 split: {overlap}")

    dev_seqs = set(splits["development"].get("sequences", []))
    held_seqs = set(splits["held_out"].get("sequences", []))
    seq_overlap = dev_seqs.intersection(held_seqs)
    if seq_overlap:
        raise ValueError(f"RÒ RỈ DỮ LIỆU PHÁT HIỆN: Sequences xuất hiện ở cả 2 split: {seq_overlap}")

    # Kiểm tra all_detector_variants
    dev_all = set(splits["development"].get("all_detector_variants", []))
    held_all = set(splits["held_out"].get("all_detector_variants", []))
    var_overlap = dev_all.intersection(held_all)
    if var_overlap:
        raise ValueError(f"RÒ RỈ DỮ LIỆU PHÁT HIỆN: Detector variants xuất hiện ở cả 2 split: {var_overlap}")

    summary = {
        "status": "valid",
        "split_strategy": splits_data.get("split_strategy", "scene_based"),
        "development": {
            "num_base_scenes": len(dev_scenes),
            "base_scenes": sorted(list(dev_scenes)),
            "num_primary_sequences": len(dev_seqs),
            "primary_sequences": sorted(list(dev_seqs)),
            "total_sequences_with_variants": len(dev_all),
        },
        "held_out": {
            "num_base_scenes": len(held_scenes),
            "base_scenes": sorted(list(held_scenes)),
            "num_primary_sequences": len(held_seqs),
            "primary_sequences": sorted(list(held_seqs)),
            "total_sequences_with_variants": len(held_all),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo và kiểm tra dataset splits chống rò rỉ")
    parser.add_argument("--config", default="configs/splits.yaml", help="Đường dẫn file cấu hình splits")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = (project_root / args.config).resolve()

    print("=" * 60)
    print("PHÂN CHIA SPLITS DATASET MOT17 CHỐNG RÒ RỈ DỮ LIỆU")
    print("=" * 60)

    splits_data = load_splits_config(config_path)
    summary = validate_splits(splits_data)

    print("Chiến lược phân chia: Tách theo video scene (No Data Leakage)")
    print("\n1. Tập Development / Tuning (dành riêng cho GA và Fuzzy rule selection):")
    print(f"   - Số video scene gốc: {summary['development']['num_base_scenes']}")
    print(f"   - Danh sách scene   : {', '.join(summary['development']['base_scenes'])}")
    print(f"   - Sequences chính   : {', '.join(summary['development']['primary_sequences'])}")

    print("\n2. Tập Local Held-out Evaluation (KHÓA hoàn toàn cho tới khi test cuối cùng):")
    print(f"   - Số video scene gốc: {summary['held_out']['num_base_scenes']}")
    print(f"   - Danh sách scene   : {', '.join(summary['held_out']['base_scenes'])}")
    print(f"   - Sequences chính   : {', '.join(summary['held_out']['primary_sequences'])}")

    print("\n[OK] Xác minh không có bất kỳ scene hoặc biến thể detector nào bị chồng lấn giữa 2 tập.")

    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "splits_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[THÀNH CÔNG] Thông tin split đã được lưu vào: {summary_path}")


if __name__ == "__main__":
    main()

