"""Script kiểm tra tính toàn vẹn và hợp lệ của Dataset MOT17 sau khi tải.

Tuân thủ nghiêm ngặt protocol MOTChallenge:
- Kiểm tra ảnh, frame index, độ liên tục thời gian.
- Đọc thông số FPS, width, height từ seqinfo.ini.
- Kiểm tra ground truth gt/gt.txt nếu có:
  + Cột frame ID, object ID, bounding box hợp lệ (w > 0, h > 0).
  + Không trỏ tới frame không tồn tại.
  + Phân biệt class (class 1: Pedestrian, các class 2,7,8,12: Ignore/Distractor).
  + Xử lý visibility.
- Xuất kết quả ra:
  + outputs/dataset_manifest.csv
  + outputs/dataset_validation.json
"""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

MOT_COLUMNS = [
    "frame_id", "object_id", "bb_left", "bb_top", "bb_width", "bb_height",
    "conf", "class_id", "visibility",
]

# MOT17 official classes:
# 1: Pedestrian (đối tượng chính)
# 2: Person on vehicle
# 7: Static person
# 8: Distractor
# 12: Occluder on the ground
PEDESTRIAN_CLASS_ID = 1
IGNORE_CLASS_IDS = {2, 7, 8, 12}


def parse_seqinfo(seq_dir: Path) -> dict[str, Any]:
    """Đọc file seqinfo.ini của sequence."""
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"Thiếu file seqinfo.ini tại {seq_dir}")

    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    if "Sequence" not in parser:
        raise ValueError(f"seqinfo.ini tại {ini_path} thiếu section [Sequence]")

    sec = parser["Sequence"]
    return {
        "name": sec.get("name", seq_dir.name),
        "imDir": sec.get("imDir", "img1"),
        "frameRate": float(sec.get("frameRate", 30.0)),
        "seqLength": int(sec.get("seqLength", 0)),
        "imWidth": int(sec.get("imWidth", 0)),
        "imHeight": int(sec.get("imHeight", 0)),
        "imExt": sec.get("imExt", ".jpg"),
    }


def validate_sequence(seq_dir: Path, is_train: bool) -> dict[str, Any]:
    """Kiểm tra toàn diện một sequence."""
    result: dict[str, Any] = {
        "sequence_name": seq_dir.name,
        "is_train": is_train,
        "valid": False,
        "errors": [],
        "warnings": [],
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "declared_frames": 0,
        "actual_frames": 0,
        "has_gt": False,
        "total_gt_annotations": 0,
        "valid_pedestrian_annotations": 0,
        "ignored_annotations": 0,
        "unique_gt_identities": 0,
    }

    # 1. Đọc seqinfo.ini
    try:
        info = parse_seqinfo(seq_dir)
        result["fps"] = info["frameRate"]
        result["width"] = info["imWidth"]
        result["height"] = info["imHeight"]
        result["declared_frames"] = info["seqLength"]
    except Exception as exc:
        result["errors"].append(f"Lỗi đọc seqinfo.ini: {exc}")
        return result

    # 2. Kiểm tra thư mục ảnh và frame index
    img_dir = seq_dir / info["imDir"]
    if not img_dir.is_dir():
        result["errors"].append(f"Không tìm thấy thư mục ảnh: {img_dir}")
        return result

    ext = info["imExt"]
    image_files = sorted(img_dir.glob(f"*{ext}"))
    result["actual_frames"] = len(image_files)

    if len(image_files) == 0:
        result["errors"].append(f"Thư mục ảnh rỗng: {img_dir}")
        return result

    if len(image_files) != info["seqLength"]:
        result["warnings"].append(
            f"Số frame thực tế ({len(image_files)}) khác khai báo trong seqinfo ({info['seqLength']})"
        )

    # Kiểm tra thứ tự và frame index liên tiếp
    frame_indices: list[int] = []
    for f in image_files:
        stem = f.stem
        try:
            frame_indices.append(int(stem))
        except ValueError:
            result["warnings"].append(f"Tên file frame không phải dạng số: {f.name}")

    if frame_indices:
        min_idx = min(frame_indices)
        max_idx = max(frame_indices)
        if min_idx != 1:
            result["warnings"].append(f"Frame index không bắt đầu từ 1 (bắt đầu từ {min_idx})")
        # Kiểm tra xem có bị nhảy frame không
        expected_indices = set(range(min_idx, max_idx + 1))
        actual_indices = set(frame_indices)
        missing_frames = expected_indices - actual_indices
        if missing_frames:
            result["warnings"].append(f"Bị thiếu {len(missing_frames)} frame giữa chừng: {sorted(list(missing_frames))[:5]}...")

    # 3. Kiểm tra ground truth nếu có
    gt_file = seq_dir / "gt" / "gt.txt"
    if gt_file.is_file():
        result["has_gt"] = True
        try:
            df_gt = pd.read_csv(gt_file, header=None, names=MOT_COLUMNS)
            result["total_gt_annotations"] = len(df_gt)

            # Kiểm tra số cột
            if df_gt.shape[1] < 9:
                result["errors"].append(f"gt.txt chỉ có {df_gt.shape[1]} cột, yêu cầu chuẩn MOT 9 cột")

            # Kiểm tra bounding box hợp lệ
            invalid_boxes = (df_gt["bb_width"] <= 0) | (df_gt["bb_height"] <= 0)
            if invalid_boxes.any():
                count_inv = int(invalid_boxes.sum())
                result["errors"].append(f"Có {count_inv} bounding box có width hoặc height <= 0")

            # Kiểm tra frame index trong GT so với frame thực tế
            gt_max_frame = int(df_gt["frame_id"].max())
            if frame_indices and gt_max_frame > max(frame_indices):
                result["errors"].append(
                    f"GT trỏ tới frame {gt_max_frame} nhưng sequence chỉ có tới frame {max(frame_indices)}"
                )

            # Phân tích class và ignore region
            pedestrian_mask = (df_gt["class_id"] == PEDESTRIAN_CLASS_ID)
            ignored_mask = df_gt["class_id"].isin(IGNORE_CLASS_IDS)
            result["valid_pedestrian_annotations"] = int(pedestrian_mask.sum())
            result["ignored_annotations"] = int(ignored_mask.sum())

            # Số lượng object ID duy nhất (cho pedestrian)
            ped_df = df_gt[pedestrian_mask]
            result["unique_gt_identities"] = int(ped_df["object_id"].nunique())

            if result["unique_gt_identities"] == 0:
                result["errors"].append("Không tìm thấy object ID nào thuộc class Pedestrian (class 1)")

        except Exception as exc:
            result["errors"].append(f"Lỗi phân tích gt.txt: {exc}")
    else:
        if is_train:
            result["errors"].append(f"Sequence thuộc tập train nhưng thiếu ground truth: {gt_file}")

    result["valid"] = len(result["errors"]) == 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra tính toàn vẹn của dataset MOT17")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Đường dẫn file cấu hình dataset")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = (project_root / args.config).resolve()
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    extracted_dir = (project_root / config.get("extracted_dir", "data/MOT17")).resolve()
    print("=" * 60)
    print("KIỂM TRA TÍNH TOÀN VẸN VÀ HỢP LỆ DATASET MOT17")
    print("=" * 60)
    print(f"Thư mục dataset: {extracted_dir}")

    if not extracted_dir.is_dir():
        print(f"[LỖI] Thư mục dataset không tồn tại: {extracted_dir}")
        print("Vui lòng chạy script tải dữ liệu trước:")
        print("    python scripts/download_mot17.py")
        return

    # Duyệt các sequence trong train và test
    sequences_report: list[dict[str, Any]] = []

    for split_name, is_train in [("train", True), ("test", False)]:
        split_dir = extracted_dir / split_name
        if not split_dir.is_dir():
            print(f"[CẢNH BÁO] Không tìm thấy thư mục {split_dir}")
            continue

        for seq_folder in sorted(split_dir.iterdir()):
            if seq_folder.is_dir() and not seq_folder.name.startswith("."):
                seq_res = validate_sequence(seq_folder, is_train)
                seq_res["split"] = split_name
                sequences_report.append(seq_res)

    if not sequences_report:
        print("[LỖI] Không tìm thấy sequence nào trong dataset.")
        return

    # Xuất ra JSON và CSV
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "dataset_validation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(sequences_report, f, indent=2, ensure_ascii=False)

    df_manifest = pd.DataFrame(sequences_report)
    csv_path = output_dir / "dataset_manifest.csv"
    df_manifest.to_csv(csv_path, index=False)

    print(f"\n[OK] Đã quét {len(sequences_report)} sequences.")
    print(f"  - Báo cáo JSON: {json_path}")
    print(f"  - Manifest CSV: {csv_path}")

    # Tổng kết
    valid_count = sum(1 for s in sequences_report if s["valid"])
    print(f"\nTổng kết: {valid_count}/{len(sequences_report)} sequences hợp lệ.")
    for s in sequences_report:
        status_icon = "✅" if s["valid"] else "❌"
        print(
            f"  {status_icon} [{s['split'].upper()}] {s['sequence_name']}: "
            f"{s['actual_frames']} frames, FPS: {s['fps']}, "
            f"GT: {'Có' if s['has_gt'] else 'Không'} (IDs: {s['unique_gt_identities']})"
        )
        if s["errors"]:
            for err in s["errors"]:
                print(f"      LỖI: {err}")
        if s["warnings"]:
            for warn in s["warnings"]:
                print(f"      CẢNH BÁO: {warn}")


if __name__ == "__main__":
    main()

