"""Chuẩn bị dữ liệu huấn luyện YOLOv8 từ Ground Truth của MOT17 (Pedestrian Class 1).

Sử dụng symbolic link để liên kết ảnh từ data/MOT17/train/ mà không nhân bản dữ liệu,
tiết kiệm tối đa dung lượng ổ đĩa.
"""

from __future__ import annotations

import configparser
from pathlib import Path
import pandas as pd
import yaml


def prepare_mot17_yolo_dataset(
    dataset_root: Path = Path("data/MOT17"),
    output_dir: Path = Path("data/mot17_yolo"),
) -> Path:
    train_seqs = ["MOT17-02-FRCNN", "MOT17-05-FRCNN", "MOT17-10-FRCNN"]
    val_seqs = ["MOT17-09-FRCNN"]

    img_train_dir = output_dir / "images" / "train"
    lbl_train_dir = output_dir / "labels" / "train"
    img_val_dir = output_dir / "images" / "val"
    lbl_val_dir = output_dir / "labels" / "val"

    for d in [img_train_dir, lbl_train_dir, img_val_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[CHẾ BẢN DỮ LIỆU YOLO TỪ MOT17]")
    splits = [("train", train_seqs, img_train_dir, lbl_train_dir), ("val", val_seqs, img_val_dir, lbl_val_dir)]

    total_converted_frames = 0
    total_pedestrian_boxes = 0

    for split_name, seq_list, target_img_dir, target_lbl_dir in splits:
        print(f"\n---> Xử lý {split_name.upper()} split ({len(seq_list)} sequences)...")
        for seq_name in seq_list:
            seq_dir = dataset_root / "train" / seq_name
            gt_file = seq_dir / "gt" / "gt.txt"
            ini_file = seq_dir / "seqinfo.ini"

            if not gt_file.is_file() or not ini_file.is_file():
                print(f"  [Bỏ qua] Không tìm thấy dữ liệu: {seq_dir}")
                continue

            cp = configparser.ConfigParser()
            cp.read(ini_file)
            im_w = int(cp["Sequence"].get("imWidth", 1920))
            im_h = int(cp["Sequence"].get("imHeight", 1080))

            df_gt = pd.read_csv(
                gt_file,
                header=None,
                names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "class_id", "visibility"],
            )

            # Chỉ lọc Class 1 (Pedestrian) với kích thước hợp lệ
            df_ped = df_gt[(df_gt["class_id"] == 1) & (df_gt["bb_width"] > 2) & (df_gt["bb_height"] > 2)]

            img_files = sorted((seq_dir / "img1").glob("*.jpg"))
            for img_path in img_files:
                fid = int(img_path.stem)
                frame_ped = df_ped[df_ped["frame"] == fid]

                stem = f"{seq_name}_{fid:06d}"
                link_path = target_img_dir / f"{stem}.jpg"
                lbl_path = target_lbl_dir / f"{stem}.txt"

                # Tạo symlink ảnh
                if not link_path.exists():
                    link_path.symlink_to(img_path.resolve())

                # Tạo nhãn YOLO
                lines = []
                for _, row in frame_ped.iterrows():
                    bx = float(row["bb_left"])
                    by = float(row["bb_top"])
                    bw = float(row["bb_width"])
                    bh = float(row["bb_height"])

                    cx = min(max((bx + bw * 0.5) / im_w, 0.0), 1.0)
                    cy = min(max((by + bh * 0.5) / im_h, 0.0), 1.0)
                    nw = min(max(bw / im_w, 0.0), 1.0)
                    nh = min(max(bh / im_h, 0.0), 1.0)

                    lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    total_pedestrian_boxes += 1

                with lbl_path.open("w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))

                total_converted_frames += 1

            print(f"  [OK] {seq_name}: Đã liên kết {len(img_files)} frames -> {target_img_dir.name}")

    data_yaml_path = output_dir / "mot17_pedestrian.yaml"
    data_dict = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "pedestrian"},
    }
    with data_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)

    print(f"\n[THÀNH CÔNG] Đã chuẩn bị {total_converted_frames} frames ({total_pedestrian_boxes} pedestrian boxes).")
    print(f"Cấu hình dữ liệu đã lưu tại: {data_yaml_path}")
    return data_yaml_path


if __name__ == "__main__":
    prepare_mot17_yolo_dataset()

