"""Script tự động tải và giải nén Dataset MOT17 theo chuẩn an toàn.

Hỗ trợ:
- Tải từ Direct URL (MOTChallenge chính thức) hoặc Kaggle API.
- Hiển thị tiến độ tải (progress bar).
- Kiểm tra dung lượng ổ đĩa trống trước khi tải và giải nén.
- Kiểm tra tính hợp lệ của file archive đã tải (không tải lại nếu đã có và hợp lệ).
- Giải nén an toàn (chống Zip Slip).
- Ghi log chi tiết ra outputs/download_log.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml


def load_dataset_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    required_keys = [
        "dataset_name", "source_type", "source_identifier", "download_url",
        "raw_dir", "extracted_dir", "download_archive",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"File cấu hình thiếu các trường: {missing}")
    return data


def check_disk_space(target_path: Path, required_bytes: int) -> tuple[bool, int, int]:
    """Kiểm tra dung lượng ổ đĩa trống tại thư mục mục tiêu."""
    parent = target_path
    while not parent.exists():
        parent = parent.parent
    stat = shutil.disk_usage(parent)
    free_bytes = stat.free
    return free_bytes >= required_bytes, free_bytes, required_bytes


def format_bytes(num_bytes: int) -> str:
    """Format dung lượng dạng dễ đọc (B, KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def download_with_progress(url: str, output_path: Path, expected_size: int | None = None) -> None:
    """Tải file qua HTTP/HTTPS với progress reporting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) MultiObjectTracking-Downloader"},
    )

    # Hỗ trợ resume nếu file .part đã tồn tại
    initial_bytes = 0
    if temp_path.exists():
        initial_bytes = temp_path.stat().st_size
        req.add_header("Range", f"bytes={initial_bytes}-")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            # 206 Partial Content hoặc 200 OK
            is_resume = response.status == 206
            if not is_resume:
                initial_bytes = 0
                mode = "wb"
            else:
                mode = "ab"

            total_size = response.headers.get("Content-Length")
            total_bytes = (int(total_size) + initial_bytes) if total_size else expected_size

            print(f"Bắt đầu tải từ: {url}")
            print(f"Lưu vào     : {output_path}")
            if total_bytes:
                print(f"Kích thước  : {format_bytes(total_bytes)}")

            downloaded = initial_bytes
            start_time = time.time()
            last_print = 0.0

            with temp_path.open(mode) as f:
                block_size = 1024 * 1024  # 1MB
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)

                    now = time.time()
                    if now - last_print >= 1.0 or downloaded == total_bytes:
                        elapsed = now - start_time
                        speed = (downloaded - initial_bytes) / elapsed if elapsed > 0 else 0
                        percent = (downloaded / total_bytes * 100) if total_bytes else 0
                        speed_str = f"{format_bytes(int(speed))}/s"
                        sys.stdout.write(
                            f"\rTiến độ: {downloaded}/{total_bytes or '?'} bytes "
                            f"({percent:.1f}%) | Tốc độ: {speed_str} | Đã tải: {format_bytes(downloaded)}   "
                        )
                        sys.stdout.flush()
                        last_print = now

            print()
    except Exception as exc:
        print(f"\n[LỖI] Quá trình tải thất bại: {exc}")
        raise

    temp_path.rename(output_path)
    print(f"[THÀNH CÔNG] Đã tải xong: {output_path}")


def download_via_kaggle(dataset_slug: str, output_dir: Path) -> Path:
    """Tải qua Kaggle API, kiểm tra authentication."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise RuntimeError(
            "Chưa cài đặt package kaggle. Vui lòng cài đặt: python -m pip install kaggle"
        )

    # Kiểm tra credential
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env_credentials = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not kaggle_json.is_file() and not has_env_credentials:
        raise RuntimeError(
            "Không tìm thấy thông tin xác thực Kaggle API.\n"
            "Vui lòng cấu hình file ~/.kaggle/kaggle.json hoặc set biến môi trường KAGGLE_USERNAME và KAGGLE_KEY.\n"
            "Hướng dẫn: Đăng nhập Kaggle -> Settings -> API -> Create New Token -> tải về kaggle.json và đặt vào ~/.kaggle/"
        )

    api = KaggleApi()
    api.authenticate()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Đang tải dataset {dataset_slug} từ Kaggle về {output_dir}...")
    api.dataset_download_files(dataset_slug, path=str(output_dir), unzip=False, quiet=False)

    # Tìm file zip vừa tải
    archives = list(output_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"Không tìm thấy file zip sau khi tải từ Kaggle tại {output_dir}")
    return archives[0]


def is_valid_zip(zip_path: Path) -> bool:
    """Kiểm tra file zip có hợp lệ và mở được không."""
    if not zip_path.is_file():
        return False
    if not zipfile.is_zipfile(zip_path):
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Kiểm tra một số entry mẫu
            info_list = zf.infolist()
            return len(info_list) > 0
    except Exception:
        return False


def safe_extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Giải nén an toàn, chống Zip Slip vulnerability."""
    extract_to.mkdir(parents=True, exist_ok=True)
    extract_to_resolved = extract_to.resolve()

    print(f"Đang giải nén an toàn {zip_path.name} -> {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total = len(members)
        for i, member in enumerate(members):
            target_path = (extract_to / member.filename).resolve()
            # Bảo vệ chống Zip Slip
            if not str(target_path).startswith(str(extract_to_resolved)):
                raise RuntimeError(f"Cảnh báo bảo mật: file zip chứa đường dẫn nguy hiểm: {member.filename}")
            if i % 1000 == 0 or i == total - 1:
                sys.stdout.write(f"\rGiải nén: {i + 1}/{total} files ({(i + 1) / total * 100:.1f}%)")
                sys.stdout.flush()
        print()
        zf.extractall(extract_to)
    print(f"[OK] Giải nén thành công vào: {extract_to}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải và giải nén Dataset MOT17")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Đường dẫn file cấu hình dataset")
    parser.add_argument("--skip-extract", action="store_true", help="Chỉ tải file zip, không giải nén")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra cấu hình và dung lượng đĩa")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = (project_root / args.config).resolve()
    config = load_dataset_config(config_path)

    raw_dir = (project_root / config["raw_dir"]).resolve()
    extracted_dir = (project_root / config["extracted_dir"]).resolve()
    archive_path = (project_root / config["download_archive"]).resolve()
    expected_bytes = config.get("expected_size_bytes", 5860214001)

    log_entry: dict[str, Any] = {
        "dataset_name": config["dataset_name"],
        "source_type": config["source_type"],
        "source_identifier": config["source_identifier"],
        "download_url": config["download_url"],
        "start_time": datetime.datetime.now().isoformat(),
        "status": "pending",
    }

    print("=" * 60)
    print("DỰ ÁN YOLO + BYTETRACK: TẢI DATASET MOT17")
    print("=" * 60)
    print(f"Dataset      : {config['dataset_name']}")
    print(f"Nguồn        : {config['source_type']} ({config['source_identifier']})")
    print(f"URL          : {config['download_url']}")
    print(f"Archive đích : {archive_path}")
    print(f"Thư mục giải nén: {extracted_dir}")
    print(f"Dung lượng dự kiến: {format_bytes(expected_bytes)}")

    # Kiểm tra dung lượng đĩa: cần archive + giải nén (khoảng ~12 GB)
    required_bytes = expected_bytes * 2 if not args.skip_extract else expected_bytes
    has_space, free_bytes, _ = check_disk_space(archive_path.parent, required_bytes)
    print(f"Dung lượng ổ đĩa trống: {format_bytes(free_bytes)}")
    print(f"Dung lượng yêu cầu    : {format_bytes(required_bytes)}")

    if not has_space:
        msg = (
            f"\n[LỖI DUNG LƯỢNG ĐĨA] Ổ đĩa chỉ còn {format_bytes(free_bytes)}, "
            f"nhưng quá trình tải + giải nén cần tối thiểu {format_bytes(required_bytes)}.\n"
            f"Vui lòng giải phóng ổ đĩa hoặc tạo symlink thư mục data sang phân vùng khác có đủ dung lượng."
        )
        print(msg)
        log_entry["status"] = "failed_disk_space"
        log_entry["error"] = msg
        output_log = project_root / "outputs" / "download_log.json"
        output_log.parent.mkdir(parents=True, exist_ok=True)
        with output_log.open("w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2)
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] Cấu hình và dung lượng hợp lệ. Không thực hiện tải trong chế độ dry-run.")
        return

    # Kiểm tra archive đã có chưa
    if archive_path.exists() and is_valid_zip(archive_path):
        current_size = archive_path.stat().st_size
        print(f"\n[INFO] File archive đã tồn tại và hợp lệ ({format_bytes(current_size)}). Bỏ qua bước tải.")
    else:
        # Thực hiện tải
        try:
            if config["source_type"] == "kaggle":
                downloaded_file = download_via_kaggle(config["source_identifier"], raw_dir)
                if downloaded_file != archive_path:
                    shutil.move(str(downloaded_file), str(archive_path))
            else:
                download_with_progress(config["download_url"], archive_path, expected_bytes)
        except Exception as exc:
            log_entry["status"] = "failed_download"
            log_entry["error"] = str(exc)
            output_log = project_root / "outputs" / "download_log.json"
            output_log.parent.mkdir(parents=True, exist_ok=True)
            with output_log.open("w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2)
            raise

    # Xác minh tính toàn vẹn của archive
    print("\nKiểm tra tính toàn vẹn của file zip...")
    if not is_valid_zip(archive_path):
        raise ValueError(f"File zip bị lỗi hoặc không thể đọc: {archive_path}")
    print("[OK] File zip toàn vẹn.")

    # Giải nén
    if not args.skip_extract:
        # Kiểm tra xem đã giải nén chưa (nếu thư mục train/ và test/ đã có dữ liệu)
        train_dir = extracted_dir / "train"
        if train_dir.is_dir() and any(train_dir.iterdir()):
            print(f"[INFO] Dữ liệu đã được giải nén sẵn tại {extracted_dir}. Bỏ qua bước giải nén.")
        else:
            safe_extract_zip(archive_path, extracted_dir)

    log_entry["end_time"] = datetime.datetime.now().isoformat()
    log_entry["status"] = "completed"
    log_entry["archive_size_bytes"] = archive_path.stat().st_size
    output_log = project_root / "outputs" / "download_log.json"
    output_log.parent.mkdir(parents=True, exist_ok=True)
    with output_log.open("w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2)

    print(f"\n[HOÀN TẤT] Log tải và giải nén đã ghi vào: {output_log}")


if __name__ == "__main__":
    main()

