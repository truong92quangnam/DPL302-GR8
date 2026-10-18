"""Script kiểm tra môi trường chạy dự án YOLO + ByteTrack + GA + Fuzzy.

Yêu cầu:
- Môi trường Conda đang active phải là `base`.
- Kiểm tra OS, Python, PyTorch, CUDA, GPU, Ultralytics.
- Xuất báo cáo ra outputs/environment_report.json.
- Sao lưu cấu hình môi trường hiện tại:
  - outputs/environment_before.yml (nếu conda khả dụng)
  - outputs/pip_before.txt
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def check_active_conda_env() -> tuple[str, bool]:
    """Kiểm tra tên môi trường Conda đang active."""
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    is_base = env_name == "base"
    return env_name, is_base


def get_gpu_info() -> dict[str, Any]:
    """Lấy thông tin GPU từ PyTorch và nvidia-smi nếu khả dụng."""
    gpu_info: dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "nvidia_smi_available": False,
        "nvidia_smi_output": None,
    }

    # Kiểm tra qua PyTorch
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_info["cuda_available"] = cuda_avail
        if cuda_avail:
            count = torch.cuda.device_count()
            gpu_info["device_count"] = count
            for idx in range(count):
                props = torch.cuda.get_device_properties(idx)
                gpu_info["devices"].append({
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                })
    except ImportError:
        gpu_info["torch_import_error"] = "torch chưa được cài đặt"
    except Exception as exc:
        gpu_info["torch_cuda_error"] = str(exc)

    # Thử kiểm tra nvidia-smi
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            gpu_info["nvidia_smi_available"] = True
            gpu_info["nvidia_smi_output"] = res.stdout.strip()
        else:
            gpu_info["nvidia_smi_error"] = res.stderr.strip()
    except Exception as exc:
        gpu_info["nvidia_smi_error"] = str(exc)

    return gpu_info


def inspect_environment() -> dict[str, Any]:
    """Thu thập toàn bộ thông tin môi trường."""
    active_env, is_base = check_active_conda_env()

    # PyTorch version
    torch_version = None
    torch_cuda_version = None
    try:
        import torch
        torch_version = torch.__version__
        torch_cuda_version = torch.version.cuda
    except ImportError:
        pass

    # Ultralytics version
    ultralytics_version = None
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except ImportError:
        pass

    report = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "conda": {
            "active_env": active_env,
            "is_base": is_base,
            "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
            "conda_exe": os.environ.get("CONDA_EXE", ""),
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "version_info": list(sys.version_info),
        },
        "pytorch": {
            "version": torch_version,
            "cuda_version": torch_cuda_version,
        },
        "gpu": get_gpu_info(),
        "ultralytics": {
            "version": ultralytics_version,
        },
    }
    return report


def backup_environment(output_dir: Path) -> None:
    """Sao lưu cấu hình môi trường hiện tại: conda env export & pip freeze."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. pip freeze
    pip_file = output_dir / "pip_before.txt"
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        pip_file.write_text(res.stdout, encoding="utf-8")
        print(f"[OK] Đã lưu pip freeze: {pip_file}")
    except Exception as exc:
        print(f"[CẢNH BÁO] Không thể lưu pip freeze: {exc}")

    # 2. conda env export
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    conda_file = output_dir / "environment_before.yml"
    try:
        res = subprocess.run(
            [conda_exe, "env", "export"],
            capture_output=True,
            text=True,
            check=True,
        )
        conda_file.write_text(res.stdout, encoding="utf-8")
        print(f"[OK] Đã lưu conda env export: {conda_file}")
    except Exception as exc:
        print(f"[CẢNH BÁO] Không thể lưu conda env export: {exc}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KIỂM TRA MÔI TRƯỜNG DỰ ÁN (CONDA BASE)")
    print("=" * 60)

    report = inspect_environment()
    active_env = report["conda"]["active_env"]
    is_base = report["conda"]["is_base"]

    print(f"Hệ điều hành          : {report['os']['platform']}")
    print(f"Conda active env       : {active_env or '(Không xác định)'}")
    print(f"Python interpreter    : {report['python']['executable']}")
    print(f"Python version        : {report['python']['version']}")
    print(f"PyTorch version       : {report['pytorch']['version'] or 'Chưa cài đặt'}")
    print(f"PyTorch CUDA version  : {report['pytorch']['cuda_version'] or 'N/A'}")
    print(f"CUDA available        : {report['gpu']['cuda_available']}")

    if report["gpu"]["cuda_available"]:
        for dev in report["gpu"]["devices"]:
            print(f"  - GPU [{dev['index']}]: {dev['name']} ({dev['total_memory_gb']} GB VRAM)")
    else:
        if report["gpu"].get("nvidia_smi_error"):
            print(f"  - Lý do / Driver info: {report['gpu']['nvidia_smi_error']}")

    print(f"Ultralytics version   : {report['ultralytics']['version'] or 'Chưa cài đặt'}")
    print("-" * 60)

    # Ghi kết quả vào outputs/environment_report.json
    report_path = output_dir / "environment_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] Báo cáo chi tiết đã ghi vào: {report_path}")

    # Kiểm tra điều kiện bắt buộc conda base
    if not is_base:
        print("\n[LỖI NGHIÊM TRỌNG] Môi trường Conda hiện tại KHÔNG PHẢI 'base'.")
        print(f"Môi trường phát hiện: '{active_env}'")
        print("Vui lòng kích hoạt môi trường base trước khi chạy tiếp:")
        print("    conda activate base")
        sys.exit(1)

    # Sao lưu môi trường
    print("\nTiến hành sao lưu trạng thái môi trường hiện tại...")
    backup_environment(output_dir)

    print("\n[THÀNH CÔNG] Kiểm tra môi trường hoàn tất. Môi trường 'base' hợp lệ.")


if __name__ == "__main__":
    main()

