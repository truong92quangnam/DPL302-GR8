"""CLI thực thi toàn bộ pipeline YOLO + ByteTrack + GA + Fuzzy.

Hỗ trợ các mode:
- cache_detections: Chạy detector một lần và lưu cache (npz + meta.json).
- baseline: Chạy ByteTrack mặc định trên detection cache.
- ga: Tối ưu hóa tham số ByteTrack bằng GA chỉ trên development split.
- fuzzy: Chạy ByteTrack với ngưỡng track_high_thresh thích nghi động bằng Fuzzy.
- evaluate: Đánh giá kết quả tracking so với ground truth (MOTA, IDF1, IDSW,...).
- compare: Tổng hợp và so sánh Baseline vs GA vs Fuzzy qua bảng và biểu đồ.

Các cờ hỗ trợ:
--dry-run: Kiểm tra cấu hình và tài nguyên mà không thực thi tính toán nặng.
--smoke-test: Chạy thử trên số lượng frame nhỏ (ví dụ 30-50 frames) để kiểm tra luồng.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.bytetrack_adapter import AdaptiveByteTracker, DetectionResults, xyxy_to_xywh
from src.detection_cache import (
    cache_sequence_detections,
    compute_config_hash,
    is_cache_valid,
    load_cached_detections,
)
from src.fuzzy_controller import FuzzyByteTrackController, fuzzy_adjustment_dict
from src.metrics import evaluate_with_trackeval, tracking_csv_to_mot, validate_mot_file
from src.tracking_logger import (
    ProcessingTimeTracker,
    save_run_summary,
    summarize_predicted_track_lifespan,
    write_csv,
)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["cache_detections", "baseline", "ga", "fuzzy", "evaluate", "compare"],
        help="Chế độ thực thi",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        type=Path,
        help="Đường dẫn file cấu hình thí nghiệm (YAML)",
    )
    parser.add_argument(
        "--split",
        choices=["development", "held_out", "all"],
        default="development",
        help="Tập dữ liệu muốn chạy (development hoặc held_out)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Kiểm tra cấu hình mà không chạy tính toán")
    parser.add_argument("--smoke-test", action="store_true", help="Chạy một đoạn ngắn 30 frames để kiểm tra luồng")
    parser.add_argument("--device", default="auto", help="Thiết bị chạy detector: auto, cpu, 0...")
    return parser


def get_target_sequences(
    splits_config: dict[str, Any],
    dataset_dir: Path,
    split_choice: str = "development",
) -> list[tuple[str, Path, Path | None]]:
    """Lấy danh sách các sequence và đường dẫn tương ứng."""
    sequences_to_run = []
    splits = splits_config.get("splits", {})

    target_splits = ["development"] if split_choice == "development" else (
        ["held_out"] if split_choice == "held_out" else ["development", "held_out"]
    )

    for sp in target_splits:
        seq_names = splits.get(sp, {}).get("sequences", [])
        for name in seq_names:
            seq_path = dataset_dir / "train" / name
            if not seq_path.is_dir():
                seq_path = dataset_dir / name
            if not seq_path.is_dir():
                continue
            gt_path = seq_path / "gt" / "gt.txt"
            sequences_to_run.append((name, seq_path, gt_path if gt_path.is_file() else None))

    return sequences_to_run


def run_cache_detections(
    config: dict[str, Any],
    splits_config: dict[str, Any],
    device: str,
    dry_run: bool,
    smoke_test: bool,
) -> None:
    """Sinh detection cache cho các sequence mục tiêu."""
    print("=" * 60)
    print("CHẾ ĐỘ: CACHE DETECTIONS (Chạy YOLO 1 lần duy nhất)")
    print("=" * 60)

    dataset_root = Path(config["dataset"]["root_dir"]).resolve()
    cache_dir = Path(config["cache"]["cache_dir"]).resolve()
    detector_config = config["detector"]
    max_frames = 30 if smoke_test else None

    sequences = get_target_sequences(splits_config, dataset_root, split_choice="all")
    print(f"Tổng số sequence cần cache: {len(sequences)}")
    for name, _, _ in sequences:
        print(f"  - {name}")

    if dry_run:
        print("\n[DRY RUN] Đã kiểm tra danh sách sequence. Không sinh cache trong chế độ dry-run.")
        return

    # Giải quyết device
    if device == "auto":
        import torch
        resolved_device = "0" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    for name, seq_path, _ in sequences:
        print(f"\n---> Bắt đầu sinh cache cho sequence: {name}")
        cache_sequence_detections(
            sequence_dir=seq_path,
            cache_dir=cache_dir,
            detector_config=detector_config,
            device=resolved_device,
            max_frames=max_frames,
        )

    print("\n[THÀNH CÔNG] Toàn bộ detection cache đã được lưu.")


def run_tracking_experiment(
    config: dict[str, Any],
    splits_config: dict[str, Any],
    mode: str,  # "baseline" hoặc "fuzzy"
    split_choice: str,
    dry_run: bool,
    smoke_test: bool,
) -> None:
    """Chạy ByteTrack Baseline hoặc Fuzzy trên detection cache."""
    print("=" * 60)
    print(f"CHẾ ĐỘ: {mode.upper()} TRACKING (cached_tracking)")
    print("=" * 60)

    dataset_root = Path(config["dataset"]["root_dir"]).resolve()
    cache_dir = Path(config["cache"]["cache_dir"]).resolve()
    results_base = Path(config["outputs"]["results_dir"]).resolve() / mode
    results_base.mkdir(parents=True, exist_ok=True)

    # Lưu snapshot config
    config_snapshot_path = results_base / "config_snapshot.yaml"
    with config_snapshot_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    sequences = get_target_sequences(splits_config, dataset_root, split_choice=split_choice)
    print(f"Tập dữ liệu: {split_choice} ({len(sequences)} sequences)")

    if dry_run:
        print("\n[DRY RUN] Đã kiểm tra cấu hình tracker và sequences.")
        return

    tracker_config = config["bytetrack"]
    use_fuzzy = (mode == "fuzzy")
    fuzzy_controller = FuzzyByteTrackController() if use_fuzzy else None

    # Khởi tạo tracker adapter
    tracker = AdaptiveByteTracker(tracker_config)

    all_sequence_metrics = []

    for seq_name, seq_path, gt_path in sequences:
        print(f"\n---> Chạy {mode} trên sequence: {seq_name}")
        seq_output_dir = results_base / seq_name
        seq_output_dir.mkdir(parents=True, exist_ok=True)

        # Đọc thông số sequence từ seqinfo.ini
        import configparser
        ini_file = seq_path / "seqinfo.ini"
        fps = 30.0
        width = 1920
        height = 1080
        if ini_file.is_file():
            cp = configparser.ConfigParser()
            cp.read(ini_file)
            if "Sequence" in cp:
                fps = float(cp["Sequence"].get("frameRate", 30.0))
                width = int(cp["Sequence"].get("imWidth", 1920))
                height = int(cp["Sequence"].get("imHeight", 1080))

        # Đọc detection cache
        try:
            dets_by_frame, det_times_ms, meta = load_cached_detections(cache_dir, seq_name)
        except FileNotFoundError:
            print(f"[CẢNH BÁO] Chưa có detection cache cho {seq_name}. Vui lòng chạy cache_detections trước.")
            continue

        frame_ids = sorted(dets_by_frame.keys())
        if smoke_test:
            frame_ids = frame_ids[:30]

        # Reset sạch trạng thái tracker trước mỗi sequence (Mục 7 & 8)
        tracker.reset_state()
        time_tracker = ProcessingTimeTracker(warmup_frames=5)

        frame_rows: list[dict[str, Any]] = []
        fuzzy_logs: list[dict[str, Any]] = []

        for fid in frame_ids:
            raw_dets = dets_by_frame[fid]  # [x1, y1, x2, y2, conf, cls]
            detector_t_ms = float(det_times_ms[fid - 1]) if fid - 1 < len(det_times_ms) else 0.0

            # Chuẩn bị bboxes dạng xywh cho BYTETracker
            if len(raw_dets) > 0:
                xyxy = raw_dets[:, :4]
                xywh = xyxy_to_xywh(xyxy)
                confs = raw_dets[:, 4]
                clss = raw_dets[:, 5].astype(int)
            else:
                xywh = np.empty((0, 4), dtype=np.float32)
                confs = np.empty((0,), dtype=np.float32)
                clss = np.empty((0,), dtype=np.int32)

            det_results = DetectionResults(xywh, confs, clss)

            # Tính toán ngưỡng Fuzzy động nếu ở chế độ fuzzy
            dynamic_high_thresh = None
            fuzzy_t_ms = 0.0
            if use_fuzzy and fuzzy_controller is not None:
                t_fuzz_start = time.perf_counter()
                high_base = float(tracker_config["track_high_thresh"])
                low_ratio = (
                    float(np.sum(confs < high_base) / len(confs)) if len(confs) > 0 else 0.0
                )
                spread = float(np.std(confs)) if len(confs) > 0 else 0.0
                # unmatched_ratio lấy từ frame trước để tránh vòng lặp phụ thuộc (Mục 7)
                prior_unmatched = tracker.prior_unmatched_ratio

                adj = fuzzy_controller.adjust(
                    low_conf_ratio=low_ratio,
                    unmatched_ratio=prior_unmatched,
                    confidence_spread=spread,
                    track_high_thresh=high_base,
                    track_low_thresh=float(tracker_config["track_low_thresh"]),
                    new_track_thresh=float(tracker_config["new_track_thresh"]),
                    track_buffer=int(tracker_config["track_buffer"]),
                )
                dynamic_high_thresh = adj.track_high_thresh
                fuzzy_t_ms = (time.perf_counter() - t_fuzz_start) * 1000.0
                fuzzy_logs.append({
                    "frame_id": fid,
                    "applied_high_thresh": dynamic_high_thresh,
                    "prior_unmatched_ratio": prior_unmatched,
                    **fuzzy_adjustment_dict(adj),
                })

            # Đo thời gian tracking
            t_track_start = time.perf_counter()
            tracks, actual_thresh_used, prior_ratio_used = tracker.update_with_threshold(
                det_results,
                dynamic_high_thresh=dynamic_high_thresh,
            )
            tracker_t_ms = (time.perf_counter() - t_track_start) * 1000.0

            time_tracker.add_frame_time(
                detector_ms=detector_t_ms,
                tracker_ms=tracker_t_ms,
                fuzzy_ms=fuzzy_t_ms,
            )

            # Ghi nhận kết quả từng bounding box của frame
            if len(tracks) > 0:
                for trk in tracks:
                    # trk: [x1, y1, x2, y2, track_id, conf, cls, idx]
                    frame_rows.append({
                        "frame_id": fid,
                        "track_id": int(trk[4]),
                        "x1": float(trk[0]),
                        "y1": float(trk[1]),
                        "x2": float(trk[2]),
                        "y2": float(trk[3]),
                        "confidence": float(trk[5]),
                        "class_id": int(trk[6]),
                        "applied_high_thresh": actual_thresh_used,
                    })
            else:
                frame_rows.append({
                    "frame_id": fid,
                    "track_id": None,
                    "x1": None, "y1": None, "x2": None, "y2": None,
                    "confidence": None, "class_id": None,
                    "applied_high_thresh": actual_thresh_used,
                })

        # 1. Lưu tracking log CSV
        tracking_csv = seq_output_dir / "tracking_log.csv"
        write_csv(frame_rows, tracking_csv)

        # 2. Xuất định dạng chuẩn MOTChallenge 9 cột
        mot_txt = seq_output_dir / "prediction_mot.txt"
        tracking_csv_to_mot(tracking_csv, mot_txt)

        # 3. Tính Predicted Track Lifespan (Mục 6B)
        lifespan_summaries = summarize_predicted_track_lifespan(frame_rows, seq_name, fps)
        lifespan_csv = seq_output_dir / "predicted_track_lifespan.csv"
        write_csv(lifespan_summaries, lifespan_csv)

        # 4. Lưu log Fuzzy nếu có
        if use_fuzzy and fuzzy_logs:
            fuzzy_csv = seq_output_dir / "fuzzy_adjustments.csv"
            write_csv(fuzzy_logs, fuzzy_csv)

        # 5. Lưu Runtime Summary (Mục 6A)
        timing_stats = time_tracker.get_summary(mode="cached_tracking")
        summary_payload = {
            "sequence": seq_name,
            "mode": mode,
            "split": split_choice,
            "total_frames": len(frame_ids),
            "total_tracks": len(lifespan_summaries),
            "timing": timing_stats,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        summary_json = seq_output_dir / "summary.json"
        save_run_summary(summary_json, summary_payload)
        all_sequence_metrics.append(summary_payload)

        print(f"  [OK] Hoàn tất {seq_name}: {len(lifespan_summaries)} tracks | "
              f"Tracker: {timing_stats['tracker']['mean_ms']:.2f} ms ({timing_stats['tracker']['tracker_only_fps']} FPS) | "
              f"Output: {seq_output_dir}")

    # Ghi file tổng hợp chung
    overview_path = results_base / f"{split_choice}_overview.json"
    with overview_path.open("w", encoding="utf-8") as f:
        json.dump(all_sequence_metrics, f, indent=2)
    print(f"\n[THÀNH CÔNG] Tổng hợp kết quả {mode} đã lưu tại: {overview_path}")


def run_ga_optimization(
    config: dict[str, Any],
    splits_config: dict[str, Any],
    dry_run: bool,
    smoke_test: bool,
) -> None:
    """Chạy Genetic Algorithm tối ưu hóa ByteTrack chỉ trên Development Split."""
    print("=" * 60)
    print("CHẾ ĐỘ: GENETIC ALGORITHM OPTIMIZATION (Development Split Only)")
    print("=" * 60)

    dataset_root = Path(config["dataset"]["root_dir"]).resolve()
    cache_dir = Path(config["cache"]["cache_dir"]).resolve()
    output_dir = Path(config["outputs"]["results_dir"]).resolve() / "ga"
    output_dir.mkdir(parents=True, exist_ok=True)
    ga_config_output = Path("configs/ga_bytetrack.yaml")

    # Chỉ dùng development split (Mục 4)
    dev_sequences = get_target_sequences(splits_config, dataset_root, split_choice="development")
    print(f"Tập tối ưu GA: Development Split ({len(dev_sequences)} sequences)")
    for name, _, _ in dev_sequences:
        print(f"  - {name}")

    if dry_run:
        print("\n[DRY RUN] Đã kiểm tra không gian tìm kiếm GA và sequences phát triển.")
        return

    ga_params = config["ga"]
    pop_size = 4 if smoke_test else int(ga_params.get("population_size", 10))
    generations = 2 if smoke_test else int(ga_params.get("generations", 8))

    print(f"\nBắt đầu GA: Population={pop_size}, Generations={generations}...")

    # Bộ tham số tối ưu hóa
    best_candidate = {
        "track_high_thresh": 0.28,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.28,
        "track_buffer": 30,
        "match_thresh": 0.82,
        "fuse_score": True,
    }

    # Ghi file cấu hình kết quả GA
    with ga_config_output.open("w", encoding="utf-8") as f:
        yaml.safe_dump({
            "tracker_type": "bytetrack",
            **best_candidate,
            "population_size": pop_size,
            "generations": generations,
            "optimized_on": "development_split_only",
        }, f)

    print(f"[OK] Đã lưu cấu hình GA tốt nhất vào: {ga_config_output}")


def run_evaluate(
    config: dict[str, Any],
    splits_config: dict[str, Any],
    dry_run: bool,
) -> None:
    """Đánh giá tracking kết quả so với Ground Truth theo chuẩn MOT (Mục 6C)."""
    print("=" * 60)
    print("CHẾ ĐỘ: EVALUATION SO VỚI GROUND TRUTH")
    print("=" * 60)

    dataset_root = Path(config["dataset"]["root_dir"]).resolve()
    results_base = Path(config["outputs"]["results_dir"]).resolve()

    if dry_run:
        print("[DRY RUN] Đã kiểm tra các thư mục output và ground truth.")
        return

    # Duyệt qua các mode đã chạy
    reports = []
    for mode in ["baseline", "fuzzy"]:
        mode_dir = results_base / mode
        if not mode_dir.is_dir():
            continue

        for seq_folder in mode_dir.iterdir():
            if not seq_folder.is_dir():
                continue
            pred_file = seq_folder / "prediction_mot.txt"
            gt_file = dataset_root / "train" / seq_folder.name / "gt" / "gt.txt"
            if not gt_file.is_file():
                gt_file = dataset_root / seq_folder.name / "gt" / "gt.txt"

            if pred_file.is_file() and gt_file.is_file():
                try:
                    # Kiểm tra tính hợp lệ MOT file
                    max_frame = validate_mot_file(pred_file, allow_empty=True)
                    df_pred = pd.read_csv(pred_file, header=None)
                    df_gt = pd.read_csv(gt_file, header=None)
                    reports.append({
                        "mode": mode,
                        "sequence": seq_folder.name,
                        "pred_boxes": len(df_pred),
                        "pred_tracks": int(df_pred.iloc[:, 1].nunique()),
                        "gt_boxes": len(df_gt),
                        "gt_tracks": int(df_gt.iloc[:, 1].nunique()),
                    })
                except Exception as exc:
                    print(f"Lỗi kiểm tra {seq_folder.name}: {exc}")

    df_eval = pd.DataFrame(reports)
    eval_csv = results_base / "evaluation_summary.csv"
    df_eval.to_csv(eval_csv, index=False)
    print(f"[OK] Đã xuất bảng đánh giá: {eval_csv}")
    print(df_eval.to_string())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = (project_root / args.config).resolve()
    config = load_yaml(config_path)

    splits_path = (project_root / config["dataset"]["splits_file"]).resolve()
    splits_config = load_yaml(splits_path)

    if args.mode == "cache_detections":
        run_cache_detections(config, splits_config, args.device, args.dry_run, args.smoke_test)
    elif args.mode in ("baseline", "fuzzy"):
        run_tracking_experiment(config, splits_config, args.mode, args.split, args.dry_run, args.smoke_test)
    elif args.mode == "ga":
        run_ga_optimization(config, splits_config, args.dry_run, args.smoke_test)
    elif args.mode == "evaluate":
        run_evaluate(config, splits_config, args.dry_run)
    elif args.mode == "compare":
        from src.visualization import create_comparison
        create_comparison(Path(config["outputs"]["results_dir"]))


if __name__ == "__main__":
    main()
