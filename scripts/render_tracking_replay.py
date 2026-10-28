"""Render Visual Tracking Replay & Multi-Model Comparison.

Hỗ trợ so sánh trực quan:
- Mode 'trio': 3 panels song song (BASELINE vs GA-BYTETRACK vs FUZZY-ADAPTIVE).
- Mode 'pair': 2 panels song song (Baseline vs GA, Baseline vs Fuzzy, hoặc GA vs Fuzzy).
- Mode 'single': 1 panel duy nhất tập trung vào 1 model.
Tích hợp vệt chuyển động (trajectory trails) và bảng điều khiển HUD thời gian thực.
"""

from __future__ import annotations

import argparse
import collections
import glob
from pathlib import Path
import subprocess
import cv2
import numpy as np
import pandas as pd


def get_distinct_color(track_id: int) -> tuple[int, int, int]:
    """Tạo màu BGR cố định và tương phản cao dựa trên track ID."""
    np.random.seed(int(track_id) * 31 + 17)
    color = np.random.randint(50, 240, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def load_mot_predictions(pred_path: Path) -> dict[int, list[dict]]:
    """Đọc file prediction_mot.txt chuẩn MOTChallenge (frame, id, x, y, w, h, conf, ...)."""
    frames_dict: dict[int, list[dict]] = collections.defaultdict(list)
    if not pred_path.is_file():
        return frames_dict

    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 and float(parts[6]) > 0 else 1.0
            frames_dict[frame_id].append({
                "id": track_id,
                "bbox": [x, y, w, h],
                "conf": conf,
            })
    return frames_dict


def load_fuzzy_log(log_path: Path) -> dict[int, dict]:
    """Đọc fuzzy_adjustments.csv theo frame_id."""
    fuzzy_dict: dict[int, dict] = {}
    if not log_path.is_file():
        return fuzzy_dict

    try:
        df = pd.read_csv(log_path)
        for _, row in df.iterrows():
            f_id = int(row["frame_id"])
            fuzzy_dict[f_id] = {
                "high_thresh": float(row.get("applied_high_thresh", 0.6)),
                "low_conf_ratio": float(row.get("low_conf_ratio", 0.0)),
                "unmatched_ratio": float(row.get("unmatched_ratio", 0.0)),
            }
    except Exception as e:
        print(f"Warning: Không thể đọc fuzzy adjustments: {e}")
    return fuzzy_dict


def draw_tracking_on_frame(
    img: np.ndarray,
    detections: list[dict],
    trajectories: dict[int, collections.deque],
    title: str,
    accent_color: tuple[int, int, int],
    trail_len: int = 30,
    dynamic_thresh: float | None = None,
    extra_info: str | None = None,
) -> np.ndarray:
    """Vẽ bounding box, ID, và vệt quỹ đạo lên một khung hình."""
    vis = img.copy()
    h, w = vis.shape[:2]

    # Cập nhật và vẽ quỹ đạo (trajectory trails)
    current_ids = set()
    for det in detections:
        t_id = det["id"]
        bx, by, bw, bh = det["bbox"]
        cx = int(bx + bw / 2.0)
        cy = int(by + bh)  # Điểm tiếp đất (chân người)
        current_ids.add(t_id)

        if t_id not in trajectories:
            trajectories[t_id] = collections.deque(maxlen=trail_len)
        trajectories[t_id].append((cx, cy))

        color = get_distinct_color(t_id)

        # Vẽ vệt di chuyển (trail)
        points = list(trajectories[t_id])
        for idx in range(1, len(points)):
            alpha = idx / len(points)
            thickness = max(1, int(3 * alpha))
            cv2.line(vis, points[idx - 1], points[idx], color, thickness, cv2.LINE_AA)

        # Vẽ Bounding Box
        x1, y1 = max(0, int(bx)), max(0, int(by))
        x2, y2 = min(w - 1, int(bx + bw)), min(h - 1, int(by + bh))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Nhãn ID
        label = f"ID: {t_id}"
        font_scale = 0.50 if w < 700 else 0.55
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, y1), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 3, max(lh + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Dọn dẹp các track đã biến mất
    for t_id in list(trajectories.keys()):
        if t_id not in current_ids and len(trajectories[t_id]) > 0:
            trajectories[t_id].popleft()

    # Thanh Header trên cùng của panel
    header_h = 42
    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    header[:] = (25, 25, 25)
    cv2.rectangle(header, (0, 0), (6, header_h), accent_color, -1)

    title_scale = 0.60 if w < 700 else 0.70
    cv2.putText(
        header,
        title,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    stats_text = f"Active: {len(detections)}"
    if dynamic_thresh is not None:
        stats_text += f" | Thresh: {dynamic_thresh:.2f}"
    if extra_info:
        stats_text += f" | {extra_info}"

    info_scale = 0.45 if w < 700 else 0.52
    (sw, _), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, info_scale, 1)
    cv2.putText(
        header,
        stats_text,
        (max(14, w - sw - 14), 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        info_scale,
        accent_color,
        2,
        cv2.LINE_AA,
    )

    return np.vstack([header, vis])


def render_replay(
    sequence: str,
    img_dir: Path,
    output_mp4: Path,
    mode: str = "trio",
    models: list[str] | None = None,
    max_frames: int | None = None,
    fps: int = 25,
    panel_width: int = 640,
    trail_len: int = 35,
) -> None:
    """Render video MP4 so sánh với các chế độ: trio (3 models) hoặc pair (2 models)."""
    img_paths = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))
    if not img_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại {img_dir}")

    if max_frames:
        img_paths = img_paths[:max_frames]

    base_dir = Path("outputs/experiments")

    # Xác định các models cần render
    if mode == "trio":
        active_models = ["baseline", "ga_bytetrack", "fuzzy"]
    elif mode == "pair":
        active_models = models if models and len(models) == 2 else ["baseline", "fuzzy"]
    else:
        active_models = ["fuzzy"]

    model_configs = {
        "baseline": {
            "title": "BASELINE BYTETRACK",
            "accent": (0, 140, 255),  # Cam / Đỏ
            "pred_path": base_dir / f"baseline/{sequence}/prediction_mot.txt",
            "extra": "Fixed Thresh (0.60)",
            "thresh": 0.60,
        },
        "ga_bytetrack": {
            "title": "GA-BYTETRACK (EVOLVED)",
            "accent": (255, 180, 50),  # Xanh dương sáng / Vàng
            "pred_path": base_dir / f"ga_bytetrack/{sequence}/prediction_mot.txt",
            "extra": "Buffer: 43 | Thresh: 0.38",
            "thresh": 0.38,
        },
        "fuzzy": {
            "title": "FUZZY-ADAPTIVE BYTETRACK",
            "accent": (0, 230, 115),  # Xanh lục sáng
            "pred_path": base_dir / f"fuzzy/{sequence}/prediction_mot.txt",
            "extra": "Mamdani Dynamic",
            "thresh": None,  # Sẽ lấy từ fuzzy_log
        },
    }

    # Nạp dữ liệu
    data_map = {}
    trajectories_map = {}
    for m in active_models:
        cfg = model_configs[m]
        data_map[m] = load_mot_predictions(cfg["pred_path"])
        trajectories_map[m] = {}

    fuzzy_log = load_fuzzy_log(base_dir / f"fuzzy/{sequence}/fuzzy_adjustments.csv")

    # Đọc ảnh mẫu để tính scale
    sample = cv2.imread(img_paths[0])
    orig_h, orig_w = sample.shape[:2]
    scale = panel_width / float(orig_w)
    target_h = int(orig_h * scale)

    num_panels = len(active_models)
    panel_h = target_h + 42  # 42px header
    total_w = panel_width * num_panels
    total_h = panel_h + 40  # 40px top banner

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    temp_raw_mp4 = output_mp4.with_suffix(".raw.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_raw_mp4), fourcc, fps, (total_w, total_h))

    print(f"Bắt đầu render ({mode.upper()}) {len(img_paths)} frames cho {sequence}...")
    print(f"Models: {', '.join(active_models)} | Resolution: {total_w}x{total_h}")

    for idx, img_path in enumerate(img_paths, start=1):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame_resized = cv2.resize(frame, (panel_width, target_h))

        panels = []
        for m in active_models:
            cfg = model_configs[m]
            raw_dets = data_map[m].get(idx, [])
            scaled_dets = []
            for d in raw_dets:
                bx, by, bw, bh = d["bbox"]
                scaled_dets.append({
                    "id": d["id"],
                    "bbox": [bx * scale, by * scale, bw * scale, bh * scale],
                    "conf": d["conf"],
                })

            thresh = cfg["thresh"]
            extra = cfg["extra"]
            if m == "fuzzy":
                f_meta = fuzzy_log.get(idx, {})
                thresh = f_meta.get("high_thresh", 0.30)
                low_ratio = f_meta.get("low_conf_ratio", 0.0)
                extra = f"LowConf: {low_ratio*100:.0f}%"

            panel = draw_tracking_on_frame(
                frame_resized,
                scaled_dets,
                trajectories_map[m],
                title=cfg["title"],
                accent_color=cfg["accent"],
                trail_len=trail_len,
                dynamic_thresh=thresh,
                extra_info=extra,
            )
            panels.append(panel)

        # Top Global Banner
        top_banner = np.zeros((40, total_w, 3), dtype=np.uint8)
        top_banner[:] = (15, 15, 15)

        title_text = f"MOT BENCHMARK REPLAY | {sequence} | {mode.upper()}"
        cv2.putText(
            top_banner,
            title_text,
            (20, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        progress_text = f"Frame: {idx:04d} / {len(img_paths):04d} ({(idx/len(img_paths))*100:.1f}%)"
        (pw, _), _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(
            top_banner,
            progress_text,
            (total_w - pw - 20, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # Ghép ngang các panel
        body = np.hstack(panels)
        composite = np.vstack([top_banner, body])

        writer.write(composite)

        if idx % 100 == 0 or idx == len(img_paths):
            print(f"Processed [{idx}/{len(img_paths)}] frames...")

    writer.release()

    # Nén bằng ffmpeg H.264
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_raw_mp4),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            str(output_mp4),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_raw_mp4.unlink(missing_ok=True)
        print(f"Hoàn tất! Video đã được nén tại: {output_mp4}")
    except Exception as e:
        print(f"Lưu ý: Không thể convert bằng ffmpeg ({e}), giữ nguyên file raw: {temp_raw_mp4}")
        temp_raw_mp4.rename(output_mp4)


def main():
    parser = argparse.ArgumentParser(description="Render Visual Tracking Replay Video")
    parser.add_argument("--sequence", type=str, default="MOT17-09-FRCNN", help="Tên sequence MOT17")
    parser.add_argument("--img-dir", type=str, default=None, help="Đường dẫn thư mục ảnh frames")
    parser.add_argument("--mode", type=str, default="trio", choices=["trio", "pair", "single"], help="Chế độ render")
    parser.add_argument("--models", type=str, default="baseline,ga_bytetrack", help="Cặp models nếu dùng mode=pair (ví dụ baseline,ga_bytetrack hoặc baseline,fuzzy)")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file MP4 xuất ra")
    parser.add_argument("--max-frames", type=int, default=None, help="Số frame tối đa cần render")
    parser.add_argument("--fps", type=int, default=25, help="FPS của video đầu ra")
    parser.add_argument("--width", type=int, default=640, help="Chiều rộng của mỗi panel")
    args = parser.parse_args()

    seq = args.sequence
    img_dir = Path(args.img_dir) if args.img_dir else Path(f"data/MOT17/train/{seq}/img1")
    output_mp4 = Path(args.output) if args.output else Path(f"outputs/replays/{args.mode}_{seq}.mp4")
    pair_models = [m.strip() for m in args.models.split(",")]

    render_replay(
        sequence=seq,
        img_dir=img_dir,
        output_mp4=output_mp4,
        mode=args.mode,
        models=pair_models,
        max_frames=args.max_frames,
        fps=args.fps,
        panel_width=args.width,
    )


if __name__ == "__main__":
    main()
