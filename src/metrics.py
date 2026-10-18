"""Chuyển MOTChallenge và tích hợp tùy chọn TrackEval; không tự tạo metric."""

from __future__ import annotations

import configparser
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MOT_COLUMNS = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "class", "visibility"]


def tracking_csv_to_mot(tracking_csv: Path, output_txt: Path) -> Path:
    """Chuyển log box xyxy thành MOTChallenge 9 cột."""
    if not tracking_csv.is_file():
        raise FileNotFoundError(f"Không tìm thấy tracking CSV: {tracking_csv}")
    data = pd.read_csv(tracking_csv)
    required = {"frame_id", "track_id", "x1", "y1", "x2", "y2", "confidence", "class_id"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Tracking CSV thiếu cột: {', '.join(sorted(missing))}")
    data = data.dropna(subset=["track_id"])
    output = pd.DataFrame({
        "frame": data["frame_id"].astype(int),
        "id": data["track_id"].astype(int),
        "bb_left": data["x1"],
        "bb_top": data["y1"],
        "bb_width": data["x2"] - data["x1"],
        "bb_height": data["y2"] - data["y1"],
        "conf": data["confidence"],
        "class": data["class_id"].astype(int),
        "visibility": 1.0,
    }, columns=MOT_COLUMNS)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_txt, header=False, index=False, float_format="%.6f")
    return output_txt


def validate_mot_file(path: Path, *, allow_empty: bool = False) -> int:
    """Kiểm tra dạng MOTChallenge và trả về frame lớn nhất."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy MOT ground truth/prediction: {path}")
    if path.stat().st_size == 0:
        if allow_empty:
            return 0
        raise ValueError(f"Tệp MOT rỗng: {path}")
    table = pd.read_csv(path, header=None)
    if table.shape[1] < 9:
        raise ValueError(f"{path} cần 9 cột MOTChallenge (frame,id,box,conf,class,visibility), hiện có {table.shape[1]}")
    return int(table.iloc[:, 0].max())


def evaluate_with_trackeval(
    prediction_file: Path,
    gt_file: Path,
    *,
    sequence_name: str,
    work_dir: Path,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
) -> dict[str, float | int]:
    """Chạy TrackEval nếu cài đặt; trả lỗi rõ ràng nếu không có GT/tool."""
    if importlib.util.find_spec("trackeval") is None:
        raise RuntimeError(
            "Chưa cài TrackEval. Cài bằng `pip install git+https://github.com/JonathonLuiten/TrackEval.git`; "
            "xem README để biết cách chạy đánh giá."
        )
    if not gt_file.is_file():
        raise FileNotFoundError(f"Thiếu ground truth có ID theo frame: {gt_file}")
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("frame_count, width và height phải lớn hơn 0 để tạo seqinfo.ini")

    import trackeval

    eval_root = work_dir / "trackeval"
    gt_root = eval_root / "gt"
    tracker_root = eval_root / "trackers"
    split_name = "MOT17-train"
    gt_sequence = gt_root / split_name / sequence_name
    tracker_name = "FuzzyByteTrack"
    tracker_sequence = tracker_root / split_name / tracker_name / "data" / f"{sequence_name}.txt"
    (gt_sequence / "gt").mkdir(parents=True, exist_ok=True)
    tracker_sequence.parent.mkdir(parents=True, exist_ok=True)
    gt_destination = gt_sequence / "gt" / "gt.txt"
    if gt_file.resolve() != gt_destination.resolve():
        gt_destination.write_bytes(gt_file.read_bytes())
    tracker_sequence.write_bytes(prediction_file.read_bytes())
    _write_seqinfo(gt_sequence / "seqinfo.ini", sequence_name, frame_count, width, height, fps)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_ONLY_COMBINED": True})
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(gt_root),
        "TRACKERS_FOLDER": str(tracker_root),
        "TRACKERS_TO_EVAL": [tracker_name],
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "SEQ_INFO": {sequence_name: frame_count},
        "SKIP_SPLIT_FOL": False,
        "DO_PREPROC": False,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_FOLDER": str(work_dir / "trackeval" / "output"),
        "PRINT_CONFIG": False,
    })
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator = trackeval.Evaluator(eval_config)
    result, _ = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics)
    leaf = _find_combined_result(result)
    hota = _scalar(leaf["HOTA"]["HOTA"])
    idf1 = _scalar(leaf["Identity"]["IDF1"])
    clear = leaf["CLEAR"]
    return {
        "HOTA": hota,
        "IDF1": idf1,
        "MOTA": _scalar(clear["MOTA"]),
        "ID_switches": int(round(_scalar(clear["IDSW"]))),
    }


def _write_seqinfo(path: Path, name: str, frames: int, width: int, height: int, fps: float) -> None:
    """Ghi metadata sequence theo schema MOTChallenge."""
    parser = configparser.ConfigParser()
    parser["Sequence"] = {
        "name": name,
        "imDir": "img1",
        "frameRate": str(fps if fps > 0 else 30.0),
        "seqLength": str(frames),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": ".jpg",
    }
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)


def _find_combined_result(value: Any) -> dict[str, Any]:
    """Lấy metric gộp sequence từ cấu trúc lồng nhau của TrackEval."""
    candidates: list[tuple[bool, dict[str, Any]]] = []

    def visit(node: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(node, dict):
            if {"HOTA", "CLEAR", "Identity"}.issubset(node):
                candidates.append(("COMBINED_SEQ" in path, node))
            for key, child in node.items():
                visit(child, (*path, str(key)))

    visit(value)
    if not candidates:
        raise RuntimeError("Không tìm thấy kết quả HOTA/CLEAR/Identity trong phản hồi TrackEval")
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


def _scalar(value: Any) -> float:
    """Chuẩn hóa metric scalar/vector (HOTA có vector theo alpha)."""
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("TrackEval trả metric rỗng hoặc không hữu hạn")
    return float(np.mean(array))
