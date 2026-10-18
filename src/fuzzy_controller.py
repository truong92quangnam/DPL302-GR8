"""Bộ điều khiển Mamdani điều chỉnh ngưỡng ByteTrack theo từng frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


@dataclass(frozen=True)
class FuzzyAdjustment:
    """Đầu vào, delta và tham số tracker sau khi điều chỉnh."""

    low_conf_ratio: float
    unmatched_ratio: float
    confidence_spread: float
    high_thresh_delta: float
    low_thresh_delta: float
    buffer_delta: float
    track_high_thresh: float
    track_low_thresh: float
    track_buffer: int


class FuzzyByteTrackController:
    """Suy luận fuzzy Mamdani và clipping tham số cho một frame."""

    def __init__(self) -> None:
        # Universes rời rạc đủ mịn để centroid defuzzification ổn định.
        low_conf = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "low_conf_ratio")
        unmatched = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "unmatched_ratio")
        spread = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "confidence_spread")
        high_delta = ctrl.Consequent(np.linspace(-0.10, 0.10, 201), "high_thresh_delta")
        low_delta = ctrl.Consequent(np.linspace(-0.05, 0.05, 201), "low_thresh_delta")
        buffer_delta = ctrl.Consequent(np.linspace(-10.0, 10.0, 201), "buffer_delta")

        # Mỗi trimf là tam giác trái/giữa/phải; vai trái/phải kéo dài tới biên miền.
        for variable in (low_conf, unmatched, spread):
            variable["low"] = fuzz.trimf(variable.universe, [0.0, 0.0, 0.4])
            variable["medium"] = fuzz.trimf(variable.universe, [0.2, 0.5, 0.8])
            variable["high"] = fuzz.trimf(variable.universe, [0.6, 1.0, 1.0])

        # Các tập đầu ra mã hóa giảm, giữ nguyên, tăng theo miền delta tương ứng.
        high_delta["decrease"] = fuzz.trimf(high_delta.universe, [-0.10, -0.06, 0.0])
        high_delta["keep"] = fuzz.trimf(high_delta.universe, [-0.02, 0.0, 0.02])
        high_delta["increase"] = fuzz.trimf(high_delta.universe, [0.0, 0.06, 0.10])
        low_delta["decrease"] = fuzz.trimf(low_delta.universe, [-0.05, -0.03, 0.0])
        low_delta["keep"] = fuzz.trimf(low_delta.universe, [-0.01, 0.0, 0.01])
        low_delta["increase"] = fuzz.trimf(low_delta.universe, [0.0, 0.03, 0.05])
        buffer_delta["decrease"] = fuzz.trimf(buffer_delta.universe, [-10.0, -6.0, 0.0])
        buffer_delta["keep"] = fuzz.trimf(buffer_delta.universe, [-2.0, 0.0, 2.0])
        buffer_delta["increase"] = fuzz.trimf(buffer_delta.universe, [0.0, 6.0, 10.0])

        # Luật Mamdani: các cặp low-confidence/unmatched bao phủ đủ consequent high.
        high_rules = []
        for confidence_level in ("low", "medium", "high"):
            for unmatched_level in ("low", "medium", "high"):
                consequent = high_delta["keep"]
                if confidence_level == "high" and unmatched_level == "high":
                    consequent = high_delta["decrease"]
                elif confidence_level == "high" and unmatched_level == "low":
                    consequent = high_delta["increase"]
                high_rules.append(ctrl.Rule(
                    low_conf[confidence_level] & unmatched[unmatched_level], consequent
                ))
        # Mọi mức unmatched/spread đều có kết luận để centroid không bị rỗng.
        buffer_rules = [
            ctrl.Rule(unmatched["high"], buffer_delta["increase"]),
            ctrl.Rule(unmatched["medium"], buffer_delta["keep"]),
            ctrl.Rule(unmatched["low"], buffer_delta["keep"]),
        ]
        spread_rules = [
            ctrl.Rule(spread["high"], low_delta["decrease"]),
            ctrl.Rule(spread["medium"], low_delta["keep"]),
            ctrl.Rule(spread["low"], low_delta["keep"]),
        ]
        rules = [*high_rules, *buffer_rules, *spread_rules]
        self._simulation = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

    def adjust(
        self,
        *,
        low_conf_ratio: float,
        unmatched_ratio: float,
        confidence_spread: float,
        track_high_thresh: float,
        track_low_thresh: float,
        new_track_thresh: float,
        track_buffer: int,
    ) -> FuzzyAdjustment:
        """Tính Mamdani output rồi chặn tham số trong miền hợp lệ."""
        inputs = {
            "low_conf_ratio": self._unit_interval(low_conf_ratio),
            "unmatched_ratio": self._unit_interval(unmatched_ratio),
            "confidence_spread": self._unit_interval(confidence_spread),
        }
        # Fuzzification: đầu vào crisp được ánh xạ lên các membership function.
        for name, value in inputs.items():
            self._simulation.input[name] = value
        # Mamdani inference kích hoạt luật, gộp các consequent và centroid hóa.
        self._simulation.compute()
        high_change = float(self._simulation.output["high_thresh_delta"])
        low_change = float(self._simulation.output["low_thresh_delta"])
        buffer_change = float(self._simulation.output["buffer_delta"])

        # Clipping giữ ngưỡng hợp lệ, high không vượt ngưỡng tạo track mới.
        adjusted_high = float(np.clip(track_high_thresh + high_change, 0.0, new_track_thresh))
        adjusted_low = float(np.clip(track_low_thresh + low_change, 0.0, adjusted_high))
        adjusted_buffer = int(np.clip(round(track_buffer + buffer_change), 5, 90))
        return FuzzyAdjustment(
            **inputs,
            high_thresh_delta=high_change,
            low_thresh_delta=low_change,
            buffer_delta=buffer_change,
            track_high_thresh=adjusted_high,
            track_low_thresh=adjusted_low,
            track_buffer=adjusted_buffer,
        )

    @staticmethod
    def _unit_interval(value: float) -> float:
        """Bảo vệ miền Antecedent trước sai lệch số học hoặc dữ liệu biên."""
        return float(np.clip(value, 0.0, 1.0))


def fuzzy_adjustment_dict(adjustment: FuzzyAdjustment) -> dict[str, Any]:
    """Trả về dict thân thiện với CSV/JSON."""
    return adjustment.__dict__.copy()
