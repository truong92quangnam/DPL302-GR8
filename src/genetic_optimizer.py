"""Genetic Algorithm tìm tham số ByteTrack bằng validation ground truth."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

PARAMETER_BOUNDS = {
    "track_high_thresh": (0.10, 0.60),
    "track_low_thresh": (0.05, 0.30),
    "new_track_thresh": (0.10, 0.70),
    "track_buffer": (5, 90),
    "match_thresh": (0.50, 0.95),
}


@dataclass
class Candidate:
    """Một chromosome và kết quả fitness đã đo trên validation."""

    parameters: dict[str, float | int]
    fitness: float = float("-inf")
    metrics: dict[str, float] | None = None


class GeneticOptimizer:
    """GA real-coded với elitism, crossover và mutation."""

    def __init__(
        self,
        evaluator: Callable[[dict[str, float | int], str], dict[str, float]],
        *,
        population_size: int = 10,
        generations: int = 10,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elitism: int = 2,
        seed: int = 42,
        speed_aware: bool = False,
    ) -> None:
        if population_size < 2 or not 1 <= elitism < population_size:
            raise ValueError("Cần population_size >= 2 và 1 <= elitism < population_size")
        self.evaluator = evaluator
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.seed = seed
        self.speed_aware = speed_aware
        self.rng = random.Random(seed)
        self.history: list[dict[str, Any]] = []

    def optimize(self, output_dir: Path, config_output: Path, base_config: dict[str, Any]) -> Candidate:
        """Chạy GA; callback evaluator nhận tên riêng biệt cho từng cá thể."""
        output_dir.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        history_path = output_dir / f"ga_history_seed{self.seed}.csv"
        population = [Candidate(self._random_parameters()) for _ in range(self.population_size)]
        best: Candidate | None = None
        for generation in range(self.generations):
            for index, candidate in enumerate(population):
                started = time.perf_counter()
                evaluation_id = f"seed{self.seed}_g{generation:02d}_i{index:02d}"
                metrics = self.evaluator(candidate.parameters, evaluation_id)
                candidate.metrics = metrics
                candidate.fitness = self._fitness(metrics)
                record = {
                    "generation": generation,
                    "individual_id": evaluation_id,
                    "parameters": candidate.parameters.copy(),
                    "fitness": candidate.fitness,
                    "HOTA": metrics["HOTA"],
                    "IDF1": metrics["IDF1"],
                    "MOTA": metrics["MOTA"],
                    "FPS": metrics["FPS"],
                    "evaluation_time": time.perf_counter() - started,
                }
                self.history.append(record)
                print(f"[GA] {evaluation_id}: fitness={candidate.fitness:.4f}")
            # Ghi sau mỗi generation để vẫn giữ lịch sử nếu lần tối ưu bị gián đoạn.
            pd.DataFrame(self.history).to_csv(history_path, index=False)
            population.sort(key=lambda item: item.fitness, reverse=True)
            if best is None or population[0].fitness > best.fitness:
                best = Candidate(population[0].parameters.copy(), population[0].fitness, population[0].metrics)
            if generation + 1 < self.generations:
                population = self._next_generation(population)

        if best is None:
            raise RuntimeError("GA không tạo được cá thể nào")
        result_config = base_config.copy()
        result_config.update(best.parameters)
        result_config["tracker_type"] = "bytetrack"
        config_output.parent.mkdir(parents=True, exist_ok=True)
        config_output.write_text(yaml.safe_dump(result_config, sort_keys=False), encoding="utf-8")
        return best

    def _fitness(self, metrics: dict[str, float]) -> float:
        """Tính đúng hàm mục tiêu; FPS chuẩn hóa theo mức tham chiếu 30 FPS."""
        if self.speed_aware:
            normalized_fps = float(np.clip(metrics["FPS"] / 30.0, 0.0, 1.0))
            return 0.4 * metrics["HOTA"] + 0.25 * metrics["IDF1"] + 0.15 * metrics["MOTA"] + 0.2 * normalized_fps
        return 0.5 * metrics["HOTA"] + 0.3 * metrics["IDF1"] + 0.2 * metrics["MOTA"]

    def _random_parameters(self) -> dict[str, float | int]:
        """Lấy chromosome trong miền và chiếu về thứ tự ngưỡng hợp lệ."""
        values: dict[str, float | int] = {
            key: self.rng.uniform(low, high)
            for key, (low, high) in PARAMETER_BOUNDS.items()
        }
        values["track_buffer"] = int(round(float(values["track_buffer"])))
        return self._repair(values)

    def _next_generation(self, ranked: list[Candidate]) -> list[Candidate]:
        """Giữ elite, chọn tournament, lai ghép và đột biến."""
        next_population = [Candidate(item.parameters.copy()) for item in ranked[:self.elitism]]
        while len(next_population) < self.population_size:
            parent_a = self._tournament(ranked)
            parent_b = self._tournament(ranked)
            child = parent_a.parameters.copy()
            if self.rng.random() < self.crossover_rate:
                for key in PARAMETER_BOUNDS:
                    if self.rng.random() < 0.5:
                        child[key] = parent_b.parameters[key]
            for key, (low, high) in PARAMETER_BOUNDS.items():
                if self.rng.random() < self.mutation_rate:
                    if key == "track_buffer":
                        child[key] = self.rng.randint(int(low), int(high))
                    else:
                        child[key] = self.rng.uniform(low, high)
            next_population.append(Candidate(self._repair(child)))
        return next_population

    def _tournament(self, ranked: list[Candidate], size: int = 3) -> Candidate:
        """Chọn cá thể tốt nhất trong ngẫu nhiên tối đa ba ứng viên."""
        sample = self.rng.sample(ranked, min(size, len(ranked)))
        return max(sample, key=lambda item: item.fitness)

    @staticmethod
    def _repair(values: dict[str, float | int]) -> dict[str, float | int]:
        """Chiếu chromosome vào miền gene và đảm bảo thứ tự threshold."""
        repaired: dict[str, float | int] = {}
        for key, (low, high) in PARAMETER_BOUNDS.items():
            repaired[key] = int(np.clip(round(float(values[key])), low, high)) if key == "track_buffer" else float(
                np.clip(float(values[key]), low, high)
            )
        repaired["track_high_thresh"] = min(
            float(repaired["track_high_thresh"]), float(repaired["new_track_thresh"])
        )
        repaired["track_low_thresh"] = min(
            float(repaired["track_low_thresh"]), float(repaired["track_high_thresh"])
        )
        return repaired
