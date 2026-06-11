"""
metrics.py
----------
Mathematically exact calibration metrics: ECE, overconfidence gap,
and wrong-and-confident count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    ece: float
    overconfidence_gap: float
    mean_confidence: float
    mean_accuracy: float
    wrong_and_confident_count: int
    n_samples: int
    bin_stats: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ece": round(self.ece, 4),
            "overconfidence_gap": round(self.overconfidence_gap, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "wrong_and_confident_count": self.wrong_and_confident_count,
            "n_samples": self.n_samples,
        }


def calculate_ece(
    confidences: list[float],
    accuracies: list[float],
    num_bins: int = 10,
) -> tuple[float, list[dict]]:
    if not confidences or not accuracies:
        logger.warning("Empty input to calculate_ece; returning ECE=0.0.")
        return 0.0, []

    if len(confidences) != len(accuracies):
        raise ValueError(
            f"Length mismatch: confidences={len(confidences)}, "
            f"accuracies={len(accuracies)}"
        )

    confs = np.array(confidences, dtype=np.float64)
    accs = np.array(accuracies, dtype=np.float64)
    n = len(confs)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece_sum = 0.0
    bin_stats: list[dict] = []

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)

        bin_confs = confs[mask]
        bin_accs = accs[mask]
        bin_size = int(mask.sum())

        if bin_size == 0:
            bin_stats.append({
                "bin": i,
                "lower": round(lo, 2),
                "upper": round(hi, 2),
                "n": 0,
                "avg_conf": None,
                "avg_acc": None,
                "gap": None,
            })
            continue

        avg_conf = float(bin_confs.mean())
        avg_acc = float(bin_accs.mean())
        gap = abs(avg_conf - avg_acc)
        ece_sum += (bin_size / n) * gap

        bin_stats.append({
            "bin": i,
            "lower": round(lo, 2),
            "upper": round(hi, 2),
            "n": bin_size,
            "avg_conf": round(avg_conf, 4),
            "avg_acc": round(avg_acc, 4),
            "gap": round(gap, 4),
        })

    return float(ece_sum), bin_stats


def calculate_overconfidence_gap(
    confidences: list[float],
    accuracies: list[float],
) -> float:
    if not confidences or not accuracies:
        return 0.0
    mean_conf = float(np.mean(confidences))
    mean_acc = float(np.mean(accuracies))
    return round(mean_conf - mean_acc, 4)


def count_wrong_and_confident(
    confidences: list[float],
    accuracies: list[float],
    threshold: float = 0.90,
) -> int:
    return sum(
        1 for c, a in zip(confidences, accuracies)
        if c >= threshold and a == 0.0
    )


def compute_metrics(
    confidences: list[float],
    accuracies: list[float],
    num_bins: int = 10,
    wrong_conf_threshold: float = 0.90,
) -> CalibrationMetrics:
    if not confidences:
        logger.warning("No valid samples; returning zero metrics.")
        return CalibrationMetrics(
            ece=0.0,
            overconfidence_gap=0.0,
            mean_confidence=0.0,
            mean_accuracy=0.0,
            wrong_and_confident_count=0,
            n_samples=0,
        )

    ece, bin_stats = calculate_ece(confidences, accuracies, num_bins=num_bins)
    og = calculate_overconfidence_gap(confidences, accuracies)
    wac = count_wrong_and_confident(confidences, accuracies, threshold=wrong_conf_threshold)

    return CalibrationMetrics(
        ece=ece,
        overconfidence_gap=og,
        mean_confidence=float(np.mean(confidences)),
        mean_accuracy=float(np.mean(accuracies)),
        wrong_and_confident_count=wac,
        n_samples=len(confidences),
        bin_stats=bin_stats,
    )


def aggregate_metrics_across_seeds(
    per_seed_metrics: list[CalibrationMetrics],
) -> dict:
    if not per_seed_metrics:
        return {}

    eces = [m.ece for m in per_seed_metrics]
    ogs = [m.overconfidence_gap for m in per_seed_metrics]
    accs = [m.mean_accuracy for m in per_seed_metrics]

    return {
        "ece_mean": round(float(np.mean(eces)), 4),
        "ece_std": round(float(np.std(eces)), 4),
        "og_mean": round(float(np.mean(ogs)), 4),
        "og_std": round(float(np.std(ogs)), 4),
        "acc_mean": round(float(np.mean(accs)), 4),
        "acc_std": round(float(np.std(accs)), 4),
        "n_seeds": len(per_seed_metrics),
    }
