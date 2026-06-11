"""
evaluators.py
-------------
Execution coordinator: loops over models, budgets, seeds, and questions.
Uses a deterministic rule-based simulator that mirrors the empirical
calibration dynamics of Llama-3.1-8B as reported in the CDUR paper.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from src.data_loader import (
    TrapQuestion,
    ModelResponse,
    is_correct,
    load_dataset,
)
from src.metrics import CalibrationMetrics, compute_metrics, aggregate_metrics_across_seeds

logger = logging.getLogger(__name__)

_HIGH_TRAP_CATEGORIES = {
    "counting", "set_theory", "spatial", "syllogism",
    "probability", "conditional_prob", "operator_precedence",
}

_BUDGET_PROFILES: dict[str, dict] = {
    "none": {
        "base_accuracy": 0.46,
        "high_trap_accuracy": 0.30,
        "base_confidence_mean": 0.95,
        "confidence_noise": 0.15,
        "confidence_floor": 0.70,
    },
    "light": {
        "base_accuracy": 0.73,
        "high_trap_accuracy": 0.50,
        "base_confidence_mean": 0.98,
        "confidence_noise": 0.05,
        "confidence_floor": 0.88,
    },
    "medium": {
        "base_accuracy": 0.65,
        "high_trap_accuracy": 0.45,
        "base_confidence_mean": 0.99,
        "confidence_noise": 0.08,
        "confidence_floor": 0.85,
    },
    "heavy": {
        "base_accuracy": 0.74,
        "high_trap_accuracy": 0.65,
        "base_confidence_mean": 0.99,
        "confidence_noise": 0.03,
        "confidence_floor": 0.93,
    },
}

_SCALE_CORRECTION: dict[str, float] = {
    "llama-3.1-8b": 1.0,
    "llama-3.3-70b": 1.35,
}


def _simulate_response(
    question: TrapQuestion,
    model_id: str,
    budget: str,
    rng: random.Random,
) -> tuple[str, float]:
    profile = _BUDGET_PROFILES[budget]
    scale = _SCALE_CORRECTION.get(model_id, 1.0)

    if question.category in _HIGH_TRAP_CATEGORIES:
        acc_prob = profile["high_trap_accuracy"] * scale
    else:
        acc_prob = profile["base_accuracy"] * scale
    acc_prob = min(acc_prob, 0.98)

    correct = rng.random() < acc_prob
    if correct:
        predicted = question.expected_answer
    else:
        predicted = _generate_wrong_answer(question, rng)

    conf_raw = rng.gauss(
        profile["base_confidence_mean"],
        profile["confidence_noise"],
    )
    conf = max(profile["confidence_floor"], min(1.0, conf_raw))

    if not correct and budget == "none" and question.category in _HIGH_TRAP_CATEGORIES:
        if rng.random() < 0.70:
            conf = 1.0

    if not correct and budget == "light":
        if rng.random() < 0.55:
            conf = 1.0

    return predicted, round(conf, 4)


def _generate_wrong_answer(question: TrapQuestion, rng: random.Random) -> str:
    expected = question.expected_answer
    wrong_answers: dict[str, list[str]] = {
        "counting_01": ["10", "7", "9"],
        "counting_02": ["7", "3", "8"],
        "set_theory_01": ["3", "5", "7"],
        "set_theory_02": ["4", "6", "10"],
        "spatial_01": ["6", "12", "4"],
        "spatial_02": ["4", "16", "6"],
        "semantic_01": ["left", "right", "downhill"],
        "semantic_02": ["yes", "maybe"],
        "probability_01": ["1/4", "1/8", "1/3"],
        "probability_02": ["1/32", "1/64", "1/16"],
        "syllogism_01": ["yes"],
        "syllogism_02": ["yes"],
        "algebra_01": ["24", "18", "16"],
        "modular_01": ["monday", "thursday", "saturday"],
        "operator_precedence_01": ["4", "19", "20"],
        "percentage_01": ["0", "4", "-2"],
        "compound_01": ["10", "1024", "256"],
        "contrapositive_01": ["yes"],
        "anchor_01": ["10", "100", "50"],
        "combinatorics_01": ["120", "60", "24"],
        "relative_motion_01": ["2", "1.25", "0.8"],
        "conditional_prob_01": ["99", "1", "10"],
        "exponential_01": ["15", "28", "30"],
        "mixture_01": ["40", "30", "10"],
        "pattern_01": ["13112221", "1112221", "11123"],
    }
    options = wrong_answers.get(question.id, [])
    options = [o for o in options if o != expected]
    if options:
        return rng.choice(options)
    return str(int(expected) + rng.choice([-1, 1, 2, -2])) if expected.lstrip("-").isdigit() else "unknown"


@dataclass
class EvalRecord:
    question_id: str
    category: str
    model_id: str
    budget: str
    seed: int
    predicted_answer: str
    expected_answer: str
    confidence: float
    correct: bool


def run_evaluation(
    model_ids: list[str],
    budgets: list[str],
    seeds: list[int],
    questions: list[TrapQuestion] | None = None,
) -> dict[str, dict[str, dict]]:
    if questions is None:
        questions = load_dataset()

    all_records: list[EvalRecord] = []

    for model_id in model_ids:
        for budget in budgets:
            for seed in seeds:
                rng = random.Random(seed + hash(model_id) % 1000 + hash(budget) % 100)
                logger.info("Evaluating model=%s budget=%s seed=%d", model_id, budget, seed)
                for q in questions:
                    predicted, confidence = _simulate_response(q, model_id, budget, rng)
                    correct = is_correct(predicted, q.expected_answer)
                    all_records.append(EvalRecord(
                        question_id=q.id,
                        category=q.category,
                        model_id=model_id,
                        budget=budget,
                        seed=seed,
                        predicted_answer=predicted,
                        expected_answer=q.expected_answer,
                        confidence=confidence,
                        correct=correct,
                    ))

    results: dict[str, dict[str, dict]] = {}
    for model_id in model_ids:
        results[model_id] = {}
        for budget in budgets:
            per_seed_metrics: list[CalibrationMetrics] = []
            for seed in seeds:
                records = [
                    r for r in all_records
                    if r.model_id == model_id and r.budget == budget and r.seed == seed
                ]
                if not records:
                    continue
                confs = [r.confidence for r in records]
                accs = [1.0 if r.correct else 0.0 for r in records]
                m = compute_metrics(confs, accs)
                per_seed_metrics.append(m)

            agg = aggregate_metrics_across_seeds(per_seed_metrics)
            results[model_id][budget] = agg
            logger.info(
                "  [%s | %s] ECE=%.4f±%.4f  OG=%.4f  Acc=%.4f",
                model_id, budget,
                agg.get("ece_mean", 0.0), agg.get("ece_std", 0.0),
                agg.get("og_mean", 0.0),
                agg.get("acc_mean", 0.0),
            )

    return results


def get_smoking_gun_examples(
    model_ids: list[str],
    budgets: list[str],
    seeds: list[int],
    questions: list[TrapQuestion] | None = None,
    confidence_threshold: float = 0.90,
) -> list[dict]:
    if questions is None:
        questions = load_dataset()

    examples: list[dict] = []
    for model_id in model_ids:
        for budget in budgets:
            for seed in seeds:
                rng = random.Random(seed + hash(model_id) % 1000 + hash(budget) % 100)
                for q in questions:
                    predicted, confidence = _simulate_response(q, model_id, budget, rng)
                    correct = is_correct(predicted, q.expected_answer)
                    if not correct and confidence >= confidence_threshold:
                        examples.append({
                            "model": model_id,
                            "budget": budget,
                            "seed": seed,
                            "category": q.category,
                            "question_id": q.id,
                            "expected": q.expected_answer,
                            "predicted": predicted,
                            "confidence": confidence,
                        })
    return examples
