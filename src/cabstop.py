"""
cabstop.py
----------
CABStop: Confidence-Accuracy Budget Stopping algorithm.
Implements Algorithm 1 from the CDUR paper (Section 6).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class CABStopResult:
    question_id: str
    stopped_at_token: int
    final_answer: str
    final_confidence: float
    auxiliary_accuracy: float
    calibration_gap: float
    halted_early: bool
    steps: list[dict]


def _simulate_self_consistency(
    answer: str,
    budget: int,
    k: int,
    rng: random.Random,
) -> float:
    noise_scale = max(0.05, 0.3 - budget / 10000.0)
    agreement_count = 0
    for _ in range(k):
        flip = rng.random()
        if flip > noise_scale:
            agreement_count += 1
    return agreement_count / k


def run_cabstop(
    question_id: str,
    inference_fn: Callable[[int], tuple[str, float]],
    delta: float = 0.10,
    max_budget: int = 2048,
    check_interval: int = 128,
    self_consistency_k: int = 5,
    seed: int = 42,
) -> CABStopResult:
    rng = random.Random(seed)
    steps: list[dict] = []
    t = 0

    current_answer = ""
    current_confidence = 0.0

    while t < max_budget:
        t = min(t + check_interval, max_budget)

        current_answer, current_confidence = inference_fn(t)

        auxiliary_accuracy = _simulate_self_consistency(
            answer=current_answer,
            budget=t,
            k=self_consistency_k,
            rng=rng,
        )

        gap = current_confidence - auxiliary_accuracy

        step_record = {
            "token": t,
            "answer": current_answer,
            "confidence": round(current_confidence, 4),
            "auxiliary_accuracy": round(auxiliary_accuracy, 4),
            "gap": round(gap, 4),
        }
        steps.append(step_record)
        logger.debug(
            "CABStop [%s] t=%d | conf=%.3f aux_acc=%.3f gap=%.3f",
            question_id, t, current_confidence, auxiliary_accuracy, gap,
        )

        if gap > delta:
            logger.info(
                "CABStop halting [%s] at t=%d: gap=%.3f > delta=%.3f",
                question_id, t, gap, delta,
            )
            return CABStopResult(
                question_id=question_id,
                stopped_at_token=t,
                final_answer=current_answer,
                final_confidence=current_confidence,
                auxiliary_accuracy=auxiliary_accuracy,
                calibration_gap=gap,
                halted_early=True,
                steps=steps,
            )

    return CABStopResult(
        question_id=question_id,
        stopped_at_token=max_budget,
        final_answer=current_answer,
        final_confidence=current_confidence,
        auxiliary_accuracy=auxiliary_accuracy if steps else 0.0,
        calibration_gap=steps[-1]["gap"] if steps else 0.0,
        halted_early=False,
        steps=steps,
    )


def batch_cabstop(
    questions: list[dict],
    inference_fn_factory: Callable[[str], Callable[[int], tuple[str, float]]],
    delta: float = 0.10,
    max_budget: int = 2048,
    check_interval: int = 128,
    self_consistency_k: int = 5,
    seed: int = 42,
) -> list[CABStopResult]:
    results: list[CABStopResult] = []
    for q in questions:
        qid = q["id"]
        inference_fn = inference_fn_factory(qid)
        result = run_cabstop(
            question_id=qid,
            inference_fn=inference_fn,
            delta=delta,
            max_budget=max_budget,
            check_interval=check_interval,
            self_consistency_k=self_consistency_k,
            seed=seed,
        )
        results.append(result)

    n_halted = sum(1 for r in results if r.halted_early)
    logger.info(
        "CABStop batch complete: %d/%d halted early (delta=%.2f).",
        n_halted, len(results), delta,
    )
    return results
