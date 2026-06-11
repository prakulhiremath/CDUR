"""
data_loader.py
--------------
Hardcoded reasoning-trap dataset loader and response validity filter.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrapQuestion:
    id: str
    category: str
    question: str
    expected_answer: str


@dataclass
class ModelResponse:
    question_id: str
    model_id: str
    budget: str
    seed: int
    raw_answer: str
    parsed_answer: Optional[str]
    confidence: Optional[float]
    is_valid: bool
    validity_reason: str


_TRAP_QUESTIONS: list[dict] = [
    {
        "id": "counting_01",
        "category": "counting",
        "question": (
            "A frog climbs 3 meters up a 10-meter wall each day and slides back "
            "2 meters each night. How many days does it take to reach the top?"
        ),
        "expected_answer": "8",
    },
    {
        "id": "counting_02",
        "category": "counting",
        "question": (
            "How many integers from 1 to 100 inclusive are divisible by both 3 and 5?"
        ),
        "expected_answer": "6",
    },
    {
        "id": "set_theory_01",
        "category": "set_theory",
        "question": (
            "In a class of 30 students, 18 play football, 15 play cricket, and "
            "5 play both. How many students play neither sport?"
        ),
        "expected_answer": "2",
    },
    {
        "id": "set_theory_02",
        "category": "set_theory",
        "question": (
            "Set A has 20 elements, Set B has 15 elements, and their union has "
            "30 elements. How many elements are in their intersection?"
        ),
        "expected_answer": "5",
    },
    {
        "id": "spatial_01",
        "category": "spatial",
        "question": (
            "A cube has side length 2. What is the number of unit cubes it can "
            "be divided into?"
        ),
        "expected_answer": "8",
    },
    {
        "id": "spatial_02",
        "category": "spatial",
        "question": (
            "If you fold a square piece of paper in half three times and then "
            "unfold it, how many equal rectangles are created?"
        ),
        "expected_answer": "8",
    },
    {
        "id": "semantic_01",
        "category": "semantic",
        "question": (
            "A rooster lays an egg on the peak of a roof. Which way does the "
            "egg roll?"
        ),
        "expected_answer": "roosters do not lay eggs",
    },
    {
        "id": "semantic_02",
        "category": "semantic",
        "question": (
            "Is it legal for a man to marry his widow's sister in the state of "
            "California?"
        ),
        "expected_answer": "no",
    },
    {
        "id": "probability_01",
        "category": "probability",
        "question": (
            "A box contains 2 red and 2 blue balls. You draw 2 balls without "
            "replacement. What is the probability both are red? Express as a fraction."
        ),
        "expected_answer": "1/6",
    },
    {
        "id": "probability_02",
        "category": "probability",
        "question": (
            "You flip a fair coin 5 times and get heads every time. What is the "
            "probability of getting heads on the 6th flip?"
        ),
        "expected_answer": "1/2",
    },
    {
        "id": "syllogism_01",
        "category": "syllogism",
        "question": (
            "All A are B. All B are C. Is it necessarily true that all C are A? "
            "Answer yes or no."
        ),
        "expected_answer": "no",
    },
    {
        "id": "syllogism_02",
        "category": "syllogism",
        "question": (
            "Some cats are black. All black things absorb light. Does it follow "
            "that all cats absorb light? Answer yes or no."
        ),
        "expected_answer": "no",
    },
    {
        "id": "algebra_01",
        "category": "algebra",
        "question": (
            "If x + y = 10 and x - y = 4, what is the value of x * y?"
        ),
        "expected_answer": "21",
    },
    {
        "id": "modular_01",
        "category": "modular",
        "question": (
            "If today is Monday, what day of the week will it be 100 days from now?"
        ),
        "expected_answer": "wednesday",
    },
    {
        "id": "operator_precedence_01",
        "category": "operator_precedence",
        "question": (
            "What is the value of 2 + 3 * 4 - 1?"
        ),
        "expected_answer": "13",
    },
    {
        "id": "percentage_01",
        "category": "percentage",
        "question": (
            "A price increases by 20% and then decreases by 20%. What is the "
            "net percentage change?"
        ),
        "expected_answer": "-4",
    },
    {
        "id": "compound_01",
        "category": "compound",
        "question": (
            "A snail doubles the distance it travels each day. On day 1 it travels "
            "1 cm. How far does it travel on day 10?"
        ),
        "expected_answer": "512",
    },
    {
        "id": "contrapositive_01",
        "category": "contrapositive",
        "question": (
            "If it rains, the ground is wet. The ground is wet. Does it necessarily "
            "follow that it rained? Answer yes or no."
        ),
        "expected_answer": "no",
    },
    {
        "id": "anchor_01",
        "category": "anchor",
        "question": (
            "A bat and a ball together cost $1.10. The bat costs $1.00 more than "
            "the ball. How much does the ball cost in cents?"
        ),
        "expected_answer": "5",
    },
    {
        "id": "combinatorics_01",
        "category": "combinatorics",
        "question": (
            "How many ways can you arrange the letters in the word LEVEL?"
        ),
        "expected_answer": "30",
    },
    {
        "id": "relative_motion_01",
        "category": "relative_motion",
        "question": (
            "Two trains start 200 km apart and head toward each other. Train A "
            "travels at 80 km/h and Train B at 120 km/h. How many hours until they meet?"
        ),
        "expected_answer": "1",
    },
    {
        "id": "conditional_prob_01",
        "category": "conditional_prob",
        "question": (
            "A disease affects 1% of the population. A test is 99% accurate. "
            "You test positive. What is the approximate probability you have the "
            "disease? Answer as a percentage rounded to the nearest whole number."
        ),
        "expected_answer": "50",
    },
    {
        "id": "exponential_01",
        "category": "exponential",
        "question": (
            "A lily pad doubles in size every day. It covers the whole pond on "
            "day 30. On what day did it cover half the pond?"
        ),
        "expected_answer": "29",
    },
    {
        "id": "mixture_01",
        "category": "mixture",
        "question": (
            "A solution is 40% alcohol. You add pure water to double the volume. "
            "What is the new alcohol percentage?"
        ),
        "expected_answer": "20",
    },
    {
        "id": "pattern_01",
        "category": "pattern",
        "question": (
            "What is the next number in this sequence: 1, 11, 21, 1211, 111221, ?"
        ),
        "expected_answer": "312211",
    },
]


def load_dataset() -> list[TrapQuestion]:
    questions = []
    for item in _TRAP_QUESTIONS:
        questions.append(
            TrapQuestion(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                expected_answer=item["expected_answer"],
            )
        )
    logger.info("Loaded %d trap questions across %d categories.",
                len(questions),
                len({q.category for q in questions}))
    return questions


def parse_and_validate_response(
    question_id: str,
    model_id: str,
    budget: str,
    seed: int,
    raw_text: str,
) -> ModelResponse:
    parsed_answer: Optional[str] = None
    confidence: Optional[float] = None
    is_valid = False
    validity_reason = "unknown"

    if not raw_text or not raw_text.strip():
        return ModelResponse(
            question_id=question_id,
            model_id=model_id,
            budget=budget,
            seed=seed,
            raw_answer=raw_text,
            parsed_answer=None,
            confidence=None,
            is_valid=False,
            validity_reason="empty response",
        )

    refusal_patterns = [
        r"\bi (cannot|can't|won't|will not) answer\b",
        r"\bi refuse\b",
        r"\bi am unable\b",
    ]
    for pat in refusal_patterns:
        if re.search(pat, raw_text, re.IGNORECASE):
            return ModelResponse(
                question_id=question_id,
                model_id=model_id,
                budget=budget,
                seed=seed,
                raw_answer=raw_text,
                parsed_answer=None,
                confidence=None,
                is_valid=False,
                validity_reason="refusal detected",
            )

    conf_match = re.search(
        r"confidence[:\s]+([01](?:\.\d+)?|\.\d+|0?\.\d+|1\.0)",
        raw_text,
        re.IGNORECASE,
    )
    if conf_match:
        try:
            val = float(conf_match.group(1))
            if 0.0 <= val <= 1.0:
                confidence = val
        except ValueError:
            pass

    if confidence is None:
        pct_match = re.search(r"(\d{1,3})\s*%", raw_text)
        if pct_match:
            try:
                val = float(pct_match.group(1)) / 100.0
                if 0.0 <= val <= 1.0:
                    confidence = val
            except ValueError:
                pass

    answer_match = re.search(
        r"(?:answer[:\s]+|therefore[,\s]+|result[:\s]+|=\s*)([^\n\.]+)",
        raw_text,
        re.IGNORECASE,
    )
    if answer_match:
        parsed_answer = answer_match.group(1).strip()
    else:
        lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]
        if lines:
            last = lines[-1]
            if re.search(r"confidence", last, re.IGNORECASE):
                parsed_answer = lines[-2] if len(lines) >= 2 else lines[-1]
            else:
                parsed_answer = last

    if parsed_answer and confidence is not None:
        is_valid = True
        validity_reason = "ok"
    elif parsed_answer is None:
        validity_reason = "no parseable answer"
    else:
        validity_reason = "no parseable confidence"

    return ModelResponse(
        question_id=question_id,
        model_id=model_id,
        budget=budget,
        seed=seed,
        raw_answer=raw_text,
        parsed_answer=parsed_answer,
        confidence=confidence,
        is_valid=is_valid,
        validity_reason=validity_reason,
    )


def normalize_answer(answer: str) -> str:
    answer = answer.lower().strip()
    answer = re.sub(r"[^\w/\.]", "", answer)
    return answer


def is_correct(parsed_answer: Optional[str], expected_answer: str) -> bool:
    if parsed_answer is None:
        return False
    return normalize_answer(parsed_answer) == normalize_answer(expected_answer)
