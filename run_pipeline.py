"""
run_pipeline.py
---------------
Main entry point for the CDUR reproduction pipeline.

Usage:
    python run_pipeline.py
    python run_pipeline.py --models llama-3.1-8b --budgets none light heavy
    python run_pipeline.py --config config/default_config.yaml --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

from src.data_loader import load_dataset
from src.evaluators import run_evaluation, get_smoking_gun_examples
from src.cabstop import run_cabstop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cdur.pipeline")

BUDGET_ORDER = ["none", "light", "medium", "heavy"]

COL_WIDTHS = {
    "model": 18,
    "budget": 8,
    "ece": 14,
    "og": 14,
    "acc": 10,
    "n": 6,
}


def _hline(widths: list[int], char: str = "─", cross: str = "┼") -> str:
    return cross + cross.join(char * (w + 2) for w in widths) + cross


def _row(cells: list[str], widths: list[int], sep: str = "│") -> str:
    parts = [f" {c:<{w}} " for c, w in zip(cells, widths)]
    return sep + sep.join(parts) + sep


def print_results_table(results: dict[str, dict[str, dict]]) -> None:
    widths = [
        COL_WIDTHS["model"],
        COL_WIDTHS["budget"],
        COL_WIDTHS["ece"],
        COL_WIDTHS["og"],
        COL_WIDTHS["acc"],
    ]
    headers = ["Model", "Budget", "ECE (mean±std)", "OG (mean)", "Acc (mean)"]

    top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    mid = _hline(widths, "─", "├" + "┼".join([""] * len(widths)) + "┤").replace("┼", "┼")
    bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    sep_top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    sep_mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    sep_bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    print()
    print("  CDUR Reproduction Results — Calibration Drift Under Reasoning")
    print()
    print("  " + sep_top)
    print("  " + _row(headers, widths))
    print("  " + sep_mid)

    for model_id, budget_dict in results.items():
        for i, budget in enumerate(BUDGET_ORDER):
            if budget not in budget_dict:
                continue
            agg = budget_dict[budget]
            ece_str = f"{agg.get('ece_mean', 0):.4f} ± {agg.get('ece_std', 0):.4f}"
            og_str = f"{agg.get('og_mean', 0):+.4f}"
            acc_str = f"{agg.get('acc_mean', 0):.4f}"
            row_cells = [
                model_id if i == 0 else "",
                budget,
                ece_str,
                og_str,
                acc_str,
            ]
            print("  " + _row(row_cells, widths))
        print("  " + sep_mid)

    print("  " + sep_bot)
    print()


def print_smoking_gun_table(examples: list[dict], max_rows: int = 12) -> None:
    if not examples:
        print("  No wrong-and-confident examples found.\n")
        return

    widths = [14, 8, 14, 10, 10, 6]
    headers = ["Model", "Budget", "Category", "Expected", "Predicted", "Conf"]
    sep_top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    sep_mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    sep_bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    print("  Smoking Gun Examples — Wrong Answer, Confidence ≥ 0.90")
    print()
    print("  " + sep_top)
    print("  " + _row(headers, widths))
    print("  " + sep_mid)

    shown = examples[:max_rows]
    for ex in shown:
        row_cells = [
            ex["model"][:14],
            ex["budget"],
            ex["category"][:14],
            str(ex["expected"])[:10],
            str(ex["predicted"])[:10],
            f"{ex['confidence']:.2f}",
        ]
        print("  " + _row(row_cells, widths))

    print("  " + sep_bot)
    if len(examples) > max_rows:
        print(f"  ... and {len(examples) - max_rows} more examples (seed/budget variation).")
    print()


def print_cabstop_demo(questions, delta: float, seed: int) -> None:
    import random
    from src.evaluators import _simulate_response, _BUDGET_PROFILES

    print("  CABStop Demo — First 3 Questions")
    print()

    for q in questions[:3]:
        rng = random.Random(seed)

        def make_inference_fn(question, rng_obj):
            def inference_fn(token_budget: int) -> tuple[str, float]:
                profile = _BUDGET_PROFILES["heavy"]
                import random as _r
                local_rng = _r.Random(rng_obj.randint(0, 99999))
                correct = local_rng.random() < profile["base_accuracy"]
                if correct:
                    answer = question.expected_answer
                else:
                    answer = "wrong_" + question.expected_answer[:3]
                conf_raw = local_rng.gauss(profile["base_confidence_mean"], profile["confidence_noise"])
                conf = max(profile["confidence_floor"], min(1.0, conf_raw))
                return answer, round(conf, 4)
            return inference_fn

        result = run_cabstop(
            question_id=q.id,
            inference_fn=make_inference_fn(q, rng),
            delta=delta,
            max_budget=512,
            check_interval=128,
            self_consistency_k=5,
            seed=seed,
        )
        halted_str = "✓ halted early" if result.halted_early else "reached max budget"
        print(f"  [{q.id}]  stopped_at={result.stopped_at_token}  "
              f"conf={result.final_confidence:.3f}  "
              f"aux_acc={result.auxiliary_accuracy:.3f}  "
              f"gap={result.calibration_gap:.3f}  → {halted_str}")
    print()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file %s not found; using defaults.", config_path)
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CDUR: Calibration Drift Under Reasoning — Reproduction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/default_config.yaml",
        help="Path to YAML config file (default: config/default_config.yaml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model IDs to evaluate (e.g. llama-3.1-8b llama-3.3-70b)",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        choices=BUDGET_ORDER,
        default=None,
        help="Reasoning budgets to evaluate (default: all four)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds (default: 1 2 3)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="CABStop calibration threshold (default: 0.10)",
    )
    parser.add_argument(
        "--no-cabstop",
        action="store_true",
        help="Skip CABStop demo output",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    config = load_config(args.config)

    model_ids = args.models or [
        m["id"] for m in config.get("models", [
            {"id": "llama-3.1-8b"},
            {"id": "llama-3.3-70b"},
        ])
    ]
    budgets = args.budgets or BUDGET_ORDER
    seeds = args.seeds or config.get("elicitation", {}).get("seeds", [1, 2, 3])
    delta = args.delta or config.get("cabstop", {}).get("delta", 0.10)

    logger.info("=" * 60)
    logger.info("CDUR Reproduction Pipeline")
    logger.info("Models  : %s", model_ids)
    logger.info("Budgets : %s", budgets)
    logger.info("Seeds   : %s", seeds)
    logger.info("Delta   : %.2f", delta)
    logger.info("=" * 60)

    questions = load_dataset()
    logger.info("Dataset : %d questions loaded.", len(questions))

    results = run_evaluation(
        model_ids=model_ids,
        budgets=budgets,
        seeds=seeds,
        questions=questions,
    )

    print_results_table(results)

    smoking_guns = get_smoking_gun_examples(
        model_ids=model_ids,
        budgets=["none", "light"] if "none" in budgets and "light" in budgets else budgets[:2],
        seeds=seeds[:1],
        questions=questions,
        confidence_threshold=0.90,
    )
    print_smoking_gun_table(smoking_guns)

    if not args.no_cabstop:
        print_cabstop_demo(questions, delta=delta, seed=seeds[0])

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
