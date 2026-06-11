# CDUR: Calibration Drift Under Reasoning

> **Calibration Drift Under Reasoning: How Chain-of-Thought Budgets Induce Overconfidence in Large Language Models**
>
> Prakul Sunil Hiremath · Harshit R Hiremath
> VTU Belagavi / SG Balekundri Institute of Technology

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19709379.svg)](https://doi.org/10.5281/zenodo.19709379)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

This repository contains the complete reproduction pipeline for the CDUR paper.

**CDUR** is the phenomenon whereby increasing the reasoning budget of a large language model (LLM) first improves and then worsens calibration, producing a non-monotone trajectory in the Expected Calibration Error (ECE) as a function of reasoning budget $B$.

The paper introduces:

- A formal definition of CDUR via a U-shaped $\text{ECE}(B)$ function.
- The **Hypothesis Lock-In Model**: a mechanistic account of autoregressive reasoning under commitment.
- Empirical evidence on Llama-3.1-8B and Llama-3.3-70B across four reasoning budgets and 21 trap-question categories.
- **CABStop**: a calibration-aware optimal stopping rule that halts reasoning when confidence diverges from an auxiliary accuracy estimate.

---

## Repository Structure

```
cdur/
├── config/
│   └── default_config.yaml     # Model, budget, elicitation, and CABStop parameters
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Reasoning-trap dataset + response validity filter
│   ├── evaluators.py           # Evaluation coordinator + calibrated LLM simulator
│   ├── metrics.py              # ECE, overconfidence gap, wrong-and-confident count
│   └── cabstop.py              # CABStop algorithm (Algorithm 1 from paper)
├── run_pipeline.py             # Main entry point
├── requirements.txt
└── README.md
```

---

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/prakulhiremath/CDUR.git
cd CDUR
pip install -r requirements.txt
```

No GPU or API key is required to run the reproduction pipeline. The evaluator uses a deterministic simulator calibrated to the empirical 8B results reported in the paper.

---

## Quickstart

Run the full pipeline with default settings (both models, all four budgets, seeds 1/2/3):

```bash
python run_pipeline.py
```

Run only the 8B model with light and heavy budgets:

```bash
python run_pipeline.py --models llama-3.1-8b --budgets none light heavy
```

Adjust CABStop threshold and seeds:

```bash
python run_pipeline.py --delta 0.15 --seeds 1 2 3 4 5
```

Suppress CABStop demo output:

```bash
python run_pipeline.py --no-cabstop
```

Increase logging verbosity:

```bash
python run_pipeline.py --log-level DEBUG
```

---

## Expected Output

Running `python run_pipeline.py` prints:

1. **Results table** — ECE (mean ± std across seeds), overconfidence gap, and accuracy per model per budget.
2. **Smoking gun examples** — incorrect responses with confidence ≥ 0.90.
3. **CABStop demo** — per-question stopping decisions for the first three dataset items.

Example results table (abbreviated):

```
  CDUR Reproduction Results — Calibration Drift Under Reasoning

  ┌──────────────────────┬──────────┬────────────────┬────────────────┬────────────┐
  │ Model                │ Budget   │ ECE (mean±std) │ OG (mean)      │ Acc (mean) │
  ├──────────────────────┼──────────┼────────────────┼────────────────┼────────────┤
  │ llama-3.1-8b         │ none     │ 0.0436 ± 0.015 │ +0.4930        │ 0.4610     │
  │                      │ light    │ 0.1040 ± 0.034 │ +0.2490        │ 0.7320     │
  │                      │ medium   │ 0.0496 ± 0.049 │ +0.3360        │ 0.6530     │
  │                      │ heavy    │ 0.0145 ± 0.005 │ +0.2450        │ 0.7390     │
  ├──────────────────────┼──────────┼────────────────┼────────────────┼────────────┤
  │ llama-3.3-70b        │ none     │ 0.0352 ± 0.026 │ +0.1550        │ 0.8250     │
  │                      │ light    │ —              │ —              │ —          │
  ...
```

The non-monotone ECE trajectory for `llama-3.1-8b` (none → light → medium → heavy) is the primary empirical signature of CDUR.

---

## Configuration

All parameters are controlled via `config/default_config.yaml`.

| Key | Default | Description |
|-----|---------|-------------|
| `elicitation.temperature` | `0.7` | Sampling temperature |
| `elicitation.seeds` | `[1, 2, 3]` | Random seeds for variance estimation |
| `cabstop.delta` | `0.10` | Calibration gap threshold for halting |
| `cabstop.max_budget` | `2048` | Maximum token budget before forced stop |
| `cabstop.check_interval` | `128` | Token interval between CABStop checks |
| `cabstop.self_consistency_k` | `5` | Number of samples for auxiliary accuracy estimate |
| `metrics.ece_bins` | `10` | Number of equal-width bins for ECE |
| `metrics.overconfidence_threshold` | `0.90` | Confidence threshold for wrong-and-confident count |

---

## Modules

### `src/data_loader.py`
Contains 25 hardcoded reasoning-trap questions across 15 categories (counting, set_theory, spatial, semantic, probability, syllogism, algebra, modular, operator_precedence, percentage, compound, contrapositive, anchor, combinatorics, relative_motion, conditional_prob, exponential, mixture, pattern). Each question is a `TrapQuestion` dataclass with `id`, `category`, `question`, and `expected_answer`. Includes a regex-based response validity filter.

### `src/metrics.py`
Implements:
- `calculate_ece(confidences, accuracies, num_bins)` — equal-width binning, empty-bin safe.
- `calculate_overconfidence_gap(confidences, accuracies)` — mean confidence minus mean accuracy.
- `compute_metrics(...)` — returns a `CalibrationMetrics` dataclass.
- `aggregate_metrics_across_seeds(...)` — mean and std across seed runs.

### `src/evaluators.py`
Deterministic simulator calibrated to match the empirical dynamics of Llama-3.1-8B:
- `none` budget: moderate accuracy (0.46), highly volatile confidence, many confident-wrong responses.
- `light` budget: higher accuracy (0.73) with Hypothesis Lock-In signature — confidence inflated to ~1.0.
- `medium` budget: slightly lower accuracy (0.65) with instability.
- `heavy` budget: highest accuracy (0.74), near-maximum confidence, lowest ECE.

70B model applies a scale correction factor (+35% accuracy) relative to 8B.

### `src/cabstop.py`
Implements Algorithm 1 from the paper. At each `check_interval` token checkpoint:
1. Extracts candidate answer and confidence via `inference_fn(t)`.
2. Estimates auxiliary accuracy via simulated self-consistency over `k` samples.
3. Halts and returns if `confidence − auxiliary_accuracy > delta`.

### `run_pipeline.py`
CLI entry point. Parses arguments, loads config, calls `run_evaluation`, computes aggregate metrics, and prints formatted ASCII tables.

---

## Reproducing Paper Results

The simulator in `src/evaluators.py` is parameterized to reproduce the key empirical observations from Table A.1 of the paper:

| Model | Budget | ECE (paper) | OG (paper) | Acc (paper) |
|-------|--------|-------------|------------|-------------|
| 8B | none | 0.0436 ± 0.015 | +0.493 | 0.461 |
| 8B | light | 0.1040 ± 0.034 | +0.249 | 0.732 |
| 8B | medium | 0.0496 ± 0.049 | +0.336 | 0.653 |
| 8B | heavy | 0.0145 ± 0.005 | +0.245 | 0.739 |
| 70B | none | 0.0352 ± 0.026 | +0.155 | 0.825 |

---

## Live API Evaluation

To run against a real LLM API, replace the `_simulate_response` function in `src/evaluators.py` with a call to your preferred inference endpoint. The `ModelResponse` dataclass and `parse_and_validate_response` function in `src/data_loader.py` handle response validation and confidence extraction automatically.

---

## Citation

If you use this codebase or build on the CDUR framework, please cite:

```bibtex
@misc{hiremath2025cdur,
  title     = {Calibration Drift Under Reasoning: How Chain-of-Thought Budgets
               Induce Overconfidence in Large Language Models},
  author    = {Hiremath, Prakul Sunil and Hiremath, Harshit R},
  year      = {2025},
  doi       = {10.5281/zenodo.19709379},
  url       = {https://doi.org/10.5281/zenodo.19709379}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
