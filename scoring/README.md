# `scoring/` — Unified scorer for the three answer types

## Overview

`score.py` is the single entry point. Given a BrainTRACE dataset directory
and a predictions blob, it:

1. **Closed-form items** → exact-match against `gt_value` (with light
   normalisation), reports accuracy + 95% bootstrap CI.
2. **Open-ended items + case-level reasoning items** → defers to `llm_judge.py` (LLM judge
   with the rubric stored in the parquet's `rubric_json` column), reports
   per-slot pass-rate, item pass-rate, and a derived 1–5 quality score.

```bash
python scoring/score.py \
  --dataset    ./braintrace_dataset \
  --predictions ./outputs/<model>/main_outputs \
  --out-dir    ./scores/<model>
```

Predictions can be:
- a **directory** of `<item_id>.json` files each containing an `"output"`
  field (the layout `eval/vlm_eval_api.py` writes), or
- a **JSONL file** with one `{"item_id": ..., "output": "..."}` per line.

The scorer deduces `answer_type` per item from the parquet and dispatches
accordingly.

## Files

| File | Purpose |
|---|---|
| `score.py` | Unified entry point. Auto-dispatches each item to the right scorer. |
| `score_closed_form.py` | Strict closed-form scorer (exact match + MCQ letter resolution + bootstrap CI + per-level CDS). |
| `score_open_judge.py` | Open-ended + case-reasoning judge wrapper (4 metrics: item_pass, slot_match, mean_quality, faithfulness). |
| `llm_judge.py` | LLM-as-judge implementation (default model `gpt-4o-mini-2024-07-18`). |
| `few_shot_examples.json` | Few-shot exemplars baked into the judge prompt. |

The judge wrapper materialises the per-item parquet rubric into a temp
file-tree that `llm_judge.py` consumes; you do not need to hand-build that
layout yourself.

## Closed-form normalisation

Default normalisation strips:
- a leading `Answer:` / `Final answer:` label (case-insensitive),
- a leading MCQ option prefix `(A)` / `A.` / `B)` / etc.,
- trailing whitespace and a single trailing period,
- case (lowercase-fold).

Pass `--strict` to disable normalisation (exact byte match against
`gt_value`).

## LLM judge model and cost

Default judge: `gpt-4o-mini-2024-07-18`. Roughly:

| Track | Items judged | Approx total cost |
|---|---:|---:|
| broadQA open-ended | 1,600 | ~$2.00 |
| 3D-track open-ended | 77 | ~$0.10 |
| clinical_reasoning_QA (case-level reasoning) | 70 × 6 decomposed VQA steps = 420 | ~$0.25 |

Override via `--judge-model <name>` if you want to swap to a different
judge model.

## Reported metrics

Per `score.py` output (`scores/<model>/summary.json`):

```json
{
  "closed_form": {
    "broadQA":               {"n": 4053, "accuracy_pct": ..., "ci95_pct": [..., ...], "random_baseline_pct": ..., "delta_vs_random_pp": ...},
    "3D":                    {"n": 1123, ...},
    "clinical_reasoning_QA": {"n": 0,    ...}
  },
  "open_ended_case_judge_dir": "scores/<model>/_llm_judge"
}
```

Inside `_llm_judge/`:
- `per_item_assessments.jsonl` — one row per item with the judge's
  per-slot decisions, `item_pass`, and the 1–5 derived quality score.
- `per_template_summary.json` — pass-rate per template-id.
- `run.log` — judge call log (token counts, JSON-parse retries, cost).

## Reproducing the published numbers

The leaderboard numbers for every model and every track are in
`../reproduction/leaderboard_runs.csv`. After running
`scoring/score.py` against the same predictions, your numbers should agree
within bootstrap-CI tolerance for closed-form, and within ±2 pp for
LLM-judge metrics (which have inherent stochasticity in the judge).

If you observe a larger drift, check:
- `--judge-model` matches what the leaderboard used (`gpt-4o-mini-2024-07-18`).
- The predictions file is from the **same model run** (not a different
  reasoning_effort or a different render version).
- The scorer version matches (`git rev-parse HEAD` recorded in
  `_llm_judge/run.log`).

## Methodology

For the full LLM-judge prompt, slot-rubric semantics, case-level grading
policy, and the derivation of the 1–5 quality score, see
[`docs/SCORING.md`](../docs/SCORING.md).
