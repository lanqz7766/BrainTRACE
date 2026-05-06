# BrainTRACE Scoring Specification

This document is the public, self-contained scoring contract for
BrainTRACE. Every metric reported in the accompanying paper or the
public leaderboards is reproducible from the rules below; no internal
documents are required.

The implementation lives in three modules under `scoring/`:

| Module | Role |
| --- | --- |
| `score_closed_form.py` | deterministic exact-match for `closed_form` items |
| `score_open_judge.py`  | LLM-as-judge wrapper for `open_ended` and `chain` items |
| `score.py`             | unified CLI dispatcher; produces one `summary.json` |

The judge implementation lives in `scoring/llm_judge.py`.


## 1. Three answer types

The parquet defines three values of the `answer_type` column. Each maps
to a single scoring mode:

| `answer_type` | What the model returns | Scoring mode |
| --- | --- | --- |
| `closed_form` | one categorical or ordinal answer | deterministic exact-match (Section 2) |
| `open_ended`  | one paragraph clinical answer | LLM judge with slot rubric (Section 3) |
| `chain`       | one six-step longitudinal case analysis | per-step LLM judge + all-or-nothing case success (Section 4) |

Notes on terminology:

- `chain` is the parquet field name. Earlier draft material used the
  longer name `case_reasoning`; the two refer to the same task family.
  Public outputs use `chain` for the parquet column and "case
  reasoning" or "L5.5" in prose.
- `closed_form` items are split internally into Pattern A (MCQ
  categorical) and Pattern B (ordinal bucket). Both are scored by the
  same exact-match rule; the public leaderboard does not replace
  exact-match with distance-aware scoring even for ordinal buckets.


## 2. Closed-form scoring

### 2.1 Algorithm

Given a model `prediction` string and a parquet row with `gt_value` and
optional `options` array, the scorer applies these steps in order:

1. **Strip inline thinking.** If the prediction contains the literal
   substring `</think>` (case-insensitive), drop everything up to and
   including the *last* `</think>` and keep only the text after it.
   Predictions without the tag are unchanged.
2. **Letter-then-paren echo.** If the result still matches the pattern
   `^[letter]\s*\(<phrase>\)$` (e.g. `"D (FLAIR)"`, `"B) FLAIR (post-contrast)"`),
   replace the prediction with `<phrase>`.
3. **Strip a leading answer label** of the form `Answer:` /
   `The correct answer is` / `Final answer is` (case-insensitive).
4. **Strip a leading MCQ prefix** in any of these forms, where the
   letter is `A`-`E`: `(A) `, `A) `, `A. `, `A: `, `[A] `, `A - `.
5. **Lowercase, trim whitespace, strip one trailing period.**
6. Compare the normalised prediction to the normalised `gt_value`.
7. **Letter-only fallback.** If steps 1-6 did not match and the
   prediction *starts* with a single letter `A`-`E` (with optional
   `(`/`[`/`)` brackets) followed by whitespace, punctuation, or
   end-of-string, resolve that letter against `options` and compare the
   resolved option text to `gt_value`. Items with no `options` array
   skip this fallback.

This is intentionally strict: the scorer never searches inside a
`<think>` block for a candidate letter, never accepts a single letter
that is the start of an English word (`"Diffusion ..."` is *not*
parsed as letter `D`), and never grades multiple-letter responses as
correct.

### 2.2 Worked examples

For an MCQ item with `gt_value = "FLAIR"` and `options =
["T1", "T2", "DWI", "FLAIR"]`, all of the following predictions match:

| Prediction | Resolution path | Match? |
| --- | --- | --- |
| `D` | leading-letter fallback → `options[3]` = `"FLAIR"` | yes |
| `[D]` | leading-letter fallback → `options[3]` | yes |
| `(D)` | leading-letter fallback → `options[3]` | yes |
| `D. it is a FLAIR sequence` | leading-letter fallback → `options[3]` | yes |
| `D - explanation` | leading-letter fallback → `options[3]` | yes |
| `Final answer: (D) FLAIR.` | answer-label + MCQ-prefix strip → `flair` | yes |
| `D (FLAIR)` | letter-then-paren echo → `flair` | yes |
| `<think>...</think>D` | think-strip + leading-letter fallback | yes |

Predictions that should *not* match:

| Prediction | Reason |
| --- | --- |
| `Diffusion-weighted imaging` | leading character `D` is followed by `i` (alphanumeric); fallback declines |
| `B or D` | leading letter `B` resolves to `T2`, not `FLAIR` |
| `<think>D is correct</think>` then end | nothing after `</think>` → empty, no match |

### 2.3 Reported metrics

Per-track, per-`(track, sub_category)`, and per-level:

- **Accuracy** = `correct / scored_items`. Items with no submitted
  prediction are excluded from the denominator (they are not counted
  as failures); this matches the contract documented in the README.
- **95% CI** by 1,000-iteration percentile bootstrap on the per-item
  correctness vector.
- **Random baseline** = `mean over scored items of 1 / |options|`.
- **Chance-debiased score (CDS)** = `max(0, (acc - chance) / (1 - chance))`,
  reported per level only. CDS is identical to "Cohen-style"
  agreement-above-chance; we report it because levels mix items with
  different option-set sizes.
- **Significance gate**: an accuracy is "significant above random" iff
  the lower bound of the 95% bootstrap CI is strictly greater than the
  random baseline.


## 3. Open-ended LLM-judge scoring

### 3.1 Judge runtime

- **Judge model**: `gpt-4o-mini-2024-07-18` (pinned version)
- **Decoding**: `temperature=0`, `top_p=1`, `max_tokens=600`
- **Output format**: a single JSON object per call (Section 3.3)
- **Prompt version**: `v0.6.2` ("factual auditor")
- **Inline thinking**: predictions are stripped of any `<think>...</think>`
  block before the judge sees them, exactly as in Section 2.1 step 1.
  This is applied uniformly to *all* models so that inline-CoT and
  hidden-CoT model surfaces are scored on the same textual basis.

### 3.2 Rubric structure

Each open-ended item carries a rubric of the following shape:

```json
{
  "reference_answer": "...",
  "rubric_slots": [
    {
      "slot_name": "dominant_finding_naming",
      "weight": 1.0,
      "expected_keywords_regex": ["(?is)(?=.*\\bdemyelinating\\b)(?=.*\\bplaque[s]?\\b)"],
      "llm_judge_criterion": "Name the dominant finding."
    }
  ],
  "critical_slots": ["trajectory_descriptor", "overall_status_or_recommendation"],
  "total_weight": 5.5,
  "pass_threshold": 3.5
}
```

Field semantics:

- `slot_name` is a stable identifier.
- `weight` is the slot's contribution to `slot_score`.
- `expected_keywords_regex` is a canonical lexical description of
  acceptable evidence; the judge uses it as a *reference*, not as a
  hard regex gate (paraphrase is allowed if clinically equivalent).
- `llm_judge_criterion` is a human-readable slot rule.
- `critical_slots` must pass regardless of total score.
- `pass_threshold` is the minimum weighted score for an item pass.

### 3.3 Judge response schema

The judge is asked to return one JSON object:

```json
{
  "required_fact_results": [{"fact": "...", "met": true, "evidence": "...", "issue": null}],
  "slot_results": [{"slot_name": "...", "criterion_met": true, "evidence": "...", "issue": null}],
  "critical_errors": [{"type": "wrong_anatomy|wrong_laterality|wrong_direction|wrong_timepoint|wrong_modality_reading|wrong_temporal_phase|wrong_severity|wrong_diagnosis|hallucinated_landmark|other", "description": "..."}],
  "unsupported_claims": [{"claim": "...", "rationale": "..."}],
  "clinically_acceptable": false,
  "reasoning": "..."
}
```

If the response is not valid JSON, the call is retried once with a
JSON-only reminder. If retry still fails, the item is counted as a
failure with `critical_errors = [{"type": "other"}]`.

### 3.4 Per-item pass/fail

Let slot `j` have weight `w_j` and let `passed_j = 1` iff the judge
returns `criterion_met=true`. Let `C` be the set of critical slots.

```text
slot_score = sum_j w_j * passed_j
item_pass  = (slot_score >= pass_threshold) AND (every j in C passes)
```

### 3.5 Aggregation buckets

Open-ended items are aggregated into seven paper-table buckets
(union of the broadQA and 3D tracks):

| Bucket | Parquet `(level, sub_category)` | Concept |
| --- | --- | --- |
| L1-ABN | (1, Det)    | abnormality detection |
| L2-APP | (2, App)    | appearance characterisation |
| L3-EFF | (3, Effect) | mass-effect / interval characterisation |
| L4-TRJ | (4, Traj)   | trajectory description |
| IMP    | (5, Imp)    | diagnostic impression (synthesis) |
| CMP    | (5, Comp)   | comparison / interval reasoning |
| DIF    | (5, Diff)   | differential diagnosis |

### 3.6 Reported metrics (per bucket and per track)

- **Item pass rate** (headline): fraction of items with `item_pass=true`.
- **Mean slot-match rate**: arithmetic mean over items of
  `n_slots_met / n_slots_total`.
- **Mean Quality 1-5**: see Section 3.7.
- **Faithfulness composite** in `[0, 1]`:

  ```text
  faith = slot_match
        * (1 - 0.3 * min(1, unsupported_claims / 3))
        * (1 -        min(1, critical_errors  / 3))
  ```

  averaged over items.

### 3.7 Derived 1-5 Quality Score

The judge does *not* emit a numeric quality score. BrainTRACE derives
one deterministically from structured judge fields, following recent
clinical-text rubric work:

| Score | Rule (highest match wins, evaluated top to bottom) |
| --- | --- |
| 5 | `clinically_acceptable=true` AND `slot_rate == 1` AND `unsupported_claims == 0` |
| 4 | `clinically_acceptable=true` AND `slot_rate < 1` |
| 3 | `critical_errors == 0` AND `slot_rate > 0.5` AND (any of: missing required fact / unsupported claim / `slot_rate < pass_threshold/total_weight`) |
| 2 | `critical_errors == 0` AND `slot_rate <= 0.5` |
| 1 | `critical_errors > 0` |

Operational notes:
- Wrong-anatomy / wrong-laterality / wrong-direction etc. enter through
  `critical_errors`, so any nonzero contradiction count yields Quality 1.
- A critical-slot failure (without a critical error) keeps the item at
  Quality 2-3 because `clinically_acceptable` is false.

### 3.8 Failure modes

- Malformed judge JSON: retry once with the JSON-only reminder; if
  parsing still fails, count the item as a failure.
- Refusal output: counted as failure.
- Empty answer: counted as failure if scored (missing answers are
  excluded from denominators instead).
- Generic keyword overlap without clinically correct meaning does not
  satisfy a slot.


## 4. Case-reasoning (chain) scoring

### 4.1 Per-step rubric

Every L5.5 case carries six steps:

1. `Anchor localization`
2. `Early interval change`
3. `Interval characterization`
4. `Trajectory or burden evolution`
5. `Latest status interpretation`
6. `Final synthesis` (`final_step`)

Each step has its own slot rubric (same shape as Section 3.2), plus
step-specific metadata: `subclass`, `timepoints_used`,
`accepted_answer_variants`, `rejection_criteria`, `semantic_targets`.

### 4.2 Grading policy

The case-level grading rule is:

```json
{
  "rule": "all_or_nothing",
  "final_step_requires_all_prior_steps_correct": true,
  "contradictory_extra_claim_fails_step": true,
  "keyword_overlap_without_semantic_match_fails_step": true
}
```

Public contract:
- Each step is judged independently against its own rubric.
- `step_pass` requires the step to satisfy its rubric *and* its
  critical-slot rules.
- `case_pass` requires every step to pass.
- `final_step` is only eligible to pass if all prior steps already pass.
- A contradictory extra claim fails that step.
- Keyword overlap without semantic match fails that step.

### 4.3 Reported metrics

Across all 70 cases (`70 * 6 = 420` steps):

- **Step Pass** = `steps_passed / 420`.
- **Chain End-to-end Success Rate (CESR)** = `cases_with_all_steps_passed / 70`.
  Equivalent to "Case Success" in some prose.
- **`>= k` of 6 distribution** for `k = 1..6`: fraction of cases that
  pass at least `k` steps. Reported in the detailed per-model report
  for completeness; CESR equals the `>= 6` row.

The `<think>` strip described in Section 3.1 is applied to every step's
prediction before the judge sees it.


## 5. Per-level CDS

CDS (chance-debiased score) is reported only for closed-form per-level
aggregation, where mixing option-set sizes makes raw accuracy hard to
read against chance.

```text
chance_l = mean over scored items in level l of 1 / |options_i|
CDS_l    = max(0, (acc_l - chance_l) / (1 - chance_l))
```

Open-ended and chain raw percentages are already calibrated against
zero (no model trivially passes a slot rubric), so we report raw
percentages without a CDS adjustment.


## 6. Inline-thinking handling (uniform policy)

Some open-weights "thinking" model variants emit reasoning text as a
literal `<think>...</think>` block before their final answer. API
"thinking" models hide their reasoning server-side and return only the
final answer. To make these comparable on identical scoring code:

- **Always-on, uniform stripping.** Both the closed-form scorer and the
  LLM judge drop everything up to and including the *last* `</think>`
  before any further normalisation or judging. This is applied to
  every model regardless of family.
- **Empirical observation.** During public evaluation, only one model
  family (a "thinking"-mode open-weights variant) was observed to emit
  `<think>` tags in raw output; for every other model the strip is a
  no-op. The unconditional rule still applies so that future models
  using the same convention score consistently.
- **No internal `<think>` substring search.** The closed-form scorer
  *never* searches inside the `<think>` block for a candidate letter;
  if the post-strip output is empty, the item simply fails to match.


## 7. Reproduction

Single command, end to end:

```bash
python scoring/score.py \
  --dataset path/to/braintrace_dataset \
  --predictions path/to/predictions \
  --out-dir path/to/scores
```

Accepted prediction inputs:

- one JSONL or NDJSON file with one `{"item_id": "...", ...}` per line, or
- one directory tree scanned recursively for `*.json` / `*.output.json`.

Accepted text fields for non-chain answers (first non-empty wins):
`raw_output`, `output`, `response`, `text`, `completion`.

Useful flags:

- `--mode closed` (or `--skip-judge`): closed-form only, no API calls.
- `--mode open`: open-ended + chain only.
- `--judge-model <id>`: override the pinned judge model id.
- `--max-parallel <int>`: judge call concurrency (default 8).
- `--reuse-judge-dir <path>`: re-aggregate from existing judge output
  without re-running the API.
- `--n-bootstrap <int>`: closed-form bootstrap iterations (default 1000).

Output layout under `out-dir/`:

```text
scored_closed_form.jsonl       # per-item closed-form rows
summary_closed_form.json       # per-track / per-subcat / per-level closed-form aggregates
_llm_judge/                    # raw judge artefacts (audit trail)
summary_open_judge.json        # open-ended + chain aggregates
summary.json                   # merged headline summary (this is the canonical leaderboard input)
```

Closed-form scoring is fully deterministic. Open-ended and chain
scoring depend on the judge model; with `temperature=0` and `top_p=1`
the same predictions and judge model id reproduce numbers within the
~0.1 pp run-to-run noise inherent to the judge API.


## 8. Edge cases

- **Missing prediction**: the parquet row is excluded from the
  denominator (not counted as failure).
- **Empty answer string**: counted as failure if the item is included
  in scoring.
- **Refusal**: counted as failure.
- **Closed-form item with empty or missing `options`**: the
  letter-only fallback is skipped; only direct text comparison applies.
- **Judge JSON missing a requested slot**: that slot is inserted as
  failed with an `issue` of `missing from judge response`.


## 9. Versioning

This file is **BrainTRACE Scoring v1.0**.

Versioning rules:

- Bump the **minor** version when the scoring methodology changes in a
  way that can change reported numbers.
- Do **not** bump the version for implementation-only refactors that
  leave reported numbers unchanged.


## 10. References

The public scoring design is informed by:

- Liu et al., *G-Eval: NLG Evaluation using GPT-4 with Better Human
  Alignment* (2023).
- Singhal et al., *Med-PaLM 2: Towards Expert-Level Medical Question
  Answering with Large Language Models* (clinical generation rubrics).
- Yang et al., *RadBench: Radiology-grounded Fact Auditing for
  Generative Models*.
- MIMIC-CXR-VQA stepwise rubric evaluation.
