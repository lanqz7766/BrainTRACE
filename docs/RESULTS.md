# BrainTRACE paper results

This page reproduces the track-level columns of the paper's closed-form diagnostic table (`tab:closed-diagnostic`) and open-ended/case-reasoning table (`tab:open-v06`). It covers all **20 model configurations**, in the paper's group order. Values below come from those manuscript tables; the run manifest is the companion record for adapter settings.

[Paper / OpenReview](https://openreview.net/forum?id=hXktXOmwp6) · [README](../README.md) · [Run manifest](../reproduction/leaderboard_runs.csv)

## Track-level results

All values are percentages. **—** reproduces an unreported paper entry; it does not mean zero.

| Model | Closed<br>Broad | Closed<br>3D | Open pass<br>Broad | Open pass<br>3D | Step<br>Pass | Case<br>Success |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 33.2 | 33.2 | 4.9 | 3.9 | 9.3 | 1.4 |
| Qwen3-VL-4B | 32.7 | 35.4 | 11.8 | 5.2 | 8.3 | 0.0 |
| Qwen3-VL-8B | 33.7 | 33.4 | 13.8 | 9.1 | 3.8 | 0.0 |
| Qwen3-VL-30B | 35.1 | 34.8 | 14.4 | 11.7 | 11.7 | 0.0 |
| Qwen3.5-27B | 37.3 | 12.6 | 8.1 | 3.9 | 2.9 | 0.0 |
| InternVL3-8B | 35.8 | 35.3 | 6.6 | — | 8.3 | 0.0 |
| InternVL3-14B | 35.1 | 34.0 | 8.6 | — | 11.2 | 0.0 |
| InternVL3-78B | 35.5 | 33.3 | 7.6 | — | 0.0 | 0.0 |
| Janus-Pro-7B | 10.9 | 17.6 | 1.8 | 0.0 | 0.5 | 0.0 |
| HuatuoGPT-Vision-7B | 36.3 | 29.1 | 9.9 | 0.0 | 0.2 | 0.0 |
| Lingshu-7B | 36.5 | 35.4 | 6.8 | 3.9 | 6.9 | 0.0 |
| MedGemma-4B-it | 24.9 | 18.9 | 5.4 | 3.9 | 7.6 | 0.0 |
| LLaVA-Med-v1.5 | 0.6 | 0.0 | 3.8 | 1.3 | 0.2 | 0.0 |
| GPT-5.4 | 40.4 | 42.2 | 19.8 | 15.6 | 24.0 | 0.0 |
| GPT-5 | 41.9 | 38.3 | 10.5 | 11.7 | 21.7 | 1.4 |
| GPT-5-mini | 38.2 | 37.5 | 10.6 | 10.4 | 24.5 | 0.0 |
| Claude Opus 4.6 | 29.8 | 1.2 | 9.6 | 6.5 | 23.1 | 0.0 |
| Gemini 2.5 Pro | 36.5 | 32.0 | 14.2 | 13.0 | 10.7 | 0.0 |
| M3D-LaMed | — | 29.9 | — | 0.0 | 4.8 | 0.0 |
| RadFM | 18.1 | 15.0 | 4.3 | 2.6 | 0.0 | 0.0 |

## Reading the metrics

- **Closed Broad / 3D:** exact-match accuracy in the broad-compatible and 3D-evidence tracks, respectively.
- **Open pass Broad / 3D:** clinical rubric pass rate for open-ended items, evaluated with the release judge `gpt-4o-mini-2024-07-18`.
- **Step Pass:** percentage of passing steps among 420 decomposed VQA steps across 70 longitudinal cases.
- **Case Success:** percentage of the 70 cases in which every one of the six steps passes.

The first nine rows are general open-weight models, the next four are medical open-weight models, the next five are frontier API models, and the final two are native-volume medical baselines. This preserves the source ordering; it is not a ranking by one combined score.

## Evaluation boundaries

Supported rendered-view and native-volume interfaces are evaluated on shared clinical targets. Their scores do not isolate the effect of native 3D representation. Model-specific image limits also differ. Open-ended and case scores depend on the judge and rubric: human-anchored and matched cross-judge audits in the paper show sensitivity of exact scores to judge choice.

The independent physician answerability check uses image-grounded MCQs: 55/60 steps correct and 7/10 cases with all six steps correct. Its answer format and sample differ from the model evaluation, so it is not a comparable row in this table.

These are benchmark results under the paper's protocol, not evidence of clinical deployment readiness. See the [scoring specification](SCORING.md) and paper for the detailed rubric, failure handling, and sensitivity analyses.
