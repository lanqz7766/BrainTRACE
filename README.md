# BrainTRACE — Companion Code

This repository contains everything needed to **render, evaluate, and score**
a vision-language model on the BrainTRACE benchmark.

The dataset itself (task definitions, ground truth, rubrics) is hosted
separately on Hugging Face. Imagery is **not** redistributed; users render
images and 3D volumes locally from their own MR-RATE download — see
[`DUA_NOTICE.md`](./DUA_NOTICE.md) for upstream credit and Data Use Agreement
terms.

```
.
├── README.md             ← you are here
├── DUA_NOTICE.md         ← upstream credit + MR-RATE DUA
├── LICENSE               ← Apache-2.0 (code only)
├── requirements.txt      ← Python deps
├── reproduction/
│   ├── render_images.py  ← MR-RATE → PNG mosaics + .npy volumes
│   └── reproduce.sh      ← end-to-end render → infer → score
├── eval/
│   └── vlm_eval_api.py   ← OpenAI / Anthropic / Google adapter
├── scoring/
│   ├── score.py          ← unified scorer: closed-form + open-ended + case-level reasoning
│   ├── llm_judge.py      ← rubric-based LLM judge
│   └── few_shot_examples.json
└── docs/
    ├── SCORING.md        ← scoring methodology details
    └── TAXONOMY.md       ← 30 templates × 5 cognitive levels
```

## Quickstart

```bash
# 1. Get the dataset metadata
huggingface-cli download BrainTRACE-anon/BrainTRACE --repo-type=dataset \
    --local-dir ./braintrace_dataset

# 2. Get MR-RATE upstream (sign the DUA at huggingface.co/datasets/Forithmus/MR-RATE first)
huggingface-cli download Forithmus/MR-RATE --repo-type=dataset \
    --local-dir ./mr_rate

# 3. Render images and (for 3D track) volumes locally
python reproduction/render_images.py \
    --dataset ./braintrace_dataset \
    --mr-rate-root ./mr_rate \
    --out-root ./braintrace_dataset

# 4. Run inference (example: GPT-5.4 medium reasoning)
python eval/vlm_eval_api.py \
    --dataset ./braintrace_dataset \
    --model gpt-5.4 \
    --reasoning-effort medium \
    --out-dir ./outputs/gpt-5.4

# 5. Score
python scoring/score.py \
    --dataset ./braintrace_dataset \
    --predictions ./outputs/gpt-5.4 \
    --out-dir ./scores/gpt-5.4
```

The end-to-end orchestration script is `reproduction/reproduce.sh`:

```bash
./reproduction/reproduce.sh gpt-5.4   # one-shot: render → infer → score
```

## Three answer types, three scoring modes

BrainTRACE contains 6,923 source clinical items. Case-level reasoning
items are scored through six decomposed VQA steps, so the released
evaluation reports 7,273 scored VQA instances.

| Type | n | How scored |
|---|---:|---|
| `closed_form` | 5,176 | Exact-match against `gt_value` with light normalisation. Reports accuracy and 95% bootstrap CI. |
| `open_ended` | 1,677 | LLM judge (default `gpt-4o-mini-2024-07-18`) using the `rubric_json` slot rubric and `critical_slots` gating. Reports per-slot pass-rate, item pass-rate, and a derived 1–5 quality score. |
| `case_reasoning` | 70 cases / 420 decomposed VQA steps | Stepwise rubric per sub-question plus an all-steps case success metric. Reports Step Pass, Case Success, and per-step subclass breakdown. |

Methodology details in [`docs/SCORING.md`](./docs/SCORING.md).

## Adapters

Bundled `eval/vlm_eval_api.py` covers OpenAI, Anthropic, Google. For
open-weight VLMs (Qwen, InternVL, MedGemma, Lingshu, HuatuoGPT, Janus,
LLaVA-Med, M3D-LaMed, RadFM), reference implementations targeting vLLM /
HuggingFace runtimes are described in [`eval/README.md`](./eval/README.md).
The adapter contract is intentionally small (single function: `infer(item,
images, **kwargs) -> str`) so swapping in new models is a few-dozen-line
change.

## License

- **Code (this repository):** Apache-2.0 — see [`LICENSE`](./LICENSE).
- **Dataset (parquet + metadata, hosted on HF):** CC-BY-NC-SA 4.0.
- **Upstream MR-RATE imagery:** governed by the MR-RATE DUA — see
  [`DUA_NOTICE.md`](./DUA_NOTICE.md).

## Reproducibility

The numbers in the BrainTRACE leaderboard were generated with the exact
scripts in this repo. Each model in `reproduction/leaderboard_runs.csv`
includes the adapter flags, max-tokens, parallelism, and total wall time.
Re-running `reproduce.sh <model>` should reproduce a model's headline
numbers within bootstrap-CI tolerance.

For non-deterministic API models (default temperatures), exact byte-level
reproduction is not guaranteed; bootstrap-CI agreement is.

## Citing

See the dataset card on Hugging Face for the BibTeX block. Anonymous during
the NeurIPS review period.

## Contributing & issues

Bug reports and clarification requests via the repository issue tracker
(anonymous mirror at anonymous.4open.science during review). The dataset
itself is non-live; bug fixes to scripts ship as patch versions and do not
change the dataset version.
