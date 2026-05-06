# `eval/` — VLM inference adapters

## What's here

| File | Purpose |
|---|---|
| `vlm_eval_api.py` | Single-turn adapter for OpenAI / Anthropic / Google. Reads a sample-file (BrainTRACE `data/test.parquet` or a per-item JSONL), feeds each item to the chosen API, writes one `<item_id>.json` of model output per item. |
| `chain_inference_wrapper.py` | Multi-turn driver for case-level reasoning. For each case it runs 6 sequential sub-question turns through the underlying adapter (above), preserving conversational state. |

## Adapter contract

Both scripts speak the same simple file convention:

- **Input**: any sample-file (parquet or JSONL) where each row carries
  `item_id`, `question`, optional `options`, and `image_relpaths`.
- **Output dir**: `<out_dir>/<item_id>.json` with at minimum
  `{"item_id": ..., "output": "raw model output text"}`.
- **Idempotent**: pass `--skip-existing` to safely re-fire on partial runs.

The scoring stage (`scoring/score.py`) reads from this output dir directly.

## Examples

### Closed API (OpenAI)
```bash
python eval/vlm_eval_api.py \
  --model gpt-5.4 \
  --provider openai \
  --sample-file ./braintrace_dataset/data/test.parquet \
  --images-root ./braintrace_dataset/images \
  --out-dir ./outputs/gpt-5.4/main_outputs \
  --max-tokens 2048 \
  --reasoning-effort medium \
  --parallel 4 \
  --skip-existing
```

### Closed API (Anthropic)
```bash
python eval/vlm_eval_api.py \
  --model claude-opus-4-6 \
  --provider anthropic \
  --sample-file ./braintrace_dataset/data/test.parquet \
  --images-root ./braintrace_dataset/images \
  --out-dir ./outputs/claude/main_outputs \
  --max-tokens 2048 \
  --parallel 4 \
  --skip-existing
```

### Case-level reasoning
```bash
python eval/chain_inference_wrapper.py \
  --sample-file ./braintrace_dataset/data/test.parquet \
  --images-root ./braintrace_dataset/images \
  --out-dir ./outputs/gpt-5.4/chain_outputs \
  --adapter-script ./eval/vlm_eval_api.py \
  --adapter-flags "--model gpt-5.4 --provider openai --temperature 0 --max-tokens 2048 --parallel 4" \
  --max-images 6 \
  --mode 2d \
  --skip-existing
```

## Open-weight VLMs

The bundled adapters cover commercial APIs only. For open-weight VLMs you
have two options:

1. **Roll your own adapter** that produces the same `<item_id>.json` output
   layout. The contract is the only thing the scorer cares about; the
   adapter is otherwise free to use vLLM, HuggingFace transformers, or
   anything else.
2. **Reference our adapter set.** A representative vLLM-backed adapter for
   the Qwen / InternVL / MedGemma / Lingshu / HuatuoGPT / Janus / LLaVA-Med
   families is described in `docs/OPEN_WEIGHT_ADAPTERS.md` (forthcoming).
   Native-volume models (M3D-LaMed, RadFM) consume `volume_relpath` instead
   of `image_relpaths`.

## Cost / wall-time notes

The published leaderboard cost / wall-time for every API model is in
`reproduction/leaderboard_runs.csv`. As a rough rule of thumb (USD per full
benchmark, snappy summer-2026 pricing):

| Model | broad-compatible + 3D + case-level inference cost |
|---|---:|
| GPT-5-mini | ~$116 |
| Gemini 2.5 Pro | ~$147 |
| GPT-5 (default) | ~$584 |
| GPT-5.4 (medium) | ~$697 |
| Claude Opus 4.6 | ~$1,200+ |

Pass `--max-items N` for a smoke test before committing budget to a full run.
