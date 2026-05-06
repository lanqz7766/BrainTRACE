#!/usr/bin/env bash
# BrainTRACE — end-to-end reproduction: render → infer → score.
#
# Usage:
#   ./reproduction/reproduce.sh <model> [--dataset DIR] [--mr-rate MR_RATE_DIR] \
#                               [--out OUT_DIR] [--render-only] [--score-only]
#
# Defaults assume:
#   - dataset       : ./braintrace_dataset
#   - MR-RATE root  : ./mr_rate            (you must download MR-RATE first)
#   - out           : ./outputs/<model>
#
# The script is idempotent: each phase respects --skip-existing flags so a
# rerun only does new work. A model name is required so caller chooses the
# adapter flags explicitly. Examples:
#
#   ./reproduction/reproduce.sh gpt-5.4
#   ./reproduction/reproduce.sh gemini-2.5-pro --out ./outputs/gemini
#
set -euo pipefail

usage() {
  sed -n '2,20p' "$0"
  exit 1
}

MODEL="${1:-}"
[ -z "${MODEL}" ] && usage
shift

DATASET="./braintrace_dataset"
MR_RATE="./mr_rate"
OUT_BASE="./outputs"
RENDER_ONLY=0
SCORE_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)     DATASET="$2"; shift 2 ;;
    --mr-rate)     MR_RATE="$2"; shift 2 ;;
    --out)         OUT_BASE="$2"; shift 2 ;;
    --render-only) RENDER_ONLY=1; shift ;;
    --score-only)  SCORE_ONLY=1; shift ;;
    *) echo "unknown flag: $1"; usage ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_BASE}/${MODEL}"
mkdir -p "${OUT_DIR}"

echo "[reproduce] model=${MODEL}"
echo "[reproduce] dataset=${DATASET}  mr_rate=${MR_RATE}  out=${OUT_DIR}"

# ---- Phase 1 — render (if needed) -----------------------------------------
if [ "${SCORE_ONLY}" -eq 0 ]; then
  if [ -d "${DATASET}/images" ] && [ "$(find "${DATASET}/images" -name '*.png' | head -1)" ]; then
    echo "[reproduce] images already rendered under ${DATASET}/images — skipping render"
  else
    echo "[reproduce] rendering images and volumes from MR-RATE..."
    python3 "${REPO_ROOT}/reproduction/render_images.py" \
      --dataset    "${DATASET}" \
      --mr-rate-root "${MR_RATE}" \
      --out-root   "${DATASET}"
  fi
fi
[ "${RENDER_ONLY}" -eq 1 ] && exit 0

# ---- Phase 2 — inference --------------------------------------------------
if [ "${SCORE_ONLY}" -eq 0 ]; then
  case "${MODEL}" in
    gpt-5.4|gpt-5|gpt-5-mini|claude-opus-4*|gemini-2.5-pro)
      PROVIDER="auto"
      [[ "${MODEL}" == gpt-* ]]                 && PROVIDER="openai"
      [[ "${MODEL}" == claude-* ]]              && PROVIDER="anthropic"
      [[ "${MODEL}" == gemini-* ]]              && PROVIDER="google"

      echo "[reproduce] running ${PROVIDER} inference on broadQA + 3D..."
      python3 "${REPO_ROOT}/eval/vlm_eval_api.py" \
        --model "${MODEL}" \
        --provider "${PROVIDER}" \
        --sample-file "${DATASET}/data/test.parquet" \
        --images-root "${DATASET}/images" \
        --out-dir "${OUT_DIR}/main_outputs" \
        --max-tokens 2048 \
        --parallel 4 \
        --skip-existing

      echo "[reproduce] running chain inference on clinical_reasoning_QA..."
      python3 "${REPO_ROOT}/eval/chain_inference_wrapper.py" \
        --sample-file "${DATASET}/data/test.parquet" \
        --images-root "${DATASET}/images" \
        --out-dir "${OUT_DIR}/chain_outputs" \
        --adapter-script "${REPO_ROOT}/eval/vlm_eval_api.py" \
        --adapter-flags "--model ${MODEL} --provider ${PROVIDER} --temperature 0 --max-tokens 2048 --parallel 4" \
        --max-images 6 \
        --mode 2d \
        --skip-existing
      ;;
    *)
      echo "[reproduce] open-weight model ${MODEL}: see eval/README.md for vLLM/HF setup"
      echo "[reproduce] expecting predictions JSONL at ${OUT_DIR}/predictions.jsonl"
      [ -f "${OUT_DIR}/predictions.jsonl" ] || {
        echo "[reproduce] ERROR: ${OUT_DIR}/predictions.jsonl not found. Generate it first."
        exit 1
      }
      ;;
  esac
fi

# ---- Phase 3 — score ------------------------------------------------------
echo "[reproduce] scoring..."
PRED_PATH="${OUT_DIR}/predictions.jsonl"
[ -d "${OUT_DIR}/main_outputs" ] && PRED_PATH="${OUT_DIR}/main_outputs"
python3 "${REPO_ROOT}/scoring/score.py" \
  --dataset    "${DATASET}" \
  --predictions "${PRED_PATH}" \
  --out-dir    "${OUT_DIR}/scores"

echo "[reproduce] done — see ${OUT_DIR}/scores/summary.json"
