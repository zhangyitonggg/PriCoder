#!/usr/bin/env bash
set -euo pipefail

# This wraps:
#   data_generation/sft_data_gen_vllm/scripts/generate_sft_fusion_loop.py
#
# Required inputs:
#   - SPEC: synthesis repo spec json
#   - D1_JSONL: path to an existing D1 jsonl (already generated)
#   - MAX_D: generate until D{MAX_D} (inclusive)
#   - NUM_PER_D: how many accepted samples per D (for D2..D{MAX_D})
#   - OUT_ROOT: output root (writes ${OUT_ROOT}_D2.jsonl ...)
#
# Other knobs are kept aligned with synthesis_generation.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY_SCRIPT="../sft_data_gen_vllm/scripts/generate_sft_fusion_loop.py"

SPEC="${SPEC:-../sft_data_gen_vllm/specs/numba/repo_spec.json}"

# D1 is provided externally
D1_JSONL="${D1_JSONL:-../synthesis_sft.jsonl}"

MAX_D="${MAX_D:-3}"
NUM_PER_D="${NUM_PER_D:-2000}"

OUT_ROOT="${OUT_ROOT:-../compositional_fusion/compositional_fusion}"
FAIL_ROOT="${FAIL_ROOT:-../compositional_fusion/compositional_fusion_fail}"

# BASE_URL="${BASE_URL:-,http://127.0.0.1:8003,http://127.0.0.1:8004,http://127.0.0.1:8005,http://127.0.0.1:8006,http://127.0.0.1:8001,http://127.0.0.1:8008}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
# BASE_URL="${BASE_URL:-http://127.0.0.1:17720}"
MODEL="${MODEL:-Qwen2.5-Coder-7B-Instruct}"
EXEC_PYTHON="${EXEC_PYTHON:-python}"

CANDIDATES="${CANDIDATES:-1}"
Q_WORKERS="${Q_WORKERS:-8}"

BIASED="${BIASED:-0}"
TAIL_ALPHA="${TAIL_ALPHA:-0.7}"

OVERWRITE="${OVERWRITE:-0}"

ARGS=(
  --spec "$SPEC"
  --d1 "$D1_JSONL"
  --out-root "$OUT_ROOT"
  --fail-root "$FAIL_ROOT"
  --max-d "$MAX_D"
  --num-per-d "$NUM_PER_D"
  --base-url "$BASE_URL"
  --model "$MODEL"
  --exec-python "$EXEC_PYTHON"
  --candidates-per-question "$CANDIDATES"
  --question-workers "$Q_WORKERS"
  # --debug
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

if [[ "$BIASED" == "1" ]]; then
  ARGS+=(--biased-api-sampling --tail-alpha "$TAIL_ALPHA")
else
  ARGS+=(--no-biased-api-sampling)
fi

python "$PY_SCRIPT" "${ARGS[@]}"
