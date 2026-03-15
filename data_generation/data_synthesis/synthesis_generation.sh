#!/usr/bin/env bash
set -euo pipefail

# PriCoder synthesis ：//， NUM  SFT 。
#
# ：
#   NUM=200 MODEL=Qwen2.5-Coder-7B-Instruct \
#   EXEC_PYTHON=/home/xxx/miniconda3/envs/ndonnx/bin/python \
#   Q_WORKERS=4 CANDIDATES=10 \
#   bash synthesis_generation.sh
#
# ：
#   OVERWRITE=1    （ append）
#   BIASED=0       （）
#   TAIL_ALPHA=0.7 
#   Q_WORKERS=4    （）。 Q_WORKERS * CANDIDATES
#
# Base URL（）：
#   BASE_URL  URL， URL：
#     BASE_URL="http://127.0.0.1:8001,http://127.0.0.1:8002"
#    Q_WORKERS  URL。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY_SCRIPT="../sft_data_gen_vllm/scripts/generate_sft_loop.py"
SPEC="${SPEC:-../sft_data_gen_vllm/specs/numba/repo_spec.json}"

OUT_JSONL="${OUT_JSONL:-../synthesis_sft.jsonl}"
FAIL_JSONL="${FAIL_JSONL:-../synthesis_fail.jsonl}"

NUM="${NUM:-6000}"

# ✅ k  {1,2,3} （ python ）
K_LIST="${K_LIST:-1,2,3}"

# ✅  8001..8008
# BASE_URL="${BASE_URL:-http://127.0.0.1:8005,http://127.0.0.1:8006,http://127.0.0.1:8007,http://127.0.0.1:8008,http://127.0.0.1:17720,http://127.0.0.1:17722,http://127.0.0.1:17723,http://127.0.0.1:17725,http://127.0.0.1:17721,http://127.0.0.1:17727}"
# BASE_URL="${BASE_URL:-http://127.0.0.1:8102,http://127.0.0.1:8103,http://127.0.0.1:8104,http://127.0.0.1:8105,http://127.0.0.1:8106,http://127.0.0.1:8107}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

MODEL="${MODEL:-Qwen2.5-Coder-7B-Instruct}"
EXEC_PYTHON="${EXEC_PYTHON:-python}"

# （answer/test candidates）
CANDIDATES="${CANDIDATES:-3}"

#  worker （）
Q_WORKERS="${Q_WORKERS:-8}"

# ：1=，0=
BIASED="${BIASED:-0}"
TAIL_ALPHA="${TAIL_ALPHA:-0.1}"

# ：1=（--overwrite），0=
OVERWRITE="${OVERWRITE:-0}"

ARGS=(
  --spec "$SPEC"
  --out "$OUT_JSONL"
  --fail-out "$FAIL_JSONL"
  --num "$NUM"
  --k "$K_LIST"
  --base-url "$BASE_URL"
  --model "$MODEL"
  --exec-python "$EXEC_PYTHON"
  --candidates-per-question "$CANDIDATES"
  --question-workers "$Q_WORKERS"
  --no-exec-verify
  --no-seed-overlap-check
  --allow-module-leak
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
