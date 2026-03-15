#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Logging utilities
# =============================================================================

log() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

die() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" >&2
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}
# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR_HOST="${1:-}"
SERVED_MODEL_NAME="${2:-}"
API_DOCS="${3:-}"
BENCHMARK="${4:-}"
REPO_CONFIG="${5:-}"
USE_HUMANEVAL=0
[ "${6:-}" = "humaneval" ] && USE_HUMANEVAL=1
HUMANEVAL_ARG=""
if [ "$USE_HUMANEVAL" -eq 1 ]; then
    HUMANEVAL_ARG="--humaneval"
fi

EVAL_DIR="${EVAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

echo "Model Dir Host: $MODEL_DIR_HOST"

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.13.0}"
VLLM_CONTAINER="${VLLM_CONTAINER:-vllm_eval_$(date +%Y%m%d%H%M%S)}"

GPUS="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-9432}"
TP="${TP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
VLLM_DISABLED="${VLLM_DISABLED:-false}"


BASE_URL="http://127.0.0.1:${PORT}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-10000}"

CURL_BIN="${CURL_BIN:-curl}"

# =============================================================================
# Utility functions
# =============================================================================

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

port_in_use() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltnH "sport = :${PORT}" 2>/dev/null | grep -q .
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1
    else
        return 1
    fi
}

have_weights() {
    compgen -G "${MODEL_DIR_HOST}/*.safetensors" >/dev/null \
        || compgen -G "${MODEL_DIR_HOST}/pytorch_model*.bin" >/dev/null
}

cleanup() {
    set +e
    if ! is_true "${VLLM_DISABLED}"; then
        if docker ps -a --format '{{.Names}}' | grep -qx "${VLLM_CONTAINER}"; then
            log "Removing vLLM Docker container"
            docker rm -f "${VLLM_CONTAINER}" >/dev/null 2>&1 || true
        fi
    fi
}

trap cleanup EXIT INT TERM

# =============================================================================
# Wait for vLLM
# =============================================================================

wait_for_vllm() {

    local start_ts now_ts
    start_ts="$(date +%s)"

    log "Waiting for vLLM service to become ready: ${BASE_URL}/v1/models"

    while true; do

        if ! docker ps --format '{{.Names}}' | grep -qx "${VLLM_CONTAINER}"; then
            log "vLLM container exited unexpectedly during startup"
            docker logs --tail 200 "${VLLM_CONTAINER}" 2>&1 || true
            die "vLLM startup failed"
        fi

        if "${CURL_BIN}" -sSf --max-time 2 "${BASE_URL}/v1/models" >/dev/null 2>&1; then
            log "vLLM service is ready. BASE_URL: ${BASE_URL}"
            return 0
        fi

        now_ts="$(date +%s)"

        if (( now_ts - start_ts > WAIT_TIMEOUT_SEC )); then
            log "vLLM startup timeout"
            docker logs --tail 200 "${VLLM_CONTAINER}" 2>&1 || true
            die "vLLM inference service did not become ready within ${WAIT_TIMEOUT_SEC} seconds"
        fi

        sleep 2
    done
}

# =============================================================================
# GPU configuration
# =============================================================================

build_gpus_flag() {

    local g="${GPUS}"
    g="${g//[[:space:]]/}"

    if [[ "${g}" == "all" ]]; then
        GPUS_FLAG=(--gpus all)
        return
    fi

    if [[ -z "${g}" ]]; then
        die "GPUS is empty"
    fi

    if [[ "${g}" == device=* ]]; then
        GPUS_FLAG=(--gpus "\"${g}\"")
    else
        GPUS_FLAG=(--gpus "\"device=${g}\"")
    fi
}

# =============================================================================
# Evaluation
# =============================================================================

run_evaluation() {

    log "==================== Starting Model Evaluation ===================="

    cd "${EVAL_DIR}" || die "Evaluation directory does not exist: ${EVAL_DIR}"

    local time_stamp
    time_stamp="$(date +%Y%m%d%H%M%S)"

    mkdir -p outputs

    K_LIST=(10)
    for K in "${K_LIST[@]}"; do

        echo "========================================"
        echo "Running evaluation:k=${K}"
        echo "========================================"

        [[ -f "${BENCHMARK}" ]] || die "Benchmark file not found"

        python generate_response.py \
            --benchmark "${BENCHMARK}" \
            --api-docs "${API_DOCS}" \
            --repo-config "${REPO_CONFIG}"\
            --base-url "${BASE_URL}" \
            --model "${SERVED_MODEL_NAME}" \
            --mode "none" \
            --k "${K}" \
            --temperature 0.5 \
            --top-p 0.95 \
            --out "outputs/generations.VANILLA.$(basename "${BENCHMARK%.*}").none.${SERVED_MODEL_NAME}.${time_stamp}.k=${K}.jsonl" \
            --run-eval \
            --exec-timeout 15 \
            $HUMANEVAL_ARG
        log "Model evaluation completed. Results saved:"
        log "  → outputs/generations.VANILLA.$(basename "${BENCHMARK%.*}").none.${SERVED_MODEL_NAME}.${time_stamp}.k=${K}.jsonl"
    done
    if (( USE_HUMANEVAL == 0 )); then
        for K in "${K_LIST[@]}"; do
            echo "========================================"
            echo "Running evaluation:k=${K}"
            echo "========================================"

            [[ -f "${BENCHMARK}" ]] || die "Benchmark file not found"

            python generate_response.py \
                --benchmark "${BENCHMARK}" \
                --api-docs "${API_DOCS}" \
                --repo-config "${REPO_CONFIG}" \
                --base-url "${BASE_URL}" \
                --model "${SERVED_MODEL_NAME}" \
                --mode "gold" \
                --k "${K}" \
                --temperature 0.5 \
                --top-p 0.95 \
                --out "outputs/generations.VANILLA.$(basename "${BENCHMARK%.*}").gold.${SERVED_MODEL_NAME}.${time_stamp}.k=${K}.jsonl" \
                --run-eval \
                --exec-timeout 15 \
                $HUMANEVAL_ARG
            log "Model evaluation completed. Results saved:"
            log "  → outputs/generations.VANILLA.$(basename "${BENCHMARK%.*}").gold.${SERVED_MODEL_NAME}.${time_stamp}.k=${K}.jsonl"
        done
    else
        log "Skipping gold evaluation because HumanEval does not have Gold API"
    fi
}

# =============================================================================
# Main
# =============================================================================

need_cmd python

if ! is_true "${VLLM_DISABLED}"; then
    need_cmd docker
    need_cmd "${CURL_BIN}"
fi

log "==================== Running Pre-checks ===================="

if ! is_true "${VLLM_DISABLED}"; then
    if port_in_use; then
        die "Port ${PORT} is already in use. Cannot start the vLLM service."
    fi
fi

if ! is_true "${VLLM_DISABLED}"; then

    [[ -d "${MODEL_DIR_HOST}" ]] || die "Model directory does not exist: ${MODEL_DIR_HOST}"

    if ! have_weights; then
        die "No valid model weights found in the directory. Supported formats: *.safetensors / pytorch_model*.bin"
    fi

    [[ -f "${MODEL_DIR_HOST}/config.json" ]] || \
        die "Required config.json file missing: ${MODEL_DIR_HOST}/config.json"

else
    log "VLLM_DISABLED=${VLLM_DISABLED}: skipping local model checks. Using external BASE_URL service."
fi

[[ -d "${EVAL_DIR}" ]] || die "Evaluation directory does not exist: ${EVAL_DIR}"
[[ -f "${EVAL_DIR}/generate_response.py" ]] || \
    die "Evaluation script not found: ${EVAL_DIR}/generate_response.py"

[[ -f "${REPO_CONFIG}" ]] || \
    die "Evaluation configuration file not found"

[[ -f "${API_DOCS}" ]] || die "API documentation file not found: ${API_DOCS}"

log "All pre-checks passed successfully"

# =============================================================================
# Start vLLM
# =============================================================================

if is_true "${VLLM_DISABLED}"; then

    log "==================== Skipping vLLM Startup (VLLM_DISABLED=${VLLM_DISABLED}) ===================="
    log "Evaluation will use the external inference service at BASE_URL=${BASE_URL}"

else

    log "==================== Starting vLLM Inference Service ===================="

    GPUS_FLAG=()
    build_gpus_flag

    log "Using Docker GPU configuration: ${GPUS_FLAG[*]}"

    docker run -d \
        --name "${VLLM_CONTAINER}" \
        "${GPUS_FLAG[@]}" \
        --ipc=host \
        -p "${PORT}:8000" \
        -v "${MODEL_DIR_HOST}:/model:ro" \
        "${VLLM_IMAGE}" \
        --model /model \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --tensor-parallel-size "${TP}" \
        ${EXTRA_VLLM_ARGS} >/dev/null

    log "vLLM container started. Container name: ${VLLM_CONTAINER}"

    log "vLLM service configuration:"
    log "  → Model mount path: ${MODEL_DIR_HOST}"
    log "  → GPU devices: ${GPUS}"
    log "  → Tensor parallel size: ${TP}"
    log "  → GPU memory utilization: ${GPU_MEM_UTIL}"

    wait_for_vllm
fi

# =============================================================================
# Run evaluation
# =============================================================================

run_evaluation

log "=============================================================="
log "✅ Script execution completed successfully"
log "📌 Summary:"

if is_true "${VLLM_DISABLED}"; then
    log "  → vLLM service: not started (VLLM_DISABLED=${VLLM_DISABLED}), using external BASE_URL=${BASE_URL}"
else
    log "  → vLLM service: started successfully and used for inference"
fi

log "  → Model evaluation completed"
log "  → Evaluation results directory: ${EVAL_DIR}/outputs"
log "=============================================================="