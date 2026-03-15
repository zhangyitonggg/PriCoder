export AIHUBMIX_API_KEY="${AIHUBMIX_API_KEY:-YOUR_API_KEY}"
CONDA_PATH=$(conda info --base)
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate numba_cuda
export PYTHONNOUSERSITE=1

python generate_benchmark.py \
  --spec ./ndonnx/repo_spec.json \
  --out ndonnx_benchmark.jsonl \
  --num 1000 --k 1 --model gpt-5.2 --json-mode

