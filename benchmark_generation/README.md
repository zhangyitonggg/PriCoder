# Benchmark Generation

Generate private-library-oriented code generation benchmark samples with LLMs.

## Files

- `generate_benchmark.py`: baseline benchmark generation pipeline
- `generate_benchmark.sh`: runnable examples
- `ndonnx/`, `numba/`: repo-specific specs and prompt assets

## Quick Run

```bash
cd benchmark_generation
export AIHUBMIX_API_KEY="YOUR_KEY"
python generate_benchmark.py \
  --spec ./ndonnx/repo_spec.json \
  --out ../data/benchmark/ndonnx_benchmark.jsonl \
  --num 50 --k 3 --model gpt-5.1 --json-mode
```

## Output Schema (JSONL)

Each sample typically includes:

- `task`
- `prompt`
- `canonical_solution`
- `test`
- `api_list`
- `seed_apis`
- `repo_spec`
