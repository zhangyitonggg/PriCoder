# Inference and Evaluation

Run model inference on benchmark tasks and evaluate outputs using execution-based checks.

## Files

- `generate_response.py`: generation + evaluation main script
- `generate_response.sh`: command wrapper examples
- `configs/`: repo-specific evaluation configs
- `benchmark/`: benchmark inputs
- `docs/`: API-doc JSONL used for doc injection
- `ndonnx_environment.yml`, `numba_environment.yml`: optional env specs

## Quick Run

```bash
cd infer_and_eval
./generate_response.sh "/path/to/eval_model" "model_name" "./docs/ndonnx.jsonl" "./benchmark/NdonnxEval.jsonl" "./configs/repo_config.ndonnx.json"
```

For HumanEval-style evaluation, pass the extra mode argument expected by `generate_response.sh`.
