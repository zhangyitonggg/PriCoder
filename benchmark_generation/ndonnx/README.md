# ndonnx Benchmark Spec Assets

This folder contains prompt/config assets used by `../generate_benchmark.py` for the `ndonnx` benchmark.

## Files

- `repo_spec.json`: benchmark generation spec entry
- `overview.txt`: library overview injected into prompts
- `fewshots.jsonl`: few-shot benchmark exemplars
- `prompt_template.txt`: generation instruction template

## Example

```bash
cd benchmark_generation
python generate_benchmark.py --spec ./ndonnx/repo_spec.json --out ../data/benchmark/ndonnx_init.jsonl --num 20 --k 3
```
