# SFT Data Synthesis via vLLM

Reusable PriCoder synthesis scripts for private-library training data.

## Structure

- `scripts/`: core synthesis implementations
- `specs/`: per-library synthesis specs

## Main Command

```bash
cd data_generation/sft_data_gen_vllm
python scripts/generate_sft_loop.py \
  --spec specs/ndonnx/repo_spec.json \
  --out /tmp/synthesis_sft.jsonl \
  --fail-out /tmp/synthesis_fail.jsonl \
  --num 200 --k 3 \
  --base-url http://127.0.0.1:8000 \
  --model Qwen2.5-Coder-7B-Instruct \
  --exec-python /path/to/python_with_library
```
