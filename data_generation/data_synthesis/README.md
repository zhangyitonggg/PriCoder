# Compositional Data Synthesis Entrypoints

This directory provides runnable shell wrappers for PriCoder synthesis.

## Files

- `synthesis_generation.sh`: standard synthesis loop
- `compositional_fusion_generation.sh`: compositional fusion loop (D2..Dmax)

## Example

```bash
cd data_generation/compositional_data_synthesis
NUM=200 K=3 MODEL=Qwen2.5-Coder-7B-Instruct \
EXEC_PYTHON=/path/to/python_with_private_lib \
bash synthesis_generation.sh
```

## Common Variables

- `NUM`: target number of accepted samples
- `K`: number of seed APIs per sample (or candidate set in underlying script)
- `MODEL`: model name served by a vLLM/OpenAI-compatible endpoint
- `BASE_URL`: endpoint base URL (default `http://127.0.0.1:8000`)
- `EXEC_PYTHON`: python executable for execution verification
