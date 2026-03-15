# Script Reference

## Core Scripts

- `generate_sft_loop.py`: standard synthesis loop generation
- `generate_sft_fusion_loop.py`: compositional fusion generation
- `synthesis_common.py`: shared synthesis utilities (spec loading, prompts, validation, execution)

## Minimal Examples

```bash
python generate_sft_loop.py --spec ../specs/ndonnx/repo_spec.json --out synthesis.jsonl --num 100 --k 3
```
