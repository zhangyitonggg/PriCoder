# Data Generation

Training data synthesis pipeline for PriCoder.

## Subdirectories

- `data_synthesis/`: shell entrypoints for PriCoder data synthesis
- `sft_data_gen_vllm/`: reusable synthesis framework and specs

## Typical Flow

1. Prepare API-doc JSONL
2. Configure a synthesis spec under `sft_data_gen_vllm/specs/`
3. Run generation from `data_synthesis/`
4. Collect accepted/failed JSONL outputs
