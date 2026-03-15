# PriCoder

This repo provide a full pipeline for **private-library-oriented code generation**: benchmark construction, inference/evaluation, and training-data synthesis for teaching models to truly use private libraries effectively.

---

## 🧭 Repository Map

- `api_extract/` — API-doc extraction and filtering
- `benchmark_generation/` — benchmark instance generation
- `infer_and_eval/` — generation, execution-based evaluation, metric aggregation
- `data_generation/` — PriCoder training-data synthesis pipeline
- `data/` — reusable JSONL assets (documents and benchmarks)
- `pypi_crawling/` — auxiliary package mining/filtering tools

---

## 🚀 Quick Workflow

1. Extract API docs from a target private library
2. Generate benchmark instances from API docs
3. Run model inference and execution-based evaluation
4. Synthesize PriCoder training data and fine-tune

**Detailed commands are in each subdirectory README.**
