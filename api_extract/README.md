# API Extraction

This directory extracts API records from a target Python library into JSONL format for downstream benchmark construction or SFT.

## Files

- `extract_api_runtime.py`: runtime-based API traversal and extraction
- `filter_api.py`: filter extracted API JSONL by documentation coverage

## Example

```bash
cd api_extract
python extract_api_runtime.py --module ndonnx --recursion-depth 2 -o ../data/document/ndonnx.jsonl
```

```bash
python filter_api.py --jsonl ../data/document/ndonnx.jsonl --doc api_doc.txt --out ../data/document/ndonnx.filtered.jsonl
```
