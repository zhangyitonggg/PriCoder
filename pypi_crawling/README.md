# PyPI Crawling

Auxiliary scripts for collecting and filtering candidate public libraries from PyPI metadata.

## Files

- `crawl_pypi.py`: crawl trending packages and metadata
- `filter_pypi.py`: filter crawled records by stars/downloads/time/topic rules

## Example

```bash
cd pypi_crawling
python crawl_pypi.py --limit 100 -o pypi_trending.jsonl
python filter_pypi.py -i pypi_trending.jsonl -o pypi_trending.filtered.jsonl --min-stars 5000 --min-downloads 1000000 --cutoff 2024-01-01
```
