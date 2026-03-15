#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter API JSONL records by APIs mentioned in documentation.

Input modes:
1) `--doc`: parse API names from a raw documentation text file.
2) `--doc-apis`: read one API full name per line.

The script keeps only matching records from `functions.jsonl` and writes a
filtered JSONL file. Optionally, it adds a computed `full_name` field.
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Any


FULLNAME_RE = re.compile(r"\bndonnx(?:\.[A-Za-z_]\w*)+\b")

# Sphinx/RST patterns:
#   classndonnx.Array
#   class ndonnx.Array
CLASS_RE = re.compile(r"(?:\bclass\s*ndonnx\.|\bclassndonnx\.)([A-Za-z_]\w*)")

#   propertyT: Array
#   property values: Array (no whitespace)
PROP_RE_1 = re.compile(r"^\s*property\s*([A-Za-z_]\w*)\b")
PROP_RE_2 = re.compile(r"^\s*property([A-Za-z_]\w*)\b")

#   all(axis: ...)→ Array
METHOD_RE = re.compile(r"^\s*([A-Za-z_]\w*)\(")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_doc_api_fullnames(doc_text: str) -> Set[str]:
    """
    Extract API full names (for example `ndonnx.xxx`) from documentation text.

    It supports direct names like `ndonnx.abs` and class-based entries such as
    `class ndonnx.Array` + member lines (`all`, `astype`, `property T`).
    """
    apis: Set[str] = set()

    # 1) Direct `ndonnx.xxx` mentions
    for m in FULLNAME_RE.finditer(doc_text):
        apis.add(m.group(0))

    # 2) Parse class/member sections
    current_class: Optional[str] = None
    for line in doc_text.splitlines():
        m = CLASS_RE.search(line)
        if m:
            current_class = m.group(1)
            apis.add(f"ndonnx.{current_class}")
            continue

        if not current_class:
            continue

        mp = PROP_RE_1.match(line) or PROP_RE_2.match(line)
        if mp:
            prop = mp.group(1)
            apis.add(f"ndonnx.{current_class}.{prop}")
            continue

        mm = METHOD_RE.match(line)
        if mm:
            method = mm.group(1)
            # Skip section labels that are not methods
            if method in {"Bases"}:
                continue
            apis.add(f"ndonnx.{current_class}.{method}")
            continue

    return apis


def read_doc_api_list(path: Path) -> Set[str]:
    """
    Read API full names from a plain text list (one per line).
    """
    apis: Set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        apis.add(s)
    return apis


def compute_jsonl_full_name(rec: Dict[str, Any]) -> str:
    """
    Compute canonical API full_name for one JSONL record.

    Expected examples in `functions.jsonl`:
      {"source_file": "extensions.py", "qualname": "fill_null", ...}
      {"source_file": "_array.py", "qualname": "Array.astype", ...}
      {"source_file": "_funcs.py", "qualname": "abs", ...}

    Mapping rules:
    - `extensions.py` -> `ndonnx.extensions.<qualname>`
    - otherwise -> `ndonnx.<qualname>`
    """
    qualname = rec.get("qualname") or rec.get("name") or ""
    source_file = (rec.get("source_file") or "").replace("\\", "/")

    # Already normalized
    if qualname.startswith("ndonnx."):
        return qualname

    if source_file.endswith("extensions.py"):
        return f"ndonnx.extensions.{qualname}"
    return f"ndonnx.{qualname}"


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"[ERROR] Invalid JSON at {path}:{i}\n{e}\nline={line[:200]}...") from e


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Filter functions.jsonl by APIs contained in API documentation text.")
    p.add_argument("--jsonl", required=True, help="Path to input functions.jsonl")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--doc", help="Path to documentation text file")
    g.add_argument("--doc-apis", help="Path to API full-name list (one per line)")
    p.add_argument("--out", required=True, help="Path to output filtered JSONL")
    p.add_argument("--out-apis", default=None, help="Optional path to write kept API names")
    p.add_argument("--add-fullname-field", action="store_true", help="Add computed `full_name` to each kept record")
    args = p.parse_args()

    jsonl_path = Path(args.jsonl)
    out_path = Path(args.out)

    # Load doc API names from one of the two input modes
    if args.doc_apis:
        doc_apis = read_doc_api_list(Path(args.doc_apis))
    else:
        doc_text = read_text(Path(args.doc))
        doc_apis = extract_doc_api_fullnames(doc_text)

    if not doc_apis:
        raise SystemExit("[ERROR] No API names were extracted from documentation input.")

    kept: List[Dict[str, Any]] = []
    kept_names: Set[str] = set()

    total = 0
    for rec in iter_jsonl(jsonl_path):
        total += 1
        full = compute_jsonl_full_name(rec)
        if full in doc_apis:
            if args.add_fullname_field:
                rec = dict(rec)
                rec["full_name"] = full
            kept.append(rec)
            kept_names.add(full)

    n_written = write_jsonl(out_path, kept)

    # Optionally write kept API names
    if args.out_apis:
        Path(args.out_apis).write_text("\n".join(sorted(kept_names)) + "\n", encoding="utf-8")

    print(f"[OK] doc apis: {len(doc_apis)}")
    print(f"[OK] jsonl total records: {total}")
    print(f"[OK] kept records: {n_written}")
    print(f"[OK] kept unique api names: {len(kept_names)}")
    print(f"[OK] output: {out_path}")


if __name__ == "__main__":
    main()
