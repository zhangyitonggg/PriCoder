#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract runtime-inspected APIs from a package and write API-doc JSONL.

This script imports a target package, walks functions/classes/modules,
captures signatures/docstrings/source snippets, and emits JSONL records.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import io
import json
import os
import tokenize
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


@dataclass
class FuncInfo:
    qualname: str
    name: str
    lineno: int
    end_lineno: int
    docstring: str
    signature: str
    code_no_docstring: str
    source_file: str


# -------------------------
# Docstring removal helpers (tokenize-based)
# -------------------------

def _remove_leading_docstring_after_indent(block_src: str) -> str:
    """Remove leading docstring immediately after INDENT in def/class blocks."""
    sio = io.StringIO(block_src)
    try:
        tokens = list(tokenize.generate_tokens(sio.readline))  # type: ignore
    except tokenize.TokenError:
        return block_src

    indent_idx = None
    for i, t in enumerate(tokens):
        if t.type == tokenize.INDENT:
            indent_idx = i
            break
    if indent_idx is None:
        return block_src

    str_idx = None
    for i in range(indent_idx + 1, len(tokens)):
        if tokens[i].type in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            continue
        if tokens[i].type == tokenize.STRING:
            str_idx = i
        break

    if str_idx is None:
        return block_src

    str_tok = tokens[str_idx]
    start_line, start_col = str_tok.start
    end_line, end_col = str_tok.end

    # Also remove the trailing NEWLINE token after docstring when present
    j = str_idx + 1
    while j < len(tokens) and tokens[j].type not in (tokenize.NEWLINE, tokenize.NL):
        j += 1
    if j < len(tokens) and tokens[j].type == tokenize.NEWLINE:
        end_line, end_col = tokens[j].end

    src_lines = block_src.splitlines(True)
    before = "".join(src_lines[: start_line - 1]) + src_lines[start_line - 1][:start_col]
    after = src_lines[end_line - 1][end_col:] + "".join(src_lines[end_line:])
    return before + after


def _remove_module_docstring(src: str) -> str:
    """Remove module-level leading string literal used as module docstring."""
    sio = io.StringIO(src)
    try:
        tokens = list(tokenize.generate_tokens(sio.readline))  # type: ignore
    except tokenize.TokenError:
        return src

    first_idx = None
    for i, t in enumerate(tokens):
        if t.type in (tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            continue
        first_idx = i
        break
    if first_idx is None:
        return src

    if tokens[first_idx].type != tokenize.STRING:
        return src

    str_tok = tokens[first_idx]
    start_line, start_col = str_tok.start
    end_line, end_col = str_tok.end

    j = first_idx + 1
    while j < len(tokens) and tokens[j].type not in (tokenize.NEWLINE, tokenize.NL):
        j += 1
    if j < len(tokens) and tokens[j].type == tokenize.NEWLINE:
        end_line, end_col = tokens[j].end

    src_lines = src.splitlines(True)
    before = "".join(src_lines[: start_line - 1]) + src_lines[start_line - 1][:start_col]
    after = src_lines[end_line - 1][end_col:] + "".join(src_lines[end_line:])
    return before + after


# -------------------------
# source/signature/lineno helpers
# -------------------------

def _safe_signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return ""


def _safe_get_source(obj: Any) -> Tuple[str, int, int, Optional[str]]:
    """
    Return `(source, lineno, end_lineno, source_file)` best-effort.
    For C-extensions/builtins or failures, return empty source with zero lines.
    """
    try:
        src_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
    except Exception:
        src_file = None

    try:
        lines, start = inspect.getsourcelines(obj)
        source = "".join(lines)
        lineno = int(start)
        end_lineno = int(start + len(lines) - 1)
        return source, lineno, end_lineno, src_file
    except Exception:
        return "", 0, 0, src_file


def _rel_source_path(src_file: Optional[str], package_root: Path) -> str:
    """
    Convert an absolute source path to a path relative to package_root when possible.
    Fallback to basename if the path is outside package_root.
    """
    if not src_file:
        return ""
    try:
        p = Path(src_file).resolve()
        root = package_root.resolve()
        s = str(p)
        r = str(root)
        if os.path.commonpath([s, r]) == r:
            return os.path.relpath(s, r)
        return p.name
    except Exception:
        return os.path.basename(src_file)


# -------------------------
# Member-kind / descriptor unwrap
# -------------------------

def _unwrap_descriptor(x: Any) -> Any:
    if isinstance(x, (staticmethod, classmethod)):
        return x.__func__
    return x


def _is_function_like(x: Any) -> bool:
    return (
        inspect.isfunction(x)
        or inspect.isbuiltin(x)
        or inspect.ismethoddescriptor(x)
        or inspect.isroutine(x)
    )


def _kind_of(member: Any) -> str:
    if isinstance(member, types.ModuleType):
        return "module"
    if inspect.isclass(member):
        return "class"
    if isinstance(member, (staticmethod, classmethod)):
        return "function"
    if _is_function_like(member):
        return "function"
    return "other"


def _module_names(mod: types.ModuleType, include_private: bool) -> List[str]:
    """
    Enumerate module member names.
    - Prefer `__all__` when available.
    - Otherwise use `dir(mod)`.
    Filtering:
    - Skip dunder names (`__xxx__`).
    - If include_private=False, skip single-underscore names.
    """
    allv = getattr(mod, "__all__", None)
    if isinstance(allv, (list, tuple)) and all(isinstance(x, str) for x in allv):
        names = list(allv)
    else:
        try:
            names = list(dir(mod))
        except Exception:
            names = []

    out: List[str] = []
    seen: Set[str] = set()
    for n in names:
        if n in seen:
            continue
        seen.add(n)

        #  __xxx__
        if n.startswith("__"):
            continue
        if (not include_private) and n.startswith("_"):
            continue
        out.append(n)
    return out


def _class_member_items(cls: type, include_private: bool) -> Iterator[Tuple[str, Any]]:
    """
    Iterate class members via `inspect.getmembers_static`.
    Filters:
    - skip dunder names
    - skip private names when include_private=False
    """
    try:
        items = inspect.getmembers_static(cls)
    except Exception:
        items = []
    for name, raw in items:
        if name.startswith("__"):
            continue
        if (not include_private) and name.startswith("_"):
            continue
        yield name, raw


# -------------------------
# Record builder (function/class)
# -------------------------

def _build_record(
    full_name: str,
    attr_name: str,
    obj: Any,
    kind: str,
    package_root: Path,
) -> FuncInfo:
    doc = getattr(obj, "__doc__", "") or ""
    target = _unwrap_descriptor(obj)

    source, lineno, end_lineno, src_file = _safe_get_source(target)
    src_rel = _rel_source_path(src_file, package_root)
    sig = _safe_signature(target)

    code_no_doc = source
    if source:
        if kind == "module":
            code_no_doc = _remove_module_docstring(source)
        elif kind in ("function", "class"):
            code_no_doc = _remove_leading_docstring_after_indent(source)

    return FuncInfo(
        qualname=full_name,
        name=attr_name,
        lineno=lineno,
        end_lineno=end_lineno,
        docstring=doc,
        signature=sig,
        code_no_docstring=code_no_doc,
        source_file=src_rel,
    )


# -------------------------
# Traversal: module/package containers + function/class records
# -------------------------

def extract_from_module(
    root_module_name: str,
    recursion_depth: int,
    *,
    include_private: bool = False,
) -> List[FuncInfo]:
    root_mod = importlib.import_module(root_module_name)
    if not isinstance(root_mod, types.ModuleType):
        raise SystemExit(f"[ERROR] Failed to import module: {root_module_name}")

    pkg_root_file = getattr(root_mod, "__file__", None)
    if pkg_root_file:
        package_root = Path(pkg_root_file).resolve().parent
    else:
        package_root = Path(os.getcwd()).resolve()

    seen_records: Set[str] = set()       # dedupe function/class records
    visited_containers: Set[str] = set() # dedupe visited containers
    out: List[FuncInfo] = []

    def depth_of(full_name: str) -> int:
        if full_name == root_module_name:
            return 0
        if not full_name.startswith(root_module_name + "."):
            return 10**9
        return full_name.count(".") - root_module_name.count(".")

    def maybe_add_record(full_name: str, attr_name: str, obj: Any, kind: str):
        if kind not in ("function", "class"):
            return
        if depth_of(full_name) > recursion_depth:
            return
        if full_name in seen_records:
            return
        seen_records.add(full_name)
        out.append(_build_record(full_name, attr_name, obj, kind, package_root))

    def walk_module(mod: types.ModuleType, mod_name: str):
        if mod_name in visited_containers:
            return
        visited_containers.add(mod_name)

        if depth_of(mod_name) >= recursion_depth:
            return

        for name in _module_names(mod, include_private):
            full_name = f"{mod_name}.{name}"
            if depth_of(full_name) > recursion_depth:
                continue

            try:
                obj = getattr(mod, name)  # may trigger lazy exports
            except Exception:
                continue

            kind = _kind_of(obj)
            maybe_add_record(full_name, name, obj, kind)

            # recurse into module/class containers
            if kind == "module" and isinstance(obj, types.ModuleType):
                mname = getattr(obj, "__name__", full_name)
                if mname.startswith(root_module_name + "."):
                    walk_module(obj, mname)
            elif kind == "class" and inspect.isclass(obj):
                walk_class(obj, full_name)

    def walk_class(cls: type, cls_name: str):
        if cls_name in visited_containers:
            return
        visited_containers.add(cls_name)

        if depth_of(cls_name) > recursion_depth:
            return

        # include class object itself
        maybe_add_record(cls_name, cls_name.split(".")[-1], cls, "class")

        if depth_of(cls_name) >= recursion_depth:
            return

        for name, raw in _class_member_items(cls, include_private):
            full_name = f"{cls_name}.{name}"
            if depth_of(full_name) > recursion_depth:
                continue

            obj = _unwrap_descriptor(raw)
            kind = _kind_of(obj)

            maybe_add_record(full_name, name, obj, kind)

            # recurse into nested class definitions
            if kind == "class" and inspect.isclass(obj) and depth_of(full_name) < recursion_depth:
                walk_class(obj, full_name)

    walk_module(root_mod, root_module_name)
    return out


def write_jsonl(funcs: List[FuncInfo], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for x in funcs:
            obj: Dict[str, Any] = {
                "source_file": x.source_file,
                "qualname": x.qualname,
                "name": x.name,
                "lineno": x.lineno,
                "end_lineno": x.end_lineno,
                "signature": x.signature,
                "docstring": x.docstring,
                "code": x.code_no_docstring,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="Extract runtime APIs (ONLY function/class) into JSONL. No pkgutil. Skip __* always.")
    p.add_argument("--module", default="ndonnx", help="Module name to import (e.g., ndonnx)")
    p.add_argument("--recursion-depth", type=int, default=2, help="Max traversal depth (default: 2)")
    p.add_argument("-o", "--out", default="functions.jsonl", help="Output JSONL path (default: functions.jsonl)")
    p.add_argument("--include-private", action="store_true", help="Include _private names (still skips __dunder__)")
    args = p.parse_args()

    if args.recursion_depth < 0:
        raise SystemExit("[ERROR] --recursion-depth must be >= 0")

    funcs = extract_from_module(
        args.module,
        args.recursion_depth,
        include_private=args.include_private,
    )
    write_jsonl(funcs, args.out)
    print(f"[OK] module={args.module} depth={args.recursion_depth} -> {args.out}, records={len(funcs)}")


if __name__ == "__main__":
    main()
