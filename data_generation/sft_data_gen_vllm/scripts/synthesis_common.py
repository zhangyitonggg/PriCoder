#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared utilities for PriCoder synthesis pipelines.

This module centralizes:
- spec loading and validation
- prompt construction
- model output parsing
- API extraction and validation
- optional local/docker execution helpers
"""

from __future__ import annotations

import ast
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import requests


# ----------------------------
# vLLM message sanitization
# ----------------------------
# Purpose:
# - avoid parser failures due to malformed/unsupported characters
# - strip CJK chars from messages when required by model constraints
# - keep behavior deterministic and explicit

# CJK range + punctuation coverage
_CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3000-\u303f\uF900-\uFAFF\uFF01-\uFF60\uFFE0-\uFFEF\U00020000-\U0002EBEF]"
)


def strip_cjk(text: str) -> str:
    """Remove CJK characters from text (best-effort sanitizer)."""
    if text is None:
        return None
    sanitized = _CJK_RE.sub("", text or "")
    if sanitized != text:
        print(f"[strip_cjk] Warning: CJK characters removed from text: {text!r}", file=sys.stderr)
    return sanitized


def sanitize_messages(messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        """Return copied messages with CJK-stripped `content` fields."""
    out: List[Dict[str, str]] = []
    for m in messages or []:
        mm = dict(m or {})
        mm["content"] = strip_cjk(str(mm.get("content", "") or ""))
        out.append(mm)
    return out


# ----------------------------
# Spec + API
# ----------------------------


@dataclass(frozen=True)
class SynthesisSpec:
    library_name: str
    library_overview: str
    api_docs_path: str

    # import/module naming
    module_name: str
    module_alias: str
    required_import: str
    module_call_prefixes: Tuple[str, ...]

    # prompts
    system_message: str
    judge_system_message: str
    question_instructions: str
    answer_instructions: str
    judge_instructions: str

    # few-shot examples (benchmark schema: task/solution/test)
    few_shot_pool: List[Dict[str, str]]
    num_few_shots_q: int
    num_few_shots_a: int

    # constraints
    min_asserts: int
    seed_overlap_required: bool

    # max allowed top-level defs
    # - 1: single-function tasks
    # - 2+: allows kernel+wrapper style patterns
    max_top_level_functions: int

    def normalized(self) -> "SynthesisSpec":
        prefixes = list(self.module_call_prefixes)
        if self.module_alias and self.module_alias not in prefixes:
            prefixes.insert(0, self.module_alias)
        if self.module_name.isidentifier() and self.module_name not in prefixes:
            prefixes.append(self.module_name)

        return SynthesisSpec(
            library_name=self.library_name,
            library_overview=self.library_overview,
            api_docs_path=self.api_docs_path,
            module_name=self.module_name,
            module_alias=self.module_alias,
            required_import=self.required_import,
            module_call_prefixes=tuple(prefixes),
            system_message=self.system_message,
            judge_system_message=self.judge_system_message,
            question_instructions=self.question_instructions,
            answer_instructions=self.answer_instructions,
            judge_instructions=self.judge_instructions,
            few_shot_pool=self.few_shot_pool,
            num_few_shots_q=self.num_few_shots_q,
            num_few_shots_a=self.num_few_shots_a,
            min_asserts=self.min_asserts,
            seed_overlap_required=self.seed_overlap_required,
            max_top_level_functions=self.max_top_level_functions,
        )


@dataclass(frozen=True)
class ApiDoc:
    source_file: str
    qualname: str
    name: str
    signature: str
    docstring: str
    code: str


SeedApiDoc = Dict[str, str]  # {full_name, signature, docstring}


def _resolve_path(path_str: str, *, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


def _load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Any]:
    items: List[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _validate_few_shot_pool(pool: Any) -> List[Dict[str, str]]:
    if not isinstance(pool, list):
        raise ValueError("few_shot_pool must be a JSON list.")
    out: List[Dict[str, str]] = []
    for i, ex in enumerate(pool):
        if not isinstance(ex, dict):
            raise ValueError(f"few_shot_pool[{i}] must be an object.")
        for k in ("task", "solution", "test"):
            if k not in ex or not isinstance(ex[k], str) or not ex[k].strip():
                raise ValueError(f"few_shot_pool[{i}] missing/invalid field: {k}")
        out.append({"task": ex["task"], "solution": ex["solution"], "test": ex["test"]})
    return out


def load_synthesis_spec(spec_path: Path) -> SynthesisSpec:
        """Load synthesis spec JSON and resolve optional indirection files.

        Supported *_file indirections include overview, few-shot pool, prompt texts,
        and api_docs path.
        """
    base_dir = spec_path.parent
    raw = _load_json_file(spec_path)
    if not isinstance(raw, dict):
        raise ValueError("--spec must be a JSON object")

    library_name = str(raw.get("library_name", raw.get("module_name", "Library")))

    if "library_overview_file" in raw and raw["library_overview_file"]:
        overview_path = _resolve_path(str(raw["library_overview_file"]), base_dir=base_dir)
        library_overview = _load_text_file(overview_path)
    else:
        library_overview = str(raw.get("library_overview", "") or "")
    if not library_overview.strip():
        raise ValueError("Spec must provide library_overview or library_overview_file")

    # API docs (JSONL)
    api_docs_raw = raw.get("api_docs_file") or raw.get("api_docs") or raw.get("api_docs_path")
    if not api_docs_raw:
        raise ValueError("Spec must provide 'api_docs_file' (path to API docs JSONL).")
    api_docs_path = _resolve_path(str(api_docs_raw), base_dir=base_dir)
    if not api_docs_path.exists():
        raise ValueError(f"api_docs_file not found: {api_docs_path}")

    def _load_required_prompt(key: str) -> str:
        file_key = key + "_file"
        if file_key in raw and raw[file_key]:
            p = _resolve_path(str(raw[file_key]), base_dir=base_dir)
            return _load_text_file(p)
        t = str(raw.get(key, "") or "")
        if not t.strip():
            raise ValueError(f"Spec must provide '{key}' or '{file_key}' (no built-in defaults).")
        return t

    system_message = str(raw.get("system_message", "You are a careful data generator. Follow instructions exactly."))
    judge_system_message = str(raw.get("judge_system_message", "You are a strict semantic judge. Follow instructions exactly."))
    question_instructions = _load_required_prompt("question_instructions")
    answer_instructions = _load_required_prompt("answer_instructions")
    judge_instructions = _load_required_prompt("judge_instructions")

    if "few_shot_pool_file" in raw and raw["few_shot_pool_file"]:
        pool_path = _resolve_path(str(raw["few_shot_pool_file"]), base_dir=base_dir)
        pool_any = _load_jsonl(pool_path) if pool_path.suffix.lower() == ".jsonl" else _load_json_file(pool_path)
        few_shot_pool = _validate_few_shot_pool(pool_any)
    else:
        few_shot_pool = _validate_few_shot_pool(raw.get("few_shot_pool", []))

    module_name = str(raw.get("module_name", library_name or "mylib"))
    module_alias = str(raw.get("module_alias", "ml"))
    required_import = str(raw.get("required_import", f"import {module_name} as {module_alias}"))

    prefixes_raw = raw.get("module_call_prefixes", None)
    if prefixes_raw is None:
        prefixes: Tuple[str, ...] = (module_alias, module_name) if module_name.isidentifier() else (module_alias,)
    else:
        if not isinstance(prefixes_raw, list) or not all(isinstance(x, str) and x for x in prefixes_raw):
            raise ValueError("'module_call_prefixes' must be a list of non-empty strings.")
        prefixes = tuple(prefixes_raw)

    default_nfs = raw.get("num_few_shots", 2)
    num_few_shots_q = int(raw.get("num_few_shots_q", default_nfs))
    num_few_shots_a = int(raw.get("num_few_shots_a", default_nfs))

    # Keep compatibility with benchmark specs that omit `min_asserts`.
    # Default to >=1 assert unless explicitly set.
    min_asserts_raw = raw.get("min_asserts")
    if min_asserts_raw is None:
        min_asserts = 1
    else:
        min_asserts = int(min_asserts_raw)

    seed_overlap_required = bool(raw.get("seed_overlap_required", True))

    # Maximum top-level function count in generated answers.
    # Default is 1; CUDA/Numba workflows may use 2 (kernel + wrapper).
    max_toplevel_raw = raw.get("max_top_level_functions")
    max_top_level_functions = int(max_toplevel_raw) if max_toplevel_raw is not None else 1
    if max_top_level_functions not in (1, 2, 3, 4):
        raise ValueError("max_top_level_functions must be 1 or 2 or 3 or 4")

    spec = SynthesisSpec(
        library_name=library_name,
        library_overview=library_overview,
        api_docs_path=str(api_docs_path),
        module_name=module_name,
        module_alias=module_alias,
        required_import=required_import,
        module_call_prefixes=prefixes,
        system_message=system_message,
        judge_system_message=judge_system_message,
        question_instructions=question_instructions,
        answer_instructions=answer_instructions,
        judge_instructions=judge_instructions,
        few_shot_pool=few_shot_pool,
        num_few_shots_q=num_few_shots_q,
        num_few_shots_a=num_few_shots_a,
        min_asserts=min_asserts,
        seed_overlap_required=seed_overlap_required,
        max_top_level_functions=max_top_level_functions,
    ).normalized()

    need = max(spec.num_few_shots_q, spec.num_few_shots_a)
    if len(spec.few_shot_pool) < need:
        raise ValueError(
            f"few_shot_pool has {len(spec.few_shot_pool)} items, but need at least {need} "
            f"(max(num_few_shots_q={spec.num_few_shots_q}, num_few_shots_a={spec.num_few_shots_a}))."
        )
    if spec.min_asserts < 0:
        raise ValueError("min_asserts must be >= 0")
    if not spec.required_import.strip():
        raise ValueError("required_import must be non-empty")
    return spec

# ----------------------------
# API names + API docs
# ----------------------------


def api_full_name(d: ApiDoc, spec: SynthesisSpec) -> str:
    """Return canonical API full name from one API-doc record.

    We intentionally trust `qualname` from API docs as canonical value.
    """
    qn = (d.qualname or d.name or "").strip()
    if not qn:
        raise ValueError("API doc missing qualname/name")
    return qn


def load_api_docs(path: Path) -> List[ApiDoc]:
    docs: List[ApiDoc] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                ApiDoc(
                    source_file=obj.get("source_file", "") or obj.get("path", "") or "",
                    qualname=obj.get("qualname", obj.get("name", "")) or "",
                    name=obj.get("name", "") or "",
                    signature=obj.get("signature", "") or "",
                    docstring=obj.get("docstring", obj.get("description", "")) or "",
                    code=obj.get("code", obj.get("implementation", "")) or "",
                )
            )
    return docs


def seed_payload_from_docs(seed_docs: Sequence[ApiDoc], spec: SynthesisSpec) -> List[SeedApiDoc]:
    out: List[SeedApiDoc] = []
    for d in seed_docs:
        out.append(
            {
                "full_name": api_full_name(d, spec),
                "signature": d.signature,
                "docstring": d.docstring,
            }
        )
    return out


# ----------------------------
# Parsing helpers
# ----------------------------


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json|python)?\s*([\s\S]*?)\s*```\s*$", re.MULTILINE)


def strip_code_fences(s: str) -> str:
    s = s.strip()
    m = _CODE_FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    return s


def parse_json_object(text: str) -> Dict[str, Any]:
        """Parse one JSON object from model output text.

        Strategy: strip fences -> direct json.loads -> balanced-brace fallback.
        """
    raw = strip_code_fences(text).strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        raise ValueError("JSON root is not an object.")
    except Exception:
        pass

    # Fallback: recover first balanced JSON object from mixed text output.
    start = raw.find("{")
    if start < 0:
        raise ValueError("Failed to parse JSON object from model output.")

    in_str = False
    esc = False
    depth = 0
    end = None
    for i, ch in enumerate(raw[start:], start=start):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

    if end is None:
        raise ValueError("Failed to parse JSON object from model output.")

    sub = raw[start : end + 1]
    obj = json.loads(sub)
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object.")
    return obj


def count_asserts(test_code: str) -> int:
    return len(re.findall(r"(?m)^\s*assert\s+", test_code))


def extract_top_level_functions(tree: ast.AST, *, max_allowed: int) -> List[ast.FunctionDef]:
    """Extract top-level function defs and enforce count in [1, max_allowed]."""
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) < 1 or len(funcs) > max_allowed:
        raise ValueError(f"Expected 1..{max_allowed} top-level function(s), found {len(funcs)}")
    return funcs


def extract_api_list_from_code(code: str, *, spec: SynthesisSpec) -> List[str]:
        """Extract and normalize API call sites from generated code.

        It recognizes names under `spec.module_call_prefixes` and rewrites them to
        canonical `spec.module_name.*` for overlap checks.
        """
    tree = ast.parse(code)
    sites: List[Tuple[int, int, str]] = []
    seen: set[str] = set()
    module_prefixes = set(spec.module_call_prefixes)

    def attr_chain(node: ast.AST) -> Optional[List[str]]:
        parts: List[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return list(reversed(parts))
        return None

    def add_site(node: ast.AST, full_name: str) -> None:
        full_name = (full_name or "").strip()
        if not full_name:
            return
        if full_name in seen:
            return
        seen.add(full_name)
        lineno = getattr(node, "lineno", 0) or 0
        col = getattr(node, "col_offset", 0) or 0
        sites.append((lineno, col, full_name))

    # 1) Normal call form: cuda.xxx(...)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            chain = attr_chain(n.func)
            if chain and chain[0] in module_prefixes and len(chain) >= 2:
                # full chained call, e.g. numba.cuda.atomic.add
                add_site(n, spec.module_name + "." + ".".join(chain[1:]))
                # also include base API for overlap checks
                add_site(n, spec.module_name + "." + chain[1])

        # 2) Kernel launch/bracket call: cuda.xxx[grid, block](...)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript):
            base = n.func.value
            if isinstance(base, ast.Attribute):
                chain = attr_chain(base)
                if chain and chain[0] in module_prefixes and len(chain) >= 2:
                    add_site(n, spec.module_name + "." + ".".join(chain[1:]))
                    add_site(n, spec.module_name + "." + chain[1])

    # 3) Attribute-only API usage: cuda.blockDim.x / cuda.threadIdx.x
    #    (not necessarily wrapped in ast.Call)
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            chain = attr_chain(n)
            if chain and chain[0] in module_prefixes and len(chain) >= 2:
                # base API name (numba.cuda.blockDim / threadIdx / shared ...)
                add_site(n, spec.module_name + "." + chain[1])
                # full attribute chain (numba.cuda.blockDim.x)
                add_site(n, spec.module_name + "." + ".".join(chain[1:]))

    sites.sort(key=lambda t: (t[0], t[1]))
    return [name for _, _, name in sites]


# ----------------------------
# Prompt builders
# ----------------------------


def format_seed_api_docs_json(seed_api_docs: Sequence[SeedApiDoc]) -> str:
    return json.dumps(list(seed_api_docs), ensure_ascii=False, indent=2)


def _strip_solution_code(s: str) -> str:
    return strip_code_fences(s).strip() + "\n"


def _split_lines_keep(s: str) -> List[str]:
    return s.splitlines()


def format_few_shots_for_question(shots: Sequence[Dict[str, str]]) -> str:
    blocks: List[str] = []
    for ex in shots:
        blocks.append(json.dumps({"question": ex["task"].strip()}, ensure_ascii=False))
    return "\n".join(blocks)


def format_few_shots_for_answer(shots: Sequence[Dict[str, str]]) -> str:
    """Format few-shot examples as two Python fences: answer then test."""

    blocks: List[str] = []
    for ex in shots:
        sol = _strip_solution_code(ex["solution"]).rstrip()
        tst = _strip_solution_code(ex["test"]).rstrip()
        blocks.append("```python\n" + sol + "\n```\n\n```python\n" + tst + "\n```")
    return "\n\n".join(blocks)


_ALL_CODE_BLOCKS_RE = re.compile(r"```(?:\s*(?:python|py))?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_code_blocks(text: str) -> List[str]:
    """Extract markdown code blocks (```...```) preserving order."""
    return [m.group(1).strip() for m in _ALL_CODE_BLOCKS_RE.finditer(text or "")]


def parse_answer_test_from_model_output(raw_text: str, *, spec: Optional[SynthesisSpec] = None) -> Tuple[str, str]:
        """Parse `(answer, test)` from model output.

        Accepted formats:
            1) JSON object with answer/test fields.
            2) Two Python fenced code blocks (answer first, test second).
        """

    # Try JSON format first
    try:
        obj = parse_json_object(raw_text)
        answer, test = normalize_answer_and_test(obj)
        return answer, test
    except Exception:
        pass

    blocks = extract_code_blocks(raw_text)
    if len(blocks) >= 2:
        norm_blocks = [b.rstrip() + "\n" for b in blocks]

        # With spec available, select pair that passes validation if possible
        if spec is not None:
            # prioritize blocks containing required_import
            cand_answers = [i for i, b in enumerate(norm_blocks) if spec.required_import in b]
            if not cand_answers:
                cand_answers = list(range(len(norm_blocks)))

            for i in cand_answers:
                for j in range(len(norm_blocks)):
                    if i == j:
                        continue
                    ans = norm_blocks[i]
                    tst = norm_blocks[j]
                    try:
                        validate_answer_and_test(ans, tst, spec=spec)
                        return ans, tst
                    except Exception:
                        continue

        # fallback: first two blocks
        return norm_blocks[0], norm_blocks[1]

    # Last fallback: split by a line starting with 'test'
    low = (raw_text or "").lower()
    if "test" in low and "answer" in low:
        lines = (raw_text or "").splitlines()
        split_idx = None
        for idx, ln in enumerate(lines):
            if ln.strip().lower().startswith("test"):
                split_idx = idx
                break
        if split_idx is not None and split_idx > 0:
            answer = "\n".join(lines[:split_idx]).strip() + "\n"
            test = "\n".join(lines[split_idx + 1 :]).strip() + "\n"
            if answer and test:
                return answer, test

    raise ValueError(
        "Failed to parse answer/test. Expected either a JSON object with keys (solution,test) "
        "or two ```python``` code blocks (solution first, then test)."
    )


def safe_format_map(template: str, ctx: Dict[str, Any]) -> str:
        """Safe `str.format_map` for templates containing literal braces.

        It preserves placeholders from `ctx` while escaping unrelated `{}` pairs.
        """

    # protect known placeholders
    protected = template
    for k in ctx.keys():
        protected = protected.replace("{" + k + "}", f"@@@__{k}__@@@")

    # escape all remaining braces
    protected = protected.replace("{", "{{").replace("}", "}}")

    # restore known placeholders
    for k in ctx.keys():
        protected = protected.replace(f"@@@__{k}__@@@", "{" + k + "}")

    return protected.format_map(ctx)


def build_question_prompt(*, seed_api_docs: Sequence[SeedApiDoc], shots: Sequence[Dict[str, str]], spec: SynthesisSpec) -> str:
    ctx = {
        "library_overview": spec.library_overview.strip(),
        "seed_api_docs_json": format_seed_api_docs_json(seed_api_docs),
        "few_shots": format_few_shots_for_question(shots),
        "module_name": spec.module_name,
        "module_alias": spec.module_alias,
        "required_import": spec.required_import,
    }
    return safe_format_map(spec.question_instructions, ctx)


def build_answer_prompt(
    *, question: str, seed_api_docs: Sequence[SeedApiDoc], shots: Sequence[Dict[str, str]], spec: SynthesisSpec
) -> str:
    ctx = {
        "library_overview": spec.library_overview.strip(),
        "question": question.strip(),
        "seed_api_docs_json": format_seed_api_docs_json(seed_api_docs),
        "few_shots": format_few_shots_for_answer(shots),
        "required_import": spec.required_import,
        "min_asserts": spec.min_asserts,
        "module_name": spec.module_name,
        "module_alias": spec.module_alias,
    }
    return safe_format_map(spec.answer_instructions, ctx)


def build_judge_prompt(*, question: str, answer: str, seed_api_docs: Sequence[SeedApiDoc], spec: SynthesisSpec) -> str:
    ctx = {
        "library_overview": spec.library_overview.strip(),
        "question": question.strip(),
        "answer": answer.rstrip(),
        "answer_code": answer.rstrip(),
        "seed_api_docs_json": format_seed_api_docs_json(seed_api_docs),
        "module_name": spec.module_name,
        "module_alias": spec.module_alias,
        "required_import": spec.required_import,
    }
    return safe_format_map(spec.judge_instructions, ctx)


# ----------------------------
# vLLM client
# ----------------------------


def call_vllm_chat(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 120,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)

    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"vLLM request failed: {resp.status_code} {resp.text[:500]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ----------------------------
# Execution helpers
# ----------------------------


def run_python_local(code: str, timeout_s: int = 20, python_bin: str = "") -> Tuple[bool, str]:
    """Run code with local Python (`python -`) and return (ok, combined_output)."""
    py = (python_bin or "").strip() or sys.executable
    try:
        env = os.environ.copy()
        env.setdefault("NUMBA_ENABLE_CUDASIM", "1")
        p = subprocess.run(
            [py, "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        out = (p.stdout or "") + (p.stderr or "")
        return (p.returncode == 0), out
    except FileNotFoundError as e:
        return False, f"[PYTHON NOT FOUND] {e}\n"
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        return False, f"[TIMEOUT after {timeout_s}s]\n{out}"


def run_python_docker(
    code: str,
    *,
    image: str,
    timeout_s: int = 20,
    network_none: bool = True,
    mem_limit: str = "2g",
    cpu_limit: str = "2",
) -> Tuple[bool, str]:
    """Run code in Docker via stdin (`python -`) and return (ok, combined_output)."""
    cmd = ["docker", "run", "--rm", "-i"]
    if network_none:
        cmd += ["--network", "none"]
    if mem_limit:
        cmd += ["--memory", mem_limit]
    if cpu_limit:
        cmd += ["--cpus", cpu_limit]
    cmd += [image, "python", "-"]

    try:
        env = os.environ.copy()
        env.setdefault("NUMBA_ENABLE_CUDASIM", "1")
        p = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        out = (p.stdout or "") + (p.stderr or "")
        return (p.returncode == 0), out
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        return False, f"[TIMEOUT after {timeout_s}s]\n{out}"


# ----------------------------
# Weighted sampling
# ----------------------------


def weighted_sample_without_replacement(
    items: Sequence[Any],
    weights: Sequence[float],
    k: int,
    *,
    rng: random.Random,
) -> List[Any]:
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    idxs = list(range(len(items)))
    chosen: List[Any] = []

    for _ in range(k):
        total = 0.0
        for i in idxs:
            w = weights[i]
            if w > 0:
                total += w
        if total <= 0:
            pick_i = rng.choice(idxs)
        else:
            r = rng.random() * total
            acc = 0.0
            pick_i = idxs[-1]
            for i in idxs:
                w = weights[i]
                if w <= 0:
                    continue
                acc += w
                if acc >= r:
                    pick_i = i
                    break
        chosen.append(items[pick_i])
        idxs.remove(pick_i)
    return chosen


def sample_seed_apis(
    api_docs: Sequence[ApiDoc],
    *,
    k: int,
    counts: Dict[str, int],
    alpha: float,
    rng: random.Random,
    spec: SynthesisSpec,
) -> List[ApiDoc]:
    """Sample seed APIs with weight(api) = 1 / (count + 1) ** alpha."""
    full_names = [api_full_name(d, spec) for d in api_docs]
    weights = [1.0 / ((counts.get(fn, 0) + 1) ** alpha) for fn in full_names]
    seed_docs = weighted_sample_without_replacement(api_docs, weights, k, rng=rng)
    for d in seed_docs:
        fn = api_full_name(d, spec)
        counts[fn] = counts.get(fn, 0) + 1
    return seed_docs


# ----------------------------
# answer/test normalization and validation
# ----------------------------


def normalize_answer_and_test(obj: Dict[str, Any]) -> Tuple[str, str]:
    """Normalize answer/test from JSON fields.

    Supports:
    - answer_lines/test_lines (list[str])
    - answer/test (string; code fences allowed)
    """
    if "answer_lines" in obj and "test_lines" in obj:
        al = obj["answer_lines"]
        tl = obj["test_lines"]
        if not isinstance(al, list) or not all(isinstance(x, str) for x in al):
            raise ValueError("answer_lines must be a list[str]")
        if not isinstance(tl, list) or not all(isinstance(x, str) for x in tl):
            raise ValueError("test_lines must be a list[str]")
        answer = "\n".join(al).rstrip() + "\n"
        test = "\n".join(tl).rstrip() + "\n"
        return answer, test

    if "answer" in obj and "test" in obj:
        answer = _strip_solution_code(str(obj["answer"]))
        test = _strip_solution_code(str(obj["test"]))
        return answer, test

    raise ValueError("Model JSON must contain either (answer_lines,test_lines) or (answer,test)")


def validate_answer_and_test(answer: str, test: str, *, spec: SynthesisSpec, question: Optional[str] = None) -> str:
        """Validate answer/test pair and return target function name.

        Raises ValueError when constraints are violated.
        """
    tree = ast.parse(answer)
    fns = extract_top_level_functions(tree, max_allowed=spec.max_top_level_functions)
    target_fn = fns[-1]
    fn_name = target_fn.name

    if spec.required_import not in answer:
        raise ValueError(f"answer must include required import: `{spec.required_import}`")

    # Accept either normal call `fn(...)` or kernel launch `fn[grid, block](...)`
    called_by_parens = f"{fn_name}(" in test
    called_by_launch = f"{fn_name}[" in test
    if not (called_by_parens or called_by_launch):
        raise ValueError(f"test must call the target function `{fn_name}` at least once")

    # allow assert-free tests for print-oriented questions
    required_asserts = spec.min_asserts
    if question and "print" in (question or "").lower():
        required_asserts = 0

    if required_asserts > 0:
        n_assert = count_asserts(test)
        if n_assert < required_asserts:
            raise ValueError(f"test must contain at least {required_asserts} assert statements, found {n_assert}")

    # optional docstring checks (currently disabled)
    # if not fn.body:
    #     raise ValueError("function body empty")
    # if (
    #     not isinstance(fn.body[0], ast.Expr)
    #     or not isinstance(getattr(fn.body[0], "value", None), ast.Constant)
    #     or not isinstance(fn.body[0].value.value, str)
    # ):
    #     raise ValueError("first statement in function must be a docstring string literal")
    # if "Examples" not in answer and ">>>" not in answer:
    #     raise ValueError("docstring should include an Examples section or doctest prompts")

    # final syntax check on combined code
    ast.parse(answer + "\n\n" + test)
    return fn_name
