#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate library-oriented benchmark items from a repo spec.

`generate_library_benchmark.py` supports loading a repo-specific JSON config
and producing benchmark JSONL records with optional execution verification.

A configurable benchmark generator for *library-oriented code generation*.

Compared with earlier versions, this script can be pointed at different libraries / repos
by passing a repo spec (JSON) that contains:
- library overview text
- few-shot pool
- optional prompt template / system message
- import/module naming conventions used for validation and API-call extraction

It also optionally executes each generated sample in a fresh Python interpreter before
writing it to JSONL:

    executable = prompt + canonical_solution + "\n\n" + test

If execution fails, the error is fed back to the model and a new sample is generated.

Usage example
-------------
export AIHUBMIX_API_KEY="sk-..."
python generate_library_benchmark.py \
  --spec /path/to/repo_spec.json \
  --out /path/to/benchmark.jsonl \
  --num 50 --k 5 --model gpt-4o-mini --json-mode

Notes
-----
- Executing generated code has risks. Prefer running in a container/sandbox.
- The runtime must have the target library installed (e.g., ndonnx) for exec verification.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


# ----------------------------
# Repo spec (config) structures
# ----------------------------
DEFAULT_GENERATION_INSTRUCTIONS = """\
You are generating ONE benchmark instance for *library-oriented code generation*.

## Library overview
{library_overview}

## Your job
You will be given a list of SEED APIs (name, signature, docstring). Create a realistic programming task
and a reference implementation that uses the target library. The task should be solvable, runnable, and testable.

Return ONLY a single JSON object (no markdown, no commentary).

## Output JSON schema (ONLY these keys)
{{
  "task": "string",
  "solution": "string",
  "test": "string"
}}

### Execution rule (must satisfy)
Let:
- executable = solution + "\\n\\n" + test

The `executable` must run in a fresh Python interpreter and all asserts in `test`
must pass (no AssertionError).

### Requirements
- task:
  - 1–3 sentences, natural user request.
  - Do NOT mention specific API names or module paths.
- solution:
  - Must be valid Python code.
  - Must include: `{required_import}`
  - Must define EXACTLY ONE top-level function, and put all logic inside that function.
  - The function must have a docstring describing the task and MUST include an "Examples" section
    with 1–2 minimal examples (doctest-style `>>>` is OK). Do NOT mention specific API names.
- test:
  - MUST contain at least {min_asserts} assert-based tests (counting `assert` statements).
  - Prefer 5–8 asserts, all deterministic.
  - Should focus on observable behavior: values, shapes, edge-cases.
  - Appended and executed directly after the program.
  - Should call the function multiple times with small deterministic inputs.
  - Avoid asserts about unrelated internals (e.g., random properties, environment assumptions).

### Seed API usage
- Try to use some of the provided SEED APIs where it makes sense, but do not force unnatural usage.
- Do not invent APIs that do not exist.

## Few-shot examples (format + quality)
Below are {num_few_shots} examples of *valid* outputs (not necessarily using the same seed APIs):
{few_shots}

## SEED APIs (JSON list)
{seed_api_docs_json}
"""


@dataclass(frozen=True)
class RepoSpec:
    """    Configuration for a target library/repo.

    Most repo-specific content that used to be hardcoded in the script should live here
    and be passed in via --spec.
    """

    # Human-facing library overview inserted into the prompt
    library_overview: str

    # Few-shot pool: list of {"task": ..., "solution": ..., "test": ...}
    few_shot_pool: List[Dict[str, str]]

    # Path to the API-docs jsonl (the old --api-docs argument)
    api_docs: str

    # Import conventions in generated solutions (used for validation and prompt instructions)
    module_name: str = "ndonnx"  # canonical module path used in "full_name" strings
    module_alias: str = "ndx"  # expected alias used by the model in solutions
    required_import: str = "import ndonnx as ndx"  # validation requires this exact substring

    # Names in code that should be treated as "module prefixes" when extracting API calls
    # e.g. ["ndx", "ndonnx"]
    module_call_prefixes: Tuple[str, ...] = ("ndx", "ndonnx")

    # Prompt texts
    system_message: str = "You are a careful benchmark author. Follow instructions exactly."
    generation_instructions: str = DEFAULT_GENERATION_INSTRUCTIONS

    # Sampling / validation knobs
    num_few_shots: int = 2
    min_asserts: int = 4
    seed_overlap_required: bool = True

    def normalized(self) -> "RepoSpec":
        """Return a copy with a few derived defaults filled in."""
        prefixes = list(self.module_call_prefixes)

        # Ensure alias is always included
        if self.module_alias and self.module_alias not in prefixes:
            prefixes.insert(0, self.module_alias)
        # If module_name is a valid identifier, include it too
        if self.module_name.isidentifier() and self.module_name not in prefixes:
            prefixes.append(self.module_name)

        return RepoSpec(
            library_overview=self.library_overview,
            few_shot_pool=self.few_shot_pool,
            api_docs=self.api_docs,
            module_name=self.module_name,
            module_alias=self.module_alias,
            required_import=self.required_import,
            module_call_prefixes=tuple(prefixes),
            system_message=self.system_message,
            generation_instructions=self.generation_instructions,
            num_few_shots=self.num_few_shots,
            min_asserts=self.min_asserts,
            seed_overlap_required=self.seed_overlap_required,
        )



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


def load_repo_spec(spec_path: Path) -> RepoSpec:
    """
    Load RepoSpec from a JSON file.

    To make large multiline content manageable, the spec supports indirection via file paths:
      - library_overview_file: path to a .txt file
      - few_shot_pool_file: path to a .json or .jsonl file
      - generation_instructions_file: path to a .txt file
      - api_docs / api_docs_file: path to a .jsonl file containing API docs

    Relative paths are resolved relative to the spec file directory.
    """
    base_dir = spec_path.parent
    raw = _load_json_file(spec_path)
    if not isinstance(raw, dict):
        raise ValueError("--spec must point to a JSON object.")

    # api docs
    api_docs_raw = raw.get("api_docs", None) or raw.get("api_docs_file", None)
    if not api_docs_raw or not str(api_docs_raw).strip():
        raise ValueError("Spec must provide 'api_docs' (path to API docs .jsonl).")
    api_docs_path = _resolve_path(str(api_docs_raw), base_dir=base_dir)

    # library overview
    if "library_overview_file" in raw:
        overview_path = _resolve_path(str(raw["library_overview_file"]), base_dir=base_dir)
        library_overview = _load_text_file(overview_path)
    else:
        library_overview = str(raw.get("library_overview", "") or "")
    if not library_overview.strip():
        raise ValueError("Spec must provide either 'library_overview' or 'library_overview_file'.")

    # few shots
    if "few_shot_pool_file" in raw:
        pool_path = _resolve_path(str(raw["few_shot_pool_file"]), base_dir=base_dir)
        if pool_path.suffix.lower() == ".jsonl":
            pool_any = _load_jsonl(pool_path)
        else:
            pool_any = _load_json_file(pool_path)
        few_shot_pool = _validate_few_shot_pool(pool_any)
    else:
        few_shot_pool = _validate_few_shot_pool(raw.get("few_shot_pool", []))

    # prompt template
    if "generation_instructions_file" in raw:
        instr_path = _resolve_path(str(raw["generation_instructions_file"]), base_dir=base_dir)
        generation_instructions = _load_text_file(instr_path)
    else:
        generation_instructions = str(raw.get("generation_instructions", "") or "").strip() or DEFAULT_GENERATION_INSTRUCTIONS

    # misc fields
    module_name = str(raw.get("module_name", "ndonnx"))
    module_alias = str(raw.get("module_alias", "ndx"))
    required_import = str(raw.get("required_import", f"import {module_name} as {module_alias}"))

    prefixes_raw = raw.get("module_call_prefixes", None)
    if prefixes_raw is None:
        prefixes: Tuple[str, ...] = (module_alias, module_name) if module_name.isidentifier() else (module_alias,)
    else:
        if not isinstance(prefixes_raw, list) or not all(isinstance(x, str) and x for x in prefixes_raw):
            raise ValueError("'module_call_prefixes' must be a list of non-empty strings.")
        prefixes = tuple(prefixes_raw)

    system_message = str(raw.get("system_message", "You are a careful benchmark author. Follow instructions exactly."))
    num_few_shots = int(raw.get("num_few_shots", 2))
    min_asserts = int(raw.get("min_asserts", 4))
    seed_overlap_required = bool(raw.get("seed_overlap_required", True))

    spec = RepoSpec(
        library_overview=library_overview,
        few_shot_pool=few_shot_pool,
        api_docs=str(api_docs_path),
        module_name=module_name,
        module_alias=module_alias,
        required_import=required_import,
        module_call_prefixes=prefixes,
        system_message=system_message,
        generation_instructions=generation_instructions,
        num_few_shots=num_few_shots,
        min_asserts=min_asserts,
        seed_overlap_required=seed_overlap_required,
    )
    spec = spec.normalized()

    if spec.num_few_shots <= 0:
        raise ValueError("num_few_shots must be >= 1.")
    if len(spec.few_shot_pool) < spec.num_few_shots:
        raise ValueError(f"few_shot_pool has {len(spec.few_shot_pool)} items, but num_few_shots={spec.num_few_shots}.")
    if spec.min_asserts <= 0:
        raise ValueError("min_asserts must be >= 1.")
    if not spec.required_import.strip():
        raise ValueError("required_import must be a non-empty string.")
    if not spec.module_name.strip():
        raise ValueError("module_name must be a non-empty string.")
    if not spec.module_alias.strip():
        raise ValueError("module_alias must be a non-empty string.")
    if not spec.api_docs.strip():
        raise ValueError("api_docs must be a non-empty string.")

    return spec


# ----------------------------
# Helper: API doc record
# ----------------------------
@dataclass(frozen=True)
class ApiDoc:
    source_file: str
    qualname: str
    name: str
    signature: str
    docstring: str
    code: str


def api_full_name(d: ApiDoc, spec: RepoSpec) -> str:
    """    Canonicalize a doc record to a fully-qualified API name string.

    This function intentionally contains **no special-case logic** for any submodule
    like ``extensions``. If an API belongs to a submodule, its module path should be
    present in the doc record's ``qualname`` (e.g. ``extensions.make_nullable``).

    Rules:
    - If ``qualname`` is already fully-qualified (starts with ``{module_name}.``), keep it.
    - If ``qualname`` refers to an Array method (e.g. ``Array.mean``), prefix with ``module_name``.
    - If ``qualname`` contains a module path (e.g. ``extensions.foo``), prefix with ``module_name``.
    - Otherwise treat it as a top-level symbol and prefix with ``module_name``.
    """
    q = (d.qualname or "").strip() or (d.name or "").strip()
    if not q:
        return spec.module_name

    # Already fully-qualified
    if q.startswith(spec.module_name + "."):
        return q
    # Any other dotted path is treated as a normal submodule path.
    if "." in q:
        return f"{spec.module_name}.{q}"

    return f"{spec.module_name}.{q}"


# ----------------------------
# Parsing utilities
# ----------------------------
_CODE_FENCE_RE = re.compile(r"^\s*```(?:python)?\s*([\s\S]*?)\s*```\s*$", re.MULTILINE)


def strip_code_fences(s: str) -> str:
    m = _CODE_FENCE_RE.search(s.strip())
    if m:
        return m.group(1).strip() + "\n"
    return s


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = _CODE_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def count_asserts(test_code: str) -> int:
    return len(re.findall(r"(?m)^\s*assert\s+", test_code))


def extract_top_level_function(tree: ast.AST) -> ast.FunctionDef:
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        raise ValueError(f"Expected exactly 1 top-level function, found {len(funcs)}")
    return funcs[0]


def split_prompt_and_completion(solution_code: str) -> Tuple[str, str]:
    tree = ast.parse(solution_code)
    fn = extract_top_level_function(tree)

    if not fn.body:
        raise ValueError("Function body is empty; must include a docstring and implementation.")
    if (
        not isinstance(fn.body[0], ast.Expr)
        or not isinstance(getattr(fn.body[0], "value", None), ast.Constant)
        or not isinstance(fn.body[0].value.value, str)
    ):
        raise ValueError("First statement in the function must be a docstring string literal.")

    doc_expr: ast.Expr = fn.body[0]
    if doc_expr.end_lineno is None:
        raise ValueError("AST end_lineno missing; need Python 3.8+ with end positions.")

    lines = solution_code.splitlines(keepends=True)
    doc_end_line_idx = doc_expr.end_lineno  # 1-based
    prompt = "".join(lines[:doc_end_line_idx])
    if not prompt.endswith("\n"):
        prompt += "\n"
    completion = "".join(lines[doc_end_line_idx:])
    return prompt, completion


def extract_api_list_from_code(code: str, *, spec: RepoSpec) -> List[str]:
    tree = ast.parse(code)
    call_sites: List[Tuple[int, int, str]] = []
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

    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            lineno = getattr(n, "lineno", 0) or 0
            col = getattr(n, "col_offset", 0) or 0

            if isinstance(n.func, ast.Attribute):
                chain = attr_chain(n.func)
                if chain and chain[0] in module_prefixes:
                    # Canonicalize under spec.module_name even if code uses alias.
                    full = spec.module_name + "." + ".".join(chain[1:])
                    call_sites.append((lineno, col, full))

    call_sites.sort(key=lambda t: (t[0], t[1]))
    return [name for _, _, name in call_sites]


# ----------------------------
# Prompt builder
# ----------------------------
def format_seed_api_docs(seed_docs: Sequence[ApiDoc], spec: RepoSpec) -> str:
    payload = []
    for d in seed_docs:
        payload.append(
            {
                "full_name": api_full_name(d, spec),
                "signature": d.signature,
                "docstring": d.docstring,
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_few_shots(shots: Sequence[Dict[str, str]]) -> str:
    blocks = []
    for ex in shots:
        blocks.append(json.dumps(ex, ensure_ascii=False, indent=2))
    return "\n\n".join(blocks)


def build_generation_prompt(seed_docs: Sequence[ApiDoc], shots: Sequence[Dict[str, str]], spec: RepoSpec) -> str:
    seed_api_docs_json = format_seed_api_docs(seed_docs, spec)
    few_shots_json = format_few_shots(shots)

    ctx = {
        "library_overview": spec.library_overview.strip(),
        "few_shots": few_shots_json,
        "seed_api_docs_json": seed_api_docs_json,
        "required_import": spec.required_import,
        "module_name": spec.module_name,
        "module_alias": spec.module_alias,
        "min_asserts": spec.min_asserts,
        "num_few_shots": spec.num_few_shots,
    }

    try:
        return spec.generation_instructions.format_map(ctx)
    except KeyError as e:
        raise ValueError(f"generation_instructions template missing placeholder: {e}")


# ----------------------------
# LLM API client
# ----------------------------
def call_aihubmix_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    json_mode: bool = True,
    temperature: float = 0.7,
    timeout: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ----------------------------
# Runtime execution check
# ----------------------------
def run_python(code: str, timeout_s: int = 20) -> Tuple[bool, str]:
    """
    Run code in a fresh python process. Return (ok, combined_output).
    """
    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        out = (p.stdout or "") + (p.stderr or "")
        return (p.returncode == 0), out
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        return False, f"[TIMEOUT after {timeout_s}s]\n{out}"


# ----------------------------
# Main generation loop
# ----------------------------
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
                    source_file=obj.get("source_file", "") or "",
                    qualname=obj.get("qualname", obj.get("name", "")) or "",
                    name=obj.get("name", "") or "",
                    signature=obj.get("signature", "") or "",
                    docstring=obj.get("docstring", "") or "",
                    code=obj.get("code", "") or "",
                )
            )
    return docs


def validate_model_item(item: Dict[str, Any], spec: RepoSpec) -> Tuple[str, str, str]:
    if not isinstance(item, dict):
        raise ValueError("Model output is not a JSON object.")
    for k in ("task", "solution", "test"):
        if k not in item:
            raise ValueError(f"Missing key: {k}")
        if not isinstance(item[k], str):
            raise ValueError(f"Field {k} must be a string.")

    task = item["task"].strip()
    solution = strip_code_fences(item["solution"])
    test = strip_code_fences(item["test"])

    tree = ast.parse(solution)
    fn = extract_top_level_function(tree)
    fn_name = fn.name

    if f"{fn_name}(" not in test:
        raise ValueError(f"Test must call the target function `{fn_name}` at least once.")
    if not task:
        raise ValueError("Empty task.")
    if spec.required_import not in solution:
        raise ValueError(f"Solution must include required import: `{spec.required_import}`.")

    n_assert = count_asserts(test)
    if n_assert < spec.min_asserts:
        raise ValueError(f"Test must contain at least {spec.min_asserts} assert statements, found {n_assert}.")

    if "Examples" not in solution and ">>>" not in solution:
        raise ValueError("Solution docstring should include an Examples section (or doctest prompts).")

    # Syntax check
    ast.parse(solution)
    ast.parse(solution + "\n\n" + test)
    return task, solution, test


def build_final_benchmark_item(
    *,
    task: str,
    solution: str,
    test: str,
    seed_docs: Sequence[ApiDoc],
    spec: RepoSpec,
) -> Dict[str, Any]:
    prompt, completion = split_prompt_and_completion(solution)
    api_list = extract_api_list_from_code(solution, spec=spec)

    return {
        "task": task,
        "prompt": prompt,
        "canonical_solution": completion,
        "test": test,
        "api_list": api_list,
        "seed_apis": [api_full_name(d, spec) for d in seed_docs],
        "repo_spec": {
            "module_name": spec.module_name,
            "module_alias": spec.module_alias,
            "required_import": spec.required_import,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=str, required=True, help="Path to repo spec JSON (overview, few-shots, templates).")
    ap.add_argument("--out", type=str, required=True, help="Output benchmark jsonl path.")
    ap.add_argument("--num", type=int, default=50, help="Number of instances to generate.")
    ap.add_argument("--k", type=int, default=5, help="Number of seed APIs per instance.")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="Chat model name.")
    ap.add_argument("--base-url", type=str, default="https://aihubmix.com", help="API base URL.")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    ap.add_argument("--max-tries", type=int, default=20, help="Max retries per instance.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--json-mode", action="store_true", help="Use JSON mode (response_format=json_object).")
    ap.add_argument("--exec-timeout", type=int, default=3, help="Timeout seconds for running prompt+completion+test.")
    ap.add_argument("--no-exec-verify", action="store_true", help="Disable executing tests during generation.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between tries (optional).")

    args = ap.parse_args()

    random.seed(args.seed)

    api_key = os.environ.get("AIHUBMIX_API_KEY", "").strip()
    if not api_key:
        print("ERROR: Missing AIHUBMIX_API_KEY env var.", file=sys.stderr)
        return 2

    spec = load_repo_spec(Path(args.spec))

    api_docs = load_api_docs(Path(spec.api_docs))
    if len(api_docs) < args.k:
        print(f"ERROR: Not enough APIs in {spec.api_docs} for k={args.k}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for idx in range(args.num):
            seed_docs = random.sample(api_docs, args.k)
            shots = random.sample(spec.few_shot_pool, k=spec.num_few_shots)

            prompt_text = build_generation_prompt(seed_docs, shots, spec)

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": spec.system_message},
                {"role": "user", "content": prompt_text},
            ]

            success = False
            last_err: Optional[str] = None
            # print(messages)
            for attempt in range(1, args.max_tries + 1):
                try:
                    raw = call_aihubmix_chat(
                        base_url=args.base_url,
                        api_key=api_key,
                        model=args.model,
                        messages=messages,
                        json_mode=args.json_mode,
                        temperature=args.temperature,
                    )
                    model_obj = parse_json_object(raw)
                    task, solution, test = validate_model_item(model_obj, spec)

                    item = build_final_benchmark_item(
                        task=task,
                        solution=solution,
                        test=test,
                        seed_docs=seed_docs,
                        spec=spec,
                    )

                    # Optional seed overlap check (keeps older behavior configurable)
                    if spec.seed_overlap_required:
                        used = set(item["api_list"])
                        seed = set(item["seed_apis"])
                        # print(used,seed)
                        if used and seed and len(used.intersection(seed)) == 0:
                            raise ValueError("No overlap between used APIs and provided seed_apis; regenerate.")

                    # Runtime verify: prompt + canonical_solution + test must PASS
                    if not args.no_exec_verify:
                        executable = item["prompt"] + item["canonical_solution"] + "\n\n" + item["test"]
                        ok, out = run_python(executable, timeout_s=args.exec_timeout)
                        if not ok:
                            tail = out[-2000:] if len(out) > 2000 else out
                            raise ValueError(
                                "Runtime check failed for prompt+canonical_solution+test. Output tail:\n" + tail
                            )

                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    written += 1
                    success = True
                    print(f"[{idx+1}/{args.num}] ok (attempt {attempt})", file=sys.stderr)
                    break

                except Exception as e:
                    last_err = str(e)
                    # print(last_err)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous output failed validation.\n"
                                f"Error:\n{last_err}\n\n"
                                "Please regenerate a NEW JSON object that satisfies all requirements, "
                                "including that running `solution + \"\\n\\n\" + test` passes all asserts "
                                f"and the test contains at least {spec.min_asserts} assert statements."
                            ),
                        }
                    )
                    if args.sleep:
                        time.sleep(args.sleep)

            if not success:
                print(
                    f"[{idx+1}/{args.num}] FAILED after {args.max_tries} attempts. Last error: {last_err}",
                    file=sys.stderr,
                )

    print(f"Done. Wrote {written} instances to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
