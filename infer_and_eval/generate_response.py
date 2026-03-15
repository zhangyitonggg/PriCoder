#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate model responses for benchmark items and write them to JSONL."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import copy
import json
import os
import random
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
DEFAULT_REPO_CONFIG: Dict[str, Any] = {
    # Human readable name for logging / titles
    "repo_name": "ndonnx",
    # The top-level import/package name shown in API full names & prompt text
    "library_name": "ndonnx",
    # If the starter code imports the library as an alias (e.g. `import ndonnx as ndx`),
    # we can mention that alias in the prompt. If not found, we fall back to this.
    "default_import_alias": "ndx",
    "infer_import_alias_from_starter": True,
    # How to read and name API docs records
    "api_docs": {
        # If this field exists and is non-empty in each record, it will be used directly.
        # Otherwise, full_name will be inferred using `full_name_inference_rules`.
        "full_name_field": "full_name",
        "signature_field": "signature",
        "docstring_field": "docstring",
        "max_doc_chars": 4096,
        # If an API-doc record doesn't provide `full_name`, we treat `qualname`
        # as the full name (identity). Your `qualname` is expected to already
        # include any desired prefix (e.g. `ndonnx.`).
        "full_name_inference_rules": [
            {"when": {"always": True}, "template": "{qualname}"},
        ],
    },
    # Benchmark JSONL field mapping (so different benchmarks can reuse this script)
    "benchmark_fields": {
        "task": "task",
        "prompt": "prompt",
        "test": "test",
        "api_list": "api_list",
        "seed_apis": "seed_apis",
    },
    # Titles for injected API blocks (format-able with library_name)
    "api_block_titles": {
        "gold": "Golden {library_name} API docs (from reference api_list/seed_apis):",
    },
    # Prompt behavior
    "prompt": {
        # If you want to fully customize the library-usage instruction, override this template.
        # Available fields: library_name, library_alias
        "library_usage_instruction_template": (
            "5) You MUST use/call functions from the given {library_name} library "
            "(the provided API docs / {library_alias}.*) to implement the solution.\n"
            "   Do NOT reimplement {library_name} primitives yourself.\n\n"
        ),
        # System prompt can be overridden if needed.
        "system_prompt": (
            "You are a coding assistant.\n"
            "You MUST output exactly one Python fenced code block using ```python ... ```.\n"
            "Do NOT include any explanations or text outside the code block.\n"
        ),
    },
}


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst (in place) and return dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def load_repo_config(path: Optional[Path]) -> Dict[str, Any]:
    """Load a repo config JSON and deep-merge it over the default config."""
    cfg = copy.deepcopy(DEFAULT_REPO_CONFIG)
    if path is None:
        return cfg
    if not path.exists():
        raise FileNotFoundError(f"--repo-config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--repo-config must be a JSON object (dict) at top-level.")
    return _deep_update(cfg, data)


# -----------------------------
# Random seed helpers
# -----------------------------
def set_global_seed(seed: int) -> None:
    """Best-effort global seeding for reproducibility."""
    if seed is None or seed < 0:
        return

    # NOTE: PYTHONHASHSEED is only fully effective if set before Python starts,
    # but setting it here can still help downstream tools that read the env var.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)

    # Keep seeding limited to Python-level sources.
    # This runner delegates all model inference to an external vLLM server.


def make_item_seed(base_seed: int, idx: int) -> Optional[int]:
    """Derive a deterministic per-item seed from a base seed and item index."""
    if base_seed is None or base_seed < 0:
        return None
    # keep in 32-bit range for compatibility
    return int((base_seed + idx) % (2**31 - 1))


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class ApiDoc:
    full_name: str
    signature: str
    docstring: str

    def as_text(self) -> str:
        return f"{self.full_name}\n{self.signature}\n{self.docstring}".strip()


# -----------------------------
# JSONL I/O
# -----------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl_line(path_fh, obj: Dict[str, Any]) -> None:
    path_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    path_fh.flush()


# -----------------------------
# API docs loading (repo-configurable)
# -----------------------------
class _SafeFormatMap(dict):
    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return ""


def _rule_matches(rec: Dict[str, Any], when: Dict[str, Any]) -> bool:
    """Return True if a rule condition matches an API-doc record."""
    if not when:
        return False

    # Fallback rule
    if bool(when.get("always", False)):
        return True

    source_file = str(rec.get("source_file", "") or "")
    qualname = str(rec.get("qualname", "") or "").strip()
    name = str(rec.get("name", "") or "").strip()

    # All specified conditions must match
    for k, v in when.items():
        if k == "source_file_endswith":
            if not source_file.endswith(str(v)):
                return False
        elif k == "source_file_contains":
            if str(v) not in source_file:
                return False
        elif k == "qualname_startswith":
            if not qualname.startswith(str(v)):
                return False
        elif k == "qualname_contains":
            if str(v) not in qualname:
                return False
        elif k == "qualname_regex":
            if re.search(str(v), qualname) is None:
                return False
        elif k == "name_regex":
            if re.search(str(v), name) is None:
                return False
        elif k == "always":
            # already handled
            continue
        else:
            raise ValueError(f"Unknown full_name_inference_rules.when key: {k}")
    return True


def infer_full_name(rec: Dict[str, Any], repo_cfg: Dict[str, Any]) -> str:
    """Infer a fully qualified API name from a doc record using repo config."""
    lib_name = str(repo_cfg.get("library_name", "") or "").strip()
    api_cfg = repo_cfg.get("api_docs", {}) or {}
    rules = api_cfg.get("full_name_inference_rules", []) or []
    if not isinstance(rules, list):
        rules = []

    source_file = str(rec.get("source_file", "") or "")
    qualname = str(rec.get("qualname", "") or "").strip()
    name = str(rec.get("name", "") or "").strip()

    fmt_vars = _SafeFormatMap(
        {
            "library_name": lib_name,
            "name": name,
            "qualname": qualname,
            "source_file": source_file,
        }
    )

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        when = rule.get("when", {}) or {}
        template = str(rule.get("template", "") or "").strip()
        if not template:
            continue
        if not isinstance(when, dict):
            continue
        if _rule_matches(rec, when):
            out = template.format_map(fmt_vars).strip()
            if out:
                return out

    # Last resort fallback
    # Identity behavior: full_name == qualname (or name if qualname is missing).
    return qualname or name


def load_api_docs(api_docs_jsonl: Path, *, repo_cfg: Dict[str, Any]) -> Tuple[List[ApiDoc], Dict[str, ApiDoc]]:
    api_cfg = repo_cfg.get("api_docs", {}) or {}
    full_name_field = str(api_cfg.get("full_name_field", "full_name") or "full_name")
    signature_field = str(api_cfg.get("signature_field", "signature") or "signature")
    docstring_field = str(api_cfg.get("docstring_field", "docstring") or "docstring")

    docs: List[ApiDoc] = []
    with api_docs_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            full_name = str(rec.get(full_name_field, "") or "").strip()
            if not full_name:
                full_name = infer_full_name(rec, repo_cfg)

            docs.append(
                ApiDoc(
                    full_name=full_name,
                    signature=str(rec.get(signature_field, "") or "").strip(),
                    docstring=str(rec.get(docstring_field, "") or "").strip(),
                )
            )

    # Deduplicate by full_name (keep first)
    seen = set()
    uniq: List[ApiDoc] = []
    by_name: Dict[str, ApiDoc] = {}
    for d in docs:
        if d.full_name in seen:
            continue
        seen.add(d.full_name)
        uniq.append(d)
        by_name[d.full_name] = d
    return uniq, by_name


# -----------------------------
# Benchmark helpers
# -----------------------------
def extract_target_function_name(prompt_code: str) -> str:
    tree = ast.parse(prompt_code)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        raise ValueError(f"Expected exactly 1 top-level function in prompt, found {len(funcs)}")
    return funcs[0].name


def infer_import_alias_from_starter(starter_code: str, library_name: str) -> Optional[str]:
    """
    Try to infer the import alias used for `library_name` from the starter code.

    Example: `import ndonnx as ndx` -> returns "ndx"
    """
    if not starter_code or not library_name:
        return None

    # Match: import <lib> as <alias>
    pat1 = re.compile(rf"(?m)^\s*import\s+{re.escape(library_name)}\s+as\s+([A-Za-z_]\w*)\s*$")
    m = pat1.search(starter_code)
    if m:
        return m.group(1)

    # Match: import <lib>.<sub> as <alias>
    pat2 = re.compile(rf"(?m)^\s*import\s+{re.escape(library_name)}\.[A-Za-z_]\w*\s+as\s+([A-Za-z_]\w*)\s*$")
    m = pat2.search(starter_code)
    if m:
        return m.group(1)

    return None


# -----------------------------
# API info block formatting
# -----------------------------
def _wrap_comment_lines(s: str, width: int = 92) -> List[str]:
    out: List[str] = []
    for line in s.splitlines():
        if not line.strip():
            out.append("#")
            continue
        chunks = textwrap.wrap(line, width=width - 2) or [""]
        for c in chunks:
            out.append("# " + c)
    return out


def build_api_info_block(
    title: str,
    api_names: Sequence[str],
    by_name: Dict[str, ApiDoc],
    *,
    max_doc_chars: int = 360,
) -> str:
    lines: List[str] = []
    if(len(api_names)!=0):
        lines.append(f"# {title}")
    uniq_names: List[str] = []
    seen = set()
    for n in api_names:
        if n in seen:
            continue
        seen.add(n)
        uniq_names.append(n)

    for i, name in enumerate(uniq_names, start=1):
        d = by_name.get(name)
        if d is None:
            lines.extend(_wrap_comment_lines(f"{i})The api name is:{name}"))
            lines.append("#")
            continue

        header = f"{i})The api signature and name is:{d.full_name} {d.signature}".strip()
        lines.extend(_wrap_comment_lines(header))
        if d.docstring:
            doc = re.sub(r"\s+", " ", d.docstring.strip())
            if len(doc) > max_doc_chars:
                doc = doc[:max_doc_chars].rstrip() + " ..."
            lines.extend(_wrap_comment_lines("The api doc is:" + doc))
        lines.append("#")

    return "\n".join(lines).rstrip() + "\n\n"


# This script intentionally contains no local model inference code.
# Generation is done by calling an external vLLM server over HTTP.

# -----------------------------
# vLLM generation (OpenAI-compatible)
# -----------------------------


def vllm_chat_completions(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    n: int,
    seed: Optional[int],
    api_key: str,
    timeout_s: int,
) -> List[str]:
    """Call a vLLM OpenAI-compatible Chat Completions endpoint and return `n` raw texts."""
    import requests

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n": int(n),
    }
    if seed is not None and seed >= 0:
        payload["seed"] = int(seed)

    headers = {"Content-Type": "application/json"}
    # vLLM often ignores auth, but keep compatibility.
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        print("==== ERROR BODY ====")
        print(resp.text)
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices", []) or []
    outs: List[str] = []
    for ch in choices:
        msg = (ch.get("message") or {}) if isinstance(ch, dict) else {}
        content = msg.get("content")
        if isinstance(content, str):
            outs.append(content)
    # Be robust: if server returned fewer, pad by reusing last (rare) or empty.
    if len(outs) < n:
        outs.extend([outs[-1] if outs else ""] * (n - len(outs)))
    return outs


def generate_k_answers(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
    api_key: str,
    timeout_s: int,
) -> List[str]:
    """Generate k answers via vLLM."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Ask server to produce `k` completions in one request when possible.
    return vllm_chat_completions(
        base_url=base_url,
        model=model,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        n=k,
        seed=seed,
        api_key=api_key,
        timeout_s=timeout_s,
    )

def build_user_prompt(
    task: str,
    starter_prompt: str,
    api_block: str,
    target_function: str,
    *,
    repo_cfg: Dict[str, Any],
    library_alias: str,
    is_humaneval:bool
) -> str:
    """
    Ask for a COMPLETED version of the starter module, in a single ```python fenced block,
    without changing imports / signature / docstring / existing lines.

    Repo-specific parts (library name, alias, library-usage instructions) are controlled
    via `repo_cfg`.
    """
    starter = starter_prompt.rstrip("\n") + "\n"
    lib_name = str(repo_cfg.get("library_name", "") or "").strip()
    prompt_cfg = repo_cfg.get("prompt", {}) or {}

    instr_tpl_raw = str(prompt_cfg.get("library_usage_instruction_template", "") or "")
    if instr_tpl_raw.strip():
        library_usage_instr = instr_tpl_raw.format(library_name=lib_name, library_alias=library_alias)
        # Ensure the instruction block ends with a blank line before the next section.
        library_usage_instr = library_usage_instr.rstrip() + "\n\n"
    else:
        # Generic fallback instruction
        library_usage_instr = (
            "5) You MUST use/call functions from the target library for this benchmark "
            "(see the injected API docs below).\n"
            "   Use the same import/alias as in the starter code and do NOT reimplement its primitives.\n\n"
        )
    if(is_humaneval==False):
        return (
            "You are given an incomplete Python module.\n"
            "You must return a COMPLETED version of that module.\n\n"
            "Hard requirements:\n"
            "1) Your entire response MUST be exactly ONE fenced code block:\n"
            "   ```python\n"
            "   ...\n"
            "   ```\n"
            "   No text outside the code block.\n"
            "2) Inside the code block, you MUST reproduce the Starter code EXACTLY as given\n"
            "   (do NOT change imports, do NOT change any existing lines, do NOT change the function\n"
            "   name/signature/docstring).\n"
            "3) Only add the missing function body for the target function, immediately after its docstring.\n"
            "4) Do NOT add new top-level code (no new imports, no extra functions/classes).\n"
            "   Put all logic inside the target function.\n\n"
            + library_usage_instr
            + f"Target function: {target_function}\n\n"
            f"Task:\n{task}\n\n"
            + (api_block if api_block else "")
            + "Starter code (incomplete) — copy this EXACTLY and only fill in the missing body:\n"
            "```python\n"
            + starter
            + "```\n"
        )
    else:
        return(
            "Please provide a self-contained Python script that solves the following problem in a markdown code block\n"
            + (api_block if api_block else "")
            +starter
            +"Output a Python script with a self-contained function that solves the problem and passes corresponding tests"
        )


def _generate_one_item_record(
    idx: int,
    item: Any,
    args: argparse.Namespace,
    repo_cfg: Dict[str, Any],
    api_by_name: Dict[str, ApiDoc],
    system_prompt: str,
    lib_name: str,
    default_alias: str,
    infer_alias: bool,
    max_doc_chars: int,
    gold_title_tpl: str,
    none_title_tpl: str,
    task_key: str,
    prompt_key: str,
    test_key: str,
    api_list_key: str,
    seed_apis_key: str,
    is_humaneval:bool
) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(item, dict):
        return idx, None, None
    if task_key not in item or prompt_key not in item or test_key not in item:
        return idx, None, None

    task = str(item[task_key])
    prompt_prefix = str(item[prompt_key])
    test_code = str(item[test_key])
    api_list = item.get(api_list_key, []) or []
    seed_apis = item.get(seed_apis_key, []) or []
    retrive_api_list=item.get("retrive_api",[]) or []
    if not isinstance(api_list, list):
        api_list = []
    if not isinstance(seed_apis, list):
        seed_apis = []

    try:
        if("entry_point" in item):fn_name=item["entry_point"]
        else:fn_name = extract_target_function_name(prompt_prefix)
    except Exception as e:
        print(prompt_prefix)
        return idx, None, f"[{idx}] SKIP: cannot parse target function: {e}"

    retrieved_api_names: List[str] = []
    api_block = ""

    if args.mode == "none":
        #api_block = ""
        retrieve_api = [str(x) for x in retrive_api_list]
        retrieved_api_names = retrieve_api
        title = none_title_tpl.format(library_name=lib_name)
        api_block = build_api_info_block(
            title=title,
            api_names=retrieve_api,
            by_name=api_by_name,
            max_doc_chars=max_doc_chars,
        )
    elif args.mode == "gold":
        # Golden API docs from reference api_list (fallback to seed_apis if api_list empty)
        golden = [str(x) for x in (api_list if api_list else seed_apis)]
        retrieved_api_names = golden
        title = gold_title_tpl.format(library_name=lib_name)
        api_block = build_api_info_block(
            title=title,
            api_names=golden,
            by_name=api_by_name,
            max_doc_chars=max_doc_chars,
        )

    # Infer library alias for prompt text (optional)
    lib_alias = ""
    if infer_alias:
        lib_alias = infer_import_alias_from_starter(prompt_prefix, lib_name) or ""
    if not lib_alias:
        lib_alias = default_alias or lib_name or "lib"

    user_prompt = build_user_prompt(
        task,
        prompt_prefix,
        api_block,
        target_function=fn_name,
        repo_cfg=repo_cfg,
        library_alias=lib_alias,
        is_humaneval=is_humaneval
    )
    # vLLM uses Chat Completions; we send system + user messages directly.
    item_seed = make_item_seed(int(args.seed), idx)

    raw_answers = generate_k_answers(
        base_url=args.base_url,
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        k=max(1, args.k),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=item_seed,
        api_key=args.api_key,
        timeout_s=int(args.request_timeout),
    )
    record = {
        "idx": idx,
        "mode": args.mode,
        "task": task,
        "target_function": fn_name,
        "prompt_prefix": prompt_prefix,
        "test": test_code,
        "gold_api_list": [str(x) for x in api_list],
        "seed_apis": [str(x) for x in seed_apis],
        "retrieved_api_names": retrieved_api_names,
        "api_block": api_block,
        "model_user_prompt": user_prompt,
        "gen_params": {
            "base_url": args.base_url,
            "model": args.model,
            "k": args.k,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "item_seed": item_seed,
        },
        "attempts": [{"attempt": j + 1, "raw_answer": raw_answers[j]} for j in range(len(raw_answers))],
    }

    return idx, record, None


def _extract_python_fenced_blocks(text: str) -> List[str]:
    if not text:
        return []

    normalized = re.sub(
        r"```[ \s]*python[ \t]*",
        "```",
        text,
        flags=re.IGNORECASE,
    )

    fences = list(re.finditer(r"```", normalized))

    if not fences:
        return [text]

    candidates: List[str] = []

    if len(fences) >= 2:
        for i in range(len(fences) - 1):
            start = fences[i].end()
            end = fences[i + 1].start()
            content = normalized[start:end].strip("\n")
            if content:
                candidates.append(content)

        if candidates:
            return [max(candidates, key=len)]

    fence = fences[0]
    tail = normalized[fence.end():].strip("\n")
    if tail:
        return [tail]

    return [text]


def _safe_exec_with_timeout(code: str, timeout_s: int) -> Tuple[bool, str]:
    """Execute Python code in a fresh interpreter with a hard timeout.

    We run `python -` and pass the code via stdin to avoid:
    - multiprocessing pickling issues under the 'spawn' start method
    - command line length limits (`-c ...`) when prompts/tests are large

    Returns:
      (ok, info) where ok is True iff exit code == 0. If ok is False, info contains
      combined stdout/stderr (best-effort), or a timeout marker.
    """
    import subprocess
    import sys

    try:
        p = subprocess.run(
            [sys.executable, "-"],
            input=code,
            text=True,
            capture_output=True,
            timeout=int(timeout_s),
        )
        if p.returncode == 0:
            return True, ""
        out = (p.stdout or "") + (p.stderr or "")
        out = out.strip()
        return False, out if out else f"returncode_{p.returncode}"
    except subprocess.TimeoutExpired as e:
        out = ""
        # e.stdout/e.stderr are strings when text=True, but be defensive.
        if getattr(e, "stdout", None):
            out += str(e.stdout)
        if getattr(e, "stderr", None):
            out += str(e.stderr)
        out = out.strip()
        msg = f"timeout_after_{timeout_s}s"
        if out:
            msg += "\n" + out
        return False, msg
    except Exception as ex:
        return False, f"runner_error: {ex}"

def _evaluate_one_record(
    pos: int,
    rec: Dict[str, Any],
    exec_timeout: int,
    k: int,
    safe_import:str
) -> Tuple[int, Dict[str, Any], bool]:
    # `prompt_prefix` is kept in the generations JSONL for debugging, but evaluation
    # runs the model-produced module directly.
    test_code = str(rec.get("test", "") or "")
    attempts = rec.get("attempts", []) or []
    # evaluate up to k attempts
    passed = False
    attempt_details = []
    total_passed=0
    for a_i, a in enumerate(attempts[: max(1, k)]):
        raw = str((a or {}).get("raw_answer", "") or "")
        blocks = _extract_python_fenced_blocks(raw)
        code_block = blocks[0] if blocks else raw  # fallback: treat whole as code
        candidate = safe_import+"\n"+code_block.strip() + "\n\n" + test_code + "\n"
        ok, info = _safe_exec_with_timeout(candidate, int(exec_timeout))
        attempt_details.append({"attempt": a_i + 1, "passed": ok, "info": "" if ok else info})
        if ok:
            total_passed+=1
    result = {
        "idx": rec.get("idx"),
        "mode": rec.get("mode"),
        "passed": passed,
        "k_total_evluate_attempt": int(k),
        "total_passed":total_passed,
        "attempts_evaluated": attempt_details,
    }
    return pos, result, total_passed
import math
def comb(n, k):
    if n < 0 or k < 0 or n < k:
        return 0
    return math.comb(n, k)

def pass_at_k(total_passed, evaluated_attempt, k):
    if evaluated_attempt < k:
        return -1
    if total_passed >= evaluated_attempt:
        return 1.0
    return 1 - comb(evaluated_attempt - total_passed, k) / comb(evaluated_attempt, k)

def evaluate_generations_jsonl(
    generations_path: str,
    *,
    exec_timeout: int,
    k: int,
    safe_import:str,
    parallel: int = 100,
) -> Dict[str, Any]:
    """
    Evaluate a generations JSONL (as produced by this script).
    Metric: pass@k (for fixed k samples per item): success if ANY of first k attempts passes.
    """
    items: List[Dict[str, Any]] = []
    with open(generations_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    results_by_pos: List[Optional[Dict[str, Any]]] = [None] * len(items)
    max_workers = max(1, int(parallel))
    pass_1=0.0
    pass_3=0.0
    pass_5=0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_evaluate_one_record, pos, rec, int(exec_timeout), int(k),safe_import)
            for pos, rec in enumerate(items)
        ]
        for fut in concurrent.futures.as_completed(futures):
            pos, result, total_passed = fut.result()
            results_by_pos[pos] = result
            pass_1+=pass_at_k(total_passed,k,1)
            pass_3+=pass_at_k(total_passed,k,3)
            pass_5+=pass_at_k(total_passed,k,5)
    results: List[Dict[str, Any]] = [r for r in results_by_pos if r is not None]

    total = len(items)
    return {"total": total, "pass@1": pass_1/total,"pass@3": pass_3/total,"pass@5": pass_5/total, "k_total_evluate_attempt": int(k), "results": results}

# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=str, required=True, help="Benchmark JSONL.")
    ap.add_argument("--api-docs", type=str, required=True, help="API docs JSONL (e.g., in_doc.jsonl).")
    ap.add_argument("--out", type=str, required=True, help="Output generations JSONL.")
    ap.add_argument("--max-items", type=int, default=0, help="If >0, generate only first N items.")

    ap.add_argument("--mode", type=str, default="gold", choices=["none", "gold"], help="Prompt variant.")

    # pass@k generation
    ap.add_argument("--k", type=int, default=1, help="Generate k answers per item.")
    ap.add_argument("--max-new-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gen-parallel", type=int, default=200, help="Parallel workers for generation (HTTP requests).")

    # reproducibility
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible generation. Use a negative value to disable seeding.",
    )

    # model serving (vLLM OpenAI-compatible server)
    ap.add_argument("--base-url", type=str, required=True, help="vLLM base URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("--model", type=str, required=True, help="Served model name, e.g. Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--api-key", type=str, default="", help="Optional API key (vLLM usually ignores).")
    ap.add_argument("--request-timeout", type=int, default=3600, help="HTTP request timeout seconds for each generation call.")

    # evaluation (merged: generate + evaluate)
    ap.add_argument("--run-eval", action="store_true", help="After generation, execute tests and write result JSON.")
    ap.add_argument("--out-json", type=str, default="", help="Evaluation JSON output path (default: <out>_answer.json).")
    ap.add_argument("--exec-timeout", type=int, default=10, help="Timeout seconds for executing candidate code.")
    ap.add_argument("--eval-parallel", type=int, default=10, help="Parallel workers for evaluation (executions).")
    ap.add_argument("--humaneval", action="store_true", help="humaneval_evaluation")

    # repo-specific config (external JSON)
    ap.add_argument(
        "--repo-config",
        type=str,
        default="",
        help="Path to repo config JSON that defines library-specific behavior (optional).",
    )
    ap.add_argument(
        "--dump-effective-repo-config",
        type=str,
        default="",
        help="Write the effective repo config JSON to this path and exit.",
    )
    # quick overrides (optional)
    ap.add_argument("--library-name", type=str, default="", help="Override repo config: library_name.")
    ap.add_argument("--default-import-alias", type=str, default="", help="Override repo config: default_import_alias.")

    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    # Load repo config (and apply overrides)
    repo_cfg = load_repo_config(Path(args.repo_config) if args.repo_config else None)
    if args.library_name:
        repo_cfg["library_name"] = args.library_name
        repo_cfg.setdefault("repo_name", args.library_name)
    if args.default_import_alias:
        repo_cfg["default_import_alias"] = args.default_import_alias
    if("ndonnx" in repo_cfg["library_name"] or "ndx" in repo_cfg["library_name"]):
        SAFE_IMPORTS = """
from typing import *
import math
import itertools
import functools
import collections
import heapq
import bisect
import re
import sys
import random
import ndonnx as ndx
            """
    elif("numba" in repo_cfg["library_name"] or "numba" in repo_cfg["repo_name"]):
        SAFE_IMPORTS = """
from typing import *
import math
import itertools
import functools
import collections
import heapq
import bisect
import re
import sys
import random
import numba.cuda as cuda
            """
    else:
        raise AssertionError("unrecognized library")
    if args.dump_effective_repo_config:
        out_cfg_path = Path(args.dump_effective_repo_config)
        out_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        out_cfg_path.write_text(json.dumps(repo_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[RepoConfig] wrote effective config to: {out_cfg_path}", file=sys.stderr)
        return 0

    # Global seeding (best-effort).
    if args.seed is not None and args.seed >= 0:
        set_global_seed(int(args.seed))
        print(f"[Seed] base_seed={args.seed} (item_seed = base_seed + idx)", file=sys.stderr)
    else:
        print("[Seed] disabled (args.seed < 0)", file=sys.stderr)

    bench = read_jsonl(Path(args.benchmark))
    if args.max_items > 0:
        bench = bench[: args.max_items]

    _, api_by_name = load_api_docs(Path(args.api_docs), repo_cfg=repo_cfg)

    # vLLM server is started elsewhere. We only call it via --base-url/--model.
    system_prompt = str((repo_cfg.get("prompt", {}) or {}).get("system_prompt", "") or "").strip()
    if not system_prompt:
        system_prompt = DEFAULT_REPO_CONFIG["prompt"]["system_prompt"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Benchmark field mapping
    bf = repo_cfg.get("benchmark_fields", {}) or {}
    task_key = str(bf.get("task", "task"))
    prompt_key = str(bf.get("prompt", "prompt"))
    test_key = str(bf.get("test", "test"))
    api_list_key = str(bf.get("api_list", "api_list"))
    seed_apis_key = str(bf.get("seed_apis", "seed_apis"))

    lib_name = str(repo_cfg.get("library_name", "") or "").strip()
    default_alias = str(repo_cfg.get("default_import_alias", "") or "").strip()
    infer_alias = bool(repo_cfg.get("infer_import_alias_from_starter", True))
    api_cfg = repo_cfg.get("api_docs", {}) or {}
    max_doc_chars = int(api_cfg.get("max_doc_chars", 360) or 360)
    titles = repo_cfg.get("api_block_titles", {}) or {}
    gold_title_tpl = str(titles.get("gold", "Golden API docs (from reference api_list/seed_apis):"))
    none_title_tpl = str(titles.get("none", "Retrieve API docs (from reference api_list/seed_apis):"))

    with out_path.open("w", encoding="utf-8") as fout:
        max_workers = max(1, int(args.gen_parallel))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures: Dict[int, concurrent.futures.Future] = {}
            for i, item in enumerate(bench, start=1):
                futures[i] = ex.submit(
                    _generate_one_item_record,
                    i,
                    item,
                    args,
                    repo_cfg,
                    api_by_name,
                    system_prompt,
                    lib_name,
                    default_alias,
                    infer_alias,
                    max_doc_chars,
                    gold_title_tpl,
                    none_title_tpl,
                    task_key,
                    prompt_key,
                    test_key,
                    api_list_key,
                    seed_apis_key,
                    args.humaneval
                )

            for i in range(1, len(bench) + 1):
                _, record, skip_msg = futures[i].result()
                if skip_msg:
                    print(skip_msg, file=sys.stderr)
                    continue
                if record is None:
                    continue

                write_jsonl_line(fout, record)
                print(f"[{i}/{len(bench)}] wrote generations (k={args.k}) mode={args.mode}", file=sys.stderr)

                if args.sleep:
                    time.sleep(args.sleep)

    if args.run_eval:
        out_json = args.out_json.strip()
        if not out_json:
            out_json = str(out_path.with_suffix("").as_posix()) + "_answer.json"
        res = evaluate_generations_jsonl(
            str(out_path),
            exec_timeout=int(args.exec_timeout),
            k=int(args.k) if int(args.k) > 0 else 1,
            safe_import=SAFE_IMPORTS,
            parallel=int(args.eval_parallel),
        )
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Eval] wrote: {out_json} | pass@1: {res.get('pass@1')}", file=sys.stderr)

    print(f"Done. Wrote generations to: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
