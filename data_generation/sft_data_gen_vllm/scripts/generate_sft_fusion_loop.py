#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate PriCoder Stage-2 SFT data with iterative fusion (D2..Dmax).

`generate_sft_fusion_loop.py` builds higher-depth datasets by repeatedly
fusing examples from earlier depths and validating generated answer/test pairs.

Synthesis (Fusion / Iterative)
----------------------------

This script implements the "second type" Stage-2 training data generation.

Conceptually:

  - D1 is NOT generated here. It is provided as a jsonl file (already generated).
  - For d = 2..max_d:
      - Build a seed pool = union(D1..D{d-1}).
      - Repeatedly sample two examples from the seed pool.
      - Ask the model to *fuse* them into a NEW question.
      - Ask the model to generate (answer, test) for that question.
      - Validate + (optional) execute + (optional) judge.
      - Keep writing accepted samples until D{d} has `num_per_d` samples.

Key requirement (user):
  When generating the new question AND the new code/test, the model must be able
  to see BOTH source examples' question, code and test.

We keep hyperparameters and parallelism strategy consistent with
`generate_sft_loop.py`:
  - question-level workers (--question-workers)
  - candidate-level workers (--candidates-per-question)
  - same retry logic & vLLM parameters
  - same validation / optional execution / optional judge
  - same multi-base-url slot assignment

Outputs:
  - Accepted samples:  <out_root>_D{d}.jsonl
  - Fail logs (optional): <fail_root>_D{d}.jsonl

Each output line keeps the PriCoder SFT schema
  (question/answer/test/messages/seed_apis/used_apis/judge/meta)
and adds fusion provenance into meta.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from synthesis_common import (
    ApiDoc,
    api_full_name,
    build_answer_prompt,
    build_judge_prompt,
    build_question_prompt,
    call_vllm_chat,
    extract_api_list_from_code,
    load_api_docs,
    load_synthesis_spec,
    parse_answer_test_from_model_output,
    parse_json_object,
    run_python_docker,
    run_python_local,
    sanitize_messages,
    strip_code_fences,
    validate_answer_and_test,
    weighted_sample_without_replacement,
)


# ----------------------------
# Helpers kept consistent with generate_sft_loop.py
# ----------------------------

def _sleep_with_jitter(max_s: float) -> None:
    if max_s <= 0:
        return
    time.sleep(random.uniform(0.0, max_s))

def _normalize_question(q: str) -> str:
    return " ".join((q or "").split())


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    recs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                # Allow a truncated last line due to crash.
                continue
    return recs


def _write_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


def _parse_base_urls(base_url_arg: Any) -> List[str]:
    """Parse --base-url exactly like generate_sft_loop.py."""

    def _add_from_string(s: str, out: List[str]) -> None:
        s = (s or "").strip()
        if not s:
            return

        # JSON list
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    for it in obj:
                        if isinstance(it, str):
                            _add_from_string(it, out)
                        else:
                            _add_from_string(str(it), out)
                    return
            except Exception:
                pass

        # comma separated
        if "," in s:
            for p in [p.strip() for p in s.split(",")]:
                if p:
                    out.append(p)
            return

        out.append(s)

    raw: List[str] = []
    if base_url_arg is None:
        raw = []
    elif isinstance(base_url_arg, str):
        _add_from_string(base_url_arg, raw)
    elif isinstance(base_url_arg, (list, tuple)):
        for item in base_url_arg:
            if item is None:
                continue
            if isinstance(item, str):
                _add_from_string(item, raw)
            else:
                _add_from_string(str(item), raw)
    else:
        _add_from_string(str(base_url_arg), raw)

    # dedup (stable)
    seen: set[str] = set()
    dedup: List[str] = []
    for u in raw:
        u = (u or "").strip()
        if not u:
            continue
        if u in seen:
            continue
        seen.add(u)
        dedup.append(u)
    return dedup


def _assign_base_urls_to_slots(base_urls: Sequence[str], n_slots: int) -> List[str]:
    """Assign base_urls to question-worker slots exactly like generate_sft_loop.py."""
    if n_slots <= 0:
        raise ValueError("n_slots must be positive")
    if not base_urls:
        raise ValueError("base_urls is empty")

    m = len(base_urls)
    if m >= n_slots:
        return list(base_urls[:n_slots])

    base = n_slots // m
    rem = n_slots % m

    slots: List[str] = []
    for i, u in enumerate(base_urls):
        cnt = base + (1 if i < rem else 0)
        slots.extend([u] * cnt)

    if len(slots) != n_slots:
        raise RuntimeError(f"slot assignment bug: len(slots)={len(slots)} != n_slots={n_slots}")
    return slots




# ----------------------------
# API name normalization (compat with older datasets)
# ----------------------------


def _normalize_api_name(fn: str, *, spec) -> str:
    """Normalize an API full name to the canonical prefix used in api-docs.

    Motivation:
      - Older jsonl files might store used_apis/seed_apis using module_alias
        (e.g., `cuda.atomic.add`) or other prefixes.
    - synthesis_common.extract_api_list_from_code normalizes detected calls to
        `spec.module_name.*`.

    To keep fusion overlap checks and api_doc_map lookup consistent, we map any
    known prefix in `spec.module_call_prefixes` (except the canonical
    `spec.module_name`) onto `spec.module_name`.
    """
    s = str(fn or '').strip()
    if not s:
        return ''

    # Already canonical
    if s == spec.module_name or s.startswith(spec.module_name + '.'):
        return s

    prefixes = list(getattr(spec, 'module_call_prefixes', ()) or [])
    if getattr(spec, 'module_alias', None) and spec.module_alias not in prefixes:
        prefixes.insert(0, spec.module_alias)

    # Match longer prefixes first to avoid partial replacement issues.
    prefixes = sorted(set([p for p in prefixes if isinstance(p, str)]), key=len, reverse=True)

    for p in prefixes:
        p = (p or '').strip()
        if not p or p == spec.module_name:
            continue
        if s == p:
            return spec.module_name
        if s.startswith(p + '.'):
            return spec.module_name + s[len(p):]

    return s


def _normalize_api_names(
    api_names: Sequence[str],
    *,
    spec,
    add_base: bool = False,
) -> List[str]:
    """Normalize a list of API names (stable-dedup) and optionally add base APIs.

    When add_base=True, for a name like:
        <module_name>.foo.bar.baz
    we also add:
        <module_name>.foo

    This mirrors synthesis_common.extract_api_list_from_code, which records both the
    full attribute chain and the first attribute after the module prefix.

    Important: we do NOT recursively add shorter prefixes (we won't add
    `module_name` alone), to avoid making overlap checks meaningless.
    """
    out: List[str] = []
    seen: set[str] = set()

    mod_parts = [p for p in str(getattr(spec, 'module_name', '') or '').split('.') if p]
    mod_len = len(mod_parts)

    for fn in api_names or []:
        n = _normalize_api_name(str(fn), spec=spec)
        if not n:
            continue
        if n not in seen:
            out.append(n)
            seen.add(n)

        if not add_base or mod_len <= 0:
            continue

        parts = [p for p in n.split('.') if p]
        if parts[:mod_len] != mod_parts:
            continue

        # Need at least 2 segments after module prefix to have a distinct base.
        if len(parts) <= mod_len + 1:
            continue

        base = '.'.join(mod_parts + [parts[mod_len]])
        if base and base != n and base not in seen:
            out.append(base)
            seen.add(base)

    return out


def _lookup_api_doc(fn: str, *, api_doc_map: Dict[str, ApiDoc], spec) -> Optional[ApiDoc]:
    """Best-effort doc lookup for a possibly non-canonical API name.

    - First try exact match (after normalization)
    - Then progressively strip trailing segments until a doc is found
      (down to at least module_name + 1 segment)
    """
    key = _normalize_api_name(fn, spec=spec)
    d = api_doc_map.get(key)
    if d is not None:
        return d

    parts = [p for p in key.split('.') if p]
    mod_parts = [p for p in str(getattr(spec, 'module_name', '') or '').split('.') if p]
    min_len = max(len(mod_parts) + 1, 1)

    for i in range(len(parts) - 1, min_len - 1, -1):
        cand = '.'.join(parts[:i])
        d = api_doc_map.get(cand)
        if d is not None:
            return d
    return None

# ----------------------------
# Fusion example representation
# ----------------------------


@dataclass(frozen=True)
class FusionExample:
    question: str
    answer_code: str
    test_code: str
    answer_md: str
    test_md: str
    seed_apis: Tuple[str, ...]
    used_apis: Tuple[str, ...]
    meta: Dict[str, Any]
    source_path: str


def _parse_fusion_example(obj: Dict[str, Any], *, spec, source_path: str) -> Optional[FusionExample]:
    q = str(obj.get("question", "") or "").strip()
    if not q:
        return None

    answer_md = str(obj.get("answer", "") or "").strip()
    test_md = str(obj.get("test", "") or "").strip()
    if not answer_md or not test_md:
        return None

    answer_code = strip_code_fences(answer_md).strip()
    test_code = strip_code_fences(test_md).strip()
    if not answer_code or not test_code:
        return None

    seed_apis_any = obj.get("seed_apis", []) or []
    if not isinstance(seed_apis_any, list):
        seed_apis_any = []
    seed_apis = tuple(str(x).strip() for x in seed_apis_any if isinstance(x, str) and str(x).strip())

    used_apis_any = obj.get("used_apis", []) or []
    if not isinstance(used_apis_any, list):
        used_apis_any = []
    used_apis = tuple(str(x).strip() for x in used_apis_any if isinstance(x, str) and str(x).strip())

    # Normalize API names for compatibility with older datasets.
    seed_apis = tuple(_normalize_api_names(seed_apis, spec=spec, add_base=False))
    used_apis = tuple(_normalize_api_names(used_apis, spec=spec, add_base=True))

    if not used_apis:
        # best-effort: recompute
        try:
            used_apis = tuple(_normalize_api_names(extract_api_list_from_code(answer_code, spec=spec), spec=spec, add_base=True))
        except Exception:
            used_apis = tuple()

    # For fusion, we rely on used_apis to construct seed_apis (= union of two sources' used_apis).
    # If we still cannot obtain used_apis, skip this record.
    if not used_apis:
        return None

    meta_any = obj.get("meta", {}) or {}
    meta = meta_any if isinstance(meta_any, dict) else {}

    return FusionExample(
        question=q,
        answer_code=answer_code,
        test_code=test_code,
        answer_md=answer_md,
        test_md=test_md,
        seed_apis=seed_apis,
        used_apis=used_apis,
        meta=meta,
        source_path=source_path,
    )


def _load_examples_jsonl(path: Path, *, spec, dedup_questions: bool) -> List[FusionExample]:
    out: List[FusionExample] = []
    seen: set[str] = set()
    for rec in _iter_jsonl(path):
        ex = _parse_fusion_example(rec, spec=spec, source_path=str(path))
        if ex is None:
            continue
        if dedup_questions:
            key = _normalize_question(ex.question)
            if key in seen:
                continue
            seen.add(key)
        out.append(ex)
    return out


def _update_api_use_counts_from_examples(api_use_counts: Dict[str, int], examples: Sequence[FusionExample]) -> None:
    for ex in examples:
        for a in ex.used_apis:
            if a:
                api_use_counts[a] = api_use_counts.get(a, 0) + 1


def _format_one_source_example(ex: FusionExample, label: str) -> str:
    seed = ", ".join(ex.seed_apis) if ex.seed_apis else "(none)"
    used = ", ".join(ex.used_apis) if ex.used_apis else "(unknown)"
    return (
        f"### Source Example {label}\n"
        f"Question:\n{ex.question.strip()}\n\n"
        "Answer (Python):\n"
        f"```python\n{ex.answer_code.rstrip()}\n```\n\n"
        "Test (Python):\n"
        f"```python\n{ex.test_code.rstrip()}\n```\n\n"
        f"Seed APIs: {seed}\n"
        f"Used APIs: {used}\n"
    )


def _fusion_context_block(ex_a: FusionExample, ex_b: FusionExample) -> str:
    return (
        "\n\n"
        "# Fusion context\n"
        "You will be given two existing solved training examples. Your job is to\n"
        "create a NEW task by fusing their key ideas into ONE coherent problem.\n"
        "Do NOT copy their text or code verbatim; write something new.\n\n"
        + _format_one_source_example(ex_a, "A")
        + "\n"
        + _format_one_source_example(ex_b, "B")
        + "\n"
    )


def _fusion_question_suffix() -> str:
    return (
        "Now generate ONE new question that fuses Source Example A and B.\n"
        "Constraints:\n"
        "- The new question MUST combine important elements from BOTH examples.\n"
        "- The new question MUST be different from both original questions.\n"
        "- The question must be in English.\n"
        "- The task must be solvable by implementing exactly ONE top-level Python function.\n"
        "- The task must be testable with asserts (unit-test style).\n"
        "Output format:\n"
        "Return ONLY a JSON object with a single key `question`. Example: {\"question\": \"...\"}\n"
    )


def _fusion_answer_suffix() -> str:
    return (
        "Use the two source examples above as inspiration to solve the NEW question.\n"
        "Your solution MUST fuse key ideas from BOTH examples.\n"
        "API usage constraint:\n"
        "- Your final implementation MUST use at least ONE API that overlaps with Source Example A's Used APIs.\n"
        "- Your final implementation MUST use at least ONE API that overlaps with Source Example B's Used APIs.\n"
        "Do NOT copy-paste code; write an original implementation and original tests.\n"
    )


def _example_weight(ex: FusionExample, *, api_use_counts: Dict[str, int], alpha: float) -> float:
    apis = list(dict.fromkeys(list(ex.used_apis) or list(ex.seed_apis)))
    if not apis:
        return 1.0
    a = max(float(alpha), 0.0)
    if a == 0.0:
        return 1.0
    w_sum = 0.0
    for fn in apis:
        w_sum += 1.0 / ((api_use_counts.get(fn, 0) + 1) ** a)
    return w_sum / max(1, len(apis))


def _sample_two_examples(
    pool: Sequence[FusionExample],
    *,
    rng: random.Random,
    biased: bool,
    alpha: float,
    api_use_counts: Dict[str, int],
) -> Tuple[FusionExample, FusionExample]:
    if len(pool) < 2:
        raise ValueError("seed pool must contain at least 2 examples to fuse")
    if not biased:
        a, b = rng.sample(list(pool), k=2)
        return a, b
    weights = [_example_weight(ex, api_use_counts=api_use_counts, alpha=alpha) for ex in pool]
    chosen = weighted_sample_without_replacement(list(pool), weights, 2, rng=rng)
    return chosen[0], chosen[1]


def _seed_apis_from_source_used_apis(
    ex_a: FusionExample,
    ex_b: FusionExample,
    *,
    fallback_api_names: Sequence[str],
) -> List[str]:
    """Seed APIs strategy (fusion):

    The seed APIs provided to the model are the **union of the two source
    examples' `used_apis`**.

    This matches the user's requirement: no need for a "--k" parameter.
    """

    union_used = list(dict.fromkeys(list(ex_a.used_apis) + list(ex_b.used_apis)))
    if union_used:
        return union_used

    # Fallbacks (should be rare): try seed_apis union, otherwise pick a small
    # subset from the global api list to avoid an empty prompt.
    union_seed = list(dict.fromkeys(list(ex_a.seed_apis) + list(ex_b.seed_apis)))
    if union_seed:
        return union_seed

    if fallback_api_names:
        return [str(fallback_api_names[0])]
    return []


def _seed_api_docs_from_names(seed_api_names: Sequence[str], *, api_doc_map: Dict[str, ApiDoc], spec) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for fn in seed_api_names:
        fn_norm = _normalize_api_name(fn, spec=spec)
        d = _lookup_api_doc(fn_norm, api_doc_map=api_doc_map, spec=spec)
        if d is None:
            docs.append({"full_name": fn_norm, "signature": "", "docstring": ""})
        else:
            docs.append({"full_name": fn_norm, "signature": d.signature, "docstring": d.docstring})
    return docs


# ----------------------------
# Generation for one Dx
# ----------------------------


def _generate_for_one_d(
    *,
    d_idx: int,
    target_num: int,
    seed_pool: Sequence[FusionExample],
    out_path: Path,
    fail_path: Optional[Path],
    spec,
    api_docs: Sequence[ApiDoc],
    api_doc_map: Dict[str, ApiDoc],
    args,
    rng_master: random.Random,
    seen_questions: set[str],
    api_use_counts: Dict[str, int],
) -> List[FusionExample]:
    """Generate samples for a single dataset D{d_idx}.

    Returns: list of *new* FusionExample records generated in this call.
    """

    dedup_questions = not args.no_dedup_question
    q_workers = max(1, int(args.question_workers))

    # Parse base_urls (same logic)
    base_urls = _parse_base_urls(args.base_url)
    if not base_urls:
        base_urls = ["http://127.0.0.1:8000"]
    slot_base_urls = _assign_base_urls_to_slots(base_urls, q_workers)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fail_path:
        fail_path.parent.mkdir(parents=True, exist_ok=True)

    # Existing count is computed outside; here we just open output in append mode
    # and keep generating until `target_num` unique questions are written.
    written = 0
    if out_path.exists() and not args.overwrite:
        # Count unique questions in existing file (consistent with synthesis loop)
        seen_local: set[str] = set()
        for rec in _iter_jsonl(out_path):
            q = str(rec.get("question", "") or "").strip()
            if not q:
                continue
            key = _normalize_question(q)
            if dedup_questions and key in seen_local:
                continue
            if dedup_questions:
                seen_local.add(key)
            written += 1

    if target_num <= written:
        print(
            f"[skip] D{d_idx}: already has written={written} >= target={target_num} (dedup={dedup_questions})",
            file=sys.stderr,
        )
        return []

    fout = out_path.open("w" if args.overwrite else "a", encoding="utf-8")
    ffail = fail_path.open("w" if args.overwrite else "a", encoding="utf-8") if fail_path else None

    state_lock = threading.Lock()
    stop_event = threading.Event()

    issued = int(written)
    new_examples: List[FusionExample] = []

    fallback_api_names = [api_full_name(d, spec) for d in api_docs]

    def _generate_one_sample(*, task_id: int, task_seed: int, base_url: str) -> Dict[str, Any]:
        local_rng = random.Random(int(task_seed))

        if stop_event.is_set():
            return {"ok": False, "stage": "cancelled", "question_id": f"D{d_idx}_q{task_id:07d}"}

        # Snapshot for biased sampling
        with state_lock:
            counts_snapshot = dict(api_use_counts)

        try:
            ex_a, ex_b = _sample_two_examples(
                seed_pool,
                rng=local_rng,
                biased=bool(args.biased_api_sampling),
                alpha=float(args.tail_alpha),
                api_use_counts=counts_snapshot,
            )
        except Exception as e:
            return {
                "ok": False,
                "stage": "seed_sampling",
                "fail_items": [{"stage": "seed_sampling", "error": str(e), "d": d_idx}],
            }

        # Seed APIs for fusion prompts: union(source A used_apis, source B used_apis)
        seed_api_names = _seed_apis_from_source_used_apis(
            ex_a,
            ex_b,
            fallback_api_names=fallback_api_names,
        )
        # Normalize/dedup seed APIs so overlap checks & api-doc lookup are consistent.
        seed_api_names = _normalize_api_names(seed_api_names, spec=spec, add_base=False)
        seed_api_docs = _seed_api_docs_from_names(seed_api_names, api_doc_map=api_doc_map, spec=spec)

        # ------------ 1) Question generation ------------
        qid = f"D{d_idx}_q{task_id:07d}"
        nfs = int(getattr(args, "num_few_shot", -1))
        n_shots_q = nfs if nfs >= 0 else int(spec.num_few_shots_q)
        shots_q = local_rng.sample(spec.few_shot_pool, k=n_shots_q)
        prompt_q = build_question_prompt(seed_api_docs=seed_api_docs, shots=shots_q, spec=spec)
        prompt_q = prompt_q + _fusion_context_block(ex_a, ex_b) + _fusion_question_suffix()

        q_messages: List[Dict[str, str]] = [
            {"role": "system", "content": spec.system_message},
            {"role": "user", "content": prompt_q},
        ]

        question: Optional[str] = None
        question_key: Optional[str] = None
        last_q_err: Optional[str] = None

        for q_try in range(1, int(args.max_tries_q) + 1):
            if stop_event.is_set():
                return {"ok": False, "stage": "cancelled", "question_id": qid}
            try:
                raw_q = call_vllm_chat(
                    base_url=base_url,
                    model=args.model,
                    messages=sanitize_messages(q_messages),
                    temperature=args.temperature_q,
                    max_tokens=args.max_tokens_q,
                )
                obj_q = parse_json_object(raw_q)
                q = str(obj_q.get("question", "") or "").strip()
                if not q:
                    raise ValueError("empty question")
                if (not args.allow_module_leak) and (spec.module_name in q or spec.module_alias in q):
                    raise ValueError("question mentions module name/alias")

                key = _normalize_question(q)
                if dedup_questions:
                    with state_lock:
                        if key in seen_questions:
                            raise ValueError("duplicate question")
                        # pre-commit to block other workers
                        seen_questions.add(key)
                    question_key = key

                question = q
                break
            except Exception as e:
                last_q_err = str(e)
                if args.debug:
                    print(f"[debug][Q][D{d_idx}] qid={qid} try={q_try} err={last_q_err}", file=sys.stderr)
                q_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output failed parsing/validation.\n"
                            f"Error: {last_q_err}\n\n"
                            "Please regenerate a NEW JSON object with ONLY the key `question`.\n"
                            "The question must be in English and must fuse BOTH source examples."
                        ),
                    }
                )
                if args.sleep:
                    _sleep_with_jitter(float(args.sleep))

        if question is None:
            return {
                "ok": False,
                "stage": "question",
                "question_id": qid,
                "seed_apis": seed_api_names,
                "fail_items": [
                    {
                        "stage": "question",
                        "question_id": qid,
                        "d": d_idx,
                        "seed_apis": seed_api_names,
                        "error": f"Q failed after {args.max_tries_q} tries: {last_q_err}",
                    }
                ],
            }

        # ------------ 2) Answer/Test generation (multi-candidates) ------------
        n_cand = max(1, int(args.candidates_per_question))
        cand_messages: List[Tuple[int, List[Dict[str, str]]]] = []
        for cand_id in range(n_cand):
            nfs = int(getattr(args, "num_few_shot", -1))
            n_shots_a = nfs if nfs >= 0 else int(spec.num_few_shots_a)
            shots_a = local_rng.sample(spec.few_shot_pool, k=n_shots_a)
            prompt_a = build_answer_prompt(question=question, seed_api_docs=seed_api_docs, shots=shots_a, spec=spec)
            prompt_a = prompt_a + _fusion_context_block(ex_a, ex_b) + _fusion_answer_suffix()
            messages_a: List[Dict[str, str]] = [
                {"role": "system", "content": spec.system_message},
                {"role": "user", "content": prompt_a},
            ]
            cand_messages.append((cand_id, messages_a))

        def _one_candidate_chain(*, cand_id: int, messages_a: List[Dict[str, str]]) -> Dict[str, Any]:
            last_err: Optional[str] = None
            last_raw: str = ""
            for attempt in range(1, int(args.max_tries_a) + 1):
                if stop_event.is_set():
                    return {"ok": False, "error": "cancelled", "candidate_id": cand_id}
                try:
                    raw = call_vllm_chat(
                        base_url=base_url,
                        model=args.model,
                        messages=sanitize_messages(messages_a),
                        temperature=args.temperature_a,
                        max_tokens=args.max_tokens_a,
                    )
                    last_raw = raw

                    answer, test = parse_answer_test_from_model_output(raw, spec=spec)
                    fn_name = validate_answer_and_test(answer, test, spec=spec, question=question)
                    used_apis = extract_api_list_from_code(answer, spec=spec)

                    # keep order + dedup
                    used_apis = list(dict.fromkeys(used_apis))

                    # Fusion constraint: overlap with BOTH source examples.
                    used_set = set(used_apis)
                    # if ex_a.used_apis and used_set.isdisjoint(ex_a.used_apis):
                    #     raise ValueError("No overlap between used_apis and Source Example A used_apis")
                    # if ex_b.used_apis and used_set.isdisjoint(ex_b.used_apis):
                    #     raise ValueError("No overlap between used_apis and Source Example B used_apis")
                    if not used_apis:
                        raise ValueError("Do not use any APIs")
                    if spec.seed_overlap_required and not args.no_seed_overlap_check:
                        if seed_api_names and (set(used_apis).intersection(seed_api_names) == set()):
                            raise ValueError("No overlap between used_apis and seed_apis")

                    if not args.no_exec_verify:
                        executable = answer + "\n\n" + test
                        if args.docker_image.strip():
                            ok, out = run_python_docker(
                                executable,
                                image=args.docker_image.strip(),
                                timeout_s=int(args.exec_timeout),
                                network_none=bool(args.docker_network_none),
                                mem_limit=str(args.docker_mem),
                                cpu_limit=str(args.docker_cpus),
                            )
                        else:
                            ok, out = run_python_local(
                                executable,
                                timeout_s=int(args.exec_timeout),
                                python_bin=str(args.exec_python),
                            )
                        if not ok:
                            tail = out[-2000:] if len(out) > 2000 else out
                            raise ValueError("Runtime check failed. Output tail:\n" + tail)

                    judge_obj: Optional[Dict[str, Any]] = None
                    if not args.no_judge:
                        judge_prompt = build_judge_prompt(
                            question=question, answer=answer, seed_api_docs=seed_api_docs, spec=spec
                        )
                        judge_msgs = [
                            {"role": "system", "content": getattr(spec, "judge_system_message", spec.system_message)},
                            {"role": "user", "content": judge_prompt},
                        ]
                        judge_raw = call_vllm_chat(
                            base_url=base_url,
                            model=args.model,
                            messages=sanitize_messages(judge_msgs),
                            temperature=args.temperature_judge,
                            max_tokens=args.max_tokens_judge,
                        )
                        judge_any = parse_json_object(judge_raw)
                        label = int(judge_any.get("label", 0))
                        reason = str(judge_any.get("reason", "") or "").replace("\n", " ").strip()
                        judge_obj = {"label": label, "reason": reason, "model": args.model}
                        if label != 1:
                            raise ValueError("Judge rejected sample: " + reason)

                    answer_md = f"```python\n{answer.rstrip()}\n```"
                    test_md = f"```python\n{test.rstrip()}\n```"

                    # provenance
                    src_a_qid = None
                    src_b_qid = None
                    if isinstance(ex_a.meta, dict):
                        src_a_qid = ex_a.meta.get("question_id")
                    if isinstance(ex_b.meta, dict):
                        src_b_qid = ex_b.meta.get("question_id")

                    record = {
                        "question": question,
                        "answer": answer_md,
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer_md},
                        ],
                        "test": test_md,
                        "seed_apis": seed_api_names,
                        "used_apis": used_apis,
                        "judge": judge_obj,
                        "meta": {
                            "question_id": qid,
                            "candidate_id": cand_id,
                            "attempts": attempt,
                            "generator_model": args.model,
                            "judge_model": (args.model if not args.no_judge else None),
                            "k": len(seed_api_names),
                            "fn_name": fn_name,
                            "source_question_file": "<fusion_loop>",
                            "loop_attempt": task_id,
                            "fusion_d": int(d_idx),
                            "seed_api_strategy": "union_source_used_apis",
                            "fusion_overlap_required": True,
                            "fusion_sources": [
                                {
                                    "label": "A",
                                    "question_id": src_a_qid,
                                    "question": ex_a.question,
                                    "source_path": ex_a.source_path,
                                    "used_apis": list(ex_a.used_apis),
                                },
                                {
                                    "label": "B",
                                    "question_id": src_b_qid,
                                    "question": ex_b.question,
                                    "source_path": ex_b.source_path,
                                    "used_apis": list(ex_b.used_apis),
                                },
                            ],
                            "biased_api_sampling": bool(args.biased_api_sampling),
                            "tail_alpha": float(args.tail_alpha),
                            "question_workers": int(args.question_workers),
                        },
                    }
                    return {"ok": True, "record": record}

                except Exception as e:
                    last_err = str(e)
                    if args.debug:
                        print(
                            f"[debug][A][D{d_idx}] qid={qid} cand={cand_id} attempt={attempt} err={last_err}",
                            file=sys.stderr,
                        )
                    messages_a.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous output failed validation or execution.\n"
                                f"Error: {last_err}\n\n"
                                "Please regenerate a NEW output and make sure the format and constraints are satisfied:\n"
                                "- You may start with a brief plan (3–8 lines, plain text, English only; "
                                "do NOT include ```; do NOT output JSON).\n"
                                "- Then output EXACTLY two Python code blocks using markdown triple backticks:\n"
                                "  1) The first code block is the answer (implementation).\n"
                                "  2) The second code block is the test (tests; asserts recommended).\n"
                                "- Besides the plan and these two code blocks, do NOT output anything else, "
                                "and do NOT output any additional code blocks.\n\n"
                                f"1) The answer must include `{spec.required_import}`.\n"
                                "2) Define exactly ONE top-level function.\n"
                                "3) Provide runnable tests (any number of asserts; at least one is recommended unless the spec requires 0).\n"
                                "4) Running answer + \"\\n\\n\" + test must pass all asserts.\n"
                                "Note: The question is unchanged and you must still fuse BOTH source examples."
                            ),
                        }
                    )
                    if args.sleep:
                        _sleep_with_jitter(float(args.sleep))

            return {
                "ok": False,
                "error": last_err or "unknown error",
                "last_model_output_tail": (last_raw[-2000:] if len(last_raw) > 2000 else last_raw),
                "candidate_id": cand_id,
            }

        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=n_cand) as ex:
            futs = [
                ex.submit(_one_candidate_chain, cand_id=cand_id, messages_a=messages_a)
                for cand_id, messages_a in cand_messages
            ]
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({"ok": False, "error": f"worker_crash: {e}", "candidate_id": -1})

        ok_records = [r["record"] for r in results if r.get("ok")]
        if not ok_records:
            if dedup_questions and question_key:
                with state_lock:
                    seen_questions.discard(question_key)
            fail_items: List[Dict[str, Any]] = []
            for r in results:
                if r.get("ok"):
                    continue
                fail_items.append(
                    {
                        "stage": "answer",
                        "d": d_idx,
                        "question_id": qid,
                        "question": question,
                        "seed_apis": seed_api_names,
                        "candidate_id": r.get("candidate_id"),
                        "error": r.get("error"),
                        "last_model_output_tail": r.get("last_model_output_tail", ""),
                    }
                )
            return {
                "ok": False,
                "stage": "answer",
                "question_id": qid,
                "question": question,
                "seed_apis": seed_api_names,
                "fail_items": fail_items,
            }

        chosen = local_rng.choice(ok_records)
        return {"ok": True, "record": chosen, "question_id": qid}

    # --- Logging config ---
    counts_per_url: Dict[str, int] = {}
    for u in slot_base_urls:
        counts_per_url[u] = counts_per_url.get(u, 0) + 1
    print(
        f"[config][D{d_idx}] target_num={target_num} existing_written={written} question_workers={q_workers} candidates_per_question={args.candidates_per_question} base_urls={len(base_urls)}",
        file=sys.stderr,
    )
    if len(base_urls) > 1:
        parts = " | ".join([f"{u} x{counts_per_url.get(u,0)}" for u in base_urls])
        print(f"[config][D{d_idx}] base_url_slots: {parts}", file=sys.stderr)

    inflight: Dict[Any, int] = {}

    try:
        with ThreadPoolExecutor(max_workers=q_workers) as ex:
            # Fill each slot with 1 task
            for slot_id in range(q_workers):
                if written >= int(target_num) or stop_event.is_set():
                    break
                issued += 1
                task_seed = rng_master.randrange(0, 2**31 - 1)
                fut = ex.submit(
                    _generate_one_sample,
                    task_id=issued,
                    task_seed=task_seed,
                    base_url=slot_base_urls[slot_id],
                )
                inflight[fut] = slot_id

            while inflight and written < int(target_num) and not stop_event.is_set():
                done, _ = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)

                for fut in done:
                    slot_id = inflight.pop(fut, None)
                    if slot_id is None:
                        continue

                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {
                            "ok": False,
                            "stage": "worker_crash",
                            "fail_items": [{"stage": "worker_crash", "d": d_idx, "error": str(e)}],
                        }

                    if not res.get("ok"):
                        if ffail and res.get("fail_items"):
                            for item in res.get("fail_items"):
                                _write_jsonl_line(ffail, item)

                        if res.get("stage") == "question":
                            print(
                                f"[Q][D{d_idx}] failed qid={res.get('question_id')}",
                                file=sys.stderr,
                            )
                        elif res.get("stage") == "answer":
                            print(
                                f"[fail][D{d_idx}] qid={res.get('question_id')} written={written}/{target_num} : no candidates accepted (n={max(1, int(args.candidates_per_question))})",
                                file=sys.stderr,
                            )

                        if written < int(target_num) and not stop_event.is_set():
                            issued += 1
                            task_seed = rng_master.randrange(0, 2**31 - 1)
                            nfut = ex.submit(
                                _generate_one_sample,
                                task_id=issued,
                                task_seed=task_seed,
                                base_url=slot_base_urls[slot_id],
                            )
                            inflight[nfut] = slot_id
                        continue

                    if written >= int(target_num):
                        stop_event.set()
                    else:
                        record = res["record"]
                        _write_jsonl_line(fout, record)

                        # Update global api_use_counts based on final written data
                        with state_lock:
                            for a in record.get("used_apis", []) or []:
                                if isinstance(a, str) and a:
                                    api_use_counts[a] = api_use_counts.get(a, 0) + 1

                        # Track new example in memory for later seed pools
                        ex_new = _parse_fusion_example(record, spec=spec, source_path=str(out_path))
                        if ex_new is not None:
                            with state_lock:
                                new_examples.append(ex_new)

                        written += 1
                        meta = record.get("meta", {}) or {}
                        print(
                            f"[ok][D{d_idx}] written={written}/{target_num} qid={meta.get('question_id')} cand={meta.get('candidate_id')} seed_cnt={meta.get('k')}",
                            file=sys.stderr,
                        )

                        if written >= int(target_num):
                            stop_event.set()

                    if written < int(target_num) and not stop_event.is_set():
                        issued += 1
                        task_seed = rng_master.randrange(0, 2**31 - 1)
                        nfut = ex.submit(
                            _generate_one_sample,
                            task_id=issued,
                            task_seed=task_seed,
                            base_url=slot_base_urls[slot_id],
                        )
                        inflight[nfut] = slot_id

    finally:
        fout.close()
        if ffail:
            ffail.close()

    print(f"Done [D{d_idx}]. wrote_new={len(new_examples)} total_written={written} out={out_path}", file=sys.stderr)
    return new_examples


def main() -> int:
    ap = argparse.ArgumentParser()

    # Core inputs
    ap.add_argument("--spec", type=str, required=True, help="Synthesis spec JSON")
    ap.add_argument("--d1", type=str, required=True, help="Path to existing D1 jsonl (already generated)")
    ap.add_argument("--out-root", type=str, required=True, help="Output root. Writes <out_root>_D{d}.jsonl")
    ap.add_argument(
        "--fail-root",
        type=str,
        default="",
        help="Optional failure log root. Writes <fail_root>_D{d}.jsonl (empty to disable)",
    )

    # Iteration control
    ap.add_argument("--max-d", type=int, required=True, help="Generate until D{max_d} (inclusive). Must be >= 2")
    ap.add_argument(
        "--num-per-d",
        type=int,
        required=True,
        help="How many accepted samples to generate for EACH D2..Dmax (D1 is provided).",
    )
    ap.add_argument(
        "--start-d",
        type=int,
        default=2,
        help="Start generating from D{start_d}. Default: 2. Useful for partial resume.",
    )

    # NOTE: No --k. In fusion mode, seed_apis = union(source A used_apis, source B used_apis).
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--num-few-shot",
        "--num_few_shot",
        dest="num_few_shot",
        type=int,
        default=-1,
        help=(
            "Override the number of few-shot examples sampled from spec.few_shot_pool. "
            "If set to >= 0, it overrides BOTH spec.num_few_shots_q and spec.num_few_shots_a. "
            "Set to 0 to disable few-shot. Default: -1 (use spec)."
        ),
    )

    ap.add_argument(
        "--question-workers",
        type=int,
        default=1,
        help="Question-level worker count (outer pool size).",
    )
    ap.add_argument(
        "--biased-api-sampling",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "In fusion mode, this biases which source examples are selected: "
            "prefer examples that contain currently low-frequency used_apis."
        ),
    )
    ap.add_argument("--tail-alpha", type=float, default=0.7, help="Long-tail alpha (larger -> more bias)")

    # vLLM args
    ap.add_argument(
        "--base-url",
        type=str,
        action="append",
        default=None,
        help=(
            "vLLM base URL. Can be provided multiple times to form a list. "
            "Also supports comma-separated string or JSON list string."
        ),
    )
    ap.add_argument("--model", type=str, default="model")
    ap.add_argument("--temperature-q", type=float, default=0.7)
    ap.add_argument("--max-tokens-q", type=int, default=512)
    ap.add_argument("--temperature-a", type=float, default=0.7)
    ap.add_argument("--max-tokens-a", type=int, default=2048)
    ap.add_argument("--temperature-judge", type=float, default=0.0)
    ap.add_argument("--max-tokens-judge", type=int, default=512)
    ap.add_argument("--no-judge", action="store_true")

    # retries
    ap.add_argument("--max-tries-q", type=int, default=5)
    ap.add_argument("--max-tries-a", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=3.0, help="Sleep seconds after failures")

    ap.add_argument(
        "--candidates-per-question",
        type=int,
        default=5,
        help="How many answer/test candidates to generate per question (in parallel).",
    )

    # misc
    ap.add_argument("--overwrite", action="store_true", help="Overwrite all D2.. outputs (start fresh)")
    ap.add_argument("--no-dedup-question", action="store_true", help="Disable global question dedup")
    ap.add_argument(
        "--allow-module-leak",
        action="store_true",
        help="Allow module_name/module_alias to appear in the generated question (not recommended)",
    )
    ap.add_argument("--debug", action="store_true")

    # overlap check
    ap.add_argument("--no-seed-overlap-check", action="store_true")

    # execution verify
    ap.add_argument("--no-exec-verify", action="store_true")
    ap.add_argument("--exec-timeout", type=int, default=10)
    ap.add_argument(
        "--exec-python",
        type=str,
        default="",
        help="Python interpreter to run tests (empty -> sys.executable)",
    )
    ap.add_argument("--docker-image", type=str, default="", help="If set, run tests inside docker")
    ap.add_argument("--docker-network-none", action="store_true")
    ap.add_argument("--docker-mem", type=str, default="2g")
    ap.add_argument("--docker-cpus", type=str, default="2")

    args = ap.parse_args()

    if int(args.max_d) < 2:
        print("ERROR: --max-d must be >= 2", file=sys.stderr)
        return 2
    if int(args.start_d) < 2 or int(args.start_d) > int(args.max_d):
        print("ERROR: --start-d must be in [2, max_d]", file=sys.stderr)
        return 2
    if int(args.num_per_d) <= 0:
        print("ERROR: --num-per-d must be positive", file=sys.stderr)
        return 2

    rng_master = random.Random(int(args.seed))

    spec = load_synthesis_spec(Path(args.spec))
    # Validate --num-few-shot override early (if provided).
    nfs = int(getattr(args, "num_few_shot", -1))
    if nfs >= 0 and len(spec.few_shot_pool) < nfs:
        print(
            f"ERROR: --num-few-shot={nfs} but spec.few_shot_pool has only {len(spec.few_shot_pool)} items",
            file=sys.stderr,
        )
        return 2

    api_docs = load_api_docs(Path(spec.api_docs_path))
    if not api_docs:
        print("ERROR: api-docs is empty", file=sys.stderr)
        return 2
    api_doc_map: Dict[str, ApiDoc] = {api_full_name(d, spec): d for d in api_docs}

    dedup_questions = not args.no_dedup_question

    # Load D1
    d1_path = Path(args.d1)
    if not d1_path.exists():
        print(f"ERROR: D1 file not found: {d1_path}", file=sys.stderr)
        return 2
    d1_examples = _load_examples_jsonl(d1_path, spec=spec, dedup_questions=dedup_questions)
    if len(d1_examples) < 2:
        print(f"ERROR: D1 must contain at least 2 usable examples, got {len(d1_examples)}", file=sys.stderr)
        return 2

    # datasets[1] = D1, datasets[d] = existing Dd (if resume) + newly generated
    datasets: Dict[int, List[FusionExample]] = {1: d1_examples}

    # Global dedup + global api usage counts
    seen_questions: set[str] = set()
    api_use_counts: Dict[str, int] = {}

    for ex in d1_examples:
        seen_questions.add(_normalize_question(ex.question))
    _update_api_use_counts_from_examples(api_use_counts, d1_examples)

    out_root = Path(args.out_root)
    fail_root = Path(args.fail_root) if str(args.fail_root).strip() else None

    # Pre-load existing outputs (resume mode) so we can:
    #  - avoid duplicates across all D2..Dmax
    #  - include them in later seed pools
    if not args.overwrite:
        for d in range(2, int(args.max_d) + 1):
            out_path = Path(str(out_root) + f"_D{d}.jsonl")
            if not out_path.exists():
                continue
            exs = _load_examples_jsonl(out_path, spec=spec, dedup_questions=dedup_questions)
            if not exs:
                continue
            datasets[d] = exs
            for ex in exs:
                seen_questions.add(_normalize_question(ex.question))
            _update_api_use_counts_from_examples(api_use_counts, exs)

    # Safety: if user starts from D>2, ensure previous datasets exist; otherwise
    # we'd accidentally generate (e.g.) D3 using only D1.
    if int(args.start_d) > 2:
        if args.overwrite:
            print("ERROR: --overwrite cannot be used together with --start-d > 2", file=sys.stderr)
            return 2
        missing_prev: List[int] = []
        for prev in range(2, int(args.start_d)):
            if prev not in datasets or len(datasets.get(prev, [])) == 0:
                missing_prev.append(prev)
        if missing_prev:
            ms = ", ".join([f"D{p}" for p in missing_prev])
            print(
                f"ERROR: --start-d={args.start_d} requires existing outputs for {ms}. "
                "Generate from D2 first (or lower --start-d).",
                file=sys.stderr,
            )
            return 2

    # Generate sequentially from start_d..max_d
    for d in range(int(args.start_d), int(args.max_d) + 1):
        seed_pool: List[FusionExample] = []
        for prev in range(1, d):
            seed_pool.extend(datasets.get(prev, []))
        if len(seed_pool) < 2:
            print(f"ERROR: seed pool for D{d} has <2 examples (got {len(seed_pool)})", file=sys.stderr)
            return 2

        out_path = Path(str(out_root) + f"_D{d}.jsonl")
        fail_path = Path(str(fail_root) + f"_D{d}.jsonl") if fail_root else None

        # NOTE: For Dd generation, we should ONLY fuse examples from D1..D{d-1}.
        # We intentionally do NOT include existing Dd as seeds.
        new_exs = _generate_for_one_d(
            d_idx=d,
            target_num=int(args.num_per_d),
            seed_pool=seed_pool,
            out_path=out_path,
            fail_path=fail_path,
            spec=spec,
            api_docs=api_docs,
            api_doc_map=api_doc_map,
            args=args,
            rng_master=rng_master,
            seen_questions=seen_questions,
            api_use_counts=api_use_counts,
        )

        # Update datasets[d] for later iterations.
        if args.overwrite:
            # We started fresh, so Dd is only the newly generated in this run.
            datasets[d] = list(new_exs)
        else:
            # Resume: append, but keep dedup by question.
            merged: List[FusionExample] = []
            local_seen: set[str] = set()
            for ex in datasets.get(d, []):
                key = _normalize_question(ex.question)
                if dedup_questions and key in local_seen:
                    continue
                local_seen.add(key)
                merged.append(ex)
            for ex in new_exs:
                key = _normalize_question(ex.question)
                if dedup_questions and key in local_seen:
                    continue
                local_seen.add(key)
                merged.append(ex)
            datasets[d] = merged

    print(f"All done. Generated/updated up to D{args.max_d}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
