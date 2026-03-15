#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate PriCoder SFT samples in a continuous loop.

Pipeline:
`seed APIs -> question -> (answer, test) -> validation -> optional exec -> optional judge`.

Key features:
- Supports one or multiple `k` values for seed API count.
- Supports question-level and candidate-level parallelism.
- Supports optional API-tail-biased sampling.
- Supports multi-base-url assignment for distributed vLLM endpoints.
- Writes accepted samples to JSONL with deterministic metadata fields.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    sanitize_messages,  # sanitize message content before sending to vLLM
    seed_payload_from_docs,
    validate_answer_and_test,
    weighted_sample_without_replacement,
)


def _normalize_question(q: str) -> str:
    return " ".join((q or "").split())


def _iter_jsonl(path: Path) -> Sequence[Dict[str, Any]]:
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
                # tolerate a truncated trailing line
                continue
    return recs

def _sleep_with_jitter(max_s: float) -> None:
    if max_s <= 0:
        return
    time.sleep(random.uniform(0.0, max_s))

def _load_existing_state(
    out_path: Path,
    *,
    dedup_questions: bool,
) -> Tuple[int, set[str], Dict[str, int]]:
    """Load existing output state: (written, seen_questions, api_use_counts)."""

    written = 0
    seen: set[str] = set()
    api_use_counts: Dict[str, int] = {}

    for rec in _iter_jsonl(out_path):
        q = str(rec.get("question", "") or "").strip()
        if not q:
            continue
        key = _normalize_question(q)
        if dedup_questions:
            if key in seen:
                # skip duplicated normalized questions
                continue
            seen.add(key)
        written += 1

        used = rec.get("used_apis", [])
        if isinstance(used, list):
            for a in used:
                if isinstance(a, str) and a:
                    api_use_counts[a] = api_use_counts.get(a, 0) + 1

    return written, seen, api_use_counts


def _sample_seed_docs(
    api_docs: Sequence[ApiDoc],
    *,
    k: int,
    rng: random.Random,
    spec,
    biased: bool,
    alpha: float,
    api_use_counts: Dict[str, int],
) -> List[ApiDoc]:
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= len(api_docs):
        return list(api_docs)

    full_names = [api_full_name(d, spec) for d in api_docs]
    if biased:
        # tail-biased sampling: prefer less-used APIs
        a = max(float(alpha), 0.0)
        weights = [1.0 / ((api_use_counts.get(fn, 0) + 1) ** a) for fn in full_names]
    else:
        weights = [1.0 for _ in full_names]

    return weighted_sample_without_replacement(api_docs, weights, k, rng=rng)


def _write_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


def _parse_k_values(k_arg: Any) -> List[int]:
    """Parse `--k` values from scalar/comma-separated/JSON-list/append forms."""

    def _add_from_any(x: Any, out: List[int]) -> None:
        if x is None:
            return

        if isinstance(x, (list, tuple)):
            for it in x:
                _add_from_any(it, out)
            return

        s = str(x).strip()
        if not s:
            return

        # JSON list
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
            except Exception:
                obj = None
            if isinstance(obj, list):
                for it in obj:
                    _add_from_any(it, out)
                return
            # invalid JSON list falls through to other parsers

        # comma-separated values
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            for p in parts:
                if p:
                    _add_from_any(p, out)
            return

        # single integer
        try:
            v = int(s)
        except Exception as e:
            raise ValueError(f"invalid k value: {s!r}") from e
        out.append(v)

    raw: List[int] = []
    _add_from_any(k_arg, raw)

    # stable de-dup (preserve order)
    seen: set[int] = set()
    dedup: List[int] = []
    for v in raw:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)

    # validate positivity
    for v in dedup:
        if v <= 0:
            raise ValueError(f"k must be positive, got {v}")

    return dedup


def _parse_base_urls(base_url_arg: Any) -> List[str]:
    """Parse `--base-url` values from scalar/list/comma-separated/JSON-list forms."""

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
                # invalid JSON list falls through
                pass

        # comma-separated values
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            for p in parts:
                if p:
                    out.append(p)
            return

        # single URL
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

    # stable de-dup and drop empty values
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
        """Assign base URLs to worker slots as evenly as possible."""
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

    # 
    if len(slots) != n_slots:
        raise RuntimeError(f"slot assignment bug: len(slots)={len(slots)} != n_slots={n_slots}")
    return slots


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--spec", type=str, required=True, help="Synthesis spec JSON")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL (accepted samples)")
    ap.add_argument("--fail-out", type=str, default="", help="Optional failure-log JSONL path")

    ap.add_argument("--num", type=int, required=True, help="Target number of accepted samples")

    # `--k` supports repeated values, comma lists, and JSON lists
    ap.add_argument(
        "--k",
        type=str,
        action="append",
        required=True,
        help=(
            "Number of seed APIs per question. "
            "Supports repeated --k, comma-separated values, or JSON list."
        ),
    )

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--question-workers",
        type=int,
        default=1,
        help=(
            "Question-level worker count. >1 enables pipeline mode."
        ),
    )

    ap.add_argument(
        "--biased-api-sampling",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable tail-biased API sampling using used_apis counts",
    )
    ap.add_argument("--tail-alpha", type=float, default=0.7, help="Tail-bias exponent alpha")

    # vLLM endpoints
    # `--base-url` supports repeated values/comma list/JSON list
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

    # Retry controls
    ap.add_argument("--max-tries-q", type=int, default=5)
    ap.add_argument("--max-tries-a", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=3.0, help="Max jitter sleep (seconds) between retries")

    ap.add_argument(
        "--candidates-per-question",
        type=int,
        default=5,
        help=(
            "Answer/test candidates per question (>=1)."
        ),
    )

    # Output behavior
    ap.add_argument("--overwrite", action="store_true", help="Overwrite --out/--fail-out instead of append")
    ap.add_argument(
        "--no-dedup-question",
        action="store_true",
        help="Disable question de-duplication",
    )
    ap.add_argument(
        "--allow-module-leak",
        action="store_true",
        help="Allow question text to mention module_name/module_alias",
    )
    ap.add_argument("--debug", action="store_true", help="Enable debug logs")

    # overlap behavior (overrides spec)
    ap.add_argument("--no-seed-overlap-check", action="store_true", help="Disable used_apis ∩ seed_apis overlap check")

    # Execution verification
    ap.add_argument("--no-exec-verify", action="store_true", help="Skip runtime execution of answer+test")
    ap.add_argument("--exec-timeout", type=int, default=20)
    ap.add_argument(
        "--exec-python",
        type=str,
        default="",
        help="Python executable for local runtime checks (default: current interpreter)",
    )
    ap.add_argument("--docker-image", type=str, default="", help="Docker image for sandbox execution")
    ap.add_argument("--docker-network-none", action="store_true", help="Run docker with --network none")
    ap.add_argument("--docker-mem", type=str, default="2g")
    ap.add_argument("--docker-cpus", type=str, default="2")

    args = ap.parse_args()

    # Parse and validate k values
    try:
        k_choices = _parse_k_values(args.k)
    except Exception as e:
        print(f"ERROR: invalid --k: {e}", file=sys.stderr)
        return 2
    if not k_choices:
        print("ERROR: --k is empty after parsing", file=sys.stderr)
        return 2

    # Master RNG for deterministic task seeds
    rng_master = random.Random(args.seed)

    spec = load_synthesis_spec(Path(args.spec))
    api_docs = load_api_docs(Path(spec.api_docs_path))
    if not api_docs:
        print("ERROR: api-docs is empty", file=sys.stderr)
        return 2

    # Parse base URLs; fallback to local endpoint
    base_urls = _parse_base_urls(args.base_url)
    if not base_urls:
        base_urls = ["http://127.0.0.1:8000"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path = Path(args.fail_out) if args.fail_out.strip() else None
    if fail_path:
        fail_path.parent.mkdir(parents=True, exist_ok=True)

    dedup_questions = not args.no_dedup_question

    if args.overwrite:
        written = 0
        seen_questions: set[str] = set()
        api_use_counts: Dict[str, int] = {}
        fout = out_path.open("w", encoding="utf-8")
        ffail = fail_path.open("w", encoding="utf-8") if fail_path else None
    else:
        written, seen_questions, api_use_counts = _load_existing_state(out_path, dedup_questions=dedup_questions)
        fout = out_path.open("a", encoding="utf-8")
        ffail = fail_path.open("a", encoding="utf-8") if fail_path else None

    # Early stop if target already reached
    print(written)
    if args.num <= written:
        print(
            f"Nothing to do: out already has written={written} >= num={args.num} (dedup={dedup_questions})",
            file=sys.stderr,
        )
        fout.close()
        if ffail:
            ffail.close()
        return 0

    # Shared worker state
    state_lock = threading.Lock()
    stop_event = threading.Event()

    # Start question IDs after existing written count
    issued = int(written)

    def _generate_one_sample(*, task_id: int, task_seed: int, base_url: str) -> Dict[str, Any]:
                """Worker pipeline for one sample: seeds -> question -> answer/test.

                Returns:
                    - {ok: True, record: {...}}
                    - {ok: False, fail_items: [...], stage: 'question'|'answer'|'cancelled'|'worker_crash', ...}
                """

        local_rng = random.Random(int(task_seed))

        # Sample one k value if multiple are provided
        k_sampled = int(k_choices[0]) if len(k_choices) == 1 else int(local_rng.choice(k_choices))

        if stop_event.is_set():
            return {
                "ok": False,
                "stage": "cancelled",
                "question_id": f"q{task_id:07d}",
                "k_sampled": k_sampled,
            }

        # --- 1) sample seed APIs (tail-biased by api_use_counts) ---
        with state_lock:
            counts_snapshot = dict(api_use_counts)

        seed_docs = _sample_seed_docs(
            api_docs,
            k=int(k_sampled),
            rng=local_rng,
            spec=spec,
            biased=bool(args.biased_api_sampling),
            alpha=float(args.tail_alpha),
            api_use_counts=counts_snapshot,
        )
        seed_api_names = [api_full_name(d, spec) for d in seed_docs]
        seed_api_docs = seed_payload_from_docs(seed_docs, spec)

        # --- 2) generate question ---
        qid = f"q{task_id:07d}"
        shots_q = local_rng.sample(spec.few_shot_pool, k=spec.num_few_shots_q)
        prompt_q = build_question_prompt(seed_api_docs=seed_api_docs, shots=shots_q, spec=spec)
        q_messages: List[Dict[str, str]] = [
            {"role": "system", "content": spec.system_message},
            {"role": "user", "content": prompt_q},
        ]

        question: Optional[str] = None
        question_key: Optional[str] = None
        last_q_err: Optional[str] = None

        for q_try in range(1, int(args.max_tries_q) + 1):
            if stop_event.is_set():
                return {
                    "ok": False,
                    "stage": "cancelled",
                    "question_id": qid,
                    "seed_apis": seed_api_names,
                    "k_sampled": k_sampled,
                }
            try:
                raw_q = call_vllm_chat(
                    base_url=base_url,
                    model=args.model,
                    messages=sanitize_messages(q_messages),  #  vLLM  sanitize
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
                        # reserve question key to avoid duplicate generation across workers
                        seen_questions.add(key)
                    question_key = key

                question = q
                break

            except Exception as e:
                last_q_err = str(e)
                if args.debug:
                    print(f"[debug][Q] qid={qid} try={q_try} err={last_q_err}", file=sys.stderr)
                q_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output failed parsing/validation.\n"
                            f"Error: {last_q_err}\n\n"
                            "Please regenerate a NEW JSON object with ONLY the key `question` "
                            "and make sure it satisfies all constraints. "
                            "The question should be written in English."
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
                "k_sampled": k_sampled,
                "fail_items": [
                    {
                        "stage": "question",
                        "question_id": qid,
                        "seed_apis": seed_api_names,
                        "k_sampled": k_sampled,
                        "error": f"Q failed after {args.max_tries_q} tries: {last_q_err}",
                    }
                ],
            }

        # --- 3) generate answer/test candidates and keep one valid record ---
        n_cand = max(1, int(args.candidates_per_question))

        # Build candidate prompt/messages using local RNG
        cand_messages: List[Tuple[int, List[Dict[str, str]]]] = []
        for cand_id in range(n_cand):
            shots_a = local_rng.sample(spec.few_shot_pool, k=spec.num_few_shots_a)
            prompt_a = build_answer_prompt(question=question, seed_api_docs=seed_api_docs, shots=shots_a, spec=spec)
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
                        messages=sanitize_messages(messages_a),  #  vLLM  sanitize
                        temperature=args.temperature_a,
                        max_tokens=args.max_tokens_a,
                    )
                    last_raw = raw

                    answer, test = parse_answer_test_from_model_output(raw, spec=spec)
                    fn_name = validate_answer_and_test(answer, test, spec=spec, question=question)

                    used_apis = extract_api_list_from_code(answer, spec=spec)

                    used_apis_unique = list(dict.fromkeys(used_apis))
                    # if len(used_apis_unique) < int(k_sampled):
                    #     # Optionally enforce minimum used API count
                    #     pass

                    # De-duplicated used APIs
                    used_apis = used_apis_unique
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
                            messages=sanitize_messages(judge_msgs),  #  vLLM  sanitize
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
                            # Keep historical behavior: k equals current seed count
                            "k": len(seed_api_names),
                            # Preserve sampled k for analysis
                            "k_sampled": int(k_sampled),
                            "fn_name": fn_name,
                            "source_question_file": "<loop>",
                            "loop_attempt": task_id,
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
                            f"[debug][A] qid={qid} cand={cand_id} attempt={attempt} err={last_err}",
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
                                "Note: The question is unchanged. Only output the plan + the two code blocks."
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
        # Run candidate chains in parallel for this question
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
            # No valid candidate: release reserved question key
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
                        "question_id": qid,
                        "question": question,
                        "seed_apis": seed_api_names,
                        "k_sampled": k_sampled,
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
                "k_sampled": k_sampled,
                "fail_items": fail_items,
            }

        chosen = local_rng.choice(ok_records)
        return {"ok": True, "record": chosen, "question_id": qid, "k_sampled": k_sampled}

    q_workers = max(1, int(args.question_workers))

    # Assign base URLs to question workers
    slot_base_urls = _assign_base_urls_to_slots(base_urls, q_workers)

    # Print compact runtime config
    counts_per_url: Dict[str, int] = {}
    for u in slot_base_urls:
        counts_per_url[u] = counts_per_url.get(u, 0) + 1

    k_desc = str(k_choices[0]) if len(k_choices) == 1 else json.dumps(k_choices, ensure_ascii=False)
    print(
        f"[config] target_num={args.num} k={k_desc} question_workers={q_workers} candidates_per_question={args.candidates_per_question} base_urls={len(base_urls)}",
        file=sys.stderr,
    )
    if len(base_urls) > 1:
        parts = " | ".join([f"{u} x{counts_per_url.get(u,0)}" for u in base_urls])
        print(f"[config] base_url_slots: {parts}", file=sys.stderr)

    # inflight: future -> slot_id mapping
    inflight: Dict[Any, int] = {}

    try:
        with ThreadPoolExecutor(max_workers=q_workers) as ex:
            # Keep one inflight task per worker slot
            for slot_id in range(q_workers):
                if written >= int(args.num) or stop_event.is_set():
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

            while inflight and written < int(args.num) and not stop_event.is_set():
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
                            "fail_items": [
                                {
                                    "stage": "worker_crash",
                                    "error": str(e),
                                }
                            ],
                        }

                    # write failures to fail-out when provided
                    if not res.get("ok"):
                        if ffail and res.get("fail_items"):
                            for item in res.get("fail_items"):
                                _write_jsonl_line(ffail, item)

                        # compact stderr status
                        if res.get("stage") == "question":
                            print(
                                f"[Q] failed qid={res.get('question_id')} k={res.get('k_sampled')}",
                                file=sys.stderr,
                            )
                        elif res.get("stage") == "answer":
                            print(
                                f"[fail] qid={res.get('question_id')} written={written}/{args.num} : no candidates accepted (n={max(1, int(args.candidates_per_question))}) k={res.get('k_sampled')}",
                                file=sys.stderr,
                            )

                        # refill this worker slot unless stopping
                        if written < int(args.num) and not stop_event.is_set():
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

                    # accepted sample
                    if written >= int(args.num):
                        stop_event.set()
                    else:
                        record = res["record"]
                        _write_jsonl_line(fout, record)

                        # update API usage counts for tail-biased sampling
                        with state_lock:
                            for a in record.get("used_apis", []) or []:
                                if isinstance(a, str) and a:
                                    api_use_counts[a] = api_use_counts.get(a, 0) + 1

                        written += 1
                        meta = record.get("meta", {}) or {}
                        print(
                            f"[ok] written={written}/{args.num} qid={meta.get('question_id')} cand={meta.get('candidate_id')} k={meta.get('k')} k_sampled={meta.get('k_sampled')}",
                            file=sys.stderr,
                        )

                        if written >= int(args.num):
                            stop_event.set()

                    # refill this worker slot unless stopping
                    if written < int(args.num) and not stop_event.is_set():
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

    print(f"Done. wrote={written}, out={out_path}", file=sys.stderr)
    return 0 if written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
