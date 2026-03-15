#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filter PyPI metadata JSONL produced by the crawler.

Supported filters:
- `--min-stars`: keep records with `github_stars >= min_stars`.
- `--min-downloads`: keep records with `download_count >= min_downloads`.
- `--date-after`: keep records newer than the cutoff date.

Always excludes:
1) records with missing/empty `github_repo`
2) records whose topics contain "Artificial Intelligence"
3) records whose name or summary contains "agent"

Date format in JSONL is expected to be `YYYY-MM-DD HH:MM:SS`.
"""

import argparse
import json
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, Any, List


def parse_cutoff_datetime(s: str) -> datetime:
    s = s.strip()

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m-%d":
                dt = dt.replace(hour=0, minute=0, second=0)
            return dt
        except ValueError:
            pass

    if s.isdigit():
        if len(s) == 8:  # YYYYMMDD
            year = int(s[:4])
            month = int(s[4:6])
            day = int(s[6:8])
            return datetime(year, month, day)
        elif len(s) == 6:  # YYYYMM
            year = int(s[:4])
            month = int(s[4:6])
            return datetime(year, month, 1)
        elif len(s) == 4:  # YYMM -> 20YYMM
            year = 2000 + int(s[:2])
            month = int(s[2:4])
            return datetime(year, month, 1)

    raise ValueError(f"Unsupported date format: {s!r}. Example: '2024-04-01'.")


def parse_pkg_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def topics_contains_ai(topics_val: Any) -> bool:
    """Return True if topics include 'Artificial Intelligence'."""
    if not topics_val:
        return False
    if isinstance(topics_val, list):
        for t in topics_val:
            if isinstance(t, str) and "Artificial Intelligence" in t:
                return True
        return False
    if isinstance(topics_val, str):
        return "Artificial Intelligence" in topics_val
    return False


def name_or_summary_contains_agent(obj: Any) -> bool:
    """
    Return True if package name or summary contains 'agent' (case-insensitive).
    """
    name = obj.get("name")
    summary = obj.get("summary")

    if isinstance(name, str) and "agent" in name.lower():
        return True
    if isinstance(summary, str) and "agent" in summary.lower():
        return True
    return False


def filter_record(
    idx_and_line: Tuple[int, str],
    min_stars: Optional[int],
    min_downloads: Optional[int],
    cutoff_dt: Optional[datetime],
    date_field: str,
) -> Optional[Tuple[int, Any]]:
    idx, line = idx_and_line
    line = line.strip()
    if not line:
        return None

    try:
        obj = json.loads(line)
    except Exception as e:
        print(f"[WARN] line {idx} JSON parse failed: {e}", file=sys.stderr)
        return None

    # ---------- Rule 1: github_repo must be present ----------
    repo = obj.get("github_repo")
    if repo is None:
        return None
    if isinstance(repo, str) and repo.strip() == "":
        return None

    # ---------- Rule 2: exclude AI-topic packages ----------
    if topics_contains_ai(obj.get("topics")):
        return None

    # ---------- Rule 3: exclude 'agent' in name/summary ----------
    if name_or_summary_contains_agent(obj):
        return None

    # ---------- Optional stars filter ----------
    if min_stars is not None:
        stars = obj.get("github_stars")
        if stars is not None:
            try:
                stars_val = int(stars)
            except (TypeError, ValueError):
                print(
                    f"[WARN] line {idx} github_stars is not int: {stars!r}, skipping this check.",
                    file=sys.stderr,
                )
            else:
                if stars_val < min_stars:
                    return None

    # ---------- Optional download_count filter ----------
    if min_downloads is not None:
        dl = obj.get("download_count")
        if dl is not None:
            try:
                dl_val = int(dl)
            except (TypeError, ValueError):
                print(
                    f"[WARN] line {idx} download_count is not int: {dl!r}, skipping this check.",
                    file=sys.stderr,
                )
            else:
                if dl_val < min_downloads:
                    return None

    # ---------- Date filter ----------
    if cutoff_dt is None:
        return idx, obj

    time_str = obj.get(date_field)
    dt = parse_pkg_datetime(time_str)
    if dt is None:
        return None

    if dt > cutoff_dt:
        return idx, obj
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter pypi_trending.jsonl by stars, downloads, and date."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="15000github.jsonl",
        help="Input JSONL path (default: pypi_trending.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="pypi_trending.filtered.jsonl",
        help="Output JSONL path (default: pypi_trending.filtered.jsonl)",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=None,
        help="Minimum GitHub stars; records with missing stars are kept.",
    )
    parser.add_argument(
        "--min-downloads",
        type=int,
        default=None,
        help="Minimum download_count; records with missing download_count are kept.",
    )
    parser.add_argument(
        "--date-after",
        type=str,
        help=(
            "Keep records strictly newer than this date. "
            "Supports: 2024-04-01, 202404, 2404."
        ),
    )
    parser.add_argument(
        "--date-field",
        choices=["first_release_time", "last_release_time", "github_created_at"],
        default="github_created_at",
        help="Date field used for --date-after (default: github_created_at).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Thread workers for filtering (default: 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cutoff_dt: Optional[datetime] = None
    if args.date_after:
        try:
            cutoff_dt = parse_cutoff_datetime(args.date_after)
            print(
                f"[INFO] date-after : {cutoff_dt} ()",
                f"[INFO] date-after parsed as: {cutoff_dt}",
                file=sys.stderr,
            )
        except ValueError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

    if args.min_stars is not None:
        print(
            f"[INFO] Enforce GitHub stars >= {args.min_stars} (if field exists).",
            file=sys.stderr,
        )
    else:
        print("[INFO] --min-stars not set; skip stars filter.", file=sys.stderr)

    if args.min_downloads is not None:
        print(
            f"[INFO] Enforce download_count >= {args.min_downloads} (if field exists).",
            file=sys.stderr,
        )
    else:
        print("[INFO] --min-downloads not set; skip downloads filter.", file=sys.stderr)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            lines: List[str] = f.readlines()
    except OSError as e:
        print(f"[ERROR] Failed to read input file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loaded {len(lines)} lines. Filtering...", file=sys.stderr)

    filtered_results: List[Tuple[int, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                filter_record,
                (idx, line),
                args.min_stars,
                args.min_downloads,
                cutoff_dt,
                args.date_field,
            ): idx
            for idx, line in enumerate(lines)
        }

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                filtered_results.append(result)

    filtered_results.sort(key=lambda x: x[0])

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            for _, obj in filtered_results:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[ERROR] Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[INFO] Kept {len(filtered_results)} records. Wrote: {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
