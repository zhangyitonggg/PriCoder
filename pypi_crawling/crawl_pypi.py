#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect PyPI package metadata and export JSONL.

The crawler can use either:
- Top PyPI package list (default), or
- a custom package list via `--packages`.

For each package it records summary/version/release-time range/topics and
optionally GitHub repository + stars.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup  # dependency imported for compatibility

# --------- Constants ---------

USER_AGENT = "pypi-trending-crawler/0.1 (+https://example.com)"

# Hugo van Kemenade's Top PyPI Packages dataset
TOP_PYPI_URL = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.json"

# PyPI project JSON API
PYPI_JSON_PROJECT_URL = "https://pypi.org/pypi/{name}/json"

# GitHub repo API (stargazers_count)
GITHUB_API_REPO_URL = "https://api.github.com/repos/{owner}/{repo}"


# --------- Date parsing ---------

def normalize_datetime_str(s: Optional[str]) -> Optional[str]:
    """
    Normalize PyPI timestamp strings to `YYYY-MM-DD HH:MM:SS`.

    Supports values like `...Z` and `...+00:00`; output is UTC naive time.
    """
    if not s:
        return None
    try:
        raw = s.strip()
        # Convert trailing Z to UTC offset for fromisoformat
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"

        dt = datetime.fromisoformat(raw)

        # Convert to UTC and drop tzinfo
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # Keep original value on parse failure
        return s


# --------- Step 1: fetch package names ---------

def fetch_trending_project_names(limit: int = 50) -> List[str]:
    """
    Fetch package names from Top PyPI Packages JSON and keep first `limit`.

    Expected JSON shape:
    {
      "last_update": "...",
      "rows": [
        {"project": "boto3", "download_count": 1380415659},
        ...
      ]
    }
    """
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(TOP_PYPI_URL, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch Top PyPI Packages: {e}", file=sys.stderr)
        return []

    try:
        data = resp.json()
    except ValueError:
        print("[ERROR] Failed to parse Top PyPI Packages JSON.", file=sys.stderr)
        return []

    rows = data.get("rows", [])
    if not rows:
        print("[ERROR] Top PyPI Packages JSON has no rows.", file=sys.stderr)
        return []

    names: List[str] = []
    for row in rows[:limit]:
        name = row.get("project")
        if name:
            names.append(name)

    print(
        f"[INFO] Fetched {len(names)} package names from Top PyPI Packages.",
        file=sys.stderr,
    )
    return names


# --------- Step 2: fetch PyPI metadata ---------

def fetch_project_metadata(name: str) -> Optional[Dict]:
    """
    Fetch metadata JSON from PyPI API.
    """
    url = PYPI_JSON_PROJECT_URL.format(name=name)
    headers = {"User-Agent": USER_AGENT}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as e:
        print(f"[WARN] Request failed for {url}: {e}", file=sys.stderr)
        return None

    if resp.status_code == 404:
        print(f"[INFO] Package {name} not found on PyPI (404).", file=sys.stderr)
        return None

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"[WARN] HTTP error for {url}: {e}", file=sys.stderr)
        return None

    try:
        return resp.json()
    except ValueError:
        print(f"[WARN] Invalid JSON payload from {url}.", file=sys.stderr)
        return None


def extract_release_time_range(
    releases: Dict[str, List[Dict]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract first and last release timestamps from `releases`.

    Example `releases` shape:
        {
          "1.0.0": [ { "upload_time_iso_8601": "...", ... }, ... ],
          "1.1.0": [ ... ],
          ...
        }

    Returns:
        (first_release_time, last_release_time)
    in `YYYY-MM-DD HH:MM:SS` (UTC naive).
    """
    all_version_times: List[str] = []

    for files in releases.values():
        if not files:
            continue

        candidate_times: List[str] = []

        for file_info in files:
            t = file_info.get("upload_time_iso_8601") or file_info.get("upload_time")
            if not t:
                continue
            normalized = normalize_datetime_str(t)
            if normalized:
                candidate_times.append(normalized)

        if candidate_times:
            # Earliest file upload time in one version
            version_first = min(candidate_times)
            all_version_times.append(version_first)

    if not all_version_times:
        return None, None

    first_release = min(all_version_times)
    last_release = max(all_version_times)
    return first_release, last_release


def extract_topics(info: Dict) -> List[str]:
    """
        Extract Topic classifiers from `info.classifiers`.
    """
    classifiers = info.get("classifiers") or []
    topics = [c for c in classifiers if c.startswith("Topic ::")]
    return topics


# --------- Step 3: GitHub repo + stars ---------

def guess_github_repo_from_info(info: Dict) -> Optional[str]:
    """
    Infer `owner/repo` from project URLs or home page.
    Return None if no GitHub URL is found.
    """
    urls: List[str] = []

    project_urls = info.get("project_urls") or {}
    urls.extend(list(project_urls.values()))

    home_page = info.get("home_page")
    if home_page:
        urls.append(home_page)

    for url in urls:
        if not url:
            continue
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host not in ("github.com", "www.github.com"):
            continue

        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) < 2:
            continue

        owner, repo = parts[0], parts[1]
        # Skip non-repository pseudo paths
        if owner.lower() in ("issues", "pulls", "actions"):
            continue

        # Strip trailing .git
        if repo.endswith(".git"):
            repo = repo[:-4]

        return f"{owner}/{repo}"

    return None


def fetch_github_stars(repo_full_name: str, token: Optional[str] = None) -> Optional[int]:
    """
    Query GitHub API and return `stargazers_count`.
    """
    if not repo_full_name:
        return None

    owner, repo = repo_full_name.split("/", 1)
    url = GITHUB_API_REPO_URL.format(owner=owner, repo=repo)

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as e:
        print(f"[WARN] GitHub request failed for {url}: {e}", file=sys.stderr)
        return None

    if resp.status_code == 404:
        print(f"[INFO] GitHub repo {repo_full_name} not found (404).", file=sys.stderr)
        return None
    if resp.status_code == 403:
        print(f"[WARN] GitHub API rate-limited or forbidden (403), skip stars.", file=sys.stderr)
        return None

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"[WARN] GitHub HTTP error for {url}: {e}", file=sys.stderr)
        return None

    try:
        data = resp.json()
    except ValueError:
        print(f"[WARN] Invalid GitHub JSON response from {url}", file=sys.stderr)
        return None

    return data.get("stargazers_count")


# --------- Main crawl ---------

def crawl_pypi(
    packages: List[str],
    github_token: Optional[str] = None,
    delay: float = 0.2,
) -> List[Dict]:
    """
    Crawl metadata for all packages and return list of dict records.
    """
    results: List[Dict] = []

    for idx, name in enumerate(packages, start=1):
        print(f"[INFO] ({idx}/{len(packages)}) processing: {name}", file=sys.stderr)

        meta = fetch_project_metadata(name)
        if not meta:
            continue

        info = meta.get("info") or {}
        releases = meta.get("releases") or {}

        first_release, last_release = extract_release_time_range(releases)
        topics = extract_topics(info)

        github_repo = guess_github_repo_from_info(info)
        github_stars: Optional[int] = None
        if github_repo:
            github_stars = fetch_github_stars(github_repo, token=github_token)
        else:
            print(
                f"[INFO] {name} has no GitHub repo; skip stars lookup.",
                file=sys.stderr,
            )

        pkg_entry = {
            "name": info.get("name") or name,
            "summary": info.get("summary"),
            "latest_version": info.get("version"),
            # normalized format: YYYY-MM-DD HH:MM:SS
            "first_release_time": first_release,
            "last_release_time": last_release,
            "topics": topics,
            "github_repo": github_repo,
            "github_stars": github_stars,
        }
        results.append(pkg_entry)

        if delay > 0:
            time.sleep(delay)

    return results


# --------- CLI ---------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl PyPI packages and export metadata JSONL."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="pypi_trending.jsonl",
        help="Output JSONL path (default: pypi_trending.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Top packages to fetch (default: 50). Ignored when --packages is provided.",
    )
    parser.add_argument(
        "--packages",
        nargs="*",
        help="Explicit package names; bypasses Top PyPI source.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Sleep seconds between packages (default: 0.2).",
    )
    parser.add_argument(
        "--no-github",
        action="store_true",
        help="Disable GitHub API calls (only collect PyPI metadata).",
    )
    parser.add_argument(
        "--github-token",
        help="GitHub Personal Access Token. Fallback env: GITHUB_TOKEN.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.packages:
        packages = args.packages
        print(
            f"[INFO] Using explicit package list, count={len(packages)}.",
            file=sys.stderr,
        )
    else:
        print(
            f"[INFO] Fetching Top PyPI Packages (limit={args.limit})...",
            file=sys.stderr,
        )
        packages = fetch_trending_project_names(limit=args.limit)

    if not packages:
        print("[ERROR] No packages available to crawl.", file=sys.stderr)
        sys.exit(1)

    github_token = None
    if not args.no_github:
        github_token = args.github_token or os.getenv("GITHUB_TOKEN")

    packages_data = crawl_pypi(
        packages=packages,
        github_token=github_token,
        delay=args.delay,
    )

    # Write JSONL output
    with open(args.output, "w", encoding="utf-8") as f:
        for pkg in packages_data:
            line = json.dumps(pkg, ensure_ascii=False)
            f.write(line + "\n")

    print(
        f"[INFO] Wrote {len(packages_data)} records to {args.output} (JSONL).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
