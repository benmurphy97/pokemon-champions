"""
fetch_replays.py — Download Showdown replay logs for gen9championsvgc2026regma.

Usage:
    python src/data/imitation/fetch_replays.py --out-dir data/replays --pages 10

This will:
  1. Hit the Showdown replay search API to list replays for the format
  2. Download each replay's raw log to --out-dir as <replay-id>.log
  3. Skip replays already downloaded (safe to re-run)
"""

import argparse
import json
import os
import time
import urllib.request
import urllib.parse
from pathlib import Path

FORMAT = "gen9championsvgc2026regma"
SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
LOG_URL    = "https://replay.pokemonshowdown.com/{replay_id}.log"
DEFAULT_DELAY   = 0.5   # seconds between requests
DEFAULT_TIMEOUT = 30    # seconds per request
MAX_RETRIES     = 3     # retries on timeout/error before skipping


def fetch_json(url: str, timeout: int = DEFAULT_TIMEOUT) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "pokemon-champions-rl/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "pokemon-champions-rl/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_with_retry(url: str, timeout: int = DEFAULT_TIMEOUT, retries: int = MAX_RETRIES) -> str:
    """Fetch URL text with exponential backoff retries on failure."""
    for attempt in range(1, retries + 1):
        try:
            return fetch_text(url, timeout=timeout)
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"    attempt {attempt}/{retries} failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def list_replay_ids(format_str: str, n_pages: int, delay: float) -> list[str]:
    """Page through the search API and collect replay IDs."""
    ids: list[str] = []
    before = None

    for page in range(n_pages):
        params: dict = {"format": format_str, "output": "json"}
        if before is not None:
            params["before"] = str(before)

        url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
        print(f"  [page {page + 1}/{n_pages}] GET {url}")

        try:
            data = fetch_json(url)
        except Exception as e:
            print(f"  WARNING: failed to fetch page {page + 1}: {e}")
            break

        if not data:
            print("  No more results.")
            break

        for entry in data:
            replay_id = entry.get("id")
            if replay_id:
                ids.append(replay_id)
                before = entry.get("uploadtime", before)

        time.sleep(delay)

    return ids


def download_replays(
    out_dir: Path,
    format_str: str = FORMAT,
    n_pages: int = 20,
    delay: float = DEFAULT_DELAY,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[Path]:
    """Download all replay logs and return paths to saved files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing replays for format: {format_str}")
    replay_ids = list_replay_ids(format_str, n_pages, delay)
    print(f"Found {len(replay_ids)} replay IDs across {n_pages} page(s).")

    saved: list[Path] = []
    skipped = 0

    for i, replay_id in enumerate(replay_ids):
        out_path = out_dir / f"{replay_id}.log"

        if out_path.exists():
            skipped += 1
            saved.append(out_path)
            continue

        url = LOG_URL.format(replay_id=replay_id)
        try:
            log_text = fetch_with_retry(url)
            out_path.write_text(log_text, encoding="utf-8")
            saved.append(out_path)
            print(f"  [{i + 1}/{len(replay_ids)}] saved {out_path.name}  ({len(log_text):,} bytes)")
        except Exception as e:
            print(f"  [{i + 1}/{len(replay_ids)}] WARN: skipping {replay_id} after retries: {e}")

        time.sleep(delay)

    print(f"\nDone. {len(saved)} logs saved ({skipped} already existed) in {out_dir}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Showdown replay logs")
    parser.add_argument("--out-dir", default="data/replays", help="Directory to save .log files")
    parser.add_argument("--pages", type=int, default=20, help="Number of search result pages (~50 replays each)")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between HTTP requests")
    parser.add_argument("--format", default=FORMAT, help="Showdown format string")
    args = parser.parse_args()

    download_replays(
        out_dir=Path(args.out_dir),
        format_str=args.format,
        n_pages=args.pages,
        delay=args.delay,
    )
