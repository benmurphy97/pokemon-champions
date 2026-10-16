"""Collect real tournament teams for the current regulation from MetaVGC + pokepast.es.

    PYTHONPATH=src python3 -m champions.import_teams --limit 50

Writes data/teams/<regulation>/<paste-id>.txt (exact Showdown paste) and an index.json
recording where each team came from and whether it passes Showdown's M-C validation.
Re-running skips teams already imported. Requests are spaced out (robots.txt allows crawling).
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from champions.config import REGULATION, REGULATION_START
from champions.simulate import validate_team

SITE = "https://metavgc.com"
SITEMAPS = [f"{SITE}/sitemap-teams/{n}" for n in range(4)]
USER_AGENT = "champions-team-research/0.1 (personal project; 1 request/second)"
DELAY_SECONDS = 1.0

SITE_REGULATION_ID = "pokemonchampions-reg" + REGULATION.split("-")[1].lower()  # e.g. pokemonchampions-regc

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "teams" / REGULATION.lower()
INDEX = OUT_DIR / "index.json"
URL_CACHE = OUT_DIR / "urls.json"  # the sitemaps are ~14 MB and slow, so download them once


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def team_urls(refresh: bool = False) -> list[str]:
    """Team page URLs whose slug names this regulation (e.g. '...reg-m-c...'), cached on disk."""
    if URL_CACHE.exists() and not refresh:
        return json.loads(URL_CACHE.read_text())

    tag = f"reg-{REGULATION.lower()}"
    urls = []
    for sitemap in SITEMAPS:
        try:
            xml = fetch(sitemap)
        except OSError:
            continue
        urls += [u for u in re.findall(r"<loc>([^<]+)</loc>", xml) if tag in u.lower()]
        time.sleep(DELAY_SECONDS)
    URL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    URL_CACHE.write_text(json.dumps(urls, indent=0) + "\n")
    return urls


def team_record(html: str, slug: str) -> str:
    """The slice of embedded page JSON holding this team's own fields.

    The page also embeds other teams, so read from this team's publicId up to the next one.
    """
    key = '\\"publicId\\":\\"'
    at = html.find(f"{key}{slug}")
    if at < 0:
        return ""
    end = html.find(key, at + len(key) + len(slug))
    return html[at: end if end > 0 else at + 80000]


def field(record: str, key: str) -> str | None:
    """Read a string field from page JSON, where quotes appear escaped as \\"."""
    match = re.search(rf'\\"{key}\\":\\"(.*?)\\"', record)
    return match.group(1) if match else None


def import_team(url: str) -> dict | None:
    html = fetch(url)
    record = team_record(html, url.rstrip("/").rsplit("/", 1)[-1])
    paste_url = field(record, "pokepaste")
    date = (field(record, "tournamentDate") or "")[:10]
    if not paste_url or field(record, "regulation") != SITE_REGULATION_ID or date < REGULATION_START:
        return None

    time.sleep(DELAY_SECONDS)
    paste_id = paste_url.rstrip("/").rsplit("/", 1)[-1]
    text = fetch(f"{paste_url.rstrip('/')}/raw")
    (OUT_DIR / f"{paste_id}.txt").write_text(text.replace("\r\n", "\n"))

    return {
        "id": paste_id,
        "source": url,
        "pokepaste": paste_url,
        "player": field(record, "player"),
        "tournament": field(record, "tournament"),
        "date": date,
        "title": field(record, "title"),
        "record": field(record, "record"),
        "problems": validate_team(text),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=50, help="max new teams to import this run")
    parser.add_argument("--refresh-urls", action="store_true", help="re-download the sitemaps (slow)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX.read_text()) if INDEX.exists() else []
    seen_sources = {entry["source"] for entry in index}

    todo = [u for u in team_urls(args.refresh_urls) if u not in seen_sources]
    print(f"{len(todo)} new {REGULATION} team pages found; importing up to {args.limit}")

    added = 0
    for url in todo:
        if added >= args.limit:
            break
        try:
            entry = import_team(url)
        except (OSError, ValueError) as err:
            print(f"  skip {url}: {err}")
            continue
        finally:
            time.sleep(DELAY_SECONDS)
        # Record skipped pages too (entry None) so they aren't refetched next run.
        index.append(entry or {"source": url, "skipped": True})
        added += 1
        INDEX.write_text(json.dumps(index, indent=1) + "\n")
        if entry:
            status = "ok" if not entry["problems"] else f"{len(entry['problems'])} problems"
            print(f"  {entry['id']}  {entry['player']} @ {entry['tournament']}  [{status}]")

    print(f"done: {sum(1 for e in index if not e.get('skipped'))} teams in {INDEX}")


if __name__ == "__main__":
    main()
