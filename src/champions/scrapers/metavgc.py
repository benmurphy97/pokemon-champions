"""Collect real tournament teams for the current regulation from MetaVGC + pokepast.es.

    PYTHONPATH=src python3 -m champions.scrapers.metavgc --limit 700

MetaVGC's sitemap lists every team page; each page links to the exact Showdown paste. Writes
data/teams/<regulation>/<paste-id>.txt and an index.json recording the tournament, date and placement
of each team (player names are not kept), whether it carries Trick Room, and whether it is legal
under the format's rules. Re-running skips teams already imported.

Requests go through curl at about one per second (robots.txt allows crawling). Python's own
downloader stalled on this site's large sitemaps, curl did not.
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from champions.config import REGULATION, REGULATION_START

SITE = "https://metavgc.com"
SITEMAPS = [f"{SITE}/sitemap-teams/{n}" for n in range(4)]
USER_AGENT = "champions-team-research/0.1 (personal project; 1 request/second)"
DELAY_SECONDS = 0.7

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "teams" / REGULATION.lower()
INDEX = OUT_DIR / "index.json"
URL_CACHE = OUT_DIR / "urls.json"   # the sitemaps are ~14 MB, so download them once
SITE_REGULATION_ID = "pokemonchampions-reg" + REGULATION.split("-")[1].lower()   # e.g. pokemonchampions-regc


def fetch(url: str, timeout: int = 90) -> str:
    run = subprocess.run(["curl", "-s", "--fail", "--max-time", str(timeout), "-A", USER_AGENT, url],
                         capture_output=True, text=True)
    if run.returncode:
        raise OSError(f"curl failed ({run.returncode}) for {url}")
    return run.stdout


def team_urls(refresh: bool = False) -> list[str]:
    """Team page URLs whose slug names this regulation (e.g. '...reg-m-c...'), cached on disk."""
    if URL_CACHE.exists() and not refresh and json.loads(URL_CACHE.read_text()):
        return json.loads(URL_CACHE.read_text())
    tag = f"reg-{REGULATION.lower()}"
    urls = []
    for sitemap in SITEMAPS:
        try:
            xml = fetch(sitemap, timeout=180)
        except OSError:
            continue
        urls += [u for u in re.findall(r"<loc>([^<]+)</loc>", xml) if tag in u.lower()]
        time.sleep(DELAY_SECONDS)
    if urls:   # never cache an empty result from a failed download
        URL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        URL_CACHE.write_text(json.dumps(urls, indent=0) + "\n")
    return urls


def team_record(html: str, slug: str) -> str:
    """The slice of embedded page JSON holding this team's own fields (the page also embeds other teams)."""
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


def placement(title: str | None) -> str | None:
    """'#1st - Someone at Some Event' -> '1st'. The player's name is discarded."""
    match = re.match(r"#(\S+)", title or "")
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
    text = fetch(f"{paste_url.rstrip('/')}/raw").replace("\r\n", "\n")
    (OUT_DIR / f"{paste_id}.txt").write_text(text)
    return {"id": paste_id, "source": url, "pokepaste": paste_url, "tournament": field(record, "tournament"),
            "date": date, "placement": placement(field(record, "title")), "trick_room": "- Trick Room" in text}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=700, help="max new teams to import this run")
    parser.add_argument("--refresh-urls", action="store_true", help="re-download the sitemaps (slow)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX.read_text()) if INDEX.exists() else []
    seen = {entry["source"] for entry in index}
    todo = [u for u in team_urls(args.refresh_urls) if u not in seen]
    print(f"{len(todo)} new {REGULATION} team pages; importing up to {args.limit}", flush=True)

    added = 0
    for url in todo:
        if added >= args.limit:
            break
        try:
            entry = import_team(url)
        except OSError as err:
            print(f"  skip {url}: {err}", flush=True)
            time.sleep(DELAY_SECONDS)
            continue
        time.sleep(DELAY_SECONDS)
        index.append(entry or {"source": url, "skipped": True})   # remember skipped pages so they are not refetched
        added += 1
        if added % 25 == 0:
            INDEX.write_text(json.dumps(index, indent=1) + "\n")
            print(f"  {added} pages done, {sum(1 for e in index if not e.get('skipped'))} teams, "
                  f"{sum(1 for e in index if e.get('trick_room'))} with Trick Room", flush=True)
    INDEX.write_text(json.dumps(index, indent=1) + "\n")

    # Check every paste against the format's rules in one go.
    from champions.sim.simulate import validate_many
    teams = [e for e in index if not e.get("skipped")]
    problems = validate_many([(OUT_DIR / f"{e['id']}.txt").read_text() for e in teams])
    for entry, issues in zip(teams, problems):
        entry["problems"] = issues
    INDEX.write_text(json.dumps(index, indent=1) + "\n")
    legal = [e for e in teams if not e["problems"]]
    print(f"done: {len(teams)} teams, {len(legal)} legal, {sum(1 for e in legal if e['trick_room'])} legal with Trick Room → {INDEX}")


if __name__ == "__main__":
    main()
