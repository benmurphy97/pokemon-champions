"""Collect tournament teams from Limitless VGC for the current regulation.

    PYTHONPATH=src python3 -m champions.scrapers.limitless 445          # Regional Louisville, KY

A Limitless team page lists species, item, ability, nature and moves but NOT stat points, so each
Pokémon gets an estimated spread that fits its nature and moves (champions.teams.spreads: in-game Ranked
Battle data first). Those teams are
marked "estimated_spreads" in data/teams/<regulation>/index.json: the sets are real, the spreads are
a guess. Writes data/teams/<regulation>/lim-<team id>.txt beside the MetaVGC imports.
Player names are not kept. Requests are one per second.
"""

import argparse
import html
import json
import re
import subprocess
import time
from datetime import datetime

from champions.config import REGULATION
from champions.scrapers.metavgc import INDEX, OUT_DIR
from champions.teams.spreads import estimate
from champions.teams.parse import serialise_team

SITE = "https://limitlessvgc.com"
USER_AGENT = "champions-team-research/0.1 (personal project; 1 request/second)"


def fetch(url: str) -> str:
    run = subprocess.run(["curl", "-s", "--fail", "--max-time", "60", "-A", USER_AGENT, url], capture_output=True, text=True)
    if run.returncode:
        raise OSError(f"curl failed ({run.returncode}) for {url}")
    return run.stdout


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def standings(page: str) -> list[tuple[int, int]]:
    """[(placement, team id)] from the tournament page."""
    rows = re.findall(r'<tr data-rank="(\d+)".*?href="/teams/(\d+)"', page, flags=re.S)
    return [(int(rank), int(team)) for rank, team in rows]


def parse_page(page: str) -> list[dict]:
    mons = []
    for block in page.split('<div class="pkmn"')[1:]:
        grab = lambda pattern: (m.group(1).strip() if (m := re.search(pattern, block, flags=re.S)) else None)
        name = html.unescape(grab(r'<div class="name">\s*<a[^>]*>(.*?)</a>') or "")
        slug = grab(r'data-id="([^"]+)"') or ""
        if "-" in slug and slug.replace("-", " ") != name.lower():   # a form, which the display name spells out ("Eternal Flower Floette")
            name = "-".join(part.capitalize() for part in slug.split("-"))
        nature = grab(r'<div class="nature">(\w+) Nature')
        ability = grab(r'<div class="ability">Ability:\s*(.*?)</div>')
        moves = [html.unescape(m.strip()) for m in re.findall(r"<li>(.*?)</li>", block.split('<ul class="moves">')[-1].split("</ul>")[0])]
        nature, points = estimate(name, nature, moves)
        mons.append({"name": name, "item": html.unescape(grab(r'<div class="item">(.*?)</div>') or "") or None,
                     "ability": html.unescape(ability or ""), "nature": nature, "evs": points, "moves": moves})
    return mons


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tournament", type=int, help="Limitless tournament number, e.g. 445")
    parser.add_argument("--top", type=int, default=200, help="import at most this many placings")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX.read_text()) if INDEX.exists() else []
    seen = {entry["source"] for entry in index}

    page = fetch(f"{SITE}/tournaments/{args.tournament}")
    title = re.search(r"<title>(.*?)</title>", page, flags=re.S)
    if f"Regulation Set {REGULATION}" not in page:
        raise SystemExit(f"tournament {args.tournament} is not {REGULATION}")
    day = re.search(r"(\d{1,2})(?:st|nd|rd|th) ([A-Z][a-z]+) (\d{4})", page)
    name = re.sub(r"\s*[–-]\s*Limitless VGC", "", title.group(1)).strip() if title else ""
    tournament = f"{html.unescape(name)} (Limitless #{args.tournament})"
    date = datetime.strptime(" ".join(day.groups()), "%d %B %Y").strftime("%Y-%m-%d") if day else None
    added = 0
    for rank, team in standings(page)[: args.top]:
        source = f"{SITE}/teams/{team}"
        if source in seen:
            continue
        time.sleep(0.8)
        mons = parse_page(fetch(source))
        text = serialise_team(mons)
        team_id = f"lim-{team}"
        (OUT_DIR / f"{team_id}.txt").write_text(text)
        index.append({"id": team_id, "source": source, "tournament": tournament, "date": date, "placement": ordinal(rank),
                      "trick_room": "- Trick Room" in text, "estimated_spreads": True})
        seen.add(source); added += 1
    from champions.sim.simulate import validate_many
    fresh = [e for e in index if e.get("estimated_spreads") and "problems" not in e]
    for entry, issues in zip(fresh, validate_many([(OUT_DIR / f"{e['id']}.txt").read_text() for e in fresh])):
        entry["problems"] = issues
    INDEX.write_text(json.dumps(index, indent=1) + "\n")
    legal = [e for e in index if e.get("estimated_spreads") and not e["problems"]]
    print(f"{added} new teams from {tournament}; {len(legal)} legal Limitless teams, {sum(e['trick_room'] for e in legal)} with Trick Room")
    for e in fresh:
        if e["problems"]:
            print("  illegal", e["id"], e["problems"][:2])


if __name__ == "__main__":
    main()
