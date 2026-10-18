"""Collect tournament teams from Limitless VGC for the current regulation.

    PYTHONPATH=src python3 -m champions.import_limitless 445          # Regional Louisville, KY

A Limitless team page lists species, item, ability, nature and moves but NOT stat points, so each
Pokémon gets the most popular ladder spread for its nature (data/meta usage). Those teams are
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

from champions.config import REGULATION, REGULATION_START
from champions.import_teams import INDEX, OUT_DIR
from champions.meta import load_usage
from champions.teams import serialise_team

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


def estimate_spread(usage: dict, name: str, nature: str | None) -> dict:
    """Most popular ladder spread for this Pokémon and nature, else its most popular spread."""
    spreads = usage.get(name, {}).get("spreads") or usage.get(name.split("-")[0], {}).get("spreads") or []
    for spread_nature, points, _ in spreads:
        if spread_nature == nature:
            return points
    return spreads[0][1] if spreads else {"HP": 32, "Atk": 0, "Def": 0, "SpA": 0, "SpD": 0, "Spe": 2}


def parse_page(page: str, usage: dict) -> list[dict]:
    mons = []
    for block in page.split('<div class="pkmn"')[1:]:
        grab = lambda pattern: (m.group(1).strip() if (m := re.search(pattern, block, flags=re.S)) else None)
        name = html.unescape(grab(r'<div class="name">\s*<a[^>]*>(.*?)</a>') or "")
        slug = grab(r'data-id="([^"]+)"') or ""
        if "-" in slug and slug.replace("-", " ") != name.lower():   # a form, which the display name spells out ("Eternal Flower Floette")
            name = "-".join(part.capitalize() for part in slug.split("-"))
        nature = grab(r'<div class="nature">(\w+) Nature')
        ability = grab(r'<div class="ability">Ability:\s*(.*?)</div>')
        mons.append({"name": name, "item": html.unescape(grab(r'<div class="item">(.*?)</div>') or "") or None,
                     "ability": html.unescape(ability or ""), "nature": nature,
                     "evs": estimate_spread(usage, name, nature),
                     "moves": [html.unescape(m.strip()) for m in re.findall(r"<li>(.*?)</li>", block.split('<ul class="moves">')[-1].split("</ul>")[0])]})
    return mons


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tournament", type=int, help="Limitless tournament number, e.g. 445")
    parser.add_argument("--top", type=int, default=200, help="import at most this many placings")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX.read_text()) if INDEX.exists() else []
    seen = {entry["source"] for entry in index}
    usage = load_usage()

    page = fetch(f"{SITE}/tournaments/{args.tournament}")
    title = re.search(r"<title>(.*?)</title>", page, flags=re.S)
    date = re.search(r"(\d{4}-\d{2}-\d{2})", page)
    name = re.sub(r"\s*[–-]\s*Limitless VGC", "", title.group(1)).strip() if title else ""
    tournament = f"{html.unescape(name)} (Limitless #{args.tournament})"
    date = (date.group(1) if date else None) or "2026-10-10"
    added = 0
    for rank, team in standings(page)[: args.top]:
        source = f"{SITE}/teams/{team}"
        if source in seen:
            continue
        time.sleep(0.8)
        mons = parse_page(fetch(source), usage)
        text = serialise_team(mons)
        team_id = f"lim-{team}"
        (OUT_DIR / f"{team_id}.txt").write_text(text)
        index.append({"id": team_id, "source": source, "tournament": tournament, "date": date, "placement": ordinal(rank),
                      "trick_room": "- Trick Room" in text, "estimated_spreads": True})
        seen.add(source); added += 1
    from champions.simulate import validate_many
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
