"""Pick the small set of known-good Trick Room teams the player is trained to pilot.

    PYTHONPATH=src .venv/bin/python -m champions.known_teams --count 10

Candidates are legal tournament Trick Room teams that finished in the top 8. They are ranked by
their simulated record (rate_teams.py, low end of the 95% range) and chosen greedily, skipping any
team that shares more than --max-shared Pokémon with one already chosen so the set covers
different ideas. Writes data/teams/<regulation>/known_good_tr.json.
"""

import argparse
import json
import re
from pathlib import Path

from champions.arena import wilson
from champions.config import REGULATION
from champions.rate_teams import RATINGS
from champions.web_teams import build_pool

OUT = Path(__file__).resolve().parents[2] / "data" / "teams" / REGULATION.lower() / "known_good_tr.json"


def finished_top8(placement: str | None) -> bool:
    match = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", placement or "")
    return bool(match) and int(match.group(1)) <= 8


def choose(count: int, max_shared: int) -> list[dict]:
    ratings = json.loads(RATINGS.read_text())["teams"]
    candidates = [t for t in build_pool() if t["trick_room"] and finished_top8(t["placement"]) and t["id"] in ratings]
    candidates.sort(key=lambda t: -wilson(ratings[t["id"]]["wins"], ratings[t["id"]]["games"])[0])
    chosen = []
    for team in candidates:
        if all(len(set(team["species"]) & set(c["species"])) <= max_shared for c in chosen):
            chosen.append(team)
        if len(chosen) == count:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-shared", type=int, default=3)
    args = parser.parse_args()
    teams = choose(args.count, args.max_shared)
    OUT.write_text(json.dumps(teams, indent=1) + "\n")
    for t in teams:
        print(f"{t['placement']:>4}  {t['tournament'][:34]:34}  {' / '.join(t['species'])}")
    print(f"{len(teams)} teams → {OUT}")


if __name__ == "__main__":
    main()
