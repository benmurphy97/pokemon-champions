"""Turn the imported tournament pastes (import_teams.py) into an opponent pool for training.

    PYTHONPATH=src python3 -m champions.web_teams

Keeps legal teams only. Writes results/web_teams.json with, for each team, its species, exact
text, whether it carries Trick Room, and the tournament/placement it came from.
"""

import json
from pathlib import Path

from champions.config import REGULATION
from champions.teams import parse_team

ROOT = Path(__file__).resolve().parents[2]
TEAM_DIR = ROOT / "data" / "teams" / REGULATION.lower()
OUT = ROOT / "results" / "web_teams.json"


def build_pool() -> list[dict]:
    index = json.loads((TEAM_DIR / "index.json").read_text())
    pool = []
    for entry in index:
        if entry.get("skipped") or entry.get("problems"):
            continue
        text = (TEAM_DIR / f"{entry['id']}.txt").read_text()
        pool.append({"id": f"web-{entry['id']}", "species": [m["name"] for m in parse_team(text)], "text": text,
                     "trick_room": entry["trick_room"], "tournament": entry["tournament"],
                     "placement": entry["placement"], "date": entry["date"]})
    return pool


def main():
    pool = build_pool()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(pool))
    print(f"{len(pool)} legal tournament teams, {sum(t['trick_room'] for t in pool)} with Trick Room → {OUT}")


if __name__ == "__main__":
    main()
