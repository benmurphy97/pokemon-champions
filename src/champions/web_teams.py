"""Turn the imported tournament pastes (import_teams.py) into an opponent pool for training.

    PYTHONPATH=src python3 -m champions.web_teams

Keeps legal teams only. Writes results/web_teams.json with, for each team, its species, exact
text, whether it carries Trick Room, the tournament/placement it came from, whether it counts as
"good" (finished in the top 8, or won at least 70% of at least three rounds), and a train/eval
split (by a hash of the paste id) so teams used to test the agent are never trained against.
"""

import json
import re
import zlib
from pathlib import Path

from champions.config import REGULATION
from champions.teams import parse_team

ROOT = Path(__file__).resolve().parents[2]
TEAM_DIR = ROOT / "data" / "teams" / REGULATION.lower()
OUT = ROOT / "results" / "web_teams.json"


def is_good(placement: str | None) -> bool:
    """Top-8 finish ('3rd'), or a winning record ('4-1-0' means 4 wins, 1 loss, 0 draws)."""
    if not placement:
        return False
    ordinal = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", placement)
    if ordinal:
        return int(ordinal.group(1)) <= 8
    record = re.fullmatch(r"(\d+)-(\d+)-(\d+)", placement)
    if record:
        wins, losses, draws = map(int, record.groups())
        rounds = wins + losses + draws
        return rounds >= 3 and wins / rounds >= 0.7
    return False


def build_pool() -> list[dict]:
    index = json.loads((TEAM_DIR / "index.json").read_text())
    pool, seen = [], set()
    for entry in index:
        if entry.get("skipped") or entry.get("problems"):
            continue
        text = (TEAM_DIR / f"{entry['id']}.txt").read_text()
        signature = tuple(sorted((m["name"], tuple(sorted(m["moves"]))) for m in parse_team(text)))
        if signature in seen:      # the same team is often posted to several brackets
            continue
        seen.add(signature)
        pool.append({"id": f"web-{entry['id']}", "species": [m["name"] for m in parse_team(text)], "text": text,
                     "trick_room": entry["trick_room"], "tournament": entry["tournament"],
                     "placement": entry["placement"], "date": entry["date"], "good": is_good(entry["placement"]),
                     "split": "eval" if zlib.crc32(entry["id"].encode()) % 5 == 0 else "train"})
    return pool


def main():
    pool = build_pool()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(pool))
    good = [t for t in pool if t["good"]]
    print(f"{len(pool)} legal tournament teams ({sum(t['trick_room'] for t in pool)} with Trick Room); "
          f"{len(good)} good ({sum(t['trick_room'] for t in good)} with Trick Room) → {OUT}")


if __name__ == "__main__":
    main()
