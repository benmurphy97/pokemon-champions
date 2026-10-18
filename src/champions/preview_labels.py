"""Team-preview labels from a replay: which four each player brought and which two led.

    PYTHONPATH=src .venv/bin/python -m champions.preview_labels --games 400     # pilot: check the labels

A log shows both sides' six Pokémon (in the order they were listed), the two that start on the
field, and every Pokémon that ever appeared. A brought Pokémon that never appears cannot be seen,
so a side's four may be only partly known: `brought_seen` holds the ones seen and `complete` says
whether all four were. Training should use complete sides for the bring-four labels and any side
for the leads.
"""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from champions.battle_state import BattleState

RAW_DIR = Path(__file__).resolve().parents[2] / "cache" / "raw"


def extract_preview(lines: list[str]) -> dict[str, dict]:
    """{side: {"six": names in listed order, "leads": [i, j], "brought_seen": [indices], "complete": bool}}"""
    state, leads = BattleState(), {}
    for line in lines:
        if line.startswith("|turn|") and not leads:
            leads = {side: [i for i in state.sides[side].active if i is not None] for side in ("p1", "p2")}
        state.update(line)
    out = {}
    for side in ("p1", "p2"):
        team = state.sides[side].team[:6]
        seen = [i for i, mon in enumerate(team) if mon.revealed]
        out[side] = {"six": [m.species for m in team], "leads": leads.get(side, []), "brought_seen": seen,
                     "complete": len(seen) == 4 and len(team) == 6}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=400)
    args = parser.parse_args()
    seen_counts, lead_ok, sides, examples = Counter(), 0, 0, []
    for path in sorted(RAW_DIR.glob("*.json.gz"))[: args.games]:
        replay = json.load(gzip.open(path, "rt"))
        for side, info in extract_preview(replay["log"].split("\n")).items():
            sides += 1
            seen_counts[len(info["brought_seen"])] += 1
            lead_ok += len(info["leads"]) == 2 and all(i in info["brought_seen"] for i in info["leads"]) and len(info["six"]) == 6
            if len(examples) < 3 and info["complete"]:
                examples.append({k: info[k] for k in ("six", "leads", "brought_seen")})
    print(f"{sides} sides from {args.games} games")
    for n in sorted(seen_counts):
        print(f"  {n} brought Pokémon seen: {seen_counts[n]} sides ({seen_counts[n] / sides:.1%})")
    print(f"  two leads found and among the seen: {lead_ok} sides ({lead_ok / sides:.1%})")
    for e in examples:
        print("  e.g.", [e["six"][i] for i in e["leads"]], "lead;", [e["six"][i] for i in e["brought_seen"]], "brought, of", e["six"])


if __name__ == "__main__":
    main()
