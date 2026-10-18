"""Rebuild the real teams players brought to the ladder, from the downloaded replays.

    .venv/bin/python -m champions.teams.ladder_pool

Each side of each game gives one team: its six Pokémon, with the sets reconstructed from the whole
match (hindsight.py). Teams are kept only if every Pokémon is in the ladder data and the team is
legal under the format's rules. These are coherent teams of every kind of strategy, which makes
them good opponents. Games are split by a hash of the game id, so teams in the evaluation file
never come from a game used for training.

Output: results/ladder_teams_train.json and results/ladder_teams_eval.json.
"""

import argparse
import gzip
import json
import zlib
from pathlib import Path

from champions.model.battle_state import BattleState
from champions.training.hindsight import complete_sets
from champions.teams.generate import load_usage, usage_key
from champions.sim.simulate import validate_many
from champions.teams.parse import serialise_team
from champions.model.vocab import REFERENCE

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "cache" / "raw"
RESULTS = ROOT / "results"
POINTS = {"HP": "HP", "Atk": "Atk", "Def": "Def", "SpA": "SpA", "SpD": "SpD", "Spe": "Spe"}
STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]


def team_from_state(state: BattleState, side: str, usage: dict, reference: dict) -> list[dict] | None:
    """The side's six as Showdown sets, or None if any Pokémon is missing from the ladder data."""
    info = complete_sets(state, side, usage)
    team, taken = [], set()
    for index, mon in enumerate(state.sides[side].team[:6]):
        key = usage_key(mon.species, mon.mega, usage)
        if key is None or key not in reference or index not in info:
            return None
        known = info[index]
        entry = usage[key]
        item = known["item"]
        if item in taken or item is None:      # item clause: fall back to this Pokémon's next most common item
            item = next((name for name, _ in entry["items"] if name not in taken), None)
        taken.add(item)
        nature, spread, _ = entry["spreads"][0]
        team.append({"name": key, "item": item, "ability": known["ability"] or entry["abilities"][0][0],
                     "nature": nature, "evs": {s: spread[s] for s in STATS},
                     "moves": list(dict.fromkeys(known["moves"]))[:4]})
    return team if len(team) == 6 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-rating", type=int, default=1300)
    args = parser.parse_args()

    usage = load_usage()
    reference = json.loads((REFERENCE / "pokemon.json").read_text())
    candidates, seen = [], set()
    games = skipped = 0
    for path in sorted(RAW_DIR.glob("*.json.gz")):
        replay = json.load(gzip.open(path, "rt"))
        if (replay.get("rating") or 0) < args.min_rating:
            continue
        games += 1
        state = BattleState()
        for line in replay["log"].split("\n"):
            state.update(line)
        for side in ("p1", "p2"):
            team = team_from_state(state, side, usage, reference)
            if team is None:
                skipped += 1
                continue
            signature = tuple(sorted((m["name"], tuple(sorted(m["moves"]))) for m in team))
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append({"id": f"{replay['id']}-{side}", "game": replay["id"], "team": team})

    texts = [serialise_team(c["team"]) for c in candidates]
    problems = validate_many(texts)
    train, evaluation = [], []
    for c, text, issues in zip(candidates, texts, problems):
        if issues:
            continue
        entry = {"id": c["id"], "species": [m["name"] for m in c["team"]], "text": text,
                 "trick_room": any("Trick Room" in m["moves"] for m in c["team"])}
        (evaluation if zlib.crc32(c["game"].encode()) % 5 == 0 else train).append(entry)

    RESULTS.joinpath("ladder_teams_train.json").write_text(json.dumps(train))
    RESULTS.joinpath("ladder_teams_eval.json").write_text(json.dumps(evaluation))
    legal = len(train) + len(evaluation)
    print(f"{games} games -> {len(candidates)} distinct teams ({skipped} sides skipped: Pokémon missing from ladder data); "
          f"{legal} legal ({len(train)} train, {len(evaluation)} eval); "
          f"{sum(t['trick_room'] for t in train + evaluation)} carry Trick Room")


if __name__ == "__main__":
    main()
