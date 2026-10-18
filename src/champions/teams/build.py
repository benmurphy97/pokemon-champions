"""Build teams from in-game Ranked Battle usage (scrapers/battledata.py).

    PYTHONPATH=src .venv/bin/python -m champions.teams.build --teams 300 --seed 1

The first Pokémon is drawn by in-game usage rank; each next one is drawn by rank too, boosted when
it is listed among the teammates of those already chosen (higher on that list counts more). Each
gets a coherent version of its most popular set:
  * ability and item: the most used, with no item twice on a team and at most two Mega Stones;
  * moves: the four most used, except that a Choice item or Assault Vest drops status moves
    (Protect on a Choice Scarf user is the classic mix-up when sets are read off usage tables);
  * nature and stat points: from teams/spreads.py, which fits them to the moves.
Every team is checked against the format's rules, and no two teams share five or more Pokémon.
Output: results/built_teams.json, a pool like the others ({id, species, text}).
"""

import argparse
import json
import random

from champions.config import REGULATION, ROOT, TEAM_SIZE
from champions.sim.simulate import validate_many
from champions.teams.parse import serialise_team
from champions.teams.spreads import estimate

BATTLE_DATA = ROOT / "data" / "meta" / REGULATION.lower() / "battle_data.json"
REFERENCE = ROOT / "data" / "reference"
OUT = ROOT / "results" / "built_teams.json"

RANK_POWER = 0.9          # weight of a Pokémon = 1 / rank^power: how strongly popular Pokémon are preferred
PARTNER_BOOST = 4.0       # extra weight for being high on a chosen Pokémon's teammate list
MAX_MEGA_STONES = 2
NO_STATUS_ITEMS = {"Choice Scarf", "Choice Band", "Choice Specs", "Assault Vest"}


def load() -> tuple[dict, dict, dict]:
    battle = json.loads(BATTLE_DATA.read_text())["pokemon"]
    pokemon = json.loads((REFERENCE / "pokemon.json").read_text())
    items = json.loads((REFERENCE / "items.json").read_text())
    moves = json.loads((REFERENCE / "moves.json").read_text())
    battle = {name: entry for name, entry in battle.items() if name in pokemon and len(entry["moves"]) >= 4}
    stones = {name for name, item in items.items() if item.get("megaStone")}
    categories = {name: move["category"] for name, move in moves.items()}
    return battle, stones, categories


def pick(rng: random.Random, battle: dict, chosen: list[str]) -> str:
    candidates = [name for name in battle if name not in chosen]
    weights = []
    for name in candidates:
        boost = 0.0
        for mate in chosen:
            listed = battle[mate]["teammates"]
            if name in listed:
                boost += (len(listed) - listed.index(name)) / len(listed)
        weights.append((1 + PARTNER_BOOST * boost / max(len(chosen), 1)) / battle[name]["rank"] ** RANK_POWER)
    return rng.choices(candidates, weights)[0]


def build_set(name: str, entry: dict, taken: set[str], stones_used: int, stones: set[str], categories: dict) -> dict:
    item = next((i for i, _ in entry["items"] if i not in taken and (i not in stones or stones_used < MAX_MEGA_STONES)), None)
    ranked = [m for m, _ in entry["moves"]]
    if item in NO_STATUS_ITEMS:
        ranked = [m for m in ranked if categories.get(m) != "Status"] or ranked
    moves = ranked[:4]
    nature, points = estimate(name, None, moves)
    return {"name": name, "item": item, "ability": entry["abilities"][0][0], "nature": nature, "evs": points, "moves": moves}


def build_team(rng: random.Random, battle: dict, stones: set[str], categories: dict) -> list[dict]:
    chosen: list[str] = []
    while len(chosen) < TEAM_SIZE:
        chosen.append(pick(rng, battle, chosen))
    taken: set[str] = set()
    team = []
    for name in chosen:
        mon = build_set(name, battle[name], taken, sum(m["item"] in stones for m in team), stones, categories)
        taken.add(mon["item"])
        team.append(mon)
    return team


def build_pool(count: int, seed: int) -> list[dict]:
    battle, stones, categories = load()
    rng = random.Random(seed)
    pool, rejected = [], 0
    while len(pool) < count and rejected < count * 20:
        batch = []
        while len(batch) < 2 * (count - len(pool)):
            team = build_team(rng, battle, stones, categories)
            species = {m["name"] for m in team}
            if all(len(species & set(t["species"])) < 5 for t in pool + batch):
                batch.append({"species": sorted(species), "text": serialise_team(team)})
        for team, problems in zip(batch, validate_many([t["text"] for t in batch])):
            if problems:
                rejected += 1
            elif len(pool) < count and all(len(set(team["species"]) & set(t["species"])) < 5 for t in pool):
                pool.append({"id": f"built-{seed}-{len(pool):03d}", **team})
    print(f"{len(pool)} legal teams ({rejected} rejected by the {REGULATION} validator)")
    return pool


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teams", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    pool = build_pool(args.teams, args.seed)
    OUT.parent.mkdir(exist_ok=True)
    with open(args.out, "w") as out:
        json.dump(pool, out, indent=1)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
