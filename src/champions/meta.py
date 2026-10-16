"""Build realistic teams from the ladder usage stats (see import_stats.py).

    PYTHONPATH=src python3 -m champions.meta --teams 50 --seed 1

Each Pokémon gets its most common set; teams are filled by picking partners in proportion to
how often they appear together on the ladder. Every team is checked against Showdown's
M-C rules before it is kept. Output: results/opponent_pool.json (generated, gitignored).
"""

import argparse
import json
import random
from pathlib import Path

from champions.config import REGULATION, STATS, TEAM_SIZE
from champions.simulate import validate_team
from champions.teams import serialise_team

ROOT = Path(__file__).resolve().parents[2]
USAGE = ROOT / "data" / "meta" / REGULATION.lower() / "usage.json"
REFERENCE = ROOT / "data" / "reference" / "pokemon.json"
OUT = ROOT / "results" / "opponent_pool.json"

PARTNER_BOOST = 3.0  # how strongly known pairings steer the next pick, relative to raw usage


def load_usage() -> dict:
    return json.loads(USAGE.read_text())["pokemon"]


def usage_key(species: str, mega: bool, usage: dict) -> str | None:
    """The usage entry for a team slot.

    Replays list the base species even for Pokémon that Mega Evolve on the ladder, so use the
    Mega form if the player Mega Evolved it, or if it is far more common than the base form.
    """
    family = species.split("-")[0]  # replays say Floette-Eternal; the ladder lists Floette-Mega
    forms = [name for name in usage if name.startswith((f"{species}-Mega", f"{family}-Mega"))]
    best_mega = max(forms, key=lambda n: usage[n]["usage"]) if forms else None
    base_usage = usage[species]["usage"] if species in usage else 0.0
    if best_mega and (mega or usage[best_mega]["usage"] > base_usage):
        return best_mega
    return species if species in usage else None


def build_set(species: str, entry: dict, reference: dict, taken_items: set[str]) -> dict:
    """The most common set for a species, using the best item not already on the team."""
    stone = reference[species].get("megaStone")
    item = stone or next((name for name, _ in entry["items"] if name not in taken_items), None)
    nature, spread, _ = entry["spreads"][0]
    return {
        "name": species,
        "item": item,
        "ability": entry["abilities"][0][0],
        "nature": nature,
        "evs": {stat: spread[stat] for stat in STATS},
        "moves": [name for name, _ in entry["moves"][:4]],
    }


def pick_species(rng: random.Random, usage: dict, chosen: list[str]) -> str:
    """Next teammate: weighted by usage, boosted by how often it is paired with those already chosen."""
    candidates = [s for s in usage if s not in chosen]
    weights = []
    for species in candidates:
        pairing = 0.0
        for mate in chosen:
            shares = dict(usage[mate]["teammates"])
            pairing += shares.get(species, 0.0)
        pairing /= max(len(chosen), 1)
        weights.append(usage[species]["usage"] * (1 + PARTNER_BOOST * pairing))
    return rng.choices(candidates, weights)[0]


def build_team(rng: random.Random, usage: dict, reference: dict) -> list[dict]:
    chosen = [rng.choices(list(usage), [e["usage"] for e in usage.values()])[0]]
    while len(chosen) < TEAM_SIZE:
        chosen.append(pick_species(rng, usage, chosen))

    taken: set[str] = set()
    team = []
    for species in chosen:
        mon = build_set(species, usage[species], reference, taken)
        taken.add(mon["item"])
        team.append(mon)
    return team


def generate_pool(count: int, seed: int = 1, max_attempts: int = 20) -> list[dict]:
    reference = json.loads(REFERENCE.read_text())
    usage = {name: entry for name, entry in load_usage().items() if name in reference}   # species the engine knows
    rng = random.Random(seed)
    pool, rejected = [], 0
    while len(pool) < count and rejected < count * max_attempts:
        team = build_team(rng, usage, reference)
        text = serialise_team(team)
        problems = validate_team(text)
        if problems:
            rejected += 1
            continue
        pool.append({"id": f"gen-{seed}-{len(pool):03d}", "species": [m["name"] for m in team], "text": text})
    print(f"{len(pool)} legal teams ({rejected} rejected by the M-C validator)")
    return pool


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teams", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=str(OUT), help="where to write the pool")
    args = parser.parse_args()
    pool = generate_pool(args.teams, args.seed)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(pool, indent=1) + "\n")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
