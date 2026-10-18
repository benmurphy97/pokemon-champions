"""Team-preview labels from a replay: which four each player brought and which two led.

    PYTHONPATH=src .venv/bin/python -m champions.training.preview_labels --games 400     # pilot: check the labels

A log shows both sides' six Pokémon (in the order they were listed), the two that start on the
field, and every Pokémon that ever appeared. A brought Pokémon that never appears cannot be seen,
so a side's four may be only partly known: `brought_seen` holds the ones seen and `complete` says
whether all four were. The label is the set of plans (four, lead pair; see policy.PLANS) consistent
with what was seen, and training maximises their total probability (marginal likelihood).
"""

import argparse
import copy
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from champions.model.battle_state import BattleState
from champions.model.encoding import ARRAY_KEYS, PREVIEW, encode
from champions.training.hindsight import complete_sets
from champions.teams.generate import load_usage
from champions.model.policy import PAIRS, PLANS, SUBSETS

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "cache" / "raw"
CACHE = ROOT / "cache" / "preview_examples.npz"


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


def preview_examples(lines: list[str], usage: dict) -> list[dict]:
    """One training example per side: the team-preview view and what the player did.

    The view is both teams' six species and the player's own full sets (reconstructed from the whole
    game, as in decisions.extract), with nothing revealed. Label: `plan_ok` [90] marks the plans the
    player could have chosen given what the log shows: the lead pair as seen, and a four that contains
    every brought Pokémon seen. `complete` says whether all four were seen (exactly one plan is then ok).
    """
    labels = extract_preview(lines)
    state, at_preview, final = BattleState(), None, BattleState()
    for line in lines:
        if line.startswith("|teampreview") and at_preview is None:
            at_preview = copy.deepcopy(state)
        state.update(line)
        final.update(line)
    if at_preview is None:
        return []
    out = []
    for side, info in labels.items():
        if len(info["six"]) != 6 or len(info["leads"]) != 2:
            continue
        own = complete_sets(final, side, usage)
        arrays = encode(at_preview, side, PREVIEW, own_info=own)
        seen, lead = set(info["brought_seen"]) | set(info["leads"]), tuple(sorted(info["leads"]))
        plan_ok = [seen <= set(SUBSETS[s]) and PAIRS[p] == lead for s, p in PLANS]
        if not any(plan_ok):
            continue
        out.append({"arrays": arrays, "plan_ok": plan_ok, "complete": info["complete"], "side": side})
    return out


def build_examples(limit: int | None = None, min_rating: int = 1300) -> dict:
    usage = load_usage()
    arrays: dict[str, list] = {k: [] for k in ARRAY_KEYS}
    meta = {"plan_ok": [], "complete": [], "game": [], "rating": []}
    for g, path in enumerate(sorted(RAW_DIR.glob("*.json.gz"))[:limit]):
        replay = json.load(gzip.open(path, "rt"))
        if (replay.get("rating") or 0) < min_rating:
            continue
        for example in preview_examples(replay["log"].split("\n"), usage):
            for k in ARRAY_KEYS:
                arrays[k].append(example["arrays"][k])
            for k in ("plan_ok", "complete"):
                meta[k].append(example[k])
            meta["game"].append(g)
            meta["rating"].append(replay.get("rating") or 0)
    return {**{k: np.stack(v) for k, v in arrays.items()}, **{k: np.array(v) for k, v in meta.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--build", action="store_true", help="build cache/preview_examples.npz from all raw games (training uses it)")
    args = parser.parse_args()
    if args.build:
        data = build_examples()
        np.savez_compressed(CACHE, **data)
        print(f"{len(data['plan_ok'])} preview examples ({int(data['complete'].sum())} with the full four seen) -> {CACHE}")
        return
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
