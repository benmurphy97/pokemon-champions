"""How good are our beliefs about the opponent's sets? Scored offline on real tournament teams.

    PYTHONPATH=src .venv/bin/python -m champions.evaluation.belief_bench [--seed 1] [--trials 3]

Holds out a fifth of the tournament teams. The corpus is the other four fifths. For each held-out team we
pretend to be its opponent: we see its six species, a few of the four brought Pokémon's moves, items and
abilities are revealed at random, and we score how much probability each method gave the facts still hidden.

Methods: ladder (today: independent usage shares), then the posterior with team context off (beta 0) and on.
Metrics (lower is better unless noted): cross-entropy per hidden move, per hidden item and ability; hit rate of
the top guesses (higher is better); mean absolute error of the expected Speed and main attacking stat.
"""

import argparse
import math
import random
from collections import defaultdict

from champions.model.belief import Corpus, family, infer
from champions.teams.generate import load_usage, usage_key
from champions.teams.stats import compute_stats
from champions.training.hindsight import _BASE_STATS
from champions.model.belief import _POINT_KEY

FLOOR = 1e-3


def ladder_belief(usage, species, observed):
    """Today's estimate: independent ladder shares; revealed facts pinned to 100%."""
    entry = usage.get(usage_key(family(species), False, usage) or "")
    known = observed.get("moves") or []
    moves = [{"name": m, "p": 1.0, "known": True} for m in known]
    if entry:
        moves += [{"name": n, "p": p, "known": False} for n, p in entry["moves"] if n not in known][:8]
    item = observed.get("item")
    items = [{"name": item, "p": 1.0, "known": True}] if item else ([{"name": n, "p": p, "known": False} for n, p in entry["items"][:3]] if entry else [])
    ability = observed.get("ability")
    abilities = [{"name": ability, "p": 1.0, "known": True}] if ability else ([{"name": n, "p": p, "known": False} for n, p in entry["abilities"][:2]] if entry else [])
    expected = None
    base = _BASE_STATS.get(family(species))
    if entry and base:
        nature, spread, _ = entry["spreads"][0]
        expected = compute_stats(base, nature, {_POINT_KEY[k]: v for k, v in spread.items()})
    return {"moves": moves, "items": items, "abilities": abilities, "expectedStats": expected}


def methods(corpus, usage):
    return {
        "ladder (today)": lambda six, obs: [ladder_belief(usage, s, o) for s, o in zip(six, obs)],
        "posterior, no team context": lambda six, obs: infer(corpus, usage, six, obs, beta=0.0),
        "posterior, team context 0.5": lambda six, obs: infer(corpus, usage, six, obs, beta=0.5),
        "posterior, team context 1.0": lambda six, obs: infer(corpus, usage, six, obs, beta=1.0),
        "posterior, team context 2.0": lambda six, obs: infer(corpus, usage, six, obs, beta=2.0),
    }


def reveal(team, rng):
    """Brought four (random) and what they have shown by mid-game: 1 to 3 moves; item and ability sometimes."""
    brought = set(rng.sample(range(6), 4))
    observed = []
    for i, s in enumerate(team):
        o = {"moves": [], "item": None, "ability": None}
        if i in brought:
            o["moves"] = rng.sample(s["moves"], rng.randint(1, min(3, len(s["moves"]))))
            o["item"] = s["item"] if rng.random() < 0.35 else None
            o["ability"] = s.get("ability") if rng.random() < 0.35 else None
        observed.append(o)
    return brought, observed


def score(belief, truth, observed, stats):
    hidden = [m for m in truth["moves"] if m not in observed["moves"]]
    slots = max(len(hidden), 1)
    shares = {m["name"]: m["p"] for m in belief["moves"] if not m["known"]}
    total = sum(shares.values()) or 1.0
    out = {"move_ce": [], "move_hit": [], "item_ce": [], "item_hit": [], "ability_ce": [], "spe_err": [], "atk_err": []}
    for m in hidden:
        out["move_ce"].append(-math.log(max(shares.get(m, 0.0) / total, FLOOR)))
    top = {k for k, _ in sorted(shares.items(), key=lambda kv: -kv[1])[:slots]}
    if hidden:
        out["move_hit"].append(sum(m in top for m in hidden) / len(hidden))
    if not observed["item"]:
        p = {i["name"]: i["p"] for i in belief["items"]}
        norm = sum(p.values()) or 1.0
        out["item_ce"].append(-math.log(max(p.get(truth["item"], 0.0) / norm, FLOOR)))
        out["item_hit"].append(float(bool(belief["items"]) and belief["items"][0]["name"] == truth["item"]))
    if not observed["ability"] and truth.get("ability"):
        p = {i["name"]: i["p"] for i in belief["abilities"]}
        norm = sum(p.values()) or 1.0
        out["ability_ce"].append(-math.log(max(p.get(truth["ability"], 0.0) / norm, FLOOR)))
    if belief.get("expectedStats") and stats:
        out["spe_err"].append(abs(belief["expectedStats"]["spe"] - stats["spe"]))
        main = "atk" if stats["atk"] >= stats["spa"] else "spa"
        out["atk_err"].append(abs(belief["expectedStats"][main] - stats[main]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3, help="random reveals per held-out team")
    parser.add_argument("--novel", action="store_true", help="only score held-out teams with no near-copy (3+ identical sets) in the corpus")
    args = parser.parse_args()
    from champions.config import REGULATION
    from champions.model.belief import TEAM_DIR
    ids = sorted(p.stem for p in TEAM_DIR.glob("*.txt"))
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    test_ids = set(ids[: len(ids) // 5])
    corpus, usage = Corpus.load(exclude=test_ids), load_usage()
    test = Corpus.load(exclude=set(ids) - test_ids).teams
    if args.novel:
        signature = lambda s: (tuple(sorted(s["moves"])), s["item"], s.get("ability"), s.get("nature"), tuple(sorted(s["evs"].items())))
        index = defaultdict(set)
        for t, team in enumerate(corpus.teams):
            for s in team:
                index[signature(s)].add(t)
        def near_copy(team):
            shared = defaultdict(int)
            for s in team:
                for t in index.get(signature(s), ()):
                    shared[t] += 1
            return max(shared.values(), default=0) >= 3
        before = len(test)
        test = [team for team in test if not near_copy(team)]
        print(f"novel only: kept {len(test)} of {before} held-out teams ({before - len(test)} had a near-copy in the corpus)")
    print(f"corpus {len(corpus.teams)} teams, test {len(test)} teams, {args.trials} reveals each")
    results = {name: defaultdict(lambda: defaultdict(list)) for name in methods(corpus, usage)}
    for team in test:
        six = [s["name"] for s in team]
        true_stats = []
        for s in team:
            base = _BASE_STATS.get(family(s["name"]))
            true_stats.append(compute_stats(base, s.get("nature") or "Hardy", {_POINT_KEY[k]: v for k, v in s["evs"].items()}) if base else None)
        for _ in range(args.trials):
            brought, observed = reveal(team, rng)
            for name, method in methods(corpus, usage).items():
                beliefs = method(six, observed)
                for i, belief in enumerate(beliefs):
                    group = "brought" if i in brought else "unbrought"
                    for key, values in score(belief, team[i], observed[i], true_stats[i]).items():
                        results[name][group][key] += values
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    for group in ("brought", "unbrought"):
        print(f"\n{group} Pokémon")
        print(f"{'method':32}{'move CE':>9}{'move hit':>10}{'item CE':>9}{'item hit':>10}{'abil CE':>9}{'Spe err':>9}{'Atk err':>9}")
        for name, r in results.items():
            g = r[group]
            print(f"{name:32}{mean(g['move_ce']):9.3f}{mean(g['move_hit']):10.3f}{mean(g['item_ce']):9.3f}{mean(g['item_hit']):10.3f}{mean(g['ability_ce']):9.3f}{mean(g['spe_err']):9.1f}{mean(g['atk_err']):9.1f}")


if __name__ == "__main__":
    main()
