"""Does the simulator predict who wins real ladder games?

    PYTHONPATH=src python3 -m champions.replay_check --games 20 --limit 800

For each downloaded replay, both teams are rebuilt from the six species on each side using the
most common ladder set for each Pokémon (replays do not reveal full sets). That exact matchup is
simulated many times, and the simulated win probability is compared with the real winner.

Score: AUC = the chance that a randomly chosen game won by player 1 gets a higher simulated
win probability than a randomly chosen game won by player 2. 0.5 is a coin flip, 1.0 is perfect.
Real games are noisy (player skill, misplays, unknown sets), so even a good simulator scores modestly.
Two baselines are reported for context: the same simulation with random agents, and the
players' own rating difference.
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

from champions.meta import REFERENCE, build_set, load_usage, usage_key
from champions.teams import serialise_team

ROOT = Path(__file__).resolve().parents[2]
REPLAYS = ROOT / "cache" / "replays-m-c.jsonl"
RESULTS = ROOT / "results"
TEAM_SIZE = 6


def team_text(side: dict, usage: dict, reference: dict) -> str | None:
    """A full Showdown team for one replay side, or None if a species is missing from the usage data."""
    chosen = []
    for species in side["team"]:
        key = usage_key(species, species in side["mega"], usage)
        if key is None or key not in reference:
            return None
        chosen.append(key)
    taken: set[str] = set()
    team = []
    for key in chosen:
        mon = build_set(key, usage[key], reference, taken)
        taken.add(mon["item"])
        team.append(mon)
    return serialise_team(team)


def auc(scores: list[float], won: list[int]) -> float:
    """Probability a random winner outscores a random loser (ties count half)."""
    pos = [s for s, w in zip(scores, won) if w]
    neg = [s for s, w in zip(scores, won) if not w]
    total = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return total / (len(pos) * len(neg))


def bootstrap_interval(scores, won, rounds=300, seed=0) -> tuple[float, float]:
    rng = random.Random(seed)
    n, values = len(scores), []
    if n < 2:
        return float("nan"), float("nan")
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        s, w = [scores[i] for i in idx], [won[i] for i in idx]
        if 0 < sum(w) < n:
            values.append(auc(s, w))
    values.sort()
    if not values:
        return float("nan"), float("nan")
    return values[int(0.025 * len(values))], values[max(int(0.975 * len(values)) - 1, 0)]


def simulate(matchups: list[dict], games: int, agent: str, seed: int) -> dict:
    RESULTS.mkdir(exist_ok=True)
    inp, out = RESULTS / "matchups.json", RESULTS / f"matchups_{agent}.json"
    inp.write_text(json.dumps(matchups))
    run = subprocess.run(["node", str(ROOT / "sim" / "matchups.js"), str(inp), "--games", str(games),
                          "--agent", agent, "--seed", str(seed), "--out", str(out)],
                         capture_output=True, text=True)
    if run.returncode:
        raise SystemExit(run.stderr)
    print(f"  {agent}: {run.stdout.strip()}")
    return {r["id"]: r for r in json.loads(out.read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=20, help="simulated games per real matchup")
    parser.add_argument("--limit", type=int, default=800, help="max real games to use")
    parser.add_argument("--min-rating", type=int, default=0)
    args = parser.parse_args()

    usage, reference = load_usage(), json.loads(REFERENCE.read_text())
    replays = [json.loads(line) for line in REPLAYS.read_text().splitlines()]

    matchups, real = [], {}
    for game in replays:
        if min(game["p1"]["rating"] or 0, game["p2"]["rating"] or 0) < args.min_rating:
            continue
        a, b = team_text(game["p1"], usage, reference), team_text(game["p2"], usage, reference)
        if a and b and len(game["p1"]["team"]) == TEAM_SIZE and len(game["p2"]["team"]) == TEAM_SIZE:
            matchups.append({"id": game["id"], "a": a, "b": b})
            real[game["id"]] = game
        if len(matchups) >= args.limit:
            break
    print(f"{len(matchups)} of {len(replays)} replays have all 12 species in the ladder data")

    print("simulating…")
    sims = {agent: simulate(matchups, args.games, agent, seed=5) for agent in ("greedy", "random")}

    for agent, results in sims.items():
        ids = [i for i in real if i in results and "error" not in results[i]]
        scores = [results[i]["winsA"] / results[i]["games"] for i in ids]
        won = [1 if real[i]["winner"] == "p1" else 0 for i in ids]
        low, high = bootstrap_interval(scores, won)
        print(f"{agent:>7} agent: AUC {auc(scores, won):.3f}  (95% interval {low:.3f}–{high:.3f}, {len(ids)} games)")

    ids = [i for i in real if "error" not in sims["greedy"].get(i, {"error": 1})]
    diff = [(real[i]["p1"]["rating"] or 0) - (real[i]["p2"]["rating"] or 0) for i in ids]
    won = [1 if real[i]["winner"] == "p1" else 0 for i in ids]
    print(f"rating-difference baseline: AUC {auc(diff, won):.3f}")
    skipped = sum(1 for r in sims["greedy"].values() if "error" in r)
    if skipped:
        print(f"({skipped} matchups skipped: team failed M-C validation)")


if __name__ == "__main__":
    main()
