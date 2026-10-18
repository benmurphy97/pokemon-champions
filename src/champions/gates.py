"""Check a model against the two agreed gates (docs/plan.md) and record the result.

    PYTHONPATH=src .venv/bin/python -m champions.gates --model models/il_policy_v3.pt            # both gates
    PYTHONPATH=src .venv/bin/python -m champions.gates --model models/x.pt --gate random --games 200

random:  share of paired games won against the random player (gate 1: 90%).
general: share of paired games won against greedy-blind on held-out teams of every kind (gate 2: 55%), with
         the health margin; paired games make 50% parity.
known:   (sub-score) share of games won piloting the 10 known-good Trick Room teams against held-out strong teams,
        next to the rule-based baseline piloting the same teams.
Results are appended to results/gates.json, which the operating room's plan view reads.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from champions.agent import NeuralAgent
from champions.preview import LearnedPreview
from champions.arena import challenger_wins, load_policy, paired_jobs, run_jobs, wilson
from champions.train_rl import KNOWN_TEAMS, evaluate, load_pool

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "results" / "gates.json"
POOL = ROOT / "results" / "opponent_pool.json"


def record(entry: dict) -> None:
    history = json.loads(LOG.read_text()) if LOG.exists() else []
    history.append({"time": time.time(), **entry})
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text(json.dumps(history, indent=1) + "\n")


def gate_random(net, name: str, games: int, workers: int, preview: bool = False) -> dict:
    agent = NeuralAgent(net, preview=LearnedPreview(net) if preview else None)
    results = run_jobs(paired_jobs(json.loads(POOL.read_text()), games, "neural", "random", 1), agent, workers)
    wins, decided, _ = challenger_wins(results)
    low, high = wilson(wins, decided)
    return {"model": name, "gate": "random", "value": wins / decided, "interval": [low, high], "games": decided,
            "preview": "learned" if preview else "rule"}


def gate_general(net, name: str, games: int, workers: int, preview: bool = False) -> dict:
    from champions.arena import challenger_margin
    from champions.train_rl import LADDER_EVAL, WEB_TEAMS
    pool = load_pool(LADDER_EVAL)[:400] + load_pool(WEB_TEAMS, split="eval")
    agent = NeuralAgent(net, preview=LearnedPreview(net) if preview else None)
    results = run_jobs(paired_jobs(pool, games, "neural", "greedy-blind", 11), agent, workers)
    wins, decided, _ = challenger_wins(results)
    low, high = wilson(wins, decided)
    margin, se = challenger_margin(results)
    return {"model": name, "gate": "general", "value": wins / decided, "interval": [low, high], "games": decided,
            "margin": margin, "margin_se": se, "preview": "learned" if preview else "rule"}


def gate_known(net, name: str, games: int, workers: int, preview: bool = False) -> dict:
    result = evaluate(net, games, workers, seed=7, known=load_pool(KNOWN_TEAMS), learned_preview=preview)
    return {"model": name, "gate": "known", "value": result["policy"], "interval": result["interval"],
            "baseline": result["baseline"], "games": games, "preview": "learned" if preview else "rule"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--gate", choices=["random", "general", "known", "all"], default="all")
    parser.add_argument("--games", type=int, default=None, help="default: 400 for random, 1200 for known")
    parser.add_argument("--preview", choices=["rule", "learned"], default="rule",
                        help="who picks the four and the leads: the rule-based preview (default) or the model's own")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    net, name = load_policy(Path(args.model)), Path(args.model).name
    for gate, default_games, fn in (("random", 400, gate_random), ("general", 1200, gate_general), ("known", 1200, gate_known)):
        if args.gate in (gate, "all"):
            entry = fn(net, name, args.games or default_games, args.workers, args.preview == "learned")
            record(entry)
            print(f"{name} gate {gate} ({args.preview} preview): {entry['value']:.1%} {entry['interval'][0]:.1%}-{entry['interval'][1]:.1%}"
                  + (f" (rule-based {entry['baseline']:.1%})" if "baseline" in entry else "")
                  + (f", health margin {entry['margin']:+.3f} ± {entry['margin_se']:.3f}" if "margin" in entry else ""), flush=True)


if __name__ == "__main__":
    main()
