"""Reinforcement learning: improve the policy by playing games in the simulator and learning from wins.

    .venv/bin/python -m champions.train_rl --iterations 60

Each iteration plays a batch of games against a mix of opponents (greedy-blind, random, the
all-seeing greedy), records every decision the policy made, and nudges it towards the choices
that came before wins and away from those before losses (PPO).

Details that keep it stable:
  * the start point is the imitation-learned policy, so it already plays sensibly;
  * a value estimate says how well a position was already going, so a decision is credited for
    what it changed (advantage = result - value), not for the result alone;
  * a penalty for drifting far from the imitation policy keeps it from forgetting human play;
  * training teams (results/train_pool.json) are different from the evaluation teams
    (results/opponent_pool.json).
Every few iterations the policy is played against greedy-blind with fixed settings and the
best-scoring version is saved to models/rl_policy.pt.
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from champions.agent import KEYS, NeuralAgent
from champions.arena import challenger_wins, load_policy, paired_jobs, run_jobs, wilson

ROOT = Path(__file__).resolve().parents[2]
TRAIN_POOL = ROOT / "results" / "train_pool.json"
EVAL_POOL = ROOT / "results" / "opponent_pool.json"
IL_MODEL = ROOT / "models" / "il_policy.pt"
RL_MODEL = ROOT / "models" / "rl_policy.pt"
LOG = ROOT / "results" / "rl_log.jsonl"


def training_jobs(pool: list[dict], count: int, mix: dict, rng: random.Random, tag: str) -> list[dict]:
    """Random team pairs; the policy takes a random seat against an opponent drawn from `mix`."""
    kinds, weights = list(mix), list(mix.values())
    jobs = []
    for i in range(count):
        t1, t2 = rng.sample(pool, 2)
        opponent = rng.choices(kinds, weights)[0]
        neural_first = rng.random() < 0.5
        jobs.append({"id": f"{tag}-{i}", "a": t1["text"], "b": t2["text"], "seed": rng.randrange(10**6),
                     "agentA": "neural" if neural_first else opponent,
                     "agentB": opponent if neural_first else "neural", "opponent": opponent})
    return jobs


def play_batch(net, jobs: list[dict], workers: int, seed: int) -> tuple[list[dict], dict]:
    """Play `jobs` with the policy sampling its moves; returns the recorded decisions and win rates."""
    agent = NeuralAgent(net, temperature=1.0, seed=seed)
    trajectories: dict = {}
    results = run_jobs(jobs, agent, workers, trajectories=trajectories)
    by_id = {j["id"]: j for j in jobs}
    steps, record = [], {}
    for r in results:
        job = by_id[r["id"]]
        side = "p1" if job["agentA"] == "neural" else "p2"
        won = r["winner"] == ("A" if side == "p1" else "B")
        outcome = 1.0 if won else (-1.0 if r["winner"] in ("A", "B") else 0.0)
        wins, games = record.get(job["opponent"], (0, 0))
        record[job["opponent"]] = (wins + (outcome > 0), games + 1)
        for decision in trajectories.get((r["id"], side), []):
            steps.append({**decision, "ret": outcome})
    return steps, record


def to_batch(steps: list[dict]) -> dict:
    batch = {k: torch.as_tensor(np.stack([np.asarray(s["arrays"][k]) for s in steps])) for k in KEYS}
    batch["action"] = torch.as_tensor([s["action"] for s in steps])
    batch["target"] = torch.as_tensor([s["target"] for s in steps])
    batch["target_mask"] = torch.as_tensor(np.stack([s["target_mask"] for s in steps]))
    batch["old_logp"] = torch.as_tensor([s["logp"] for s in steps], dtype=torch.float32)
    batch["ret"] = torch.as_tensor([s["ret"] for s in steps], dtype=torch.float32)
    return batch


def take(batch: dict, idx: torch.Tensor) -> dict:
    return {k: v[idx] for k, v in batch.items()}


def ppo_update(net, reference, optimiser, batch: dict, args, value_only: bool) -> dict:
    """Several passes over one batch of recorded decisions."""
    n = len(batch["ret"])
    with torch.no_grad():
        _, _, old_value, _ = net.evaluate(batch, batch["action"], batch["target"], batch["target_mask"])
        _, _, _, ref_log_probs = reference.evaluate(batch, batch["action"], batch["target"], batch["target_mask"])
    advantage = batch["ret"] - old_value
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

    stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "clipped": 0.0}
    passes = 0
    for _ in range(args.epochs):
        order = torch.randperm(n)
        for start in range(0, n, args.minibatch):
            idx = order[start:start + args.minibatch]
            b = take(batch, idx)
            logp, entropy, value, log_probs = net.evaluate(b, b["action"], b["target"], b["target_mask"])
            ratio = (logp - b["old_logp"]).exp()
            adv = advantage[idx]
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
            value_loss = ((value - b["ret"]) ** 2).mean()
            kl = (log_probs.exp() * (log_probs - ref_log_probs[idx])).sum(-1).mean()
            loss = 0.5 * value_loss if value_only else (policy_loss + 0.5 * value_loss
                                                         - args.entropy * entropy.mean() + args.kl * kl)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            passes += 1
            stats["policy"] += policy_loss.item(); stats["value"] += value_loss.item()
            stats["entropy"] += entropy.mean().item(); stats["kl"] += kl.item()
            stats["clipped"] += ((ratio - 1).abs() > args.clip).float().mean().item()
    return {k: v / max(passes, 1) for k, v in stats.items()}


def evaluate(net, games: int, workers: int, opponent: str, seed: int) -> tuple[float, tuple[float, float], int]:
    """Win rate when playing the most likely move each time, in paired games on the evaluation teams."""
    pool = json.loads(EVAL_POOL.read_text())
    results = run_jobs(paired_jobs(pool, games, "neural", opponent, seed), NeuralAgent(net, temperature=0.0), workers)
    wins, decided, fallbacks = challenger_wins(results)
    return wins / decided, wilson(wins, decided), fallbacks


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--games", type=int, default=256, help="games per iteration")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--init", default=str(IL_MODEL))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--kl", type=float, default=0.03, help="penalty for drifting from the imitation policy")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--value-warmup", type=int, default=1, help="iterations that train only the value estimate")
    parser.add_argument("--mix", default="greedy-blind:0.6,random:0.25,greedy:0.15")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    mix = {k: float(v) for k, v in (item.split(":") for item in args.mix.split(","))}
    pool = json.loads(TRAIN_POOL.read_text())

    net = load_policy(Path(args.init))
    reference = copy.deepcopy(net).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)

    best = -1.0
    LOG.parent.mkdir(exist_ok=True)
    start = time.time()
    for it in range(1, args.iterations + 1):
        t0 = time.time()
        jobs = training_jobs(pool, args.games, mix, rng, f"i{it}")
        steps, record = play_batch(net, jobs, args.workers, seed=args.seed * 1000 + it)
        t1 = time.time()
        stats = ppo_update(net, reference, optimiser, to_batch(steps), args, value_only=it <= args.value_warmup)
        entry = {"iteration": it, "decisions": len(steps), "rollout_seconds": round(t1 - t0, 1),
                 "update_seconds": round(time.time() - t1, 1),
                 "win_rate": {k: round(w / g, 3) for k, (w, g) in record.items()}, **{k: round(v, 4) for k, v in stats.items()}}

        if it % args.eval_every == 0 or it == args.iterations:
            rate, (low, high), fallbacks = evaluate(net, args.eval_games, args.workers, "greedy-blind", seed=7)
            entry["eval_vs_greedy_blind"] = {"win_rate": round(rate, 3), "interval": [round(low, 3), round(high, 3)],
                                             "invalid_choices": fallbacks}
            torch.save({"state": net.state_dict(), "dim": net.dim, "layers": len(net.encoder.layers), "iteration": it,
                        "eval": entry["eval_vs_greedy_blind"]}, RL_MODEL.with_name("rl_latest.pt"))
            if rate > best:
                best = rate
                torch.save({"state": net.state_dict(), "dim": net.dim, "layers": len(net.encoder.layers), "iteration": it,
                            "eval": entry["eval_vs_greedy_blind"]}, RL_MODEL)
        with LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
    print(f"done in {(time.time() - start) / 60:.1f} min; best evaluation win rate vs greedy-blind {best:.1%}")


if __name__ == "__main__":
    main()
