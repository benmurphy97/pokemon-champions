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
Our teams each carry two Trick Room setters; the opponents' teams are mostly real ladder teams of
every kind of strategy, plus Trick Room teams and random meta teams. Every few iterations the
policy is tested on held-out teams and the best-scoring version is saved to models/rl_policy.pt.
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
from champions.model_io import save_policy
from champions.rate_teams import RATINGS
from champions.arena import challenger_wins, load_policy, paired_jobs, run_jobs, wilson

ROOT = Path(__file__).resolve().parents[2]
TRAIN_POOL = ROOT / "results" / "train_pool_tr.json"          # our teams: each carries two Trick Room setters
EVAL_POOL = ROOT / "results" / "eval_pool_tr.json"
LADDER_TRAIN = ROOT / "results" / "ladder_teams_train.json"   # real teams of every kind, used as opponents
LADDER_EVAL = ROOT / "results" / "ladder_teams_eval.json"
GENERAL_POOL = ROOT / "results" / "train_pool.json"
WEB_TEAMS = ROOT / "results" / "web_teams.json"               # good tournament teams from the web, Trick Room and otherwise
KNOWN_TEAMS = ROOT / "data" / "teams" / "m-c" / "known_good_tr.json"   # see known_teams.py
IL_MODEL = ROOT / "models" / "il_policy_tr.pt"
RL_MODEL = ROOT / "models" / "rl_policy.pt"
LOG = ROOT / "results" / "rl_log.jsonl"


DEFAULT_OPP_POOLS = "ladder:0.45,web:0.2,webgood:0.15,gentr:0.1,general:0.1"


def load_pool(path: Path, **filters) -> list[dict]:
    """Teams from a pool file, optionally only those whose fields match, e.g. split="train"."""
    return [t for t in json.loads(path.read_text()) if all(t.get(k) == v for k, v in filters.items())]


def strong_teams(count: int = 400) -> list[dict]:
    """The tournament teams with the best simulated records (rate_teams.py): the hard opponents."""
    from champions.arena import wilson
    ratings = json.loads(RATINGS.read_text())["teams"]
    teams = [t for t in load_pool(WEB_TEAMS) if t["id"] in ratings]
    teams.sort(key=lambda t: -wilson(ratings[t["id"]]["wins"], ratings[t["id"]]["games"])[0])
    return teams[:count]


def named_pools() -> dict[str, list[dict]]:
    """The team pools training can draw opponents from, by name."""
    return {
        "ladder": load_pool(LADDER_TRAIN),                       # real ladder teams of every kind
        "web": load_pool(WEB_TEAMS, split="train"),              # tournament teams
        "webgood": load_pool(WEB_TEAMS, split="train", good=True),   # tournament teams that placed well
        "gentr": load_pool(TRAIN_POOL),                          # generated Trick Room teams
        "general": load_pool(GENERAL_POOL),
        "strong": [t for t in strong_teams() if t["split"] == "train"],   # the best-performing tournament teams                      # generated meta teams
    }


def training_jobs(pool: list[dict], count: int, mix: dict, rng: random.Random, tag: str,
                  opp_pools: list[tuple[list[dict], float]] | None = None) -> list[dict]:
    """The policy plays a team from `pool` against a team drawn from `opp_pools` (default: `pool` itself).

    It takes a random seat against an opponent agent drawn from `mix`.
    """
    kinds, weights = list(mix), list(mix.values())
    jobs = []
    for i in range(count):
        if opp_pools:
            t1 = rng.choice(pool)
            t2 = rng.choice(rng.choices([p for p, _ in opp_pools], [w for _, w in opp_pools])[0])
        else:
            t1, t2 = rng.sample(pool, 2)
        opponent = rng.choices(kinds, weights)[0]
        neural_first = rng.random() < 0.5
        jobs.append({"id": f"{tag}-{i}", "a": t1["text"], "b": t2["text"], "seed": rng.randrange(10**6),
                     "agentA": "neural" if neural_first else opponent,
                     "agentB": opponent if neural_first else "neural", "opponent": opponent})
    return jobs


def play_batch(net, jobs: list[dict], workers: int, seed: int, shaping: float = 0.0) -> tuple[list[dict], dict]:
    """Play `jobs` with the policy sampling its moves; returns the recorded decisions and win rates.

    Each decision carries a reward: the game result on a game's last decision, plus `shaping` times
    the change in health lead since the previous one (the lead counts as 0 once the game is over, so
    the extra credit cancels out over a game and never changes which play is best).
    """
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
        trajectory = trajectories.get((r["id"], side), [])
        for i, decision in enumerate(trajectory):
            last = i == len(trajectory) - 1
            later = 0.0 if last else trajectory[i + 1]["lead"]
            steps.append({**decision, "reward": shaping * (later - decision["lead"]) + (outcome if last else 0.0), "last": last})
    return steps, record


def to_batch(steps: list[dict]) -> dict:
    batch = {k: torch.as_tensor(np.stack([np.asarray(s["arrays"][k]) for s in steps])) for k in KEYS}
    batch["action"] = torch.as_tensor([s["action"] for s in steps])
    batch["target"] = torch.as_tensor([s["target"] for s in steps])
    batch["target_mask"] = torch.as_tensor(np.stack([s["target_mask"] for s in steps]))
    batch["old_logp"] = torch.as_tensor([s["logp"] for s in steps], dtype=torch.float32)
    batch["reward"] = torch.as_tensor([s["reward"] for s in steps], dtype=torch.float32)
    batch["last"] = torch.as_tensor([s["last"] for s in steps])
    return batch


def advantages(reward: torch.Tensor, last: torch.Tensor, value: torch.Tensor, lam: float) -> torch.Tensor:
    """Generalised advantage estimate. Steps are in game order and `last` marks each game's final one.

    lam = 1 credits a decision with everything that happened after it (result minus the value
    estimate); smaller values lean more on the value estimate, which is steadier but biased.
    """
    advantage, running = torch.zeros_like(value), 0.0
    for t in reversed(range(len(value))):
        following = 0.0 if last[t] else float(value[t + 1])
        running = float(reward[t]) + following - float(value[t]) + (0.0 if last[t] else lam * running)
        advantage[t] = running
    return advantage


def take(batch: dict, idx: torch.Tensor) -> dict:
    return {k: v[idx] for k, v in batch.items()}


def ppo_update(net, reference, optimiser, batch: dict, args, value_only: bool) -> dict:
    """Several passes over one batch of recorded decisions."""
    n = len(batch["reward"])
    with torch.no_grad():
        _, _, old_value, _ = net.evaluate(batch, batch["action"], batch["target"], batch["target_mask"])
        _, _, _, ref_log_probs = reference.evaluate(batch, batch["action"], batch["target"], batch["target_mask"])
    advantage = advantages(batch["reward"], batch["last"], old_value, args.gae_lambda)
    batch["ret"] = advantage + old_value
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


def evaluate(net, games: int, workers: int, seed: int, known: list[dict] | None = None,
             learned_preview: bool = False) -> dict:
    """How well the policy pilots Trick Room teams against real teams of every kind.

    Each of our evaluation teams meets a held-out real ladder team. The same pairing is played with
    the policy and, as a baseline, with greedy-blind piloting our team, so the difference is the
    pilot. Each condition is played from both seats. Everything is deterministic (most likely move).
    """
    own_pools = {"generated": load_pool(EVAL_POOL), "tournament": load_pool(WEB_TEAMS, split="eval", good=True, trick_room=True)}
    opp_pools = [load_pool(LADDER_EVAL), load_pool(WEB_TEAMS, split="eval")]
    if known:   # the known-good teams are the ones being tested, against opponents that are not among them
        ids = {t["id"] for t in known}
        own_pools = {"known": known}
        opp_pools = [[t for t in strong_teams() if t["split"] == "eval" and t["id"] not in ids]]
    rng = random.Random(seed)
    jobs = []
    for n in range(games // 4):
        source = list(own_pools)[n % len(own_pools)]
        t, o = rng.choice(own_pools[source]), rng.choice(rng.choice(opp_pools))
        base = rng.randrange(10**6)
        for cond, pilot in (("policy", "neural"), ("baseline", "greedy-blind")):
            jobs.append({"id": f"{n}-{cond}-a", "a": t["text"], "b": o["text"], "seed": base, "agentA": pilot,
                         "agentB": "greedy-blind", "cond": cond, "t_side": "A", "o_tr": o["trick_room"], "source": source})
            jobs.append({"id": f"{n}-{cond}-b", "a": o["text"], "b": t["text"], "seed": base, "agentA": "greedy-blind",
                         "agentB": pilot, "cond": cond, "t_side": "B", "o_tr": o["trick_room"], "source": source})
    from champions.preview import LearnedPreview
    agent = NeuralAgent(net, temperature=0.0, preview=LearnedPreview(net) if learned_preview else None)
    results = run_jobs(jobs, agent, workers)
    by_id = {j["id"]: j for j in jobs}
    tally: dict = {}
    fallbacks = 0
    for r in results:
        job = by_id[r["id"]]
        fallbacks += r["fallbacks"]
        for key in (job["cond"], f"{job['cond']}_vs_{'tr' if job['o_tr'] else 'other'}", f"{job['cond']}_on_{job['source']}"):
            wins, played = tally.get(key, (0, 0))
            tally[key] = (wins + (r["winner"] == job["t_side"]), played + (r["winner"] in ("A", "B")))
    rates = {k: round(w / p, 3) for k, (w, p) in tally.items() if p}
    wins, played = tally["policy"]
    low, high = wilson(wins, played)
    return {"policy": rates["policy"], "interval": [round(low, 3), round(high, 3)], "baseline": rates["baseline"],
            "by_opponent": {k: v for k, v in rates.items() if "_vs_" in k},
            "by_team_source": {k: v for k, v in rates.items() if "_on_" in k}, "invalid_choices": fallbacks}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--games", type=int, default=256, help="games per iteration")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--init", default=str(IL_MODEL))
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--kl", type=float, default=0.1, help="penalty for drifting from the imitation policy")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--known", action="store_true",
                        help="train to pilot only the known-good Trick Room teams (known_teams.py) and test on them")
    parser.add_argument("--shaping", type=float, default=0.0,
                        help="extra reward per unit of health lead gained (0 = only the game result counts)")
    parser.add_argument("--gae-lambda", type=float, default=1.0,
                        help="1 = credit decisions with the whole game's result; lower leans on the value estimate")
    parser.add_argument("--value-warmup", type=int, default=1, help="iterations that train only the value estimate")
    parser.add_argument("--mix", default="greedy-blind:0.6,random:0.25,greedy:0.15", help="opponent agents")
    parser.add_argument("--opp-pools", default=DEFAULT_OPP_POOLS,
                        help="opponent teams, name:weight,... from: ladder, web, webgood, gentr, general, strong")
    parser.add_argument("--web-own-weight", type=int, default=3,
                        help="how many times good tournament Trick Room teams are repeated in the policy's own teams")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    mix = {k: float(v) for k, v in (item.split(":") for item in args.mix.split(","))}
    named = named_pools()
    web_tr = load_pool(WEB_TEAMS, split="train", good=True, trick_room=True)
    pool = load_pool(TRAIN_POOL) + web_tr * args.web_own_weight       # our teams: generated, plus good tournament ones
    known = load_pool(KNOWN_TEAMS) if args.known else None
    if known and args.opp_pools == DEFAULT_OPP_POOLS:
        args.opp_pools = "strong:0.8,ladder:0.2"
    if known:
        ids = {t["id"] for t in known}
        pool = known
        named = {name: [t for t in teams if t["id"] not in ids] for name, teams in named.items()}
    opp_pools = [(named[name], float(w)) for name, w in (item.split(":") for item in args.opp_pools.split(","))]
    print(f"own teams: {len(pool)} entries ({len(web_tr)} good tournament Trick Room teams x{args.web_own_weight}); "
          f"opponent pools: {', '.join(f'{n} {len(named[n])}' for n, _ in (i.split(':') for i in args.opp_pools.split(',')))}", flush=True)

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
        jobs = training_jobs(pool, args.games, mix, rng, f"i{it}", opp_pools)
        steps, record = play_batch(net, jobs, args.workers, seed=args.seed * 1000 + it, shaping=args.shaping)
        t1 = time.time()
        stats = ppo_update(net, reference, optimiser, to_batch(steps), args, value_only=it <= args.value_warmup)
        entry = {"iteration": it, "decisions": len(steps), "rollout_seconds": round(t1 - t0, 1),
                 "update_seconds": round(time.time() - t1, 1),
                 "win_rate": {k: round(w / g, 3) for k, (w, g) in record.items()}, **{k: round(v, 4) for k, v in stats.items()}}

        if it % args.eval_every == 0 or it == args.iterations:
            result = evaluate(net, args.eval_games, args.workers, seed=7, known=known)
            entry["eval"] = result
            save_policy(net, RL_MODEL.with_name("rl_latest.pt"), iteration=it, eval=result)
            if result["policy"] > best:
                best = result["policy"]
                save_policy(net, RL_MODEL, iteration=it, eval=result)
        with LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
    print(f"done in {(time.time() - start) / 60:.1f} min; best evaluation win rate {best:.1%}")


if __name__ == "__main__":
    main()
