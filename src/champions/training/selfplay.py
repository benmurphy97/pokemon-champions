"""Self-play league training for a general player.

    .venv/bin/python -m champions.training.selfplay --init models/il_policy_v3.pt --iterations 40

Both sides draw teams of every kind from the pool of real tournament and ladder teams. Each game the
learner meets an opponent drawn from a league: its latest self, saved past versions (snapshots), the
rule-based greedy-blind player, and the imitation model it started from. The fixed opponents keep it
from drifting into a style only it understands.

What it is rewarded for: half the result (win/loss) and half how the game went, the health it had
left minus the opponent's (see arena_worker.js). A win by a hair and a dominant win differ, and so do
a close loss and a rout, which makes each game say more about how well it played. PPO (training/ppo.py).

Progress is judged against fixed yardsticks on held-out teams, in paired games (each side pilots both
teams, so team strength cancels): greedy-blind and the imitation model. It cannot be judged against
itself, since that is 50% by definition.
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch

from champions.model.agent import NeuralAgent
from champions.sim.arena import challenger_margin, challenger_wins, load_policy, paired_jobs, run_jobs, wilson
from champions.model.model_io import save_policy
from champions.model.vocab import MOVE_ID
from champions.teams.pools import LADDER_EVAL, LADDER_TRAIN, WEB_TEAMS, load_pool
from champions.training.ppo import ppo_update, to_batch

ROOT = Path(__file__).resolve().parents[3]
LEAGUE_DIR = ROOT / "models" / "league"
BEST = ROOT / "models" / "selfplay_best.pt"
LOG = ROOT / "results" / "rl_log.jsonl"
LATEST = ROOT / "models" / "selfplay_latest.pt"
# Moves that only protect or heal. How often the learner picks them is logged, so a drift into stalling shows.
STALL_MOVES = ["Protect", "Detect", "Spiky Shield", "Baneful Bunker", "Silk Trap", "Burning Bulwark", "Wide Guard", "Quick Guard",
               "Mat Block", "Recover", "Roost", "Soft-Boiled", "Slack Off", "Moonlight", "Morning Sun", "Synthesis", "Shore Up",
               "Milk Drink", "Heal Order", "Rest", "Wish", "Life Dew", "Jungle Healing", "Lunar Blessing", "Strength Sap"]
STALL_IDS = {MOVE_ID[m] for m in STALL_MOVES if m in MOVE_ID}
TIMEOUT_REWARD = -0.5      # a game stalled past the turn cap is worse than a draw for both sides
DEFAULT_LEAGUE = "self:0.4,snap:0.3,greedy-blind:0.2,imitation:0.1"


def make_jobs(count: int, rng: random.Random, pools: list, league: dict, snapshots: list[str], tag: str) -> list[dict]:
    kinds, weights = list(league), list(league.values())
    jobs = []
    for i in range(count):
        t1, t2 = (rng.choice(rng.choices([p for p, _ in pools], [w for _, w in pools])[0]) for _ in range(2))
        kind = rng.choices(kinds, weights)[0]
        if kind == "snap" and not snapshots:
            kind = "self"
        opponent = {"self": "learner", "imitation": "imitation", "greedy-blind": "greedy-blind",
                    "snap": rng.choice(snapshots) if snapshots else ""}[kind]
        learner_first = kind == "self" or rng.random() < 0.5
        jobs.append({"id": f"{tag}-{i}", "a": t1["text"], "b": t2["text"], "seed": rng.randrange(10**6),
                     "agentA": "learner" if learner_first else opponent, "agentB": opponent if learner_first else "learner",
                     "opponent": kind})
    return jobs


def play_batch(net, imitation, snapshot_agents: dict, jobs: list[dict], workers: int, seed: int, margin_weight: float,
               idle_penalty: float = 0.03):
    """Play the jobs; returns the learner's recorded decisions (with rewards) and per-opponent results."""
    agents = {"learner": NeuralAgent(net, temperature=1.0, seed=seed), "imitation": NeuralAgent(imitation, temperature=1.0, seed=seed + 1),
              **snapshot_agents}
    trajectories: dict = {}
    results = run_jobs(jobs, None, workers, trajectories=trajectories, agents=agents, record_names={"learner"})
    by_id = {j["id"]: j for j in jobs}
    steps, record = [], {}
    totals = {"turns": 0, "idle": 0, "timeouts": 0, "n": 0}
    for r in results:
        job = by_id[r["id"]]
        for side, name in (("p1", job["agentA"]), ("p2", job["agentB"])):
            if name != "learner":
                continue
            sign = 1 if side == "p1" else -1
            won = r["winner"] == ("A" if side == "p1" else "B")
            outcome = 1.0 if won else (-1.0 if r["winner"] in ("A", "B") else 0.0)
            margin = sign * r["margin"]
            reward = (1 - margin_weight) * outcome + margin_weight * margin - idle_penalty * r["idleTurns"]
            if r["timeout"]:
                reward = TIMEOUT_REWARD
            totals["turns"] += r["turns"]; totals["idle"] += r["idleTurns"]; totals["timeouts"] += r["timeout"]; totals["n"] += 1
            if job["opponent"] != "self":     # against itself the two seats cancel, so only count other opponents
                wins, games, total = record.get(job["opponent"], (0, 0, 0.0))
                record[job["opponent"]] = (wins + (outcome > 0), games + 1, total + margin)
            trajectory = trajectories.get((r["id"], side), [])
            for i, decision in enumerate(trajectory):
                last = i == len(trajectory) - 1
                steps.append({**decision, "reward": reward if last else 0.0, "last": last})
    stall = sum(s["action"] in STALL_IDS for s in steps) / max(len(steps), 1)
    n = max(totals["n"], 1)
    return steps, record, {"stall_share": round(stall, 3), "turns": round(totals["turns"] / n, 1),
                           "idle_turns": round(totals["idle"] / n, 2), "timeouts": totals["timeouts"], "seats": totals["n"]}


def evaluate(net, imitation, pool: list[dict], games: int, workers: int, seed: int) -> dict:
    """Paired games on held-out teams against greedy-blind and against the imitation model."""
    agents = {"neural": NeuralAgent(net, temperature=0.0), "imitation": NeuralAgent(imitation, temperature=0.0)}
    out = {}
    for name, opponent, n in (("blind", "greedy-blind", games), ("imitation", "imitation", games // 2)):
        results = run_jobs(paired_jobs(pool, n, "neural", opponent, seed), None, workers, agents=agents)
        wins, decided, fallbacks = challenger_wins(results)
        low, high = wilson(wins, decided)
        mean, se = challenger_margin(results)
        out[name] = {"turns": round(sum(r["turns"] for r in results) / len(results), 1), "timeouts": sum(r["timeout"] for r in results),
                     "win": round(wins / decided, 3), "interval": [round(low, 3), round(high, 3)], "margin": round(mean, 3),
                     "margin_se": round(se, 3), "games": decided, "invalid_choices": fallbacks}
    return out


def parse_weights(text: str) -> dict:
    return {k: float(v) for k, v in (item.split(":") for item in text.split(","))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--init", default=str(ROOT / "models" / "il_policy_v3.pt"))
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--games", type=int, default=256, help="games per iteration")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--league", default=DEFAULT_LEAGUE)
    parser.add_argument("--margin-weight", type=float, default=0.5, help="share of the reward that is the health margin, not the result")
    parser.add_argument("--idle-penalty", type=float, default=0.03, help="reward lost per turn in which nothing changed (anti-stalling)")
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--max-snapshots", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--kl", type=float, default=0.05, help="penalty for drifting from the imitation start")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--value-warmup", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", default=str(LOG))
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    league = parse_weights(args.league)
    train_pools = [(load_pool(LADDER_TRAIN), 0.5), (load_pool(WEB_TEAMS, split="train"), 0.5)]
    eval_pool = load_pool(LADDER_EVAL)[:400] + load_pool(WEB_TEAMS, split="eval")
    print(f"teams: ladder {len(train_pools[0][0])}, tournament {len(train_pools[1][0])}; evaluation pool {len(eval_pool)}; league {league}", flush=True)

    net = load_policy(Path(args.init))
    imitation = copy.deepcopy(net).eval()
    reference = copy.deepcopy(net).eval()
    for p in list(imitation.parameters()) + list(reference.parameters()):
        p.requires_grad_(False)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    LEAGUE_DIR.mkdir(parents=True, exist_ok=True)
    snapshots: list[str] = []
    snapshot_agents: dict = {}
    log = Path(args.log)
    log.parent.mkdir(exist_ok=True)

    best, start = -9.0, time.time()
    for it in range(1, args.iterations + 1):
        t0 = time.time()
        jobs = make_jobs(args.games, rng, train_pools, league, snapshots, f"s{it}")
        steps, record, play = play_batch(net, imitation, snapshot_agents, jobs, args.workers, args.seed * 1000 + it,
                                         args.margin_weight, args.idle_penalty)
        t1 = time.time()
        stats = ppo_update(net, reference, optimiser, to_batch(steps), args, value_only=it <= args.value_warmup)
        entry = {"iteration": it, "decisions": len(steps), "rollout_seconds": round(t1 - t0, 1), "update_seconds": round(time.time() - t1, 1),
                 "win_rate": {k: round(w / g, 3) for k, (w, g, _) in record.items()},
                 "margin": {k: round(m / g, 3) for k, (_, g, m) in record.items()}, "snapshots": len(snapshots), "play": play,
                 **{k: round(v, 4) for k, v in stats.items()}}

        if it % args.snapshot_every == 0:
            name = f"snap{it}"
            path = LEAGUE_DIR / f"{name}.pt"
            save_policy(net, path, iteration=it)
            snapshots.append(name)
            snapshot_agents[name] = NeuralAgent(load_policy(path), temperature=1.0, seed=it)
            while len(snapshots) > args.max_snapshots:
                snapshot_agents.pop(snapshots.pop(0))

        if it % args.eval_every == 0 or it == args.iterations:
            result = evaluate(net, imitation, eval_pool, args.eval_games, args.workers, seed=7)
            blind = result["blind"]
            entry["eval"] = {"policy": blind["win"], "interval": blind["interval"], "baseline": 0.5, "margin": blind["margin"],
                             "margin_se": blind["margin_se"], "vs_imitation": result["imitation"], "invalid_choices": blind["invalid_choices"],
                             "turns": blind["turns"], "timeouts": blind["timeouts"]}
            save_policy(net, LATEST, iteration=it, eval=entry["eval"])
            if blind["margin"] > best:
                best = blind["margin"]
                save_policy(net, BEST, iteration=it, eval=entry["eval"])
        with log.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
    print(f"done in {(time.time() - start) / 60:.1f} min; best margin against greedy-blind {best:+.3f}")


if __name__ == "__main__":
    main()
