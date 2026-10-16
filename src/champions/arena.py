"""Play the learned policy against another agent in the simulator, many games in parallel.

    .venv/bin/python -m champions.arena --vs greedy --games 400

Teams come from the generated opponent pool (champions.meta). Each pair of teams is played twice,
so both agents pilot both teams: team strength cancels out and the result reflects the pilot.
"""

import argparse
import json
import math
import queue
import random
import subprocess
import threading
from pathlib import Path

import torch

from champions.agent import NeuralAgent
from champions.battle_state import BattleState
from champions.policy import PolicyNet

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "sim" / "arena_worker.js"
POOL = ROOT / "results" / "opponent_pool.json"
MODEL = ROOT / "models" / "il_policy.pt"


def load_policy(path: Path = MODEL) -> PolicyNet:
    saved = torch.load(path, map_location="cpu")
    net = PolicyNet(dim=saved["dim"], layers=saved["layers"])
    net.load_state_dict(saved["state"])
    return net.eval()


def run_jobs(jobs: list[dict], agent: NeuralAgent | None, workers: int = 6, in_flight: int = 4) -> list[dict]:
    """Run battles across worker processes; neural decisions are answered by `agent`."""
    todo: queue.Queue = queue.Queue()
    for job in jobs:
        todo.put(job)
    results: list[dict] = []
    lock = threading.Lock()
    errors: list[str] = []

    def serve(proc):
        states: dict = {}
        sent = 0

        def feed():
            nonlocal sent
            while sent - len(done) < in_flight:
                try:
                    job = todo.get_nowait()
                except queue.Empty:
                    return
                proc.stdin.write(json.dumps({"t": "job", **job}) + "\n")
                proc.stdin.flush()
                sent += 1

        done: list = []
        feed()
        for raw in proc.stdout:
            message = json.loads(raw)
            if message["t"] == "decide":
                state = states.setdefault((message["id"], message["side"]), BattleState())
                for line in message["lines"]:
                    state.update(line)
                choice = agent.decide(state, message["side"], message["request"])
                proc.stdin.write(json.dumps({"t": "act", "key": message["key"], "choice": choice}) + "\n")
                proc.stdin.flush()
            elif message["t"] == "result":
                done.append(message["id"])
                states.pop((message["id"], "p1"), None); states.pop((message["id"], "p2"), None)
                with lock:
                    results.append(message)
                feed()
            elif message["t"] == "error":
                done.append(message["id"])
                with lock:
                    errors.append(message["message"])
                feed()
            if len(done) == sent and todo.empty():
                break
        proc.stdin.close()
        proc.wait()

    procs = [subprocess.Popen(["node", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, bufsize=1) for _ in range(min(workers, len(jobs)))]
    threads = [threading.Thread(target=serve, args=(p,)) for p in procs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"{len(errors)} games crashed; first error:\n{errors[0]}")
    return results


def paired_jobs(pool: list[dict], games: int, challenger: str, opponent: str, seed: int) -> list[dict]:
    """Two jobs per sampled team pair: each agent pilots each team once."""
    rng = random.Random(seed)
    jobs = []
    for n in range(games // 2):
        t1, t2 = rng.sample(pool, 2)
        base = {"seed": 1000 + n * 10}
        jobs.append({"id": f"{n}a", "a": t1["text"], "b": t2["text"], "agentA": challenger, "agentB": opponent, **base})
        jobs.append({"id": f"{n}b", "a": t1["text"], "b": t2["text"], "agentA": opponent, "agentB": challenger, **base})
    return jobs


def challenger_wins(results: list[dict]) -> tuple[int, int, int]:
    """(wins, decided games, fallbacks) for the challenger, which is player A in 'a' jobs and B in 'b' jobs."""
    wins = decided = fallbacks = 0
    for r in results:
        fallbacks += r["fallbacks"]
        if r["winner"] in ("A", "B"):
            decided += 1
            wins += (r["winner"] == "A") == r["id"].endswith("a")
    return wins, decided, fallbacks


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / d, (centre + half) / d


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vs", default="greedy", choices=["greedy", "random"])
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(1)
    agent = NeuralAgent(load_policy(Path(args.model)), temperature=args.temperature)
    pool = json.loads(POOL.read_text())
    results = run_jobs(paired_jobs(pool, args.games, "neural", args.vs, args.seed), agent, args.workers)
    wins, decided, fallbacks = challenger_wins(results)
    low, high = wilson(wins, decided)
    print(f"learned policy vs {args.vs}: {wins}/{decided} wins = {wins / decided:.1%} "
          f"(95% interval {low:.1%}–{high:.1%}); average game {sum(r['turns'] for r in results) / len(results):.1f} turns; "
          f"{fallbacks} invalid choices replaced by the default")


if __name__ == "__main__":
    main()
