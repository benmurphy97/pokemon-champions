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
from champions.model_io import load_policy as load_model
from champions.policy import PolicyNet

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "sim" / "arena_worker.js"
POOL = ROOT / "results" / "opponent_pool.json"
MODEL = ROOT / "models" / "il_policy.pt"


def load_policy(path: Path = MODEL) -> PolicyNet:
    return load_model(path)


def run_jobs(jobs: list[dict], agent: NeuralAgent | None, workers: int = 6, in_flight: int = 4,
             trajectories: dict | None = None, observer=None, agents: dict | None = None,
             record_names: set | None = None) -> list[dict]:
    """Run battles across worker processes; neural decisions are answered by `agent`.

    If `trajectories` is a dict, each neural decision is recorded into it under (job id, side).
    `agents` maps a player name used in a job ("learner", "imitation", "snap12", ...) to the NeuralAgent that
    answers for it, so one game can pit two different networks against each other; `agent` answers for the
    rest. Decisions are recorded only for names in `record_names` (default: every neural seat).
    `observer(job id, side, lines)` is called with the new log lines before each neural decision.
    """
    todo: queue.Queue = queue.Queue()
    by_id = {job["id"]: job for job in jobs}
    for job in jobs:
        todo.put(job)
    results: list[dict] = []
    lock = threading.Lock()
    errors: list[str] = []

    def serve(proc):
        try:
            serve_worker(proc)
        except Exception as err:   # never let a dead worker pass for missing results
            with lock:
                errors.append(f"worker thread failed: {err!r}")
            proc.kill()

    def serve_worker(proc):
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
        if sent == 0:      # other workers took every job: nothing will ever arrive from this one
            proc.stdin.close()
            proc.wait()
            proc.stdout.close()
            return
        for raw in proc.stdout:
            message = json.loads(raw)
            if message["t"] == "decide":
                state = states.setdefault((message["id"], message["side"]), BattleState())
                for line in message["lines"]:
                    state.update(line)
                if observer:
                    observer(message["id"], message["side"], message["lines"])
                job = by_id[message["id"]]
                name = job["agentA"] if message["side"] == "p1" else job["agentB"]
                recording = trajectories is not None and (record_names is None or name in record_names)
                record = trajectories.setdefault((message["id"], message["side"]), []) if recording else None
                seat_agent = (agents or {}).get(name, agent)
                choice = seat_agent.decide(state, message["side"], message["request"], record=record)
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
        proc.stdout.close()

    procs = [subprocess.Popen(["node", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, bufsize=1) for _ in range(min(workers, len(jobs)))]
    threads = [threading.Thread(target=serve, args=(p,)) for p in procs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"{len(errors)} problems; first:\n{errors[0]}")
    if len(results) != len(jobs):
        raise RuntimeError(f"only {len(results)} of {len(jobs)} games finished")
    return results


def challenger_margin(results: list[dict]) -> tuple[float, float]:
    """(mean, standard error) of the challenger's end-of-game health lead, from -1 to 1 (see arena_worker.js).

    Unlike win/loss this credits a close loss and a dominant win differently, so it needs fewer games to
    tell two players apart.
    """
    values = [r["margin"] * (1 if r["id"].endswith("a") else -1) for r in results if "margin" in r]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    return mean, math.sqrt(var / len(values))


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
    parser.add_argument("--vs", default="greedy", choices=["greedy", "greedy-blind", "random"])
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
