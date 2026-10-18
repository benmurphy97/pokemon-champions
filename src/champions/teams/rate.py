"""Keep a running record of how well each team does, so good teams are remembered.

    PYTHONPATH=src python3 -m champions.teams.rate --games 2 --limit 200

Every team plays a fixed gauntlet of real ladder teams, piloted on both sides by the same player
(greedy-blind), so a team's win rate reflects the team and not who flew it. Results add up across
runs in data/ratings/<regulation>/teams.json: run again with a new seed to sharpen the numbers,
or point --pool at another team list (a JSON list of {id, species, text}) to rate it too.

The gauntlet is stored in data/ratings/<regulation>/gauntlet.json so that every run, and every
team, faces the same opponents.
"""

import argparse
import json
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from champions.config import REGULATION
from champions.sim.simulate import RUNNER
from champions.teams.tournament_pool import build_pool

ROOT = Path(__file__).resolve().parents[3]
RATINGS_DIR = ROOT / "data" / "ratings" / REGULATION.lower()
RATINGS = RATINGS_DIR / "teams.json"
GAUNTLET = RATINGS_DIR / "gauntlet.json"
LADDER_TEAMS = ROOT / "results" / "ladder_teams_train.json"   # made by ladder_teams.py
PILOT = "greedy-blind"
GAUNTLET_SIZE = 60


def load_gauntlet() -> list[dict]:
    """The fixed opponents. Drawn once from real ladder teams, then kept on disk."""
    if not GAUNTLET.exists():
        ladder = json.loads(LADDER_TEAMS.read_text())
        chosen = random.Random(0).sample(ladder, GAUNTLET_SIZE)
        GAUNTLET.parent.mkdir(parents=True, exist_ok=True)
        GAUNTLET.write_text(json.dumps([{"id": t["id"], "text": t["text"]} for t in chosen], indent=1) + "\n")
    return json.loads(GAUNTLET.read_text())


def play(matchups: list[dict], games: int, seed: int) -> list[dict]:
    """Run sim/matchups.js on a list of {id, a, b}; returns its per-matchup results."""
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp, "in.json"), Path(tmp, "out.json")
        inp.write_text(json.dumps(matchups))
        run = subprocess.run(["node", str(RUNNER.with_name("matchups.js")), str(inp), "--games", str(games),
                              "--agent", PILOT, "--seed", str(seed), "--out", str(out)], capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr.strip() or "matchups.js failed")
        return json.loads(out.read_text())


def rate(teams: list[dict], gauntlet: list[dict], games: int, seed: int, workers: int) -> dict[str, tuple[int, int]]:
    """{team id: (wins, games)} for each team against the whole gauntlet."""
    batches = [teams[i::workers] for i in range(workers)]
    def job(batch):
        matchups = [{"id": f"{t['id']}|{o['id']}", "a": t["text"], "b": o["text"]} for t in batch for o in gauntlet]
        return play(matchups, games, seed)
    tally: dict[str, list[int]] = {}
    with ThreadPoolExecutor(workers) as pool:
        for results in pool.map(job, batches):
            for r in results:
                if "error" in r:
                    continue
                total = tally.setdefault(r["id"].split("|")[0], [0, 0])
                total[0] += r["winsA"]; total[1] += r["games"]
    return {k: (w, g) for k, (w, g) in tally.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=2, help="games per team per gauntlet opponent (seats alternate)")
    parser.add_argument("--limit", type=int, default=200, help="rate at most this many teams this run, fewest games first")
    parser.add_argument("--pool", help="JSON list of {id, species, text} to rate; default is the tournament teams")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None, help="default: a new seed each run")
    args = parser.parse_args()

    ratings = json.loads(RATINGS.read_text()) if RATINGS.exists() else {"pilot": PILOT, "teams": {}}
    seed = args.seed if args.seed is not None else ratings.get("runs", 0) + 1
    pool = json.loads(Path(args.pool).read_text()) if args.pool else build_pool()
    gauntlet = load_gauntlet()

    teams = sorted(pool, key=lambda t: ratings["teams"].get(t["id"], {}).get("games", 0))[: args.limit]
    print(f"rating {len(teams)} teams x {len(gauntlet)} opponents x {args.games} games (seed {seed})", flush=True)
    for team_id, (wins, games) in rate(teams, gauntlet, args.games, seed * 100_000, args.workers).items():
        entry = ratings["teams"].setdefault(team_id, {"wins": 0, "games": 0})
        entry["wins"] += wins; entry["games"] += games
        entry["species"] = next(t["species"] for t in pool if t["id"] == team_id)
    ratings["runs"] = seed if args.seed is None else ratings.get("runs", 0)
    RATINGS.parent.mkdir(parents=True, exist_ok=True)
    RATINGS.write_text(json.dumps(ratings, separators=(",", ":")) + "\n")
    best = sorted(ratings["teams"].items(), key=lambda kv: -kv[1]["wins"] / kv[1]["games"])[:5]
    print(f"{len(ratings['teams'])} teams on record → {RATINGS}")
    for team_id, e in best:
        print(f"  {e['wins'] / e['games']:.1%} over {e['games']} games  {' / '.join(e['species'])}")


if __name__ == "__main__":
    main()
