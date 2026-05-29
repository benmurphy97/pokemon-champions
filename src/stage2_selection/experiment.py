"""
ExperimentRunner — evaluates all 90 bring/lead combinations for each team pairing.

For each (our_team, opp_team) pair:
  For each of the 90 combinations (C(6,4)=15 bring × C(4,2)=6 lead):
    Run n_battles battles; opponent uses the auto heuristic (top-4 BST, lead fastest 2)
    Append one aggregated record to the JSONL checkpoint file

Resume-safe: skips any (our_roster, opp_roster, bring_indices, lead_indices) already saved.

Usage:
    docker-compose run --rm trainer python src/selection/experiment.py
    docker-compose run --rm trainer python src/selection/experiment.py --pilot      # 2 teams, 3 combos, 3 battles
    docker-compose run --rm trainer python src/selection/experiment.py --battles 10 --teams wolfe cybertron
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from itertools import combinations
from pathlib import Path

from config import SHOWDOWN_URL
from selection.client import run_battle

log = logging.getLogger(__name__)

_TEAMS_DIR      = Path(__file__).parent.parent.parent / "data" / "teams"
_DEFAULT_CKPT   = Path(__file__).parent.parent.parent / "data" / "experiment_results.jsonl"
_DEFAULT_BATTLES = 5


# ── helpers ──────────────────────────────────────────────────────────────────

def _all_combinations() -> list[tuple[list[int], list[int]]]:
    """All 90 (bring, lead) pairs: C(6,4)=15 × C(4,2)=6."""
    combos = []
    for bring in combinations(range(6), 4):
        for lead in combinations(range(4), 2):
            combos.append((list(bring), list(lead)))
    return combos


def _load_teams(teams_dir: Path, names: list[str] | None = None) -> dict[str, str]:
    """Return {team_name: showdown_string} for each .txt in teams_dir."""
    files = sorted(teams_dir.glob("*.txt"))
    teams = {f.stem: f.read_text() for f in files}
    if names:
        missing = [n for n in names if n not in teams]
        if missing:
            raise ValueError(f"Team files not found: {missing}")
        teams = {n: teams[n] for n in names}
    return teams


def _load_checkpoint(path: Path) -> set[tuple]:
    """Return set of (our_roster, opp_roster, bring_key, lead_key) already done."""
    done: set[tuple] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (
                    rec["our_roster"],
                    rec["opp_roster"],
                    tuple(rec["bring_indices"]),
                    tuple(rec["lead_indices"]),
                )
                done.add(key)
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _append_record(path: Path, record: dict) -> None:
    """Append one JSON record as a newline to the JSONL checkpoint file."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── core runner ──────────────────────────────────────────────────────────────

async def run_experiment(
    teams_dir:   Path  = _TEAMS_DIR,
    ckpt_path:   Path  = _DEFAULT_CKPT,
    n_battles:   int   = _DEFAULT_BATTLES,
    server_url:  str   = SHOWDOWN_URL,
    team_names:  list[str] | None = None,
    sample_n:    int   | None = None,
    pilot:       bool  = False,
) -> None:
    """
    Run the full experiment (or a subset) and checkpoint results to JSONL.

    Args:
        sample_n:  If set, randomly sample this many combinations per pairing
                   instead of all 90.
        pilot:     Quick smoke run: 2 teams × 3 combinations × 3 battles.
    """
    if pilot:
        n_battles  = 3
        sample_n   = 3
        team_names = team_names or None   # let it pick first 2 below

    teams = _load_teams(teams_dir, team_names)
    team_items = list(teams.items())
    if pilot:
        team_items = team_items[:2]

    all_combos = _all_combinations()
    done       = _load_checkpoint(ckpt_path)

    log.info("Teams: %s", [n for n, _ in team_items])
    log.info("Combinations per pairing: %d  (sample_n=%s)", len(all_combos), sample_n)
    log.info("Battles per combination: %d", n_battles)
    log.info("Checkpoint: %s  (%d records already done)", ckpt_path, len(done))

    total_pairings = len(team_items) * (len(team_items) - 1)
    pairing_num    = 0

    for our_name, our_str in team_items:
        for opp_name, opp_str in team_items:
            if our_name == opp_name:
                continue

            pairing_num += 1
            combos = all_combos
            if sample_n is not None:
                combos = random.sample(all_combos, min(sample_n, len(all_combos)))

            pending = [
                (bring, lead) for bring, lead in combos
                if (our_name, opp_name, tuple(bring), tuple(lead)) not in done
            ]

            log.info(
                "[%d/%d] %s vs %s — %d combinations (%d pending)",
                pairing_num, total_pairings,
                our_name, opp_name,
                len(combos), len(pending),
            )

            for combo_i, (bring, lead) in enumerate(pending, 1):
                t0 = time.monotonic()
                try:
                    results = await run_battle(
                        our_team_str=our_str,
                        opp_team_str=opp_str,
                        our_bring=bring,
                        our_lead=lead,
                        opp_bring=None,   # auto heuristic for opponent
                        opp_lead=None,
                        server_url=server_url,
                        n=n_battles,
                    )
                except Exception as exc:
                    log.error("  bring=%s lead=%s FAILED: %s", bring, lead, exc)
                    continue

                wins   = sum(1 for r in results if r.winner == "us")
                losses = sum(1 for r in results if r.winner == "them")
                draws  = sum(1 for r in results if r.winner == "draw")
                n_done = wins + losses + draws

                avg_t1  = sum(r.turn1_hp_delta  for r in results) / max(n_done, 1)
                avg_margin = sum(
                    r.our_remaining - r.their_remaining for r in results
                ) / max(n_done, 1)

                record = {
                    "our_roster":        our_name,
                    "opp_roster":        opp_name,
                    "bring_indices":     bring,
                    "lead_indices":      lead,
                    "wins":              wins,
                    "losses":            losses,
                    "draws":             draws,
                    "win_rate":          wins / max(n_done, 1),
                    "avg_turn1_hp_delta": round(avg_t1, 4),
                    "avg_win_margin":    round(avg_margin, 3),
                }
                _append_record(ckpt_path, record)

                elapsed = time.monotonic() - t0
                log.info(
                    "  [%d/%d] bring=%s lead=%s → %dW/%dL/%dD  wr=%.0f%%  t1Δ=%+.2f  (%.1fs)",
                    combo_i, len(pending),
                    bring, lead,
                    wins, losses, draws,
                    record["win_rate"] * 100,
                    avg_t1, elapsed,
                )

    log.info("Experiment complete. Results saved to %s", ckpt_path)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point; parses args and runs the experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("poke_env").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run bring/lead selection experiments")
    parser.add_argument(
        "--checkpoint", default=str(_DEFAULT_CKPT),
        help="JSONL file to save/resume results (default: data/experiment_results.jsonl)",
    )
    parser.add_argument("--battles", type=int, default=_DEFAULT_BATTLES,
                        help="Battles per combination (default: 5)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N combinations per pairing instead of all 90")
    parser.add_argument("--teams", nargs="+", default=None,
                        help="Restrict to these team names (stems of .txt files)")
    parser.add_argument("--server-url", default=SHOWDOWN_URL)
    parser.add_argument("--pilot", action="store_true",
                        help="Quick run: 2 teams × 3 combos × 3 battles")
    args = parser.parse_args()

    asyncio.run(run_experiment(
        ckpt_path=Path(args.checkpoint),
        n_battles=args.battles,
        server_url=args.server_url,
        team_names=args.teams,
        sample_n=args.sample,
        pilot=args.pilot,
    ))


if __name__ == "__main__":
    main()
