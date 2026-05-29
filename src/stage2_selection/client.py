"""
Battle client for the selection experiment pipeline.

Runs n controlled battles between two SelectionAgents with forced bring/lead
selections, returns BattleResult records.

Usage (smoke test — requires Showdown running):
    docker-compose run --rm trainer python src/selection/client.py --smoke-test
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
from dataclasses import dataclass

from poke_env import AccountConfiguration, ServerConfiguration

from config import BATTLE_FORMAT, SHOWDOWN_URL
from stage2_selection.heuristic_agent import SelectionAgent

log = logging.getLogger(__name__)

_run_counter = itertools.count(1)


@dataclass
class BattleResult:
    winner: str            # "us" | "them" | "draw"
    turns: int
    our_remaining: int     # our mons alive at end
    their_remaining: int   # opp mons alive at end
    turn1_hp_delta: float  # (our avg HP − opp avg HP) across both active slots after turn 1


async def run_battle(
    our_team_str: str,
    opp_team_str: str,
    our_bring: list[int],            # 4 zero-based indices into our 6-mon team
    our_lead:  list[int],            # 2 zero-based indices into our_bring
    opp_bring: list[int] | None,     # None → opponent uses auto heuristic teampreview
    opp_lead:  list[int] | None,
    server_url: str = SHOWDOWN_URL,
    n: int = 1,
) -> list[BattleResult]:
    """
    Run n battles with the given bring/lead selections and return results.

    Both agents use the heuristic for in-battle move decisions; only
    teampreview() is controlled by the bring/lead indices.
    Pass opp_bring=None / opp_lead=None to let the opponent use its auto heuristic.
    """
    run_id = next(_run_counter)
    server_cfg = ServerConfiguration(
        server_url,
        "https://play.pokemonshowdown.com/~~localhost/action.php?",
    )

    our_agent = SelectionAgent(
        bring_indices=our_bring,
        lead_indices=our_lead,
        account_configuration=AccountConfiguration(f"SelOur{run_id}", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=server_cfg,
        team=our_team_str,
        max_concurrent_battles=1,
        log_level=logging.WARNING,
    )
    opp_agent = SelectionAgent(
        bring_indices=opp_bring,
        lead_indices=opp_lead,
        account_configuration=AccountConfiguration(f"SelOpp{run_id}", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=server_cfg,
        team=opp_team_str,
        max_concurrent_battles=1,
        log_level=logging.WARNING,
    )

    await our_agent.battle_against(opp_agent, n_battles=n)

    results: list[BattleResult] = []
    for tag, battle in our_agent.battles.items():
        if battle.won is True:
            winner = "us"
        elif battle.won is False:
            winner = "them"
        else:
            winner = "draw"

        our_remaining   = sum(1 for m in battle.team.values()          if not m.fainted)
        their_remaining = sum(1 for m in battle.opponent_team.values() if not m.fainted)

        our_hp, opp_hp  = our_agent._t1_hp.get(tag, (1.0, 1.0))
        turn1_hp_delta  = our_hp - opp_hp

        results.append(BattleResult(
            winner=winner,
            turns=battle.turn,
            our_remaining=our_remaining,
            their_remaining=their_remaining,
            turn1_hp_delta=round(turn1_hp_delta, 4),
        ))

    return results


# ── smoke test ───────────────────────────────────────────────────────────────

async def _smoke_test(server_url: str, n: int) -> None:
    from pathlib import Path

    data_dir = Path(__file__).parent.parent.parent / "data"
    teams_dir = data_dir / "teams"

    team_files = sorted(teams_dir.glob("*.txt"))
    if len(team_files) < 2:
        raise RuntimeError(f"Need at least 2 team files in {teams_dir}")

    our_str = team_files[0].read_text()
    opp_str = team_files[1].read_text()
    our_name = team_files[0].stem
    opp_name = team_files[1].stem

    log.info("Smoke test: %s vs %s  (%d battles)", our_name, opp_name, n)

    results = await run_battle(
        our_team_str=our_str,
        opp_team_str=opp_str,
        our_bring=[0, 1, 2, 3],
        our_lead=[0, 1],
        opp_bring=[0, 1, 2, 3],
        opp_lead=[0, 1],
        server_url=server_url,
        n=n,
    )

    wins = sum(1 for r in results if r.winner == "us")
    log.info("Results (%d battles): %d wins / %d losses / %d draws",
             len(results), wins,
             sum(1 for r in results if r.winner == "them"),
             sum(1 for r in results if r.winner == "draw"))
    for i, r in enumerate(results, 1):
        log.info(
            "  [%d] winner=%-4s  turns=%2d  remaining=%d/%d  t1_hp_delta=%+.3f",
            i, r.winner, r.turns, r.our_remaining, r.their_remaining, r.turn1_hp_delta,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("poke_env").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--server-url", default=SHOWDOWN_URL)
    parser.add_argument("--battles", type=int, default=2)
    args = parser.parse_args()

    if args.smoke_test:
        asyncio.run(_smoke_test(args.server_url, args.battles))
