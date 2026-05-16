"""
Stage 5: Genetic Algorithm — Team Fitness Evaluation

Evaluates a team's fitness as a meta-weighted win rate across all 8 opponent
archetypes. Weights reflect real meta usage so beating Sun teams counts more
than beating Sand.

Used by both the mutation loop (train.py mutate mode) and the DEAP GA
(ga/evolve.py evaluate() callback).
"""

import logging
import time

from poke_env import AccountConfiguration

from battle.agents.heuristic import HeuristicAgent
from config import BATTLE_FORMAT
from battle.teams import (
    TEAM_WOLFE, TEAM_CYBERTRON, TEAM_ROD, TEAM_YUZU,
    TEAM_KURO, TEAM_WDMICHAEL, TEAM_HYUNGWOO, TEAM_JORGE,
)

log = logging.getLogger(__name__)

# Archetype usage weights — from Reg M-A meta data (May 2026).
# Two Sun teams share the Sun weight; Snow/TR split between Snow and TR.
_RAW_WEIGHTS = {
    "CYBERTRON": 74.5,   # Sun
    "HYUNGWOO":  74.5,   # Sun
    "YUZU":      72.1,   # Perish Trap
    "KURO":      67.7,   # Rain
    "ROD":       65.7,   # Snow/TR average
    "WDMICHAEL": 62.3,   # Trick Room
    "WOLFE":     60.0,   # Sand
    "JORGE":     60.0,   # Physical / Sand
}

_TOTAL = sum(_RAW_WEIGHTS.values())
WEIGHTS = {k: v / _TOTAL for k, v in _RAW_WEIGHTS.items()}

_OPPONENT_TEAMS = {
    "CYBERTRON": TEAM_CYBERTRON,
    "HYUNGWOO":  TEAM_HYUNGWOO,
    "YUZU":      TEAM_YUZU,
    "KURO":      TEAM_KURO,
    "ROD":       TEAM_ROD,
    "WDMICHAEL": TEAM_WDMICHAEL,
    "WOLFE":     TEAM_WOLFE,
    "JORGE":     TEAM_JORGE,
}


def evaluate_team(model, server_cfg, our_team: str, n_battles: int = 10) -> tuple[float, dict]:
    """
    Run n_battles against each archetype opponent (HeuristicAgent) and return
    (weighted_win_rate, {archetype: win_rate}).

    Uses HeuristicAgent — not a mirror of our own model — so win rates reflect
    actual team quality rather than always converging to ~50% via self-play.

    Args:
        model:      Loaded MaskablePPO model.
        server_cfg: Showdown server connection config.
        our_team:   Showdown team string for our agent.
        n_battles:  Battles per archetype.

    Returns:
        (weighted_win_rate, per_archetype_win_rates)
    """
    from battle.env import make_env

    per_archetype: dict[str, float] = {}

    for name, opp_team in _OPPONENT_TEAMS.items():
        run_id = int(time.time()) % 100000
        opponent = HeuristicAgent(
            account_configuration=AccountConfiguration(f"Fit_{name}_{run_id}", None),
            battle_format=BATTLE_FORMAT,
            server_configuration=server_cfg,
            team=opp_team,
            log_level=logging.WARNING,
            start_listening=False,
        )
        env = make_env(server_cfg, our_team, opponent)

        wins = 0
        try:
            for _ in range(n_battles):
                obs, _ = env.reset()
                done = False
                reward = 0.0
                while not done:
                    mask = env.action_masks()
                    action, _ = model.predict(obs, action_masks=mask, deterministic=False)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                if reward > 0:
                    wins += 1
        finally:
            env.close()

        win_rate = wins / n_battles
        per_archetype[name] = win_rate
        log.info("  %-12s %d/%d (%.0f%%)", name, wins, n_battles, win_rate * 100)

    weighted = sum(per_archetype[k] * WEIGHTS[k] for k in per_archetype)
    return weighted, per_archetype
