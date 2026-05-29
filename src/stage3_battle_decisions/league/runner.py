"""
Stage 6: Co-Evolution Loop — Population Based Training League

Each generation:
  1. Train every agent for round_steps
  2. Evaluate every agent → fitness score
  3. Bottom agent(s) exploit top agent(s): copy checkpoint weights
  4. Exploited agents explore: apply one mutation to their team (via ga/chromosome)
  5. Save historical snapshots

Sequential — one agent trains at a time (single Showdown server).
Parallelisation with Ray is a Phase 5 task (see ga/evolve.py island_model).
"""

import logging
import os
import shutil
import time

from poke_env import AccountConfiguration

from battle.agents.heuristic import HeuristicAgent
from stage3_battle_decisions.architecture.extractor import BattleStateEncoder
from stage3_battle_decisions.league.agent import Agent
from config import BATTLE_FORMAT, CHECKPOINT_DIR
from stage4_team_evolution.chromosome import mutate
from stage4_team_evolution.fitness import evaluate_team
from battle.teams import (
    TEAM_ALPHA, TEAM_WOLFE, TEAM_CYBERTRON, TEAM_ROD,
    TEAM_YUZU, TEAM_KURO, TEAM_WDMICHAEL, TEAM_HYUNGWOO, TEAM_JORGE,
    RotatingTeambuilder,
)

log = logging.getLogger(__name__)

ALL_OPPONENT_TEAMS = [
    TEAM_WOLFE, TEAM_CYBERTRON, TEAM_ROD, TEAM_YUZU,
    TEAM_KURO, TEAM_WDMICHAEL, TEAM_HYUNGWOO, TEAM_JORGE,
]

# One seed team per archetype; the GA will diversify these over generations
SEED_TEAMS = [
    TEAM_WOLFE,     # Agent 0: Sand
    TEAM_CYBERTRON, # Agent 1: Sun
    TEAM_ROD,       # Agent 2: Snow / TR
    TEAM_YUZU,      # Agent 3: Perish Trap
    TEAM_KURO,      # Agent 4: Rain
    TEAM_WDMICHAEL, # Agent 5: TR Golurk
    TEAM_HYUNGWOO,  # Agent 6: Sun (alt)
    TEAM_JORGE,     # Agent 7: CharX / Mega Ttar
]

_TB_LOG = os.path.join(os.path.dirname(__file__), "..", "..", "logs")


class League:
    def __init__(self, checkpoint_dir: str, n_agents: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.agents: list[Agent] = []
        for i in range(n_agents):
            agent = Agent(
                agent_id=i,
                team=SEED_TEAMS[i % len(SEED_TEAMS)],
                checkpoint_dir=os.path.join(checkpoint_dir, f"agent_{i}"),
            )
            os.makedirs(agent.checkpoint_dir, exist_ok=True)
            if not agent.load_team():
                agent.save_team()
            self.agents.append(agent)

    def _make_train_env(self, server_cfg, team: str):
        from battle.env import make_env
        run_id = int(time.time()) % 100000
        opponent = HeuristicAgent(
            account_configuration=AccountConfiguration(f"Lg_{run_id}", None),
            battle_format=BATTLE_FORMAT,
            server_configuration=server_cfg,
            team=RotatingTeambuilder(*ALL_OPPONENT_TEAMS),
            log_level=logging.WARNING,
            start_listening=False,
        )
        return make_env(server_cfg, team, opponent)

    def _load_model(self, agent: Agent, env):
        from sb3_contrib import MaskablePPO
        ckpt = agent.checkpoint_path
        if os.path.exists(ckpt + ".zip"):
            try:
                log.info("  Agent %d: resuming from checkpoint", agent.agent_id)
                return MaskablePPO.load(ckpt, env=env, tensorboard_log=_TB_LOG)
            except Exception as exc:
                log.warning(
                    "  Agent %d: checkpoint load failed (%s) — starting fresh",
                    agent.agent_id, exc,
                )
        log.info("  Agent %d: starting fresh model", agent.agent_id)
        return MaskablePPO(
            "MlpPolicy", env,
            n_steps=512, batch_size=64, n_epochs=4,
            learning_rate=3e-4, gamma=0.99,
            policy_kwargs=dict(
                features_extractor_class=BattleStateEncoder,
                features_extractor_kwargs=dict(features_dim=288),
                net_arch=[128],
            ),
            tensorboard_log=_TB_LOG,
            verbose=0,
        )

    def train_all(self, server_cfg, round_steps: int):
        """Train each agent for round_steps steps."""
        for agent in self.agents:
            log.info("Training agent %d (gen %d)...", agent.agent_id, agent.generation)
            env = self._make_train_env(server_cfg, agent.team)
            try:
                model = self._load_model(agent, env)
                model.learn(total_timesteps=round_steps, reset_num_timesteps=False)
                model.save(agent.checkpoint_path)
            finally:
                env.close()
            agent.generation += 1
            agent.update_tenure()
            log.info("  Agent %d: saved checkpoint", agent.agent_id)

    def evaluate_all(self, server_cfg, eval_battles: int):
        """Evaluate every agent and update fitness scores."""
        for agent in self.agents:
            log.info("Evaluating agent %d...", agent.agent_id)
            from sb3_contrib import MaskablePPO
            env = self._make_train_env(server_cfg, agent.team)
            try:
                model = self._load_model(agent, env)
                fitness, breakdown = evaluate_team(
                    model, server_cfg, agent.team, n_battles=eval_battles
                )
            finally:
                env.close()
            agent.fitness = fitness
            agent.fitness_history.append(fitness)
            log.info("  Agent %d fitness: %.1f%%", agent.agent_id, fitness * 100)

    def exploit_explore(self):
        """
        Bottom agent copies weights from top agent (exploit), then mutates team (explore).
        With 5 agents: bottom 1 copies top 1.
        """
        import random
        sorted_agents = sorted(self.agents, key=lambda a: a.fitness)
        n_exploit = max(1, len(self.agents) // 5)

        bottom = sorted_agents[:n_exploit]
        top = sorted_agents[-n_exploit:]

        for loser in bottom:
            winner = random.choice(top)
            log.info(
                "Exploit: agent %d (%.1f%%) inherits weights from agent %d (%.1f%%), mutates own team",
                loser.agent_id, loser.fitness * 100,
                winner.agent_id, winner.fitness * 100,
            )
            team_before = loser.team
            loser.copy_from(winner)
            loser.team = mutate(loser.team, tenure=loser.mon_tenure)
            loser.save_team()
            loser.record_history(
                event="exploit_explore",
                team_before=team_before,
                team_after=loser.team,
                fitness_before=loser.fitness,
            )

    def save_snapshot(self, generation: int):
        """Save a frozen snapshot of each agent's checkpoint."""
        snap_dir = os.path.join(self.checkpoint_dir, f"snapshot_gen{generation}")
        os.makedirs(snap_dir, exist_ok=True)
        for agent in self.agents:
            src = agent.checkpoint_path + ".zip"
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(snap_dir, f"agent_{agent.agent_id}.zip"))
            agent.save_team()
            team_src = os.path.join(agent.checkpoint_dir, "team.txt")
            if os.path.exists(team_src):
                shutil.copy2(team_src, os.path.join(snap_dir, f"agent_{agent.agent_id}_team.txt"))
        log.info("Snapshot saved → %s", snap_dir)

    def print_standings(self):
        sorted_agents = sorted(self.agents, key=lambda a: -a.fitness)
        log.info("=== Standings ===")
        for rank, agent in enumerate(sorted_agents, 1):
            bar = "█" * int(agent.fitness * 20) + "░" * (20 - int(agent.fitness * 20))
            log.info(
                "  #%d Agent %d  %s %.1f%%  (gen %d)",
                rank, agent.agent_id, bar, agent.fitness * 100, agent.generation,
            )


def run_league(
    n_generations: int,
    checkpoint_dir: str,
    round_steps: int,
    eval_battles: int,
    n_agents: int,
    server_cfg,
    snapshot_every: int = 5,
):
    league = League(checkpoint_dir, n_agents=n_agents)

    for gen in range(1, n_generations + 1):
        log.info("======== Generation %d/%d ========", gen, n_generations)

        log.info("--- Training phase ---")
        league.train_all(server_cfg, round_steps)

        log.info("--- Evaluation phase ---")
        league.evaluate_all(server_cfg, eval_battles)

        league.print_standings()

        log.info("--- Exploit / Explore ---")
        league.exploit_explore()

        if gen % snapshot_every == 0:
            league.save_snapshot(gen)

    log.info("League complete.")
    league.print_standings()
