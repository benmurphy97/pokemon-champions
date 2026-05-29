import argparse
import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime

from poke_env import AccountConfiguration, ServerConfiguration

from battle.agents.random import RandomAgent
from battle.agents.heuristic import HeuristicAgent
from battle.agents.self_play import SelfPlayAgent
from stage3_battle_decisions.architecture.extractor import BattleStateEncoder
from config import BATTLE_FORMAT, SHOWDOWN_URL, MAX_CONCURRENT_BATTLES, CHECKPOINT_DIR, LOG_DIR
from battle.teams import (
    TEAM_TR_ALPHA, TEAM_WOLFE, TEAM_CYBERTRON,
    TEAM_ROD, TEAM_YUZU, TEAM_KURO, TEAM_WDMICHAEL, TEAM_HYUNGWOO, TEAM_JORGE,
    RotatingTeambuilder,
)

ALL_OPPONENT_TEAMS = [
    TEAM_WOLFE, TEAM_CYBERTRON, TEAM_ROD, TEAM_YUZU,
    TEAM_KURO, TEAM_WDMICHAEL, TEAM_HYUNGWOO, TEAM_JORGE,
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("poke_env").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

_server_cfg = ServerConfiguration(
    SHOWDOWN_URL,
    "https://play.pokemonshowdown.com/~~localhost/action.php?",
)


# ---------------------------------------------------------------------------
# Baseline: two random agents battle each other (smoke test / benchmark)
# ---------------------------------------------------------------------------

async def run_baseline(n_battles: int, verbose: bool = False):
    if verbose:
        logging.getLogger("agents").setLevel(logging.DEBUG)

    run_id = int(time.time()) % 100000
    p1 = RandomAgent(
        account_configuration=AccountConfiguration(f"P1_{run_id}", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=_server_cfg,
        team=TEAM_TR_ALPHA,
        max_concurrent_battles=MAX_CONCURRENT_BATTLES,
        log_level=logging.WARNING,
    )
    p2 = RandomAgent(
        account_configuration=AccountConfiguration(f"P2_{run_id}", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=_server_cfg,
        team=TEAM_TR_ALPHA,
        max_concurrent_battles=MAX_CONCURRENT_BATTLES,
        log_level=logging.WARNING,
    )

    log.info("Starting %d baseline battles in %s", n_battles, BATTLE_FORMAT)
    await p1.battle_against(p2, n_battles=n_battles)

    log.info("--- Baseline Results ---")
    log.info("P1: %d wins (%.1f%%)", p1.n_won_battles, p1.win_rate * 100)
    log.info("P2: %d wins (%.1f%%)", p2.n_won_battles, p2.win_rate * 100)


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def _load_or_init_model(checkpoint_path: str, env, selfplay: bool):
    """Load checkpoint if compatible, otherwise init fresh. Returns model."""
    from sb3_contrib import MaskablePPO

    policy_kwargs = dict(
        features_extractor_class=BattleStateEncoder,
        features_extractor_kwargs=dict(features_dim=288),
        net_arch=[128],
    )

    if os.path.exists(checkpoint_path + ".zip"):
        try:
            log.info("Resuming from checkpoint %s", checkpoint_path)
            return MaskablePPO.load(checkpoint_path, env=env, tensorboard_log=LOG_DIR)
        except Exception as exc:
            log.warning(
                "Checkpoint load failed (%s) — starting fresh. "
                "This usually means the obs space changed (e.g. encoder upgrade).",
                exc,
            )

    log.info("No compatible checkpoint found, starting fresh")
    return MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )


def _make_opponent(run_id: int, checkpoint: str | None, selfplay: bool):
    team = RotatingTeambuilder(*ALL_OPPONENT_TEAMS)

    if selfplay and checkpoint and os.path.exists(checkpoint + ".zip"):
        from sb3_contrib import MaskablePPO
        log.info("Self-play opponent: loading %s", checkpoint)
        model = MaskablePPO.load(checkpoint)
        return SelfPlayAgent(
            model=model,
            account_configuration=AccountConfiguration(f"Opp_{run_id}", None),
            battle_format=BATTLE_FORMAT,
            server_configuration=_server_cfg,
            team=team,
            log_level=logging.WARNING,
            start_listening=False,
        )

    log.info("Self-play opponent: no checkpoint found, using heuristic")
    return HeuristicAgent(
        account_configuration=AccountConfiguration(f"Opp_{run_id}", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=_server_cfg,
        team=team,
        log_level=logging.WARNING,
        start_listening=False,
    )


def run_ppo(total_steps: int, checkpoint_dir: str = CHECKPOINT_DIR,
            selfplay: bool = False, round_steps: int = 5000):
    from battle.env import make_env

    checkpoint_path = os.path.join(checkpoint_dir, "ppo_latest")
    os.makedirs(checkpoint_dir, exist_ok=True)
    steps_done = 0
    model = None
    env = None

    while steps_done < total_steps:
        steps_this_round = min(round_steps, total_steps - steps_done)
        run_id = int(time.time()) % 100000

        opponent = _make_opponent(
            run_id,
            checkpoint_path if steps_done > 0 else None,
            selfplay,
        )

        log.info("Building environment (round %d/%d)...",
                 steps_done // round_steps + 1, -(-total_steps // round_steps))

        if env is not None:
            env.close()
        env = make_env(_server_cfg, TEAM_TR_ALPHA, opponent)

        if model is None:
            log.info("obs_space=%s  action_space=%s", env.observation_space, env.action_space)
            model = _load_or_init_model(checkpoint_path, env, selfplay)
        else:
            model.set_env(env)

        log.info("Training for %d steps (total %d/%d)...",
                 steps_this_round, steps_done + steps_this_round, total_steps)
        model.learn(total_timesteps=steps_this_round, reset_num_timesteps=False)
        steps_done += steps_this_round

        model.save(checkpoint_path)
        log.info("Saved checkpoint → %s.zip", checkpoint_path)

    if env is not None:
        env.close()


# ---------------------------------------------------------------------------
# Evaluation: win rate vs each opponent team
# ---------------------------------------------------------------------------

def run_evaluate(checkpoint: str, n_battles: int = 20,
                 team_file: str | None = None, selfplay: bool = False):
    from sb3_contrib import MaskablePPO
    from battle.env import make_env

    model = MaskablePPO.load(checkpoint)
    log.info("Loaded checkpoint: %s", checkpoint)

    if team_file:
        with open(team_file) as f:
            our_team = f.read()
        log.info("Using team from %s", team_file)
    else:
        our_team = TEAM_TR_ALPHA
        log.info("Using default TEAM_TR_ALPHA")

    opponents = {
        "WOLFE (Sand)":          TEAM_WOLFE,
        "CYBERTRON (Sun)":       TEAM_CYBERTRON,
        "ROD (Snow/TR)":         TEAM_ROD,
        "YUZU (Perish Trap)":    TEAM_YUZU,
        "KURO (Rain)":           TEAM_KURO,
        "WDMICHAEL (TR Golurk)": TEAM_WDMICHAEL,
        "HYUNGWOO (Sun)":        TEAM_HYUNGWOO,
        "JORGE (CharX/MegaTtar)":TEAM_JORGE,
    }

    results = {}
    for label, opp_team in opponents.items():
        run_id = int(time.time()) % 100000
        if selfplay:
            opponent = SelfPlayAgent(
                model=model,
                account_configuration=AccountConfiguration(f"Eval_{run_id}", None),
                battle_format=BATTLE_FORMAT,
                server_configuration=_server_cfg,
                team=opp_team,
                log_level=logging.WARNING,
                start_listening=False,
            )
        else:
            opponent = HeuristicAgent(
                account_configuration=AccountConfiguration(f"Eval_{run_id}", None),
                battle_format=BATTLE_FORMAT,
                server_configuration=_server_cfg,
                team=opp_team,
                log_level=logging.WARNING,
                start_listening=False,
            )
        env = make_env(_server_cfg, our_team, opponent)

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

        results[label] = (wins, n_battles)
        log.info("vs %-26s %2d/%d wins  (%.0f%%)", label, wins, n_battles, wins / n_battles * 100)

    log.info("--- Summary ---")
    for label, (wins, total) in sorted(results.items(), key=lambda x: -x[1][0]):
        bar = "█" * wins + "░" * (total - wins)
        log.info("%-26s %s %d%%", label, bar, wins / total * 100)


# ---------------------------------------------------------------------------
# Team mutation loop
# ---------------------------------------------------------------------------

def _make_train_env(server_cfg, team: str):
    from battle.env import make_env
    run_id = int(time.time()) % 100000
    opponent = HeuristicAgent(
        account_configuration=AccountConfiguration(f"Mut_{run_id}", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=server_cfg,
        team=RotatingTeambuilder(*ALL_OPPONENT_TEAMS),
        log_level=logging.WARNING,
        start_listening=False,
    )
    return make_env(server_cfg, team, opponent)


def _append_history(path: str, entry: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_mutate(n_rounds: int, checkpoint_dir: str, eval_battles: int):
    """
    Evolve the team using the current frozen policy as the fitness signal.
    The policy is never updated here — use 'train' for that.

    Intended workflow:
      1. train  — improve in-battle decision-making
      2. mutate — find a structurally better team (policy unchanged)
      3. train  — adapt policy to the new team
      4. repeat
    """
    from sb3_contrib import MaskablePPO
    from stage4_team_evolution.chromosome import mutate
    from stage4_team_evolution.fitness import evaluate_team

    checkpoint_path = os.path.join(checkpoint_dir, "ppo_latest")
    best_team_path = os.path.join(checkpoint_dir, "best_team.txt")
    history_path = os.path.join(checkpoint_dir, "evolution_log.jsonl")

    if not os.path.exists(checkpoint_path + ".zip"):
        log.info("No checkpoint — run 'train' first to get a baseline model")
        return

    if os.path.exists(best_team_path):
        with open(best_team_path) as f:
            current_team = f.read()
        log.info("Loaded best team from %s", best_team_path)
    else:
        current_team = TEAM_TR_ALPHA
        log.info("Starting from TEAM_TR_ALPHA")

    # Load the model once and keep it frozen throughout — no training here
    env = _make_train_env(_server_cfg, current_team)
    try:
        model = MaskablePPO.load(checkpoint_path, env=env)
    finally:
        env.close()

    log.info("Evaluating baseline team fitness (policy frozen)...")
    best_fitness, best_breakdown = evaluate_team(model, _server_cfg, current_team, n_battles=eval_battles)
    log.info("Baseline weighted win rate: %.1f%%", best_fitness * 100)

    for round_num in range(1, n_rounds + 1):
        log.info("=== Mutation round %d/%d ===", round_num, n_rounds)

        candidate_team = mutate(current_team)

        log.info("Evaluating candidate team (policy frozen)...")
        candidate_fitness, candidate_breakdown = evaluate_team(
            model, _server_cfg, candidate_team, n_battles=eval_battles
        )
        log.info("Candidate: %.1f%%  Baseline: %.1f%%",
                 candidate_fitness * 100, best_fitness * 100)

        accepted = candidate_fitness > best_fitness
        _append_history(history_path, {
            "ts":             datetime.utcnow().isoformat(timespec="seconds"),
            "event":          "mutation_accepted" if accepted else "mutation_rejected",
            "generation":     round_num,
            "fitness_before": best_fitness,
            "fitness_after":  candidate_fitness,
            "team_before":    current_team,
            "team_after":     candidate_team,
        })

        if accepted:
            log.info("Accepted — +%.1f%%", (candidate_fitness - best_fitness) * 100)
            current_team = candidate_team
            best_fitness = candidate_fitness
            best_breakdown = candidate_breakdown
            with open(best_team_path, "w") as f:
                f.write(current_team)
            versioned = os.path.join(
                checkpoint_dir, f"accepted_r{round_num:03d}_{int(time.time())}_team.txt"
            )
            shutil.copy2(best_team_path, versioned)
            log.info("Team snapshot → %s", versioned)
        else:
            log.info("Rejected")

    log.info("=== Mutation complete — best weighted win rate: %.1f%% ===", best_fitness * 100)
    log.info("Breakdown:")
    for name, wr in sorted(best_breakdown.items(), key=lambda x: -x[1]):
        bar = "█" * int(wr * 20) + "░" * (20 - int(wr * 20))
        log.info("  %-12s %s %.0f%%", name, bar, wr * 100)
    log.info("Best team saved to %s", best_team_path)
    log.info("Evolution log → %s", history_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    baseline = subparsers.add_parser("baseline", help="run two random agents")
    baseline.add_argument("--episodes", type=int, default=10)
    baseline.add_argument("--verbose", action="store_true")

    train = subparsers.add_parser("train", help="PPO training")
    train.add_argument("--steps", type=int, default=10000)
    train.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    train.add_argument("--selfplay", action="store_true")
    train.add_argument("--round-steps", type=int, default=5000)

    evaluate = subparsers.add_parser("evaluate", help="win rate vs each opponent team")
    evaluate.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "ppo_latest"))
    evaluate.add_argument("--battles", type=int, default=20)
    evaluate.add_argument("--team", default=None, help="path to team.txt (defaults to TEAM_TR_ALPHA)")
    evaluate.add_argument("--selfplay", action="store_true")

    mutate_p = subparsers.add_parser(
        "mutate",
        help="evolve team composition using the frozen policy — no gradient updates",
    )
    mutate_p.add_argument("--rounds", type=int, default=10)
    mutate_p.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    mutate_p.add_argument("--eval-battles", type=int, default=10)

    league_p = subparsers.add_parser("league", help="PBT league — multiple agents, exploit/explore")
    league_p.add_argument("--generations", type=int, default=10)
    league_p.add_argument("--agents", type=int, default=5)
    league_p.add_argument("--checkpoint-dir", default=os.path.join(CHECKPOINT_DIR, "league"))
    league_p.add_argument("--round-steps", type=int, default=3000)
    league_p.add_argument("--eval-battles", type=int, default=8)
    league_p.add_argument("--snapshot-every", type=int, default=5)

    args = parser.parse_args()

    if args.mode == "baseline":
        asyncio.run(run_baseline(args.episodes, verbose=args.verbose))
    elif args.mode == "train":
        run_ppo(args.steps, checkpoint_dir=args.checkpoint_dir,
                selfplay=args.selfplay, round_steps=args.round_steps)
    elif args.mode == "evaluate":
        run_evaluate(args.checkpoint, n_battles=args.battles,
                     team_file=args.team, selfplay=args.selfplay)
    elif args.mode == "mutate":
        run_mutate(args.rounds, checkpoint_dir=args.checkpoint_dir,
                   eval_battles=args.eval_battles)
    elif args.mode == "league":
        from stage3_battle_decisions.league.runner import run_league
        run_league(
            n_generations=args.generations,
            checkpoint_dir=args.checkpoint_dir,
            round_steps=args.round_steps,
            eval_battles=args.eval_battles,
            n_agents=args.agents,
            server_cfg=_server_cfg,
            snapshot_every=args.snapshot_every,
        )


if __name__ == "__main__":
    main()
