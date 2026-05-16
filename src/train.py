import argparse
import asyncio
import logging
import os
import shutil
import time

from poke_env import AccountConfiguration, ServerConfiguration

from battle.agents.random import RandomAgent
from battle.agents.heuristic import HeuristicAgent
from battle.agents.self_play import SelfPlayAgent
from battle_decisions.extractor import BattleExtractor
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

def _make_opponent(run_id: int, checkpoint: str | None):
    from sb3_contrib import MaskablePPO

    team = RotatingTeambuilder(*ALL_OPPONENT_TEAMS)

    if checkpoint and os.path.exists(checkpoint + ".zip"):
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
    from sb3_contrib import MaskablePPO
    from battle.env import make_env

    checkpoint_path = os.path.join(checkpoint_dir, "ppo_latest")
    os.makedirs(checkpoint_dir, exist_ok=True)
    steps_done = 0
    model = None
    env = None

    while steps_done < total_steps:
        steps_this_round = min(round_steps, total_steps - steps_done)
        run_id = int(time.time()) % 100000

        opponent = (
            _make_opponent(run_id, checkpoint_path if steps_done > 0 else None)
            if selfplay
            else HeuristicAgent(
                account_configuration=AccountConfiguration(f"Opp_{run_id}", None),
                battle_format=BATTLE_FORMAT,
                server_configuration=_server_cfg,
                team=RotatingTeambuilder(*ALL_OPPONENT_TEAMS),
                log_level=logging.WARNING,
                start_listening=False,
            )
        )

        log.info("Building environment (round %d/%d)...",
                 steps_done // round_steps + 1, -(-total_steps // round_steps))

        # Close the previous env before creating a new one to avoid websocket leaks
        if env is not None:
            env.close()
        env = make_env(_server_cfg, TEAM_TR_ALPHA, opponent)

        if model is None:
            log.info("obs_space=%s  action_space=%s", env.observation_space, env.action_space)
            if os.path.exists(checkpoint_path + ".zip"):
                log.info("Resuming from checkpoint %s", checkpoint_path)
                model = MaskablePPO.load(checkpoint_path, env=env, tensorboard_log=LOG_DIR)
            else:
                log.info("No checkpoint found, starting fresh")
                model = MaskablePPO(
                    "MlpPolicy",
                    env,
                    n_steps=512,
                    batch_size=64,
                    n_epochs=4,
                    learning_rate=3e-4,
                    gamma=0.99,
                    policy_kwargs=dict(
                        features_extractor_class=BattleExtractor,
                        features_extractor_kwargs=dict(features_dim=288),
                        net_arch=[128],
                    ),
                    tensorboard_log=LOG_DIR,
                    verbose=1,
                )
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
    """Build a fresh training env with a rotating HeuristicAgent opponent."""
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


def run_mutate(n_rounds: int, checkpoint_dir: str, train_steps: int, eval_battles: int):
    from sb3_contrib import MaskablePPO
    from genetic_algorithm.chromosome import mutate
    from genetic_algorithm.fitness import evaluate_team

    checkpoint_path = os.path.join(checkpoint_dir, "ppo_latest")
    temp_path = os.path.join(checkpoint_dir, "ppo_temp")
    best_team_path = os.path.join(checkpoint_dir, "best_team.txt")

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

    env = _make_train_env(_server_cfg, current_team)
    try:
        model = MaskablePPO.load(checkpoint_path, env=env, tensorboard_log=LOG_DIR)
    finally:
        env.close()

    log.info("Evaluating baseline team fitness...")
    best_fitness, best_breakdown = evaluate_team(model, _server_cfg, current_team, n_battles=eval_battles)
    log.info("Baseline weighted win rate: %.1f%%", best_fitness * 100)

    for round_num in range(1, n_rounds + 1):
        log.info("=== Mutation round %d/%d ===", round_num, n_rounds)

        candidate_team = mutate(current_team)

        model.save(temp_path)

        env = _make_train_env(_server_cfg, candidate_team)
        try:
            model.set_env(env)
            log.info("Retraining for %d steps with candidate team...", train_steps)
            model.learn(total_timesteps=train_steps, reset_num_timesteps=False)
        finally:
            env.close()

        log.info("Evaluating candidate team...")
        candidate_fitness, candidate_breakdown = evaluate_team(
            model, _server_cfg, candidate_team, n_battles=eval_battles
        )
        log.info("Candidate: %.1f%%  Baseline: %.1f%%",
                 candidate_fitness * 100, best_fitness * 100)

        if candidate_fitness > best_fitness:
            log.info("Accepted — +%.1f%%", (candidate_fitness - best_fitness) * 100)
            current_team = candidate_team
            best_fitness = candidate_fitness
            best_breakdown = candidate_breakdown
            model.save(checkpoint_path)
            with open(best_team_path, "w") as f:
                f.write(current_team)
            versioned_ckpt = os.path.join(
                checkpoint_dir, f"accepted_r{round_num:03d}_{int(time.time())}"
            )
            shutil.copy2(checkpoint_path + ".zip", versioned_ckpt + ".zip")
            with open(f"{versioned_ckpt}_team.txt", "w") as f:
                f.write(current_team)
            log.info("Versioned checkpoint → %s.zip", versioned_ckpt)
        else:
            log.info("Rejected — reverting")
            env = _make_train_env(_server_cfg, current_team)
            try:
                model = MaskablePPO.load(temp_path, env=env, tensorboard_log=LOG_DIR)
            finally:
                env.close()

    log.info("=== Mutation complete — best weighted win rate: %.1f%% ===", best_fitness * 100)
    log.info("Breakdown:")
    for name, wr in sorted(best_breakdown.items(), key=lambda x: -x[1]):
        bar = "█" * int(wr * 20) + "░" * (20 - int(wr * 20))
        log.info("  %-12s %s %.0f%%", name, bar, wr * 100)
    log.info("Best team saved to %s", best_team_path)


# ---------------------------------------------------------------------------
# Imitation learning pretrain
# ---------------------------------------------------------------------------

def run_pretrain(
    replay_dir: str,
    dataset_path: str,
    checkpoint_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    fetch_pages: int,
    player: int,
    device: str,
):
    from pathlib import Path
    from battle_decisions.imitation.parse_replay import BCDataset, parse_replay_dir
    from battle_decisions.imitation.bc_train import run_bc

    replay_path  = Path(replay_dir)
    dataset_file = Path(dataset_path)

    if fetch_pages > 0:
        from battle_decisions.imitation.fetch_replays import download_replays
        log.info("Fetching up to %d pages of replays into %s ...", fetch_pages, replay_path)
        download_replays(out_dir=replay_path, n_pages=fetch_pages)

    if dataset_file.exists():
        log.info("Loading cached dataset from %s", dataset_file)
        dataset = BCDataset.load(dataset_file)
    else:
        log.info("Parsing replays in %s ...", replay_path)
        if not replay_path.exists() or not list(replay_path.glob("*.log")):
            log.error(
                "No .log files found in %s.\n"
                "  • To fetch automatically:  --fetch-pages 20\n"
                "  • To fetch manually: python src/data/imitation/fetch_replays.py --out-dir data/replays",
                replay_path,
            )
            return
        dataset = parse_replay_dir(replay_path, player=player, save_path=dataset_file)

    if len(dataset) == 0:
        log.error("Dataset is empty — check replay logs and player index.")
        return

    log.info("Dataset ready: %d samples", len(dataset))

    run_bc(
        dataset_path=dataset_file,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device_str=device,
    )


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

    mutate_p = subparsers.add_parser("mutate", help="team mutation loop")
    mutate_p.add_argument("--rounds", type=int, default=10)
    mutate_p.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    mutate_p.add_argument("--train-steps", type=int, default=2000)
    mutate_p.add_argument("--eval-battles", type=int, default=10)

    pretrain_p = subparsers.add_parser("pretrain", help="BC pretrain from Showdown replays")
    pretrain_p.add_argument("--replay-dir",     default="data/replays")
    pretrain_p.add_argument("--dataset",        default="data/bc_dataset.npz")
    pretrain_p.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    pretrain_p.add_argument("--epochs",         type=int,   default=20)
    pretrain_p.add_argument("--batch-size",     type=int,   default=256)
    pretrain_p.add_argument("--lr",             type=float, default=1e-3)
    pretrain_p.add_argument("--fetch-pages",    type=int,   default=0)
    pretrain_p.add_argument("--player",         type=int,   default=1, choices=[1, 2])
    pretrain_p.add_argument("--device",         default="auto")

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
                   train_steps=args.train_steps, eval_battles=args.eval_battles)
    elif args.mode == "pretrain":
        run_pretrain(
            replay_dir=args.replay_dir,
            dataset_path=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fetch_pages=args.fetch_pages,
            player=args.player,
            device=args.device,
        )
    elif args.mode == "league":
        from battle_decisions.league.runner import run_league
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
