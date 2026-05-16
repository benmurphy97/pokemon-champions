import os
from pathlib import Path

# Project root is two levels up from src/config.py
_PROJECT_ROOT = Path(__file__).parent.parent

SHOWDOWN_URL = os.getenv("SHOWDOWN_URL", "ws://localhost:8000/showdown/websocket")

BATTLE_FORMAT = "gen9championsvgc2026regma"

N_BATTLES = 10
MAX_CONCURRENT_BATTLES = 1

N_MAIN_AGENTS = 5
N_EXPLOITER_AGENTS = 3

CHECKPOINT_INTERVAL = 500
CHECKPOINT_DIR = str(_PROJECT_ROOT / "model_outputs" / "checkpoints")
CACHE_DIR      = str(_PROJECT_ROOT / "model_outputs" / "cache")
LOG_DIR        = str(_PROJECT_ROOT / "logs")

# Reserved for future PBT opponent sampling once historical snapshots and
# attack-only bots are implemented. Currently the league uses HeuristicAgent
# opponents exclusively — these weights are not yet wired up.
OPPONENT_SAMPLING = {
    "main_agents": 0.35,
    "exploiters": 0.25,
    "historical": 0.25,
    "attack_only": 0.15,
}
