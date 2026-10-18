"""The team pools training and evaluation draw from, and how to load them.

Ladder teams are rebuilt from replays (ladder_pool.py); tournament teams come from the scrapers
(tournament_pool.py). Each pool file is a JSON list of {id, species, text, ...}.
"""

import json
from pathlib import Path

from champions.config import ROOT

LADDER_TRAIN = ROOT / "results" / "ladder_teams_train.json"   # real ladder teams of every kind
LADDER_EVAL = ROOT / "results" / "ladder_teams_eval.json"     # held out from training
WEB_TEAMS = ROOT / "results" / "web_teams.json"               # tournament teams, with a train/eval split


def load_pool(path: Path, **filters) -> list[dict]:
    """Teams from a pool file, optionally only those whose fields match, e.g. split="train"."""
    return [t for t in json.loads(path.read_text()) if all(t.get(k) == v for k, v in filters.items())]
