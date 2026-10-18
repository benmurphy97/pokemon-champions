"""Estimate a nature and stat points for a Pokémon whose sheet does not list them (open team sheets).

Uses in-game Ranked Battle data (import_battledata.py) first, then Showdown ladder usage (import_stats.py).
A missing nature is the most popular in-game one that does not lower the stat the moves attack with.
The in-game data lists natures and spreads separately, so the spread is the most popular one that fits
the nature and the moves: invests in the attacking stat the moves use and the nature raises, not in a
stat the nature lowers, and no Speed with a Speed-lowering (Trick Room) nature.
"""

import json
from functools import cache
from pathlib import Path

from champions.config import REGULATION
from champions.teams.generate import load_usage

ROOT = Path(__file__).resolve().parents[3]
BATTLE_DATA = ROOT / "data" / "meta" / REGULATION.lower() / "battle_data.json"
REFERENCE = ROOT / "data" / "reference"
STAT_KEY = {"atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe", "hp": "HP"}
DEFAULT = {"HP": 32, "Atk": 0, "Def": 0, "SpA": 0, "SpD": 0, "Spe": 2}


@cache
def tables() -> tuple[dict, dict, dict, dict]:
    battle = json.loads(BATTLE_DATA.read_text())["pokemon"] if BATTLE_DATA.exists() else {}
    natures = {name: (STAT_KEY.get(n["plus"]), STAT_KEY.get(n["minus"]))
               for name, n in json.loads((REFERENCE / "natures.json").read_text()).items()}
    categories = {name: m["category"] for name, m in json.loads((REFERENCE / "moves.json").read_text()).items()}
    return battle, load_usage(), natures, categories


def attacking_stat(moves: list[str]) -> str | None:
    """'Atk' or 'SpA' by which kind of attack the moves are mostly; None for an all-status set."""
    categories = tables()[3]
    kinds = [categories.get(move) for move in moves]
    physical, special = kinds.count("Physical"), kinds.count("Special")
    return None if physical == special == 0 else ("SpA" if special > physical else "Atk")


def fits(points: dict, nature: str | None, attack: str | None) -> bool:
    plus, minus = tables()[2].get(nature, (None, None))
    if minus in ("Atk", "SpA") and points[minus] > 8:
        return False
    if plus in ("Atk", "SpA") and (points[plus] < max(points["Atk"], points["SpA"]) or (plus == attack and points[plus] == 0)):
        return False
    if minus == "Spe" and points["Spe"] > 0:
        return False
    if attack and plus not in ("Atk", "SpA") and points[{"Atk": "SpA", "SpA": "Atk"}[attack]] > points[attack]:
        return False
    return True


def estimate(name: str, nature: str | None, moves: list[str]) -> tuple[str | None, dict]:
    """(nature, stat points) for this Pokémon; keeps the given nature if there is one."""
    battle, usage, natures, _ = tables()
    attack = attacking_stat(moves)
    entry = battle.get(name) or battle.get(name.split("-")[0])
    if entry:
        if not nature:
            ranked = [n for n, _ in entry["natures"]]
            nature = next((n for n in ranked if attack is None or natures.get(n, (None, None))[1] != attack), ranked[0] if ranked else None)
        spreads = [points for points, _ in entry["spreads"]]
    else:
        ladder = usage.get(name) or usage.get(name.split("-")[0]) or {}
        pairs = ladder.get("spreads") or []
        if not nature:
            nature = next((n for n, p, _ in pairs if fits(p, n, attack)), pairs[0][0] if pairs else None)
        spreads = [points for n, points, _ in pairs if n == nature] + [points for _, points, _ in pairs]
    points = next((p for p in spreads if fits(p, nature, attack)), spreads[0] if spreads else None)
    if points is None:
        nature = nature or {"Atk": "Adamant", "SpA": "Modest"}.get(attack, "Bold")
        points = dict(DEFAULT, **({attack: 32} if attack else {"Def": 32}))
    return nature, points
