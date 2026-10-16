"""Vocabularies and static lookup tables built from data/reference.

Index 0 is always "unknown / padding". Static tables describe each Pokémon and move (types, base
stats, power, ...) so the policy can generalise to ones it has seen rarely.
"""

import json
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).resolve().parents[2] / "data" / "reference"

TYPES = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground",
         "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]
CATEGORIES = ["Physical", "Special", "Status"]
TARGETS = ["normal", "self", "adjacentAlly", "adjacentAllyOrSelf", "adjacentFoe", "allAdjacentFoes",
           "allAdjacent", "allySide", "foeSide", "all", "allies", "any", "randomNormal", "scripted",
           "allyTeam"]


def _load(name: str) -> dict:
    return json.loads((REFERENCE / name).read_text())


_pokemon, _moves, _items = _load("pokemon.json"), _load("moves.json"), _load("items.json")

SPECIES = ["<unk>"] + list(_pokemon)
MOVES = ["<pad>"] + list(_moves)
ITEMS = ["<unknown>", "<gone>"] + list(_items)
ABILITIES = ["<unknown>"] + sorted({a for mon in _pokemon.values() for a in mon["abilities"]})

SPECIES_ID = {n: i for i, n in enumerate(SPECIES)}
MOVE_ID = {n: i for i, n in enumerate(MOVES)}
ITEM_ID = {n: i for i, n in enumerate(ITEMS)}
ABILITY_ID = {n: i for i, n in enumerate(ABILITIES)}


def _type_id(name: str) -> int:
    return TYPES.index(name) + 1 if name in TYPES else 0


# species_types[s] = (type1, type2) as 1-based ids (0 = none); species_stats[s] = base stats / 255
SPECIES_TYPES = np.zeros((len(SPECIES), 2), dtype=np.int64)
SPECIES_STATS = np.zeros((len(SPECIES), 6), dtype=np.float32)
for _name, _mon in _pokemon.items():
    _i = SPECIES_ID[_name]
    for _k, _t in enumerate(_mon["types"][:2]):
        SPECIES_TYPES[_i, _k] = _type_id(_t)
    SPECIES_STATS[_i] = [_mon["baseStats"][k] / 255 for k in ("hp", "atk", "def", "spa", "spd", "spe")]

# MOVE_CATS: type id, category id, target id (all 1-based so 0 = padding); MOVE_NUM: power, accuracy, priority
MOVE_CATS = np.zeros((len(MOVES), 3), dtype=np.int64)
MOVE_NUM = np.zeros((len(MOVES), 3), dtype=np.float32)
for _name, _mv in _moves.items():
    _i = MOVE_ID[_name]
    MOVE_CATS[_i] = [_type_id(_mv["type"]), CATEGORIES.index(_mv["category"]) + 1,
                     TARGETS.index(_mv["target"]) + 1 if _mv["target"] in TARGETS else 0]
    _acc = 1.0 if _mv["accuracy"] is True else (_mv["accuracy"] or 100) / 100
    MOVE_NUM[_i] = [(_mv["basePower"] or 0) / 150, _acc, max(-1.0, min(1.0, _mv["priority"] / 3))]

# Which moves each species can learn (rows of a species x move boolean table), for masking.
LEARNABLE = np.zeros((len(SPECIES), len(MOVES)), dtype=bool)
for _name, _mon in _pokemon.items():
    for _m in _mon["moves"]:
        if _m in MOVE_ID:
            LEARNABLE[SPECIES_ID[_name], MOVE_ID[_m]] = True

# TYPE_CHART[attacking id, defending id] = damage multiplier; id 0 (no type) is neutral both ways.
_chart = _load("typechart.json")
TYPE_CHART = np.ones((len(TYPES) + 1, len(TYPES) + 1), dtype=np.float32)
for _a in TYPES:
    for _d in TYPES:
        TYPE_CHART[_type_id(_a), _type_id(_d)] = _chart[_a][_d]

def to_id(name: str) -> str:
    """Showdown's id form of a name: 'Close Combat' -> 'closecombat'."""
    return "".join(c for c in name.lower() if c.isalnum())


NAME_BY_ID = {to_id(n): n for n in MOVES[1:] + ITEMS[2:] + ABILITIES[1:]}

NUM_SPECIES, NUM_MOVES, NUM_ITEMS, NUM_ABILITIES = len(SPECIES), len(MOVES), len(ITEMS), len(ABILITIES)
