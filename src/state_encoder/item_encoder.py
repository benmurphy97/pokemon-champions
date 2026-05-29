"""
Item feature vector for EncoderV2.

ITEM_DIM = 19 floats:
  has_item(1) + is_mega_stone(1) + is_choice(1) + is_berry(1)
  + berry_hp_trigger(1) + berry_threshold(1) + is_life_orb(1)
  + is_assault_vest(1) + is_rocky_helmet(1) + is_eject(1)
  + boosts_stat(1) + provides_speed_boost(1)
  + item_type_onehot(8)
  = 19

An extra confidence(1) float is appended only for opponent slots
(ITEM_DIM_OPP = 20). For your own Pokémon confidence is always 1.0 and
is omitted to keep the block layout clean.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .constants import ITEM_TYPE_ORDER, N_ITEM_TYPES
from .game_state import ItemState

# ── Item category sets (populated from champions_items.json at build time) ────

_MEGA_STONES: set[str] = set()
_CHOICE_ITEMS: set[str] = {"Choice Band", "Choice Specs", "Choice Scarf"}
_BERRIES: set[str] = set()
_PINCH_BERRIES: set[str] = {
    "Sitrus Berry", "Oran Berry",
    "Figy Berry", "Wiki Berry", "Mago Berry", "Aguav Berry", "Iapapa Berry",
    "Liechi Berry", "Ganlon Berry", "Salac Berry", "Petaya Berry",
    "Apicot Berry", "Lansat Berry", "Starf Berry", "Enigma Berry",
    "Micle Berry", "Custap Berry",
}
_LIFE_ORB: set[str] = {"Life Orb"}
_ASSAULT_VEST: set[str] = {"Assault Vest"}
_ROCKY_HELMET: set[str] = {"Rocky Helmet"}
_EJECT_ITEMS: set[str] = {"Eject Button", "Eject Pack"}
_STAT_BOOSTING: set[str] = {
    "Choice Band", "Choice Specs",
    "Muscle Band", "Wise Glasses",
    "Charcoal", "Black Belt", "Black Glasses", "Soft Sand", "Mystic Water",
    "Never-Melt Ice", "Silk Scarf", "Silver Powder", "Miracle Seed",
    "Hard Stone", "Magnet", "Metal Coat", "Poison Barb", "Sharp Beak",
    "Twisted Spoon", "Dragon Fang", "Fairy Feather",
    "Thick Club", "Deep Sea Tooth", "Deep Sea Scale", "Light Ball",
    "Soul Dew",  # in mainline; may not be in Champions
}
_CHOICE_SCARF: set[str] = {"Choice Scarf"}

# Seed items (terrain seeds — consume on terrain activation)
_SEEDS: set[str] = {
    "Electric Seed", "Grassy Seed", "Misty Seed", "Psychic Seed",
}

# Held-battle items (consumed mid-battle)
_HELD_BATTLE: set[str] = {
    "Focus Sash", "White Herb", "Power Herb", "Mental Herb",
    "Eject Button", "Eject Pack",
}

# Held-passive items (long-term passive effects, not consumed)
_HELD_PASSIVE: set[str] = {
    "Leftovers", "Shell Bell", "Rocky Helmet",
    "Assault Vest",
    "Life Orb",
    "Throat Spray",
    "Terrain Extender", "Utility Umbrella",
    "Heavy-Duty Boots", "Room Service",
    "King's Rock", "Scope Lens",
    "Safety Goggles",
    "Clear Amulet",
    "Loaded Dice",
    "Booster Energy",
}


def load_item_categories(items_json_path: str | Path) -> None:
    """
    Populate module-level category sets from champions_items.json.
    Call once at startup.
    """
    global _MEGA_STONES, _BERRIES

    with open(items_json_path) as f:
        raw = json.load(f)

    cats = raw.get("categories", {})
    _MEGA_STONES = set(cats.get("mega_stones", []))

    berry_cats = ["recovery", "defense_berries"]
    _BERRIES = set()
    for cat in berry_cats:
        _BERRIES.update(cats.get(cat, []))
    # Any item ending in "Berry" is a berry regardless of category
    for cat_items in cats.values():
        for item in cat_items:
            if item.endswith("Berry"):
                _BERRIES.add(item)


def _item_type_onehot(item_name: str) -> list[float]:
    """8-dim one-hot for item type category."""
    v = [0.0] * N_ITEM_TYPES
    if not item_name:
        v[ITEM_TYPE_ORDER.index("none")] = 1.0
        return v
    if item_name in _CHOICE_ITEMS or item_name == "Choice Scarf":
        v[ITEM_TYPE_ORDER.index("choice")] = 1.0
    elif item_name in _BERRIES or item_name in _PINCH_BERRIES:
        v[ITEM_TYPE_ORDER.index("berry")] = 1.0
    elif item_name in _LIFE_ORB:
        v[ITEM_TYPE_ORDER.index("orb")] = 1.0
    elif item_name in _MEGA_STONES:
        v[ITEM_TYPE_ORDER.index("mega_stone")] = 1.0
    elif item_name in _SEEDS:
        v[ITEM_TYPE_ORDER.index("seed")] = 1.0
    elif item_name in _HELD_BATTLE:
        v[ITEM_TYPE_ORDER.index("held_battle")] = 1.0
    elif item_name in _HELD_PASSIVE:
        v[ITEM_TYPE_ORDER.index("held_passive")] = 1.0
    else:
        v[ITEM_TYPE_ORDER.index("none")] = 1.0
    return v


# ── Dimensions ────────────────────────────────────────────────────────────────

ITEM_DIM     = 20   # 12 binary flags + 8-dim item_type one-hot
ITEM_DIM_OPP = 21   # +1 confidence float for opponent slots


def encode_item(item: ItemState, is_opponent: bool = False) -> list[float]:
    """
    Return ITEM_DIM (19) or ITEM_DIM_OPP (20) floats describing the held item.

    item.name == "" → no item held.
    """
    name = item.name if not item.is_consumed else ""

    v: list[float] = []

    # 1 — has item (0 after consumed / knocked off)
    v.append(1.0 if name else 0.0)

    # 1 — is Mega Stone (consumed on Mega Evolution)
    v.append(1.0 if name in _MEGA_STONES else 0.0)

    # 1 — is choice item
    v.append(1.0 if name in _CHOICE_ITEMS else 0.0)

    # 1 — is berry (any berry)
    is_berry = name in _BERRIES or name in _PINCH_BERRIES
    v.append(1.0 if is_berry else 0.0)

    # 1 — berry triggers on HP threshold
    is_pinch = name in _PINCH_BERRIES
    v.append(1.0 if is_pinch else 0.0)

    # 1 — berry threshold (approx fraction; Sitrus=0.5, pinch berries=0.25)
    if name == "Sitrus Berry" or name == "Oran Berry":
        berry_thresh = 0.5
    elif is_pinch:
        berry_thresh = 0.25
    else:
        berry_thresh = 0.0
    v.append(berry_thresh)

    # 1 — is Life Orb
    v.append(1.0 if name in _LIFE_ORB else 0.0)

    # 1 — is Assault Vest
    v.append(1.0 if name in _ASSAULT_VEST else 0.0)

    # 1 — is Rocky Helmet
    v.append(1.0 if name in _ROCKY_HELMET else 0.0)

    # 1 — is Eject Button / Eject Pack
    v.append(1.0 if name in _EJECT_ITEMS else 0.0)

    # 1 — boosts a specific attacking stat
    v.append(1.0 if name in _STAT_BOOSTING else 0.0)

    # 1 — provides speed boost (Choice Scarf)
    v.append(1.0 if name in _CHOICE_SCARF else 0.0)

    # 8 — item type one-hot
    v.extend(_item_type_onehot(name))

    assert len(v) == ITEM_DIM, f"item dim {len(v)} != {ITEM_DIM}"

    if is_opponent:
        v.append(item.confidence)   # 1 — confidence (opponent only)

    return v
