"""
Move feature vector for EncoderV2.

Each move slot is encoded to MOVE_SLOT_DIM floats:
  MOVE_FEAT_DIM  — the 58-dim "what does this move do" vector
  + 1            — PP remaining (normalised)
  + 1            — is_selectable this turn

Total: MOVE_SLOT_DIM = 60

For unknown/prior opponent moves the caller passes a MoveState with
is_revealed=False; this module returns the prior (zero-padded with the
revealed flag set to 0.0).

All values normalised to 0–1 (or one-hot binary).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import (
    N_TYPES, TYPE_IDX,
    N_STATUS, STATUS_IDX,
    N_SECONDARY, SECONDARY_EFFECT_ORDER,
    N_STAT_TARGET, STAT_CHANGE_TARGET_ORDER,
)
from .move_flags import (
    SPREAD_TARGETS, ALLY_HIT_TARGETS,
    CONTACT_MOVES, SOUND_MOVES, PUNCH_MOVES, BITING_MOVES,
    BYPASSES_PROTECT, PROTECT_MOVES, RECOIL_MOVES,
    HEALING_MOVES, DRAINING_MOVES,
    WEATHER_SETTING_MOVES, TERRAIN_SETTING_MOVES,
    TRICK_ROOM_MOVES, TAILWIND_MOVES, REDIRECTION_MOVES,
    MULTIHIT_MOVES, TWO_TURN_MOVES, FIXED_DAMAGE_MOVES,
    STATUS_INFLICTING_MOVES, SECONDARY_EFFECTS, STAT_CHANGE_MOVES,
)
from .game_state import MoveState

# ── Dimensions ────────────────────────────────────────────────────────────────
# Move feature vector layout (64 dims):
#   base_power(1) + type_onehot(18) + category(3) + priority(1) + accuracy(1)
#   + pp_max(1) + is_spread(1) + hits_ally(1) + makes_contact(1) + is_sound(1)
#   + is_punch(1) + is_biting(1) + bypasses_protect(1) + is_protect(1)
#   + has_recoil(1) + is_healing(1) + is_draining(1) + sets_weather(1)
#   + sets_terrain(1) + sets_trick_room(1) + sets_tailwind(1) + is_redirection(1)
#   + inflicts_status(1) + status_onehot(7) + secondary_chance(1)
#   + secondary_type_onehot(6) + causes_stat_change(1) + stat_change_target(3)
#   + stat_change_is_boost(1) + is_multihit(1) + is_two_turn(1) + is_fixed(1)
# = 1+18+3+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+7+1+6+1+3+1+1+1+1 = 64
MOVE_FEAT_DIM  = 64
MOVE_SLOT_DIM  = MOVE_FEAT_DIM + 2   # + pp_remaining + is_selectable = 66


def _zeroed_slot() -> list[float]:
    """Return a zero-padded slot for an unrevealed / empty move."""
    return [0.0] * MOVE_SLOT_DIM


class MoveDB:
    """
    Loads moves.json and provides fast lookup by move name.

    Instantiate once and pass to encode_move().
    """

    def __init__(self, moves_json_path: str | Path) -> None:
        with open(moves_json_path) as f:
            raw = json.load(f)
        moves_list = raw.get("moves", []) if isinstance(raw, dict) else raw
        # index by lowercase name AND by id
        self._by_id:   dict[str, dict] = {}
        self._by_name: dict[str, dict] = {}
        for m in moves_list:
            if not isinstance(m, dict):
                continue
            if "id" in m:
                self._by_id[m["id"].lower()] = m
            if "name" in m:
                self._by_name[m["name"].lower()] = m

    def get(self, name: str) -> Optional[dict]:
        """Look up a move by name (case-insensitive) or Showdown id."""
        key = name.lower().replace(" ", "")
        return (
            self._by_name.get(name.lower())
            or self._by_id.get(key)
            or self._by_name.get(key)
        )


def encode_move_features(move_name: str, db: MoveDB) -> list[float]:
    """
    Return the 58-dim move feature vector for a given move name.

    Looks up move metadata from the DB, then applies hardcoded flag sets.
    Returns zeros for an empty or unknown move.
    """
    rec = db.get(move_name) if move_name else None
    if rec is None:
        return [0.0] * MOVE_FEAT_DIM

    name     = rec.get("name", move_name)
    category = rec.get("category", "Status")
    m_type   = rec.get("type", "")
    power    = float(rec.get("basePower") or 0)
    acc_raw  = rec.get("accuracy", True)
    accuracy = 1.0 if acc_raw is True else (float(acc_raw) / 100.0)
    pp_max   = float(rec.get("pp") or 0)
    priority = int(rec.get("priority") or 0)
    target   = rec.get("target", "normal")

    v: list[float] = []

    # 1 — base power (status moves = 0)
    v.append(power / 250.0)

    # 18 — type one-hot
    type_vec = [0.0] * N_TYPES
    idx = TYPE_IDX.get(m_type)
    if idx is not None:
        type_vec[idx] = 1.0
    v.extend(type_vec)

    # 3 — category
    v.append(1.0 if category == "Physical" else 0.0)
    v.append(1.0 if category == "Special"  else 0.0)
    v.append(1.0 if category == "Status"   else 0.0)

    # 1 — priority tier: (priority + 7) / 12
    v.append((priority + 7) / 12.0)

    # 1 — accuracy
    v.append(accuracy)

    # 1 — PP max normalised
    v.append(min(1.0, pp_max / 40.0))

    # 1 — is spread (hits both foes; power > 0 guard keeps status moves out)
    v.append(1.0 if (target in SPREAD_TARGETS and power > 0) else 0.0)

    # 1 — hits ally too (Earthquake, Surf, etc.)
    v.append(1.0 if (target in ALLY_HIT_TARGETS and power > 0) else 0.0)

    # 1 — makes contact
    v.append(1.0 if name in CONTACT_MOVES else 0.0)

    # 1 — is sound-based
    v.append(1.0 if name in SOUND_MOVES else 0.0)

    # 1 — is punch-based
    v.append(1.0 if name in PUNCH_MOVES else 0.0)

    # 1 — is biting move
    v.append(1.0 if name in BITING_MOVES else 0.0)

    # 1 — bypasses protect
    v.append(1.0 if name in BYPASSES_PROTECT else 0.0)

    # 1 — is protect variant
    v.append(1.0 if name in PROTECT_MOVES else 0.0)

    # 1 — has recoil
    v.append(1.0 if name in RECOIL_MOVES else 0.0)

    # 1 — is healing move
    v.append(1.0 if name in HEALING_MOVES else 0.0)

    # 1 — is draining move
    v.append(1.0 if name in DRAINING_MOVES else 0.0)

    # 1 — sets weather
    v.append(1.0 if name in WEATHER_SETTING_MOVES else 0.0)

    # 1 — sets terrain
    v.append(1.0 if name in TERRAIN_SETTING_MOVES else 0.0)

    # 1 — sets Trick Room
    v.append(1.0 if name in TRICK_ROOM_MOVES else 0.0)

    # 1 — sets Tailwind
    v.append(1.0 if name in TAILWIND_MOVES else 0.0)

    # 1 — is redirection move
    v.append(1.0 if name in REDIRECTION_MOVES else 0.0)

    # 1 + 7 — inflicts primary status + which status
    status_entry = STATUS_INFLICTING_MOVES.get(name)
    v.append(1.0 if status_entry else 0.0)
    status_vec = [0.0] * N_STATUS
    if status_entry:
        s_idx = STATUS_IDX.get(status_entry[0])
        if s_idx is not None:
            status_vec[s_idx] = 1.0
    v.extend(status_vec)

    # 1 + 6 — secondary effect chance + type
    sec_entry = SECONDARY_EFFECTS.get(name)
    v.append(sec_entry[1] if sec_entry else 0.0)
    sec_type_vec = [0.0] * N_SECONDARY
    if sec_entry and sec_entry[1] > 0.0:
        s_idx = SECONDARY_EFFECT_ORDER.index(sec_entry[0]) if sec_entry[0] in SECONDARY_EFFECT_ORDER else -1
        if s_idx >= 0:
            sec_type_vec[s_idx] = 1.0
    else:
        sec_type_vec[SECONDARY_EFFECT_ORDER.index("none")] = 1.0
    v.extend(sec_type_vec)

    # 1 + 3 + 1 — causes stat change + target + is boost
    stat_entry = STAT_CHANGE_MOVES.get(name)
    v.append(1.0 if stat_entry else 0.0)
    target_vec = [0.0] * N_STAT_TARGET
    if stat_entry:
        t_idx = STAT_CHANGE_TARGET_ORDER.index(stat_entry[0]) if stat_entry[0] in STAT_CHANGE_TARGET_ORDER else -1
        if t_idx >= 0:
            target_vec[t_idx] = 1.0
    v.extend(target_vec)
    v.append(1.0 if (stat_entry and stat_entry[1]) else 0.0)

    # 1 — is multi-hit
    v.append(1.0 if name in MULTIHIT_MOVES else 0.0)

    # 1 — is two-turn
    v.append(1.0 if name in TWO_TURN_MOVES else 0.0)

    # 1 — is fixed-damage
    v.append(1.0 if name in FIXED_DAMAGE_MOVES else 0.0)

    assert len(v) == MOVE_FEAT_DIM, f"move feature dim mismatch: {len(v)} != {MOVE_FEAT_DIM}"
    return v


def encode_move_slot(move: MoveState, db: MoveDB) -> list[float]:
    """
    Return a MOVE_SLOT_DIM (60) vector for one move slot.

    For unrevealed opponent moves (is_revealed=False) the feature vector
    is zeroed and PP / selectable are also zero.
    """
    if not move.is_revealed:
        # Prior slot — all zeros; caller may blend with usage-stat priors later
        return _zeroed_slot()

    feats = encode_move_features(move.move_name, db)

    # PP remaining normalised
    if move.pp_max > 0:
        pp_ratio = max(0.0, min(1.0, move.pp_remaining / move.pp_max))
    else:
        pp_ratio = 0.0

    feats.append(pp_ratio)
    feats.append(1.0 if move.is_selectable else 0.0)
    return feats


def encode_prior_move_slot(
    move_name: str,
    usage_pct: float,
    times_used: int,
    pp_max: int,
    db: MoveDB,
) -> list[float]:
    """
    Encode an opponent prior move slot (not yet revealed in battle).

    move_name:  most likely move from usage data
    usage_pct:  0.0–1.0 usage probability (used to scale feature confidence)
    times_used: observed uses so far (for PP pressure estimate)
    """
    feats = encode_move_features(move_name, db)

    # PP pressure: times used / pp_max as proxy for depletion
    pp_pressure = min(1.0, times_used / max(1, pp_max))
    feats.append(pp_pressure)

    # is_selectable: assume selectable for prior (0.0 flag = not revealed)
    feats.append(0.0)   # 0 = not confirmed selectable (unrevealed)
    return feats
