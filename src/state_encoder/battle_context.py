"""
Battle context encoder for EncoderV2.

CONTEXT_DIM = 118 floats:

Momentum & score (8):
  turn(1) + our_remaining(1) + opp_remaining(1)
  + our_hp_total(1) + opp_hp_total(1) + ko_diff(1)
  + mega_used_ours(1) + mega_used_opp(1)

Last-turn actions — 4 slots × our_p1, our_p2, opp_p1, opp_p2 (88):
  Per slot (22):
    used_protect(1) + switched(1) + used_mega(1)
    + move_type_onehot(18) + damage_taken_ratio(1)

Revealed information tracker — per opponent roster slot × 6 (24):
  has_been_active(1) + moves_revealed_count(1)
  + item_confirmed(1) + ability_confirmed(1)

TOTAL = 8 + 4×22 + 6×4 = 8 + 88 + 24 = 120

Note: brief says ~130; exact count is 120.
"""
from __future__ import annotations

from .constants import (
    N_TYPES, TYPE_IDX,
    TURN_MAX,
)
from .game_state import BattleContext, LastTurnSlot

CONTEXT_DIM = 120


def _last_turn_slot_vec(slot: LastTurnSlot) -> list[float]:
    """22-dim vector for one active slot's last-turn actions."""
    v: list[float] = []

    v.append(float(slot.used_protect))
    v.append(float(slot.switched))
    v.append(float(slot.used_mega))

    # Move type one-hot (18) — all zero if switched or turn 1
    type_vec = [0.0] * N_TYPES
    if slot.move_type:
        idx = TYPE_IDX.get(slot.move_type)
        if idx is not None:
            type_vec[idx] = 1.0
    v.extend(type_vec)

    v.append(min(1.0, max(0.0, slot.damage_taken_ratio)))

    return v  # 22 dims


def encode_context(ctx: BattleContext) -> list[float]:
    """Return CONTEXT_DIM (120) floats for the battle context block."""
    v: list[float] = []

    # Momentum & score (8)
    v.append(min(1.0, ctx.turn / TURN_MAX))
    v.append(ctx.our_pokemon_remaining / 4.0)
    v.append(ctx.opp_pokemon_remaining / 4.0)
    v.append(min(1.0, max(0.0, ctx.our_total_hp_ratio)))
    v.append(min(1.0, max(0.0, ctx.opp_total_hp_ratio)))
    # KO differential: (our_KOs - their_KOs + 4) / 8 → 0–1
    ko_diff_norm = (ctx.ko_differential + 4) / 8.0
    v.append(min(1.0, max(0.0, ko_diff_norm)))
    v.append(float(ctx.battle_mega_used_ours))
    v.append(float(ctx.battle_mega_used_opp))

    # Last-turn actions — 4 slots × 22 = 88
    for slot in (ctx.our_p1_last, ctx.our_p2_last, ctx.opp_p1_last, ctx.opp_p2_last):
        v.extend(_last_turn_slot_vec(slot))

    # Revealed information — 6 roster slots × 4 = 24
    for i in range(6):
        v.append(float(ctx.opp_has_been_active[i] if i < len(ctx.opp_has_been_active) else False))
        revealed = ctx.opp_moves_revealed_count[i] if i < len(ctx.opp_moves_revealed_count) else 0
        v.append(min(1.0, revealed / 4.0))
        v.append(float(ctx.opp_item_confirmed[i] if i < len(ctx.opp_item_confirmed) else False))
        v.append(float(ctx.opp_ability_confirmed[i] if i < len(ctx.opp_ability_confirmed) else False))

    assert len(v) == CONTEXT_DIM, f"context dim {len(v)} != {CONTEXT_DIM}"
    return v
