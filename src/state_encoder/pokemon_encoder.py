"""
Pokemon block encoders for EncoderV2.

Four block types, each returning a flat list[float]:

  encode_your_active(mon)     → YOUR_ACTIVE_DIM   per Pokémon (×2 on field)
  encode_your_bench(mon)      → YOUR_BENCH_DIM    per Pokémon (×2 on bench)
  encode_opp_active(mon)      → OPP_ACTIVE_DIM    per Pokémon (×2 on field)
  encode_opp_bench_slot(slot) → OPP_BENCH_DIM     per slot    (×4 non-active)

All continuous values normalised to 0–1.
Fainted / unrevealed slots → zero-padded with presence flags set.

Format: Pokémon Champions — Regulation M-A
  No IVs · SP (max 32 per stat) · Stat Alignment · Mega Evolution · No Tera
"""
from __future__ import annotations

from .constants import (
    N_TYPES, TYPE_IDX, STAT_NAMES, BOOST_STATS,
    HP_MAX, STAT_MAX, BASE_MAX, SP_MAX, BOOST_MAX,
    N_STATUS, STATUS_IDX,
    N_ITEM_TYPES,
    N_PRESENCE, PRESENCE_ORDER,
    TAUNT_MAX, ENCORE_MAX, TRAP_MAX, SLEEP_MAX, TOXIC_MAX,
)
from .ability_encoder import encode_ability, ABILITY_FEAT_DIM
from .item_encoder import encode_item, ITEM_DIM, ITEM_DIM_OPP
from .move_encoder import encode_move_slot, MoveDB, MOVE_SLOT_DIM
from .game_state import (
    YourActivePokemon, YourBenchPokemon,
    OppActivePokemon, OppBenchSlot,
    MoveState, PrimaryStatus, VolatileStatus,
    StatAlignment, ItemState, AbilityState,
)


# ─────────────────────────── helpers ─────────────────────────────────────────

def _type_vec(types: list[str]) -> list[float]:
    """Two type slots → 18+18 = 36 floats. Second slot all-0 if single type."""
    v = [0.0] * (N_TYPES * 2)
    for slot, t in enumerate(types[:2]):
        idx = TYPE_IDX.get(t)
        if idx is not None:
            v[slot * N_TYPES + idx] = 1.0
    return v


def _is_single_type(types: list[str]) -> float:
    return 1.0 if len(types) == 1 else 0.0


def _status_vec(status: PrimaryStatus) -> list[float]:
    """7-dim one-hot over STATUS_ORDER."""
    v = [0.0] * N_STATUS
    idx = STATUS_IDX.get(status.status, STATUS_IDX["healthy"])
    v[idx] = 1.0
    return v


def _boosts_vec(boosts: dict[str, int]) -> list[float]:
    """7 stat boosts: (boost+6)/12 → 0–1."""
    return [(boosts.get(s, 0) + 6) / 12.0 for s in BOOST_STATS]


def _volatile_vec(vol: VolatileStatus) -> list[float]:
    """15-dim volatile status block."""
    v = [
        float(vol.confused),
        float(vol.taunted),
        min(1.0, vol.taunt_turns / TAUNT_MAX),
        float(vol.encored),
        min(1.0, vol.encore_turns / ENCORE_MAX),
        vol.encore_move_slot / 3.0,
        float(vol.leech_seeded),
        float(vol.substitute),
        float(vol.partially_trapped),
        min(1.0, vol.trap_turns / TRAP_MAX),
        float(vol.infatuated),
        float(vol.disabled),
        vol.disabled_move_slot / 3.0,
        float(vol.charging),
        float(vol.identified),
    ]
    return v


def _status_extras(status: PrimaryStatus) -> list[float]:
    """Toxic counter (1) + sleep turns (1)."""
    return [
        min(1.0, status.toxic_counter / TOXIC_MAX),
        min(1.0, status.sleep_turns / SLEEP_MAX),
    ]


def _stat_alignment_vec(sa: StatAlignment) -> list[float]:
    """Boosted stat one-hot (6) + reduced stat one-hot (6)."""
    boost_vec  = [0.0] * 6
    reduce_vec = [0.0] * 6
    stat_idx = {s: i for i, s in enumerate(STAT_NAMES)}
    b = stat_idx.get(sa.boosted_stat)
    r = stat_idx.get(sa.reduced_stat)
    if b is not None:
        boost_vec[b] = 1.0
    if r is not None:
        reduce_vec[r] = 1.0
    return boost_vec + reduce_vec


def _sp_vec(sp: dict[str, int]) -> list[float]:
    """6 SP values normalised by SP_MAX (32)."""
    return [sp.get(s, 0) / SP_MAX for s in STAT_NAMES]


def _actual_stats_vec(stats: dict[str, int]) -> list[float]:
    """5 non-HP actual stats normalised (÷350)."""
    return [stats.get(s, 0) / STAT_MAX for s in ("atk", "def", "spa", "spd", "spe")]


def _base_stats_vec(base: dict[str, int]) -> list[float]:
    """6 base stats normalised by BASE_MAX (255)."""
    return [base.get(s, 0) / BASE_MAX for s in STAT_NAMES]


def _encode_moves(moves: list[MoveState], db: MoveDB, n: int = 4) -> list[float]:
    """Encode up to n move slots (zero-pad if fewer)."""
    v: list[float] = []
    for i in range(n):
        ms = moves[i] if i < len(moves) else None
        if ms is None:
            v.extend([0.0] * MOVE_SLOT_DIM)
        else:
            v.extend(encode_move_slot(ms, db))
    return v


def _choice_lock_vec(is_locked: bool, slot: int) -> list[float]:
    return [float(is_locked), slot / 3.0]


# ─────────────────────── your active Pokémon ─────────────────────────────────
# Layout (per slot):
#   type1(18) + type2(18) + is_single(1) + mega_evolved(1)          = 38
#   hp_ratio(1) + hp_max(1) + stats×5(5)                            = 7
#   sp×6(6)                                                          = 6
#   stat_alignment(12)                                               = 12
#   boosts(7)                                                        = 7
#   status_onehot(7) + toxic_ctr(1) + sleep_turns(1)                = 9
#   volatile(15)                                                     = 15
#   ability(ABILITY_FEAT_DIM=34) + has_activated(1)                  = 35
#   item(ITEM_DIM=19)                                                = 19
#   choice_lock(2)                                                   = 2
#   moves × 4 (MOVE_SLOT_DIM=60)                                     = 240
#   field_position(1) + has_mega_stone(1)                            = 2
#   TOTAL                                                            = 392

YOUR_ACTIVE_DIM = (
    38      # identity + typing
    + 7     # current stats
    + 6     # SP
    + 12    # stat alignment
    + 7     # boosts
    + 9     # primary status
    + 15    # volatile
    + 36    # ability (35) + has_activated (1)
    + 20    # item
    + 2     # choice lock
    + 4 * MOVE_SLOT_DIM  # 4 moves × 60
    + 2     # position + mega_stone
)


def encode_your_active(mon: YourActivePokemon, db: MoveDB) -> list[float]:
    v: list[float] = []

    # identity + typing (38)
    v.extend(_type_vec(mon.types))
    v.append(_is_single_type(mon.types))
    v.append(float(mon.has_mega_evolved))

    # current stats (7)
    v.append(mon.hp_ratio)
    v.append(min(1.0, mon.hp_max / HP_MAX))
    v.extend(_actual_stats_vec(mon.stats))

    # SP distribution (6)
    v.extend(_sp_vec(mon.sp))

    # stat alignment (12)
    v.extend(_stat_alignment_vec(mon.stat_alignment))

    # in-battle boosts (7)
    v.extend(_boosts_vec(mon.boosts))

    # primary status (9)
    v.extend(_status_vec(mon.status))
    v.extend(_status_extras(mon.status))

    # volatile (15)
    v.extend(_volatile_vec(mon.volatile))

    # ability (34) + has_activated (1) = 35
    v.extend(encode_ability(mon.ability.name, mon.ability.is_confirmed))
    v.append(float(mon.ability.has_activated))

    # item (19)
    v.extend(encode_item(mon.item, is_opponent=False))

    # choice lock (2)
    v.extend(_choice_lock_vec(mon.is_choice_locked, mon.choice_locked_slot))

    # moves × 4 (240)
    v.extend(_encode_moves(mon.moves, db))

    # position + mega stone (2)
    v.append(float(mon.field_position))
    v.append(float(mon.has_mega_stone))

    assert len(v) == YOUR_ACTIVE_DIM, f"your_active dim {len(v)} != {YOUR_ACTIVE_DIM}"
    return v


# ─────────────────────── your bench Pokémon ──────────────────────────────────
# Layout (per slot):
#   slot_occupied(1)                                                  = 1
#   type1(18) + type2(18) + is_single(1)                             = 37
#   hp_ratio(1) + hp_max(1) + stats×5(5)                            = 7
#   sp×6(6)                                                          = 6
#   stat_alignment(12)                                               = 12
#   status_onehot(7) + toxic(1) + sleep(1)                          = 9
#   ability(34)                                                      = 34
#   has_mega_stone(1) + can_mega_evolve(1)                           = 2
#   item(19)                                                         = 19
#   moves × 4 (60 each)                                              = 240
#   TOTAL                                                            = 367

YOUR_BENCH_DIM = (
    1       # occupied
    + 37    # type + single flag
    + 7     # hp + stats
    + 6     # SP
    + 12    # alignment
    + 9     # status
    + 35    # ability (35)
    + 2     # mega stone + can mega
    + 20    # item
    + 4 * MOVE_SLOT_DIM  # 4 moves × 60
)


def encode_your_bench(mon: YourBenchPokemon, db: MoveDB) -> list[float]:
    if mon.is_fainted:
        return [0.0] * YOUR_BENCH_DIM

    v: list[float] = []

    # slot occupied (1)
    v.append(1.0)

    # typing (37)
    v.extend(_type_vec(mon.types))
    v.append(_is_single_type(mon.types))

    # stats (7)
    v.append(mon.hp_ratio)
    v.append(min(1.0, mon.hp_max / HP_MAX))
    v.extend(_actual_stats_vec(mon.stats))

    # SP (6)
    v.extend(_sp_vec(mon.sp))

    # alignment (12)
    v.extend(_stat_alignment_vec(mon.stat_alignment))

    # primary status (9)
    v.extend(_status_vec(mon.status))
    v.extend(_status_extras(mon.status))

    # ability (34)
    v.extend(encode_ability(mon.ability.name, mon.ability.is_confirmed))

    # mega stone + can evolve (2)
    v.append(float(mon.has_mega_stone))
    v.append(float(mon.can_mega_evolve))

    # item (19)
    v.extend(encode_item(mon.item, is_opponent=False))

    # moves × 4 (240)
    v.extend(_encode_moves(mon.moves, db))

    assert len(v) == YOUR_BENCH_DIM, f"your_bench dim {len(v)} != {YOUR_BENCH_DIM}"
    return v


# ─────────────────────── opponent active Pokémon ─────────────────────────────
# Layout (per slot):
#   type1(18)+type2(18)+is_single(1)+mega_evolved(1)                 = 38
#   hp_ratio(1) + base_stats×6(6) + est_speed(1)                    = 8
#   boosts(7)                                                        = 7
#   status_onehot(7)+toxic(1)+sleep(1)                               = 9
#   volatile(15)                                                     = 15
#   ability(34)                                                      = 34
#   item(ITEM_DIM_OPP=20)                                            = 20
#   choice_lock(2)+lock_confirmed(1)                                 = 3
#   moves × 4 (60)                                                   = 240
#   TOTAL                                                            = 374

OPP_ACTIVE_DIM = (
    38      # identity
    + 8     # hp + base stats + speed
    + 7     # boosts
    + 9     # status
    + 15    # volatile
    + 35    # ability
    + 21    # item (with confidence)
    + 3     # choice lock + confirmed
    + 4 * MOVE_SLOT_DIM
)


def encode_opp_active(mon: OppActivePokemon, db: MoveDB) -> list[float]:
    v: list[float] = []

    # identity (38)
    v.extend(_type_vec(mon.types))
    v.append(_is_single_type(mon.types))
    v.append(float(mon.has_mega_evolved))

    # observable stats (8)
    v.append(mon.hp_ratio)
    v.extend(_base_stats_vec(mon.base_stats))
    v.append(min(1.0, mon.estimated_speed / STAT_MAX))

    # boosts (7)
    v.extend(_boosts_vec(mon.boosts))

    # status (9)
    v.extend(_status_vec(mon.status))
    v.extend(_status_extras(mon.status))

    # volatile (15)
    v.extend(_volatile_vec(mon.volatile))

    # ability (34)
    v.extend(encode_ability(mon.ability.name, mon.ability.is_confirmed))

    # item (20 with confidence)
    v.extend(encode_item(mon.item, is_opponent=True))

    # choice lock (3)
    v.append(float(mon.is_choice_locked))
    v.append(mon.choice_locked_slot / 3.0)
    v.append(float(mon.choice_lock_confirmed))

    # moves × 4 (240)
    v.extend(_encode_moves(mon.moves, db))

    assert len(v) == OPP_ACTIVE_DIM, f"opp_active dim {len(v)} != {OPP_ACTIVE_DIM}"
    return v


# ─────────────────────── opponent bench slot ─────────────────────────────────
# Layout (per slot):
#   presence_onehot(4) + p_present(1)                                = 5
#   type1(18)+type2(18)+is_single(1)                                 = 37
#   base_stats×6(6)                                                  = 6
#   hp_last(1) + hp_is_known(1)                                      = 2
#   status_onehot(7)+status_is_known(1)                              = 8
#   ability(34)                                                      = 34
#   has_mega_stone(1)+has_mega_evolved(1)                            = 2
#   item(20)                                                         = 20
#   moves × 4 (60)                                                   = 240
#   has_fainted(1)                                                   = 1
#   TOTAL                                                            = 355

OPP_BENCH_DIM = (
    5       # presence
    + 37    # types
    + 6     # base stats
    + 2     # hp last + known flag
    + 8     # status + known flag
    + 35    # ability
    + 2     # mega flags
    + 21    # item
    + 4 * MOVE_SLOT_DIM
    + 1     # fainted
)


def encode_opp_bench_slot(slot: OppBenchSlot, db: MoveDB) -> list[float]:
    v: list[float] = []

    # presence (5)
    pres_vec = [0.0] * len(PRESENCE_ORDER)
    pres_idx = PRESENCE_ORDER.index(slot.presence_state) if slot.presence_state in PRESENCE_ORDER else -1
    if pres_idx >= 0:
        pres_vec[pres_idx] = 1.0
    v.extend(pres_vec)
    v.append(min(1.0, max(0.0, slot.p_present)))

    # types (37)
    v.extend(_type_vec(slot.types))
    v.append(_is_single_type(slot.types))

    # base stats (6)
    v.extend(_base_stats_vec(slot.base_stats))

    # HP last seen (2)
    v.append(max(0.0, slot.hp_ratio_last_known) if slot.hp_is_known else 0.0)
    v.append(float(slot.hp_is_known))

    # status (8)
    v.extend(_status_vec(slot.status) if slot.status_is_known else [0.0] * N_STATUS)
    v.append(float(slot.status_is_known))

    # ability (34)
    v.extend(encode_ability(slot.ability.name, slot.ability.is_confirmed))

    # mega flags (2)
    v.append(float(slot.has_mega_stone))
    v.append(float(slot.has_mega_evolved))

    # item (20)
    v.extend(encode_item(slot.item, is_opponent=True))

    # moves × 4 (240)
    v.extend(_encode_moves(slot.moves, db))

    # fainted (1)
    v.append(float(slot.has_fainted))

    assert len(v) == OPP_BENCH_DIM, f"opp_bench dim {len(v)} != {OPP_BENCH_DIM}"
    return v
