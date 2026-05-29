"""
battle_adapter.py — poke-env DoubleBattle → GameState translator.

Converts a live poke-env battle object into the GameState dataclasses
consumed by EncoderV2. Called each step from PokemonEnv.embed_battle().

Usage:
    from state_encoder.battle_adapter import battle_to_game_state
    state = battle_to_game_state(battle)
    obs   = encoder.encode(state)
"""
from __future__ import annotations

from typing import Optional

from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.pokemon import Pokemon as PokeMon
from poke_env.battle.move import Move as PokeMove
from poke_env.battle.status import Status
from poke_env.battle.effect import Effect
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition

from .game_state import (
    GameState, YourActivePokemon, YourBenchPokemon,
    OppActivePokemon, OppBenchSlot,
    FieldState, BattleContext, ScreenState, LastTurnSlot,
    MoveState, VolatileStatus, PrimaryStatus, StatAlignment,
    ItemState, AbilityState,
)

# ── Item ID → display name ────────────────────────────────────────────────────
# poke-env returns item IDs as lowercase with no spaces/hyphens/apostrophes.
# We maintain a reverse map so item_encoder flag sets (display names) match.

_ALL_ITEMS: set[str] = {
    # Choice
    "Choice Band", "Choice Specs", "Choice Scarf",
    # Berries (recovery + pinch)
    "Sitrus Berry", "Oran Berry",
    "Figy Berry", "Wiki Berry", "Mago Berry", "Aguav Berry", "Iapapa Berry",
    "Liechi Berry", "Ganlon Berry", "Salac Berry", "Petaya Berry",
    "Apicot Berry", "Lansat Berry", "Starf Berry", "Enigma Berry",
    "Micle Berry", "Custap Berry",
    "Lum Berry", "Chesto Berry", "Pecha Berry", "Rawst Berry", "Aspear Berry",
    "Cheri Berry", "Persim Berry", "Leppa Berry",
    # Combat passives
    "Life Orb", "Assault Vest", "Rocky Helmet",
    "Eject Button", "Eject Pack",
    "Muscle Band", "Wise Glasses",
    "Charcoal", "Black Belt", "Black Glasses", "Soft Sand", "Mystic Water",
    "Never-Melt Ice", "Silk Scarf", "Silver Powder", "Miracle Seed",
    "Hard Stone", "Magnet", "Metal Coat", "Poison Barb", "Sharp Beak",
    "Twisted Spoon", "Dragon Fang", "Fairy Feather",
    "Thick Club", "Deep Sea Tooth", "Deep Sea Scale", "Light Ball",
    "Soul Dew", "Expert Belt",
    # Seeds
    "Electric Seed", "Grassy Seed", "Misty Seed", "Psychic Seed",
    # Battle consumables
    "Focus Sash", "White Herb", "Power Herb", "Mental Herb",
    # Held passives
    "Leftovers", "Shell Bell", "Throat Spray",
    "Terrain Extender", "Utility Umbrella",
    "Heavy-Duty Boots", "Room Service",
    "King's Rock", "Scope Lens", "Wide Lens", "Zoom Lens",
    "Safety Goggles", "Clear Amulet", "Loaded Dice",
    "Booster Energy",
}

def _item_key(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "").replace("'", "")

_ITEM_ID_TO_NAME: dict[str, str] = {_item_key(n): n for n in _ALL_ITEMS}


def _poke_item_name(poke_id: str) -> str:
    """Convert a poke-env item ID ('lifeorb') to display name ('Life Orb')."""
    if not poke_id or poke_id == "unknown_item":
        return ""
    key = _item_key(poke_id)
    if key in _ITEM_ID_TO_NAME:
        return _ITEM_ID_TO_NAME[key]
    # Mega stones: loaded lazily from item_encoder after load_item_categories()
    try:
        from .item_encoder import _MEGA_STONES
        for stone in _MEGA_STONES:
            if _item_key(stone) == key:
                _ITEM_ID_TO_NAME[key] = stone   # cache for next time
                return stone
    except Exception:
        pass
    return ""


# ── Status ────────────────────────────────────────────────────────────────────

_STATUS_MAP: dict = {
    Status.BRN: "brn",
    Status.PAR: "par",
    Status.PSN: "psn",
    Status.TOX: "tox",
    Status.FRZ: "frz",
    Status.SLP: "slp",
    Status.FNT: "healthy",
}


def _primary_status(mon: Optional[PokeMon]) -> PrimaryStatus:
    if mon is None or mon.status is None:
        return PrimaryStatus()
    s = mon.status
    status_str = _STATUS_MAP.get(s, "healthy")
    toxic = mon.status_counter if s == Status.TOX else 0
    sleep = mon.status_counter if s == Status.SLP else 0
    return PrimaryStatus(status=status_str, toxic_counter=toxic, sleep_turns=sleep)


# ── Types ─────────────────────────────────────────────────────────────────────

def _mon_types(mon: Optional[PokeMon]) -> list[str]:
    if mon is None:
        return []
    return [t.name.title() for t in (mon.types or []) if t is not None]


# ── Volatile status ───────────────────────────────────────────────────────────

def _volatile(mon: Optional[PokeMon]) -> VolatileStatus:
    if mon is None:
        return VolatileStatus()
    fx: dict = mon.effects or {}
    return VolatileStatus(
        confused=Effect.CONFUSION in fx,
        taunted=Effect.TAUNT in fx,
        taunt_turns=fx.get(Effect.TAUNT, 0),
        encored=Effect.ENCORE in fx,
        encore_turns=fx.get(Effect.ENCORE, 0),
        leech_seeded=Effect.LEECH_SEED in fx,
        substitute=Effect.SUBSTITUTE in fx,
        partially_trapped=Effect.PARTIALLY_TRAPPED in fx,
        trap_turns=fx.get(Effect.PARTIALLY_TRAPPED, 0),
        disabled=Effect.DISABLE in fx,
        charging=getattr(mon, "preparing", False) or False,
    )


# ── Moves ─────────────────────────────────────────────────────────────────────

_EMPTY_MOVE = MoveState(
    move_id="", move_name="", pp_remaining=0, pp_max=1, is_revealed=False
)


def _move_state(move: PokeMove, selectable: bool) -> MoveState:
    pp_max = move.max_pp or 1
    pp_cur = move.current_pp if move.current_pp is not None else pp_max
    return MoveState(
        move_id=move.id,
        move_name=move.id,   # MoveDB normalises lowercase IDs
        pp_remaining=max(0, pp_cur),
        pp_max=pp_max,
        is_revealed=True,
        is_selectable=selectable,
    )


def _pad4(moves: list[MoveState]) -> list[MoveState]:
    result = moves[:4]
    while len(result) < 4:
        result.append(_EMPTY_MOVE)
    return result


def _your_moves(mon: PokeMon, available: list[PokeMove]) -> list[MoveState]:
    avail_ids = {m.id for m in available}
    return _pad4([_move_state(mv, mv.id in avail_ids) for mv in list(mon.moves.values())[:4]])


def _opp_moves(mon: PokeMon) -> list[MoveState]:
    return _pad4([_move_state(mv, True) for mv in list(mon.moves.values())[:4]])


# ── Item / ability helpers ────────────────────────────────────────────────────

def _your_item(mon: PokeMon) -> ItemState:
    raw = mon.item or ""
    name = _poke_item_name(raw)
    confirmed = bool(raw and raw != "unknown_item")
    return ItemState(name=name, is_confirmed=confirmed, is_consumed=False)


def _opp_item(mon: PokeMon) -> ItemState:
    raw = mon.item or ""
    name = _poke_item_name(raw)
    confirmed = bool(raw and raw != "unknown_item")
    return ItemState(name=name, is_confirmed=confirmed, confidence=1.0 if confirmed else 0.5)


def _your_ability(mon: PokeMon) -> AbilityState:
    return AbilityState(name=mon.ability or "", is_confirmed=True, has_activated=False)


def _opp_ability(mon: PokeMon) -> AbilityState:
    name = mon.ability or ""
    confirmed = bool(name)
    return AbilityState(name=name, is_confirmed=confirmed, has_activated=False)


def _is_mega(mon: PokeMon) -> bool:
    return mon.species != mon.base_species


def _has_mega_stone(mon: PokeMon) -> bool:
    try:
        from .item_encoder import _MEGA_STONES
        return _poke_item_name(mon.item or "") in _MEGA_STONES
    except Exception:
        return False


def _safe_stats(mon: PokeMon) -> dict[str, int]:
    raw = mon.stats or {}
    return {k: int(raw.get(k) or 0) for k in ("atk", "def", "spa", "spd", "spe")}


_EMPTY_SP = {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
_EMPTY_BOOSTS = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}


# ── Your active ───────────────────────────────────────────────────────────────

def _build_your_active(
    mon: Optional[PokeMon],
    slot: int,
    available: list[PokeMove],
) -> YourActivePokemon:
    if mon is None or mon.fainted:
        return YourActivePokemon(
            species=mon.species if mon else "",
            types=[],
            has_mega_evolved=False,
            has_mega_stone=False,
            hp_ratio=0.0,
            hp_max=0,
            stats={k: 0 for k in ("atk", "def", "spa", "spd", "spe")},
            sp=dict(_EMPTY_SP),
            stat_alignment=StatAlignment(),
            boosts=dict(_EMPTY_BOOSTS),
            status=PrimaryStatus(),
            volatile=VolatileStatus(),
            ability=AbilityState(),
            item=ItemState(),
            moves=_pad4([]),
            is_choice_locked=False,
            choice_locked_slot=0,
            field_position=slot,
        )

    mega = _is_mega(mon)
    locked = Effect.LOCKED_MOVE in (mon.effects or {})
    avail_ids = [m.id for m in available]

    choice_slot = 0
    if locked and len(avail_ids) == 1:
        for i, mv in enumerate(list(mon.moves.values())[:4]):
            if mv.id == avail_ids[0]:
                choice_slot = i
                break

    return YourActivePokemon(
        species=mon.species,
        types=_mon_types(mon),
        has_mega_evolved=mega,
        has_mega_stone=_has_mega_stone(mon) and not mega,
        hp_ratio=mon.current_hp_fraction,
        hp_max=mon.max_hp or 0,
        stats=_safe_stats(mon),
        sp=dict(_EMPTY_SP),
        stat_alignment=StatAlignment(),
        boosts=dict(mon.boosts or _EMPTY_BOOSTS),
        status=_primary_status(mon),
        volatile=_volatile(mon),
        ability=_your_ability(mon),
        item=_your_item(mon),
        moves=_your_moves(mon, available),
        is_choice_locked=locked,
        choice_locked_slot=choice_slot,
        field_position=slot,
    )


# ── Your bench ────────────────────────────────────────────────────────────────

def _build_your_bench(mon: Optional[PokeMon], battle_mega_used: bool) -> YourBenchPokemon:
    if mon is None:
        return YourBenchPokemon(
            species="", types=[], is_fainted=True,
            hp_ratio=0.0, hp_max=0,
            stats={k: 0 for k in ("atk", "def", "spa", "spd", "spe")},
            sp=dict(_EMPTY_SP), stat_alignment=StatAlignment(),
            status=PrimaryStatus(), ability=AbilityState(),
            has_mega_stone=False, can_mega_evolve=False,
            item=ItemState(), moves=_pad4([]),
        )

    is_fainted = mon.fainted
    mega = _is_mega(mon)
    stone = _has_mega_stone(mon)

    return YourBenchPokemon(
        species=mon.species,
        types=_mon_types(mon),
        is_fainted=is_fainted,
        hp_ratio=0.0 if is_fainted else mon.current_hp_fraction,
        hp_max=mon.max_hp or 0,
        stats=_safe_stats(mon),
        sp=dict(_EMPTY_SP),
        stat_alignment=StatAlignment(),
        status=_primary_status(mon),
        ability=_your_ability(mon),
        has_mega_stone=stone and not mega,
        can_mega_evolve=stone and not mega and not battle_mega_used,
        item=_your_item(mon),
        moves=_pad4([_move_state(mv, True) for mv in list(mon.moves.values())[:4]]),
        has_mega_evolved=mega,
    )


# ── Opponent active ───────────────────────────────────────────────────────────

def _build_opp_active(mon: Optional[PokeMon]) -> OppActivePokemon:
    if mon is None or mon.fainted:
        return OppActivePokemon(
            species=mon.species if mon else "",
            types=[],
            has_mega_evolved=False,
            hp_ratio=0.0,
            base_stats={k: 0 for k in ("hp", "atk", "def", "spa", "spd", "spe")},
            estimated_speed=80,
            boosts=dict(_EMPTY_BOOSTS),
            status=PrimaryStatus(),
            volatile=VolatileStatus(),
            ability=AbilityState(),
            item=ItemState(),
            moves=_pad4([]),
            is_choice_locked=False,
            choice_locked_slot=0,
            choice_lock_confirmed=False,
        )

    mega = _is_mega(mon)
    raw_base = mon.base_stats or {}
    base = {k: int(raw_base.get(k) or 0) for k in ("hp", "atk", "def", "spa", "spd", "spe")}
    locked = Effect.LOCKED_MOVE in (mon.effects or {})

    return OppActivePokemon(
        species=mon.species,
        types=_mon_types(mon),
        has_mega_evolved=mega,
        hp_ratio=mon.current_hp_fraction,
        base_stats=base,
        estimated_speed=base.get("spe", 80),
        boosts=dict(mon.boosts or _EMPTY_BOOSTS),
        status=_primary_status(mon),
        volatile=_volatile(mon),
        ability=_opp_ability(mon),
        item=_opp_item(mon),
        moves=_opp_moves(mon),
        is_choice_locked=locked,
        choice_locked_slot=0,
        choice_lock_confirmed=False,
    )


# ── Opponent bench ────────────────────────────────────────────────────────────

_ABSENT_BENCH = OppBenchSlot(
    species="",
    presence_state="possibly_brought",
    p_present=0.5,
    types=[],
    base_stats={k: 0 for k in ("hp", "atk", "def", "spa", "spd", "spe")},
    hp_ratio_last_known=-1.0,
    hp_is_known=False,
    status=PrimaryStatus(),
    status_is_known=False,
    ability=AbilityState(is_confirmed=False),
    has_mega_stone=False,
    has_mega_evolved=False,
    item=ItemState(is_confirmed=False, confidence=0.0),
    moves=_pad4([]),
    has_fainted=False,
)


def _build_opp_bench(mon: PokeMon) -> OppBenchSlot:
    raw_base = mon.base_stats or {}
    base = {k: int(raw_base.get(k) or 0) for k in ("hp", "atk", "def", "spa", "spd", "spe")}
    mega = _is_mega(mon)
    presence = "confirmed_fainted" if mon.fainted else "confirmed_benched"

    return OppBenchSlot(
        species=mon.species,
        presence_state=presence,
        p_present=0.0 if mon.fainted else 1.0,
        types=_mon_types(mon),
        base_stats=base,
        hp_ratio_last_known=mon.current_hp_fraction if not mon.fainted else 0.0,
        hp_is_known=True,
        status=_primary_status(mon),
        status_is_known=True,
        ability=_opp_ability(mon),
        has_mega_stone=_has_mega_stone(mon) and not mega,
        has_mega_evolved=mega,
        item=_opp_item(mon),
        moves=_opp_moves(mon),
        has_fainted=mon.fainted,
    )


# ── Field ─────────────────────────────────────────────────────────────────────

_WEATHER_MAP: dict = {
    Weather.SUNNYDAY:      "sun",
    Weather.DESOLATELAND:  "sun",
    Weather.RAINDANCE:     "rain",
    Weather.PRIMORDIALSEA: "rain",
    Weather.SANDSTORM:     "sand",
    Weather.SNOWSCAPE:     "snow",
    Weather.HAIL:          "snow",
}

_TERRAIN_MAP: dict = {
    Field.ELECTRIC_TERRAIN: "electric",
    Field.GRASSY_TERRAIN:   "grassy",
    Field.MISTY_TERRAIN:    "misty",
    Field.PSYCHIC_TERRAIN:  "psychic",
}

_PERM_WEATHER = {Weather.DESOLATELAND, Weather.PRIMORDIALSEA, Weather.DELTASTREAM}


def _screen(sc: SideCondition, side: dict) -> ScreenState:
    turns = side.get(sc, 0)
    return ScreenState(active=sc in side, turns_remaining=turns)


def _build_field(battle: DoubleBattle) -> FieldState:
    weather_dict: dict = battle.weather or {}
    fields: dict = battle.fields or {}
    our_sc: dict = battle.side_conditions or {}
    opp_sc: dict = battle.opponent_side_conditions or {}

    # Weather
    weather_str = "none"
    weather_turns = 0
    weather_perm = False
    for w, t in weather_dict.items():
        mapped = _WEATHER_MAP.get(w)
        if mapped:
            weather_str = mapped
            weather_turns = t
            weather_perm = w in _PERM_WEATHER
            break

    # Terrain
    terrain_str = "none"
    terrain_turns = 0
    for f, t in fields.items():
        mapped = _TERRAIN_MAP.get(f)
        if mapped:
            terrain_str = mapped
            terrain_turns = t
            break

    return FieldState(
        weather=weather_str,
        weather_turns=weather_turns,
        weather_permanent=weather_perm,
        weather_set_by_our_side=False,   # not trackable from poke-env obs alone
        terrain=terrain_str,
        terrain_turns=terrain_turns,
        trick_room_active=Field.TRICK_ROOM in fields,
        trick_room_turns=fields.get(Field.TRICK_ROOM, 0),
        our_tailwind=SideCondition.TAILWIND in our_sc,
        our_tailwind_turns=our_sc.get(SideCondition.TAILWIND, 0),
        opp_tailwind=SideCondition.TAILWIND in opp_sc,
        opp_tailwind_turns=opp_sc.get(SideCondition.TAILWIND, 0),
        our_light_screen=_screen(SideCondition.LIGHT_SCREEN, our_sc),
        our_reflect=_screen(SideCondition.REFLECT, our_sc),
        opp_light_screen=_screen(SideCondition.LIGHT_SCREEN, opp_sc),
        opp_reflect=_screen(SideCondition.REFLECT, opp_sc),
        gravity_active=Field.GRAVITY in fields,
        gravity_turns=fields.get(Field.GRAVITY, 0),
        our_safeguard=_screen(SideCondition.SAFEGUARD, our_sc),
        opp_safeguard=_screen(SideCondition.SAFEGUARD, opp_sc),
        magic_room_active=Field.MAGIC_ROOM in fields,
        magic_room_turns=fields.get(Field.MAGIC_ROOM, 0),
        wonder_room_active=Field.WONDER_ROOM in fields,
        wonder_room_turns=fields.get(Field.WONDER_ROOM, 0),
    )


# ── Battle context ────────────────────────────────────────────────────────────

def _build_context(battle: DoubleBattle) -> BattleContext:
    our_team = list(battle.team.values())
    opp_team = list(battle.opponent_team.values()) if battle.opponent_team else []

    our_alive = [m for m in our_team if not m.fainted]
    opp_alive = [m for m in opp_team if not m.fainted]
    our_fainted = sum(1 for m in our_team if m.fainted)
    opp_fainted = sum(1 for m in opp_team if m.fainted)

    our_hp = sum(m.current_hp_fraction for m in our_alive) / max(1, len(our_alive))
    opp_hp = sum(m.current_hp_fraction for m in opp_alive) / max(1, len(opp_alive)) if opp_alive else 1.0

    return BattleContext(
        turn=battle.turn,
        our_pokemon_remaining=len(our_alive),
        opp_pokemon_remaining=len(opp_alive),
        our_total_hp_ratio=our_hp,
        opp_total_hp_ratio=opp_hp,
        ko_differential=opp_fainted - our_fainted,
        battle_mega_used_ours=getattr(battle, "used_mega_evolve", False) or False,
        battle_mega_used_opp=getattr(battle, "opponent_used_mega_evolve", False) or False,
        # Last-turn actions not trackable from single-step obs; leave defaults
        our_p1_last=LastTurnSlot(),
        our_p2_last=LastTurnSlot(),
        opp_p1_last=LastTurnSlot(),
        opp_p2_last=LastTurnSlot(),
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def battle_to_game_state(battle: DoubleBattle) -> GameState:
    """
    Translate a poke-env DoubleBattle into a GameState for EncoderV2.

    Handles:
      - Your active × 2, bench × 2
      - Opponent active × 2, bench × 4 (with 6-pick-4 uncertainty for unseen slots)
      - Field state, battle context
    """
    # Active pokemon (may contain None during forced-switch turns or early state)
    try:
        our_active = list(battle.active_pokemon or [None, None])
    except (ValueError, AttributeError):
        our_active = [None, None]
    while len(our_active) < 2:
        our_active.append(None)

    try:
        opp_active = list(battle.opponent_active_pokemon or [None, None])
    except (ValueError, AttributeError):
        opp_active = [None, None]
    while len(opp_active) < 2:
        opp_active.append(None)

    # Available moves per slot (List[List[Move]])
    avail = battle.available_moves or [[], []]
    avail0 = avail[0] if len(avail) > 0 else []
    avail1 = avail[1] if len(avail) > 1 else []

    mega_used = getattr(battle, "used_mega_evolve", False) or False

    # ── Your active ──────────────────────────────────────────────────────────
    your_active = [
        _build_your_active(our_active[0], 0, avail0),
        _build_your_active(our_active[1], 1, avail1),
    ]

    # ── Your bench (2 non-active, non-leading slots) ─────────────────────────
    active_set = {id(m) for m in our_active if m is not None}
    bench_mons = [m for m in battle.team.values() if id(m) not in active_set]
    # Sort: alive first, then fainted
    bench_mons.sort(key=lambda m: (m.fainted, m.species))
    bench_slots: list[YourBenchPokemon] = []
    for i in range(2):
        mon = bench_mons[i] if i < len(bench_mons) else None
        bench_slots.append(_build_your_bench(mon, mega_used))

    # ── Opponent active ──────────────────────────────────────────────────────
    opp_active_enc = [
        _build_opp_active(opp_active[0]),
        _build_opp_active(opp_active[1]),
    ]

    # ── Opponent bench (4 non-active slots; pad with uncertainty) ────────────
    opp_active_set = {id(m) for m in opp_active if m is not None}
    opp_bench_mons = [
        m for m in (battle.opponent_team or {}).values()
        if id(m) not in opp_active_set
    ]
    # Sort: alive first
    opp_bench_mons.sort(key=lambda m: (m.fainted, m.species))

    opp_bench: list[OppBenchSlot] = []
    for i in range(4):
        if i < len(opp_bench_mons):
            opp_bench.append(_build_opp_bench(opp_bench_mons[i]))
        else:
            opp_bench.append(_ABSENT_BENCH)

    return GameState(
        your_active=your_active,
        your_bench=bench_slots,
        opp_active=opp_active_enc,
        opp_bench=opp_bench,
        field=_build_field(battle),
        context=_build_context(battle),
    )
