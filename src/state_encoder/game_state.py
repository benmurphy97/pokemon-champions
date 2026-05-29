"""
GameState dataclasses for EncoderV2.

The rest of the system (env, battle adapter) is responsible for populating
these dataclasses from the live poke-env Battle object. EncoderV2 only
reads from these; it never touches poke-env directly.

Format: Pokémon Champions — Regulation Set M-A
  - No IVs (removed from the game)
  - SP replaces EVs (66 total, max 32 per stat)
  - Stat Alignment replaces Natures (+10%/-10% on one stat each)
  - Mega Evolution available (one per battle)
  - No Terastallization
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── Move ──────────────────────────────────────────────────────────────────────

@dataclass
class MoveState:
    """One move slot on any Pokémon."""
    move_id: str       # lowercase hyphenated Showdown ID, e.g. "flamethrower"
    move_name: str     # display name, e.g. "Flamethrower"
    pp_remaining: int
    pp_max: int
    is_revealed: bool = True    # False → opponent prior slot
    times_used: int = 0         # for opponent PP pressure estimate
    is_selectable: bool = True  # False if taunted+status/choice-locked wrong/no PP/disabled/AV+status


# ── Volatile status ───────────────────────────────────────────────────────────

@dataclass
class VolatileStatus:
    """In-battle conditions that reset on switch."""
    confused: bool = False
    taunted: bool = False
    taunt_turns: int = 0
    encored: bool = False
    encore_turns: int = 0
    encore_move_slot: int = 0       # 0–3
    leech_seeded: bool = False
    substitute: bool = False
    partially_trapped: bool = False
    trap_turns: int = 0
    infatuated: bool = False
    disabled: bool = False
    disabled_move_slot: int = 0     # 0–3
    charging: bool = False          # two-turn charge turn or Power Herb used
    identified: bool = False        # Odor Sleuth / Miracle Eye


# ── Primary status ────────────────────────────────────────────────────────────

@dataclass
class PrimaryStatus:
    status: str = "healthy"     # one of STATUS_ORDER in constants.py
    toxic_counter: int = 0      # increments each turn under bad poison
    sleep_turns: int = 0        # turns already asleep


# ── Stat Alignment (replaces Nature) ─────────────────────────────────────────

@dataclass
class StatAlignment:
    boosted_stat: str = ""   # e.g. "atk"; "" for Serious (neutral)
    reduced_stat: str = ""   # e.g. "spd"; "" for Serious (neutral)


# ── Item ──────────────────────────────────────────────────────────────────────

@dataclass
class ItemState:
    name: str = ""             # display name; "" = no item
    is_confirmed: bool = True  # False → opponent prior
    is_consumed: bool = False  # Mega Stone after Mega, used berry, etc.
    confidence: float = 1.0   # 0 = pure prior, 1 = confirmed


# ── Ability ───────────────────────────────────────────────────────────────────

@dataclass
class AbilityState:
    name: str = ""             # display name; "" = unknown
    is_confirmed: bool = True  # False → prior estimate
    has_activated: bool = False


# ── Your own Pokémon (active on field) ───────────────────────────────────────

@dataclass
class YourActivePokemon:
    """Full information — your own Pokémon on the field."""
    species: str
    types: list[str]               # 1 or 2 types; post-Mega types if Mega-evolved
    has_mega_evolved: bool
    has_mega_stone: bool           # holding stone but not yet evolved
    hp_ratio: float                # current_hp / max_hp, 0.0–1.0
    hp_max: int                    # absolute max HP
    stats: dict[str, int]          # {atk, def, spa, spd, spe} actual at lv50
    sp: dict[str, int]             # {hp, atk, def, spa, spd, spe} SP values (0–32)
    stat_alignment: StatAlignment
    boosts: dict[str, int]         # {atk, def, spa, spd, spe, accuracy, evasion} -6..+6
    status: PrimaryStatus
    volatile: VolatileStatus
    ability: AbilityState
    item: ItemState
    moves: list[MoveState]         # exactly 4 slots
    is_choice_locked: bool
    choice_locked_slot: int        # 0–3; 0 if not locked
    field_position: int            # 0 = left slot, 1 = right slot


# ── Your bench Pokémon ────────────────────────────────────────────────────────

@dataclass
class YourBenchPokemon:
    """Full information — your own Pokémon currently on bench."""
    species: str
    types: list[str]
    is_fainted: bool
    hp_ratio: float
    hp_max: int
    stats: dict[str, int]          # {atk, def, spa, spd, spe}
    sp: dict[str, int]             # {hp, atk, def, spa, spd, spe}
    stat_alignment: StatAlignment
    status: PrimaryStatus
    ability: AbilityState
    has_mega_stone: bool
    can_mega_evolve: bool          # has stone AND battle Mega not yet used
    item: ItemState
    moves: list[MoveState]         # 4 slots; is_selectable reflects Taunt/no-PP
    has_mega_evolved: bool = False


# ── Opponent active Pokémon ───────────────────────────────────────────────────

@dataclass
class OppActivePokemon:
    """Partial information — opponent's Pokémon on the field."""
    species: str
    types: list[str]               # known from OTS; post-Mega if evolved
    has_mega_evolved: bool
    hp_ratio: float                # visible in battle
    base_stats: dict[str, int]     # known from species; {hp, atk, def, spa, spd, spe}
    estimated_speed: int           # base speed + common SP spread estimate
    boosts: dict[str, int]         # {atk, def, spa, spd, spe, accuracy, evasion}
    status: PrimaryStatus
    volatile: VolatileStatus       # all visible in doubles
    ability: AbilityState          # is_confirmed=True if triggered
    item: ItemState
    moves: list[MoveState]         # 4 slots; is_revealed=False for priors
    is_choice_locked: bool         # inferred
    choice_locked_slot: int        # 0 if unknown
    choice_lock_confirmed: bool


# ── Opponent bench slot ───────────────────────────────────────────────────────

@dataclass
class OppBenchSlot:
    """
    One of the four non-active roster slots for the opponent.

    The opponent registered 6, chose 4 (leads 2 + bench 2). We track all
    four non-active roster Pokémon here regardless of their presence state.
    """
    species: str
    # Presence
    presence_state: str            # one of PRESENCE_ORDER in constants.py
    p_present: float               # 0.0–1.0; probability this Pokémon is in the battle
    # Observable from OTS
    types: list[str]
    base_stats: dict[str, int]     # {hp, atk, def, spa, spd, spe}
    # Observed if ever active
    hp_ratio_last_known: float     # −1.0 flag if never seen active
    hp_is_known: bool
    status: PrimaryStatus
    status_is_known: bool
    ability: AbilityState
    has_mega_stone: bool           # confirmed or prior
    has_mega_evolved: bool
    item: ItemState
    moves: list[MoveState]         # 4 slots; is_revealed reflects what we've seen
    has_fainted: bool


# ── Field state ───────────────────────────────────────────────────────────────

@dataclass
class ScreenState:
    active: bool = False
    turns_remaining: int = 0


@dataclass
class FieldState:
    # Weather
    weather: str = "none"          # one of WEATHER_ORDER in constants.py
    weather_turns: int = 0         # turns remaining; 0 if permanent/none
    weather_permanent: bool = False
    weather_set_by_our_side: bool = False

    # Terrain
    terrain: str = "none"          # one of TERRAIN_ORDER
    terrain_turns: int = 0

    # Trick Room
    trick_room_active: bool = False
    trick_room_turns: int = 0

    # Tailwind
    our_tailwind: bool = False
    our_tailwind_turns: int = 0
    opp_tailwind: bool = False
    opp_tailwind_turns: int = 0

    # Screens
    our_light_screen: ScreenState = field(default_factory=ScreenState)
    our_reflect: ScreenState = field(default_factory=ScreenState)
    opp_light_screen: ScreenState = field(default_factory=ScreenState)
    opp_reflect: ScreenState = field(default_factory=ScreenState)

    # Other field effects
    gravity_active: bool = False
    gravity_turns: int = 0
    our_safeguard: ScreenState = field(default_factory=ScreenState)
    opp_safeguard: ScreenState = field(default_factory=ScreenState)
    magic_room_active: bool = False
    magic_room_turns: int = 0
    wonder_room_active: bool = False
    wonder_room_turns: int = 0


# ── Last-turn action tracking ─────────────────────────────────────────────────

@dataclass
class LastTurnSlot:
    used_protect: bool = False
    used_mega: bool = False
    switched: bool = False
    move_type: str = ""            # type name of move used; "" if switched or turn 1
    damage_taken_ratio: float = 0.0  # damage as fraction of max HP


@dataclass
class BattleContext:
    """Global battle context — fully observable."""
    turn: int = 0
    our_pokemon_remaining: int = 4
    opp_pokemon_remaining: int = 4
    our_total_hp_ratio: float = 1.0    # mean HP% across surviving Pokémon
    opp_total_hp_ratio: float = 1.0
    ko_differential: int = 0           # our KOs minus their KOs (-4 to +4)
    battle_mega_used_ours: bool = False
    battle_mega_used_opp: bool = False

    # Last-turn actions per active slot (4 total: our p1/p2, opp p1/p2)
    our_p1_last: LastTurnSlot = field(default_factory=LastTurnSlot)
    our_p2_last: LastTurnSlot = field(default_factory=LastTurnSlot)
    opp_p1_last: LastTurnSlot = field(default_factory=LastTurnSlot)
    opp_p2_last: LastTurnSlot = field(default_factory=LastTurnSlot)

    # Revealed-information tracker (one flag per roster slot, indexed 0–5)
    opp_has_been_active: list[bool] = field(default_factory=lambda: [False] * 6)
    opp_moves_revealed_count: list[int] = field(default_factory=lambda: [0] * 6)
    opp_item_confirmed: list[bool] = field(default_factory=lambda: [False] * 6)
    opp_ability_confirmed: list[bool] = field(default_factory=lambda: [False] * 6)


# ── Top-level GameState ───────────────────────────────────────────────────────

@dataclass
class GameState:
    """
    Complete state snapshot passed to EncoderV2 each turn.

    Caller must supply exactly:
      - your_active: 2 entries  (slots 0 and 1)
      - your_bench:  2 entries  (slots 2 and 3, or fewer if fainted)
      - opp_active:  2 entries
      - opp_bench:   4 entries  (all non-active roster slots)
      - field:       FieldState
      - context:     BattleContext
    """
    your_active: list[YourActivePokemon]   # len 2
    your_bench:  list[YourBenchPokemon]    # len 2 (use is_fainted for empty)
    opp_active:  list[OppActivePokemon]    # len 2
    opp_bench:   list[OppBenchSlot]        # len 4
    field:       FieldState
    context:     BattleContext
