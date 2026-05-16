import numpy as np
from poke_env.battle import Weather, Field, MoveCategory, Status

_BOOST_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

_STATUS_FLAGS = [Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.TOX, Status.SLP]

_SUN  = {Weather.SUNNYDAY, Weather.DESOLATELAND}
_RAIN = {Weather.RAINDANCE, Weather.PRIMORDIALSEA}
_SNOW = {Weather.SNOWSCAPE, Weather.HAIL}
_SAND = {Weather.SANDSTORM}

_FIELD_FLAGS = [
    Field.ELECTRIC_TERRAIN,
    Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN,
    Field.PSYCHIC_TERRAIN,
    Field.TRICK_ROOM,
]

# Per move: power(1) + category(3) + dmg_opp1(1) + dmg_opp2(1) + pp_low(1) = 7
_MOVE_FEATURES = 7

# Max weather/terrain duration including extended rocks/clay (8 turns)
_WEATHER_MAX_TURNS = 8.0
_FIELD_MAX_TURNS   = 8.0

STATE_SIZE = (
    4                            # active HP (all 4 slots)
    + 4 * 2                      # vulnerability: max opp type effectiveness vs each active mon
    + 4 * 7                      # stat boosts (all 4 slots × 7 stats)
    + 2 * 4 * _MOVE_FEATURES     # moves for our 2 active mons × 4 moves each
    + 4                          # bench HP (up to 4 non-active mons)
    + 4                          # weather binary flags (sun/rain/snow/sand)
    + 5                          # field binary flags (4 terrains + trick room)
    + 4 * len(_STATUS_FLAGS)     # status conditions for all 4 active slots
    + 4 * len(_STATUS_FLAGS)     # status conditions for up to 4 bench mons
    + 2 * 4 * _MOVE_FEATURES     # opponent revealed moves (2 opp slots × up to 4 moves)
    + 4                          # weather turn counts (normalized — how long each has been active)
    + 5                          # field turn counts (normalized)
)  # = 222


def encode_battle(battle) -> np.ndarray:
    features: list[float] = []

    active = list(battle.active_pokemon or [None, None])
    opp_active = list(battle.opponent_active_pokemon or [None, None])
    all_active = active + opp_active

    # HP fractions for all 4 active slots
    for mon in all_active:
        features.append(_hp_frac(mon))

    # Vulnerability: for each active mon, max type effectiveness from each opponent's typing
    for mon in all_active:
        def_types = _live_types(mon)
        features.append(_max_effectiveness(opp_active[0], def_types))
        features.append(_max_effectiveness(opp_active[1] if len(opp_active) > 1 else None, def_types))

    # Stat boosts for all 4 active slots
    for mon in all_active:
        features.extend(_encode_boosts(mon))

    # Available moves for our 2 active slots
    for slot_idx in range(2):
        mon = active[slot_idx] if slot_idx < len(active) else None
        available = list(battle.available_moves[slot_idx])[:4] if (
            mon and battle.available_moves and slot_idx < len(battle.available_moves)
        ) else []
        for i in range(4):
            move = available[i] if i < len(available) else None
            features.extend(_encode_move(move, opp_active))

    # Bench HP (up to 4 non-active, non-fainted team members)
    bench = [m for m in battle.team.values() if m not in active and not m.fainted]
    for i in range(4):
        features.append(bench[i].current_hp_fraction if i < len(bench) else 0.0)

    # Weather binary flags
    active_weather = set(battle.weather.keys()) if battle.weather else set()
    features.append(1.0 if active_weather & _SUN  else 0.0)
    features.append(1.0 if active_weather & _RAIN else 0.0)
    features.append(1.0 if active_weather & _SNOW else 0.0)
    features.append(1.0 if active_weather & _SAND else 0.0)

    # Field binary flags (terrains + trick room)
    active_fields = set(battle.fields.keys()) if battle.fields else set()
    for field in _FIELD_FLAGS:
        features.append(1.0 if field in active_fields else 0.0)

    # Status conditions for all 4 active slots
    for mon in all_active:
        features.extend(_encode_status(mon))

    # Status conditions for up to 4 bench mons
    for i in range(4):
        features.extend(_encode_status(bench[i]) if i < len(bench) else [0.0] * len(_STATUS_FLAGS))

    # Opponent revealed moves — encode what the opponent can do to our mons
    for slot_idx in range(2):
        opp_mon = opp_active[slot_idx] if slot_idx < len(opp_active) else None
        revealed = list(opp_mon.moves.values())[:4] if (opp_mon and not opp_mon.fainted) else []
        for i in range(4):
            move = revealed[i] if i < len(revealed) else None
            features.extend(_encode_move(move, active))  # damage vs our mons

    # Weather turn counts (how many turns each weather has been active, normalised)
    weather_dict = battle.weather if battle.weather else {}
    for weather_set in (_SUN, _RAIN, _SNOW, _SAND):
        turns = max((weather_dict.get(w, 0) for w in weather_set), default=0)
        features.append(min(1.0, turns / _WEATHER_MAX_TURNS))

    # Field turn counts (normalised)
    fields_dict = battle.fields if battle.fields else {}
    for field in _FIELD_FLAGS:
        turns = fields_dict.get(field, 0)
        features.append(min(1.0, turns / _FIELD_MAX_TURNS))

    return np.array(features, dtype=np.float32)


def _hp_frac(mon) -> float:
    if mon is None or mon.fainted:
        return 0.0
    return mon.current_hp_fraction


def _live_types(mon) -> list:
    if mon is None or mon.fainted:
        return []
    return [t for t in (mon.types or []) if t is not None]


def _max_effectiveness(attacker, def_types: list) -> float:
    """Max type effectiveness from attacker's types vs a defender. Normalised 0–1 (÷4)."""
    if not attacker or attacker.fainted or not def_types:
        return 0.25  # neutral fallback
    best = 0.0
    for att_type in _live_types(attacker):
        try:
            best = max(best, att_type.damage_multiplier(*def_types))
        except Exception:
            pass
    return best / 4.0


def _encode_boosts(mon) -> list[float]:
    if mon is None or mon.fainted:
        return [0.0] * len(_BOOST_STATS)
    boosts = mon.boosts or {}
    return [boosts.get(stat, 0) / 6.0 for stat in _BOOST_STATS]


def _encode_status(mon) -> list[float]:
    if mon is None or mon.fainted:
        return [0.0] * len(_STATUS_FLAGS)
    return [1.0 if mon.status == s else 0.0 for s in _STATUS_FLAGS]


def _encode_move(move, opp_active: list) -> list[float]:
    if move is None:
        return [0.0] * _MOVE_FEATURES
    power = (move.base_power or 0) / 250.0
    category = [
        float(move.category == MoveCategory.PHYSICAL),
        float(move.category == MoveCategory.SPECIAL),
        float(move.category == MoveCategory.STATUS),
    ]
    dmg1 = _damage_estimate(move, opp_active[0] if opp_active else None)
    dmg2 = _damage_estimate(move, opp_active[1] if len(opp_active) > 1 else None)
    pp_low = 1.0 if (move.current_pp is not None and move.current_pp < 5) else 0.0
    return [power] + category + [dmg1, dmg2, pp_low]


def _damage_estimate(move, target) -> float:
    """Normalised damage estimate: (power/250) × (effectiveness/4), capped at 1.0."""
    if target is None or target.fainted or not move.base_power:
        return 0.0
    power = move.base_power / 250.0
    def_types = _live_types(target)
    if not def_types or move.type is None:
        return min(1.0, power)
    try:
        mult = move.type.damage_multiplier(*def_types)
    except Exception:
        mult = 1.0
    return min(1.0, power * mult / 4.0)
