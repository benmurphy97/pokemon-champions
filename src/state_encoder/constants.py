"""
Shared constants for EncoderV2.

Pokémon Champions — Regulation Set M-A (Masters Division)
"""

POKEMON_TYPES: list[str] = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]
N_TYPES = len(POKEMON_TYPES)  # 18
TYPE_IDX: dict[str, int] = {t: i for i, t in enumerate(POKEMON_TYPES)}

STAT_NAMES: list[str] = ["hp", "atk", "def", "spa", "spd", "spe"]
BOOST_STATS: list[str] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

# ── Normalisation denominators ────────────────────────────────────────────────
HP_MAX      = 400.0   # max possible HP at level 50 in Champions
STAT_MAX    = 350.0   # max for Atk/Def/SpA/SpD/Spe at level 50
BASE_MAX    = 255.0   # max possible base stat
SP_MAX      = 32.0    # max SP per stat (66 total, max 32 per)
BOOST_MAX   = 6.0     # boosts range -6 to +6; encode as (boost+6)/12
WEATHER_MAX = 8.0     # max weather duration (turns)
TERRAIN_MAX = 8.0
TR_MAX      = 5.0     # Trick Room lasts 5 turns
TAILWIND_MAX = 4.0
TRAP_MAX    = 5.0
SCREEN_MAX  = 8.0
TAUNT_MAX   = 4.0
ENCORE_MAX  = 3.0
SLEEP_MAX   = 3.0     # average sleep turns before wake check
TOXIC_MAX   = 16.0    # toxic damage caps at 15/16 HP
TURN_MAX    = 30.0    # normalise turn number

# ── Status one-hot order ──────────────────────────────────────────────────────
# Matches both your side and opponent side — same order everywhere.
STATUS_ORDER: list[str] = [
    "healthy", "brn", "par", "psn", "tox", "frz", "slp",
]
STATUS_IDX: dict[str, int] = {s: i for i, s in enumerate(STATUS_ORDER)}
N_STATUS = len(STATUS_ORDER)  # 7

# ── Item type one-hot order ───────────────────────────────────────────────────
ITEM_TYPE_ORDER: list[str] = [
    "choice", "berry", "orb", "mega_stone", "seed", "held_battle",
    "held_passive", "none",
]
N_ITEM_TYPES = len(ITEM_TYPE_ORDER)  # 8

# ── Presence state one-hot (opponent bench slots) ─────────────────────────────
PRESENCE_ORDER: list[str] = [
    "confirmed_benched", "confirmed_fainted", "possibly_brought", "confirmed_absent",
]
N_PRESENCE = len(PRESENCE_ORDER)  # 4

# ── Secondary effect type one-hot ─────────────────────────────────────────────
SECONDARY_EFFECT_ORDER: list[str] = [
    "flinch", "burn", "paralysis", "freeze", "stat_change", "none",
]
N_SECONDARY = len(SECONDARY_EFFECT_ORDER)  # 6

# ── Stat change target one-hot ────────────────────────────────────────────────
STAT_CHANGE_TARGET_ORDER: list[str] = ["self", "opponent", "both"]
N_STAT_TARGET = len(STAT_CHANGE_TARGET_ORDER)  # 3

# ── Weather one-hot order ─────────────────────────────────────────────────────
WEATHER_ORDER: list[str] = ["none", "rain", "sun", "sand", "snow"]
N_WEATHER = len(WEATHER_ORDER)  # 5

# ── Terrain one-hot order ─────────────────────────────────────────────────────
TERRAIN_ORDER: list[str] = ["none", "electric", "grassy", "misty", "psychic"]
N_TERRAIN = len(TERRAIN_ORDER)  # 5
