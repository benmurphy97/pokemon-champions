"""
Ability feature vector for EncoderV2.

Each ability is encoded to ABILITY_FEAT_DIM floats describing what the
ability DOES, not which ability it IS. Enables generalisation to new
abilities provided they fit existing flag categories.

ABILITY_FEAT_DIM = 34 (brief approximates as ~25; actual count from spec):
  sets_weather(1) + weather_speed(1) + weather_other(1)
  + redirects_electric(1) + redirects_water(1)
  + grants_immunity(1) + immune_type_onehot(18)
  + lowers_opp_stat_entry(1) + on_contact_effect(1)
  + prevents_stat_drops(1) + sets_terrain(1)
  + boosts_priority_status(1) + ignores_abilities(1)
  + speed_related(1) + enables_multihit(1)
  + doubles_atk(1) + is_mega_exclusive(1)
  + is_confirmed(1)
  = 34

For your own Pokémon the caller also appends has_activated(1) separately.
"""
from __future__ import annotations

from .constants import N_TYPES, TYPE_IDX

# ── Ability flag sets ─────────────────────────────────────────────────────────

_WEATHER_SETTERS: set[str] = {
    "Drizzle", "Drought", "Sand Stream", "Snow Warning",
    "Primordial Sea", "Desolate Land", "Delta Stream",  # not in M-A but defensive
}

_WEATHER_SPEED_BOOST: set[str] = {
    "Swift Swim",     # doubles speed in rain
    "Chlorophyll",    # doubles speed in sun
    "Sand Rush",      # doubles speed in sand
    "Slush Rush",     # doubles speed in snow
}

_WEATHER_OTHER_BENEFIT: set[str] = {
    "Solar Power",    # SpA +50% in sun (costs HP)
    "Rain Dish",      # heals 1/16 in rain
    "Dry Skin",       # heals in rain, hurt in sun
    "Sand Force",     # boosts Rock/Ground/Steel in sand
    "Sand Veil",      # evasion in sand
    "Ice Body",       # heals in snow
    "Snow Cloak",     # evasion in snow
    "Hydration",      # cures status in rain
    "Leaf Guard",     # blocks status in sun
    "Flower Gift",    # boosts Attack/SpDef in sun (team)
    "Forecast",       # changes type with weather
}

_REDIRECTS_ELECTRIC: set[str] = {
    "Lightning Rod",
    "Motor Drive",    # gives immunity rather than redirect, but still drawn in
}

_REDIRECTS_WATER: set[str] = {
    "Storm Drain",
}

# Abilities that grant a complete type immunity (absorb or nullify)
# Maps ability name → immune type name
_TYPE_IMMUNITY_ABILITIES: dict[str, str] = {
    "Levitate":      "Ground",
    "Flash Fire":    "Fire",
    "Water Absorb":  "Water",
    "Volt Absorb":   "Electric",
    "Motor Drive":   "Electric",
    "Sap Sipper":    "Grass",
    "Lightning Rod": "Electric",
    "Storm Drain":   "Water",
    "Soundproof":    "Normal",    # sound moves; close enough to Normal for encoding
    "Wonder Guard":  "Normal",    # complex; placeholder — only non-SE hits land
    "Dry Skin":      "Water",
    "Earth Eater":   "Ground",
    "Well-Baked Body": "Fire",
    "Wind Rider":    "Flying",
    "Magnet Pull":   "Steel",     # traps but doesn't grant immunity — remove
}
# Remove Magnet Pull (trapping, not immunity)
_TYPE_IMMUNITY_ABILITIES.pop("Magnet Pull", None)

_LOWERS_OPP_STAT_ENTRY: set[str] = {
    "Intimidate",
}

_ON_CONTACT_EFFECT: set[str] = {
    "Rough Skin",
    "Iron Barbs",
    "Flame Body",
    "Static",
    "Poison Point",
    "Effect Spore",
    "Cute Charm",
    "Mummy",
    "Wandering Spirit",
    "Gooey",
    "Tangling Hair",
    "Perish Body",
    "Gulp Missile",
}

_PREVENTS_STAT_DROPS: set[str] = {
    "Clear Body",
    "White Smoke",
    "Full Metal Body",
    "Hyper Cutter",    # only protects Attack
    "Big Pecks",       # only protects Defence
    "Mirror Armor",    # reflects drops back
    "Defiant",         # boosts Attack on drop; doesn't prevent
    "Competitive",     # similar
}
# Defiant/Competitive react to drops but don't prevent them; keep for encoding
# since they're functionally related to the "stat drop" mechanic
_PREVENTS_STAT_DROPS |= {"Defiant", "Competitive"}

_TERRAIN_SETTERS: set[str] = {
    "Electric Surge",
    "Grassy Surge",
    "Misty Surge",
    "Psychic Surge",
}

_BOOSTS_PRIORITY_STATUS: set[str] = {
    "Prankster",
}

_IGNORES_ABILITIES: set[str] = {
    "Mold Breaker",
    "Turboblaze",
    "Teravolt",
    "Mycelium Might",  # ignores ability on status moves
}

_SPEED_RELATED: set[str] = {
    "Speed Boost",
    "Unburden",
    "Quick Feet",
    "Surge Surfer",     # doubles speed on electric terrain
    "Sand Rush",        # also in weather_speed — listed here for the flag
    "Slush Rush",
    "Swift Swim",
    "Chlorophyll",
    "Rattled",          # boosts Speed on certain hits
    "Tinted Lens",      # not speed but similar utility — remove
}
_SPEED_RELATED -= {"Tinted Lens"}

_ENABLES_MULTIHIT: set[str] = {
    "Parental Bond",
}

_DOUBLES_ATK: set[str] = {
    "Pure Power",
    "Huge Power",
}

# Abilities that only exist post-Mega (Mega-exclusive)
_MEGA_EXCLUSIVE: set[str] = {
    "Tough Claws",       # Mega Charizard X
    "Drought",           # Mega Charizard Y (overlaps with Ninetales — not exclusive)
    "Thick Fat",         # Mega Venusaur (overlaps — not strictly exclusive)
    "Aerilate",          # Mega Pinsir, Mega Salamence (not in M-A)
    "Pixilate",          # Mega Gardevoir
    "Parental Bond",     # Mega Kangaskhan
    "Pure Power",        # Mega Medicham
    "Trace",             # Mega Alakazam (also Gardevoir base — not exclusive)
    "Intimidate",        # Mega Gyarados (not exclusive)
    "Scrappy",           # Mega Lopunny (also Kangaskhan base — not exclusive)
    "Sand Force",        # Mega Garchomp (not exclusive)
    "Speed Boost",       # Mega Beedrill (also Yanmega — not exclusive)
    "Adaptability",      # Mega Ampharos? — not exclusive
    "Magic Bounce",      # Mega Sableye
    "Healer",            # Mega Audino (also base — not exclusive)
    "Normalize",         # Mega Skarmory? — check
    "Stench",            # Mega Sharpedo? — not exclusive
    "Sheer Force",       # Mega Camerupt (not exclusive)
    "Altruistic",        # Mega Clefable (check M-A data)
    "Soundproof",        # Mega Chimecho (check M-A data)
    "Prankster",         # Mega Banette
    "Own Tempo",         # Mega Slowbro (not exclusive)
    "Intimidate",        # Mega Salamence (not in M-A)
    "Sand Stream",       # Mega Tyranitar (not exclusive)
}
# Keep only abilities that are genuinely only found post-Mega in M-A:
_MEGA_EXCLUSIVE_STRICT: set[str] = {
    "Parental Bond",  # only Mega Kangaskhan in M-A
    "Pure Power",     # only Mega Medicham in M-A
    "Pixilate",       # only Mega Gardevoir in M-A
    "Magic Bounce",   # only Mega Sableye in M-A (and Mega Absol — not in M-A)
    "Prankster",      # Mega Banette; also appears on other mons — set to False
}
# Actually Prankster appears on Klefki etc; only mark ability as mega-exclusive
# if it literally cannot appear on any base-form Pokémon in M-A.
_MEGA_EXCLUSIVE_STRICT = {"Parental Bond", "Pure Power"}


# ── Encoding ──────────────────────────────────────────────────────────────────

ABILITY_FEAT_DIM = 35   # 1+1+1+1+1+1+18+1+1+1+1+1+1+1+1+1+1+1 = 35


def encode_ability(ability_name: str, is_confirmed: bool = True) -> list[float]:
    """
    Return a ABILITY_FEAT_DIM (34) float list describing what the ability does.

    ability_name: display name (e.g. "Drizzle", "Intimidate")
    is_confirmed: True if directly observed in battle; False = prior estimate
    """
    name = ability_name.strip()
    v: list[float] = []

    # 1 — sets weather on entry
    v.append(1.0 if name in _WEATHER_SETTERS else 0.0)

    # 1 — benefits from weather (speed doubled)
    v.append(1.0 if name in _WEATHER_SPEED_BOOST else 0.0)

    # 1 — benefits from weather (other)
    v.append(1.0 if name in _WEATHER_OTHER_BENEFIT else 0.0)

    # 1 — redirects electric moves
    v.append(1.0 if name in _REDIRECTS_ELECTRIC else 0.0)

    # 1 — redirects water moves
    v.append(1.0 if name in _REDIRECTS_WATER else 0.0)

    # 1 — grants type immunity
    immune_type = _TYPE_IMMUNITY_ABILITIES.get(name)
    v.append(1.0 if immune_type else 0.0)

    # 18 — immune type one-hot (all 0 if no immunity)
    immune_vec = [0.0] * N_TYPES
    if immune_type:
        idx = TYPE_IDX.get(immune_type)
        if idx is not None:
            immune_vec[idx] = 1.0
    v.extend(immune_vec)

    # 1 — lowers opponent stat on entry
    v.append(1.0 if name in _LOWERS_OPP_STAT_ENTRY else 0.0)

    # 1 — has on-contact effect
    v.append(1.0 if name in _ON_CONTACT_EFFECT else 0.0)

    # 1 — prevents own stat drops
    v.append(1.0 if name in _PREVENTS_STAT_DROPS else 0.0)

    # 1 — sets terrain on entry
    v.append(1.0 if name in _TERRAIN_SETTERS else 0.0)

    # 1 — boosts priority of status moves
    v.append(1.0 if name in _BOOSTS_PRIORITY_STATUS else 0.0)

    # 1 — ignores other abilities
    v.append(1.0 if name in _IGNORES_ABILITIES else 0.0)

    # 1 — has speed-related effect
    v.append(1.0 if name in _SPEED_RELATED else 0.0)

    # 1 — enables multi-hit on moves
    v.append(1.0 if name in _ENABLES_MULTIHIT else 0.0)

    # 1 — doubles effective Attack
    v.append(1.0 if name in _DOUBLES_ATK else 0.0)

    # 1 — is a Mega-exclusive ability (only appears post-Mega in M-A)
    v.append(1.0 if name in _MEGA_EXCLUSIVE_STRICT else 0.0)

    # 1 — is confirmed / revealed in battle
    v.append(1.0 if is_confirmed else 0.0)

    assert len(v) == ABILITY_FEAT_DIM, f"ability dim {len(v)} != {ABILITY_FEAT_DIM}"
    return v
