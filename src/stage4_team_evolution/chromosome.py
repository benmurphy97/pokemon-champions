"""
Stage 5: Genetic Algorithm — Team Chromosome Encoding

Parses a Showdown team string into structured dicts, applies mutations,
and serialises back to a valid Showdown string.

Constraints enforced:
- EV total ≤ 66, max 32 per stat  (Champions VGC EV format)
- No duplicate items (Item Clause)
- At most one Mega Stone holder per team
- All IVs 31 (never written to team strings)
"""

import logging
import random
import re
from copy import deepcopy

log = logging.getLogger(__name__)

EV_STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
EV_MAX_TOTAL = 66
EV_MAX_STAT = 32

# Valid items in gen9championsvgc2026regma — sourced from data/items.json
# BANNED: Life Orb, Choice Band, Choice Specs, Assault Vest
# Mega Stones excluded here — they're fixed per Pokémon in ga/pool.py
ITEM_POOL = [
    # Stat boost
    "Choice Scarf", "Focus Sash", "White Herb",
    # Recovery
    "Sitrus Berry", "Lum Berry", "Leftovers", "Shell Bell",
    "Chesto Berry", "Mental Herb",
    # Type power boost
    "Charcoal", "Mystic Water", "Silk Scarf", "Black Glasses",
    "Twisted Spoon", "Poison Barb", "Miracle Seed", "Magnet",
    "Never-Melt Ice", "Hard Stone", "Sharp Beak", "Dragon Fang",
    "Fairy Feather",
    # Defense berries
    "Chople Berry", "Occa Berry", "Passho Berry", "Wacan Berry",
    "Rindo Berry", "Yache Berry", "Coba Berry", "Payapa Berry",
    "Kasib Berry", "Colbur Berry", "Charti Berry", "Haban Berry",
]

# Candidate moves per Pokémon — curated for TEAM_ALPHA members.
# For any Pokémon not listed here the mutator falls back to ga/pool.py template moves.
MOVE_CANDIDATES = {
    "Farigiraf": [
        "Trick Room", "Psychic", "Hyper Voice", "Helping Hand",
        "Expanding Force", "Future Sight", "Dazzling Gleam",
        "Reflect", "Light Screen", "Protect",
    ],
    "Incineroar": [
        "Fake Out", "Parting Shot", "Flare Blitz", "Throat Chop",
        "Knock Off", "Will-O-Wisp", "Darkest Lariat", "Snarl",
        "U-turn", "Protect",
    ],
    "Blastoise": [
        "Fake Out", "Protect", "Flip Turn", "Yawn",
        "Water Spout", "Scald", "Ice Beam", "Rain Dance",
        "Muddy Water", "Icy Wind",
    ],
    "Venusaur": [
        "Sludge Bomb", "Sleep Powder", "Protect", "Leaf Storm",
        "Giga Drain", "Energy Ball", "Weather Ball",
        "Leech Seed", "Synthesis",
    ],
    "Charizard-Mega-Y": [
        "Heat Wave", "Solar Beam", "Weather Ball", "Protect",
        "Fire Blast", "Air Slash", "Ancient Power", "Dragon Pulse",
    ],
    "Drampa": [
        "Hyper Voice", "Protect", "Draco Meteor", "Thunderbolt",
        "Fire Blast", "Dragon Pulse", "Glare", "Flamethrower",
    ],
}


# ---------------------------------------------------------------------------
# Parsing and serialisation
# ---------------------------------------------------------------------------

def parse_team(team_str: str) -> list[dict]:
    """Parse a Showdown team string into a list of Pokemon dicts."""
    mons = []
    blocks = [b.strip() for b in team_str.strip().split("\n\n") if b.strip()]
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        mon: dict = {"moves": []}

        # First line: "Name @ Item" or just "Name"
        first = lines[0]
        if " @ " in first:
            name, item = first.split(" @ ", 1)
            mon["name"] = name.strip()
            mon["item"] = item.strip()
        else:
            mon["name"] = first.strip()
            mon["item"] = None

        evs_raw = {}
        for line in lines[1:]:
            if line.startswith("Ability:"):
                mon["ability"] = line[8:].strip()
            elif line.startswith("Level:"):
                mon["level"] = int(line[6:].strip())
            elif line.startswith("EVs:"):
                ev_str = line[4:].strip()
                for part in ev_str.split("/"):
                    part = part.strip()
                    m = re.match(r"(\d+)\s+(\w+)", part)
                    if m:
                        evs_raw[m.group(2)] = int(m.group(1))
            elif line.endswith("Nature"):
                mon["nature"] = line.replace("Nature", "").strip()
            elif line.startswith("- "):
                mon["moves"].append(line[2:].strip())
            # IVs are always 31 — skip those lines

        mon["evs"] = {stat: evs_raw.get(stat, 0) for stat in EV_STATS}
        mons.append(mon)
    return mons


def serialise_team(mons: list[dict]) -> str:
    """Serialise a list of Pokemon dicts back to a Showdown team string."""
    blocks = []
    for mon in mons:
        lines = []
        name = mon["name"]
        item = mon.get("item")
        lines.append(f"{name} @ {item}" if item else name)
        if "ability" in mon:
            lines.append(f"Ability: {mon['ability']}")
        lines.append("Level: 50")

        evs = mon.get("evs", {})
        ev_parts = [f"{v} {k}" for k, v in evs.items() if v > 0]
        if ev_parts:
            lines.append(f"EVs: {' / '.join(ev_parts)}")

        if "nature" in mon:
            lines.append(f"{mon['nature']} Nature")

        for move in mon.get("moves", []):
            lines.append(f"- {move}")

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Individual mutation operators
# ---------------------------------------------------------------------------

def mutate_evs(mons: list[dict]) -> tuple[list[dict], bool]:
    """Shift some EVs between two stats on one random Pokémon."""
    mons = deepcopy(mons)
    mon = random.choice(mons)
    evs = mon["evs"]

    donors = [s for s in EV_STATS if evs[s] > 0]
    recipients = [s for s in EV_STATS if evs[s] < EV_MAX_STAT]

    if not donors or not recipients:
        log.debug("mutate_evs: no valid donor/recipient on %s — skipped", mon["name"])
        return mons, False

    donor = random.choice(donors)
    recipient = random.choice([s for s in recipients if s != donor] or recipients)
    if donor == recipient:
        log.debug("mutate_evs: donor==recipient on %s — skipped", mon["name"])
        return mons, False

    max_transfer = min(evs[donor], EV_MAX_STAT - evs[recipient], 8)
    if max_transfer < 1:
        log.debug("mutate_evs: max_transfer<1 on %s — skipped", mon["name"])
        return mons, False
    amount = random.randint(1, max_transfer)
    evs[donor] -= amount
    evs[recipient] = min(EV_MAX_STAT, evs[recipient] + amount)

    log.debug("mutate_evs: %s %s %d→%d, %s %d→%d",
              mon["name"], donor, evs[donor] + amount, evs[donor],
              recipient, evs[recipient] - amount, evs[recipient])
    return mons, True


def mutate_item(mons: list[dict]) -> tuple[list[dict], bool]:
    """Swap one Pokémon's item for a different one not already on the team."""
    mons = deepcopy(mons)

    mega_stones = {m["item"] for m in mons if m.get("item", "").endswith("ite")}
    swappable = [m for m in mons if m.get("item") not in mega_stones]
    if not swappable:
        log.debug("mutate_item: all items are Mega Stones — skipped")
        return mons, False

    mon = random.choice(swappable)
    current_items = {m["item"] for m in mons if m.get("item")}
    candidates = [i for i in ITEM_POOL if i not in current_items]
    if not candidates:
        log.debug("mutate_item: item pool exhausted — skipped")
        return mons, False

    old_item = mon["item"]
    mon["item"] = random.choice(candidates)
    log.debug("mutate_item: %s %s → %s", mon["name"], old_item, mon["item"])
    return mons, True


def _get_move_pool(mon_name: str) -> list[str]:
    """Return candidate moves for a Pokémon — curated list first, then pool template."""
    if mon_name in MOVE_CANDIDATES:
        return MOVE_CANDIDATES[mon_name]
    try:
        from stage4_team_evolution.pool import POOL_BY_NAME
        template = POOL_BY_NAME.get(mon_name)
        if template:
            return template.moves
    except ImportError:
        pass
    return []


def mutate_move(mons: list[dict]) -> tuple[list[dict], bool]:
    """Swap one move on one Pokémon for a candidate move not already on that Pokémon."""
    mons = deepcopy(mons)

    eligible = [m for m in mons if _get_move_pool(m["name"]) and len(m["moves"]) > 0]
    if not eligible:
        log.debug("mutate_move: no eligible Pokémon with move candidates — skipped")
        return mons, False

    mon = random.choice(eligible)
    pool = _get_move_pool(mon["name"])
    current = set(mon["moves"])
    new_moves = [mv for mv in pool if mv not in current]
    if not new_moves:
        log.debug("mutate_move: %s already has all candidate moves — skipped", mon["name"])
        return mons, False

    slot = random.randrange(len(mon["moves"]))
    old_move = mon["moves"][slot]
    mon["moves"][slot] = random.choice(new_moves)
    log.debug("mutate_move: %s slot %d: %s → %s", mon["name"], slot, old_move, mon["moves"][slot])
    return mons, True


def mutate_pokemon(mons: list[dict], tenure: dict | None = None) -> tuple[list[dict], bool]:
    """
    Replace one non-Mega Pokémon with a random one from the meta pool.

    tenure maps pokemon_name → generations on team. Higher tenure = less likely
    to be replaced, so long-standing contributors are protected.
    """
    from stage4_team_evolution.pool import POOL, MEGA_STONES

    mons = deepcopy(mons)

    swappable_idx = [
        i for i, m in enumerate(mons)
        if not (m.get("item", "") or "").endswith("ite")
    ]
    if not swappable_idx:
        log.debug("mutate_pokemon: all Pokémon hold Mega Stones — skipped")
        return mons, False

    current_names = {m["name"] for m in mons}
    current_items = {m.get("item") for m in mons if m.get("item")}
    team_has_mega = any(m.get("item", "").endswith("ite") for m in mons)

    candidates = [
        p for p in POOL
        if p.name not in current_names
        and p.items[0] not in current_items
        and not (p.items[0].endswith("ite") and team_has_mega)
    ]
    if not candidates:
        log.debug("mutate_pokemon: no valid pool candidates — skipped")
        return mons, False

    # Weight slots by inverse tenure — newer members are easier to replace
    weights = [1.0 / (tenure.get(mons[i]["name"], 0) + 1) for i in swappable_idx] if tenure \
        else [1.0] * len(swappable_idx)
    idx = random.choices(swappable_idx, weights=weights)[0]
    template = random.choice(candidates)

    old_name = mons[idx]["name"]
    mons[idx] = {
        "name": template.name,
        "item": template.items[0],
        "ability": template.ability,
        "evs": {stat: template.evs.get(stat, 0) for stat in EV_STATS},
        "nature": template.nature,
        "moves": template.moves[:4],
    }
    log.debug("mutate_pokemon: slot %d %s → %s", idx, old_name, template.name)
    return mons, True


# ---------------------------------------------------------------------------
# High-level mutate() — used by the GA and PBT league
# ---------------------------------------------------------------------------

# Weighted mutation selector — swap_pokemon gets high weight for population diversity;
# ev/item/move are fine-tuning operators.
_MUTATIONS = [
    (mutate_evs,     0.30),
    (mutate_item,    0.25),
    (mutate_move,    0.15),
    (mutate_pokemon, 0.30),
]


def mutate(team_str: str, tenure: dict | None = None) -> str:
    """
    Apply one random mutation to a Showdown team string and return the new string.

    Tries each mutation type in weighted-random order. If all fail due to
    constraints (e.g. all items already unique, no move candidates), returns the
    original team string unchanged and logs a warning.
    """
    mons = parse_team(team_str)

    chosen_fn = random.choices(
        [m[0] for m in _MUTATIONS],
        weights=[m[1] for m in _MUTATIONS],
    )[0]
    remaining = [fn for fn, _ in _MUTATIONS if fn is not chosen_fn]
    random.shuffle(remaining)

    for fn in [chosen_fn] + remaining:
        if fn is mutate_pokemon:
            new_mons, applied = mutate_pokemon(mons, tenure=tenure)
        else:
            new_mons, applied = fn(mons)
        if applied:
            log.info("mutate: applied %s", fn.__name__)
            return serialise_team(new_mons)

    log.warning("mutate: all mutation types failed — returning unchanged team")
    return team_str
