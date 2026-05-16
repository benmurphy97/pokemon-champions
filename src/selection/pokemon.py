"""
Pokemon dataclass and Showdown paste format parser.

Handles the custom gen9championsvgc2026regma format where max EVs per stat = 32.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Max EV per stat in this custom format (standard VGC uses 252)
MAX_EV = 32

NATURES = [
    "Hardy", "Lonely", "Brave", "Adamant", "Naughty",
    "Bold", "Docile", "Relaxed", "Impish", "Lax",
    "Timid", "Hasty", "Serious", "Jolly", "Naive",
    "Modest", "Mild", "Quiet", "Bashful", "Rash",
    "Calm", "Gentle", "Sassy", "Careful", "Quirky",
]
NATURE_INDEX = {n: i for i, n in enumerate(NATURES)}

# Nature → (boosted_stat, reduced_stat); neutral natures have None, None.
# Stats: atk, def, spa, spd, spe
_NATURE_MODS: dict[str, tuple[Optional[str], Optional[str]]] = {
    "Hardy":   (None, None),   "Lonely":  ("atk", "def"),
    "Brave":   ("atk", "spe"), "Adamant": ("atk", "spa"),
    "Naughty": ("atk", "spd"), "Bold":    ("def", "atk"),
    "Docile":  (None, None),   "Relaxed": ("def", "spe"),
    "Impish":  ("def", "spa"), "Lax":     ("def", "spd"),
    "Timid":   ("spe", "atk"), "Hasty":   ("spe", "def"),
    "Serious": (None, None),   "Jolly":   ("spe", "spa"),
    "Naive":   ("spe", "spd"), "Modest":  ("spa", "atk"),
    "Mild":    ("spa", "def"), "Quiet":   ("spa", "spe"),
    "Bashful": (None, None),   "Rash":    ("spa", "spd"),
    "Calm":    ("spd", "atk"), "Gentle":  ("spd", "def"),
    "Sassy":   ("spd", "spe"), "Careful": ("spd", "spa"),
    "Quirky":  (None, None),
}

STAT_NAMES = ["hp", "atk", "def", "spa", "spd", "spe"]


@dataclass
class Pokemon:
    species: str
    item: str
    ability: str
    nature: str
    evs: dict[str, int]         # stat_name → EV value (0 to MAX_EV)
    moves: list[str]            # up to 4 move names
    level: int = 50

    # Filled in by encoder after PokéAPI lookup
    types: list[str] = field(default_factory=list)
    base_stats: dict[str, int] = field(default_factory=dict)

    def nature_multiplier(self, stat: str) -> float:
        """Returns 1.1, 0.9, or 1.0 for a given stat given this pokemon's nature."""
        boosted, reduced = _NATURE_MODS.get(self.nature, (None, None))
        if stat == boosted:
            return 1.1
        if stat == reduced:
            return 0.9
        return 1.0

    def actual_stat(self, stat: str) -> int:
        """Compute actual stat at level 50 with 31 IVs."""
        base = self.base_stats.get(stat, 0)
        ev = self.evs.get(stat, 0)
        lv = self.level
        if stat == "hp":
            return int((2 * base + 31 + ev // 4) * lv // 100) + lv + 10
        raw = int((2 * base + 31 + ev // 4) * lv // 100) + 5
        return int(raw * self.nature_multiplier(stat))

    @property
    def species_key(self) -> str:
        """Lowercase hyphenated species name for PokéAPI / vocab lookups."""
        return self.species.lower().replace(" ", "-")


@dataclass
class Team:
    name: str
    pokemon: list[Pokemon]

    def __len__(self):
        return len(self.pokemon)

    def __iter__(self):
        return iter(self.pokemon)

    def __getitem__(self, idx):
        return self.pokemon[idx]


def parse_showdown_team(text: str, name: str = "unknown") -> Team:
    """
    Parse a Showdown paste into a Team.

    Handles:
      - 'Species @ Item' or just 'Species'
      - 'Ability: ...'
      - 'Level: ...'
      - 'EVs: 32 HP / 32 Atk / ...'
      - 'Nature Nature' (single word before 'Nature')
      - '- Move Name'
    """
    pokemon = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        mon = _parse_block(block)
        if mon:
            pokemon.append(mon)
    return Team(name=name, pokemon=pokemon)


def _parse_block(block: str) -> Optional[Pokemon]:
    """Parse one blank-line-delimited Showdown block into a Pokemon, or None if invalid."""
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    if not lines:
        return None

    # First line: 'Species @ Item' or just 'Species'
    header = lines[0]
    if " @ " in header:
        species_raw, item = header.split(" @ ", 1)
    else:
        species_raw, item = header, ""
    species = species_raw.strip()

    ability = ""
    nature = "Hardy"
    level = 50
    evs: dict[str, int] = {s: 0 for s in STAT_NAMES}
    moves: list[str] = []

    _EV_STAT = {"HP": "hp", "Atk": "atk", "Def": "def", "SpA": "spa", "SpD": "spd", "Spe": "spe"}

    for line in lines[1:]:
        if line.startswith("Ability:"):
            ability = line.split(":", 1)[1].strip()
        elif line.startswith("Level:"):
            level = int(line.split(":", 1)[1].strip())
        elif line.startswith("EVs:"):
            ev_str = line.split(":", 1)[1].strip()
            for part in ev_str.split("/"):
                part = part.strip()
                m = re.match(r"(\d+)\s+(\w+)", part)
                if m:
                    val, stat_abbr = int(m.group(1)), m.group(2)
                    if stat_abbr in _EV_STAT:
                        evs[_EV_STAT[stat_abbr]] = val
        elif line.endswith(" Nature"):
            nature = line[: -len(" Nature")].strip()
        elif line.startswith("- "):
            moves.append(line[2:].strip())
        # Skip IVs, Happiness, Shiny, Tera Type lines — not used in this module

    return Pokemon(
        species=species,
        item=item.strip(),
        ability=ability,
        nature=nature,
        evs=evs,
        moves=moves[:4],
        level=level,
    )


def load_team_from_file(path: str | Path, name: str | None = None) -> Team:
    """Load a single Showdown team from a .txt file; uses the file stem as the team name."""
    path = Path(path)
    text = path.read_text()
    team_name = name or path.stem
    return parse_showdown_team(text, name=team_name)


def load_all_teams(teams_dir: str | Path) -> list[Team]:
    """Load all *.txt Showdown team files in a directory, sorted by filename."""
    teams_dir = Path(teams_dir)
    teams = []
    for txt in sorted(teams_dir.glob("*.txt")):
        teams.append(load_team_from_file(txt))
    return teams
