"""
Feature encoder for the gen9championsvgc2026regma selection module.

Your Pokemon  → ids: 7 ints  | floats: 90
Opp Pokemon   → ids: 1 int   | floats: 148
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from stage2_selection.pokemon import NATURES, NATURE_INDEX, STAT_NAMES, Pokemon, Team

# ── Types ────────────────────────────────────────────────────────────────────

POKEMON_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]
_TYPE_IDX: dict[str, int] = {t: i for i, t in enumerate(POKEMON_TYPES)}
N_TYPES = len(POKEMON_TYPES)  # 18

# ── Role-tag move / ability sets ─────────────────────────────────────────────

_TR_SETTER_MOVES    = {"Trick Room"}
_WEATHER_MOVES      = {"Rain Dance", "Sunny Day", "Sandstorm", "Snowscape", "Hail"}
_WEATHER_ABILITIES  = {
    "Drizzle", "Drought", "Sand Stream", "Snow Warning", "Sand Spit",
    "Cloud Nine",  # neutralises but is presence indicator
}
_TERRAIN_MOVES      = {
    "Electric Terrain", "Grassy Terrain", "Misty Terrain", "Psychic Terrain",
}
_TERRAIN_ABILITIES  = {
    "Electric Surge", "Grassy Surge", "Misty Surge", "Psychic Surge",
}
_REDIRECTOR_MOVES   = {"Follow Me", "Rage Powder"}
_PROTECT_MOVES      = {
    "Protect", "Wide Guard", "Quick Guard", "Detect",
    "Spiky Shield", "Baneful Bunker", "Silk Trap",
    "Burning Bulwark", "King's Shield",
}
_SPEED_CTRL_MOVES   = {
    "Tailwind", "Icy Wind", "Electroweb", "Thunder Wave",
    "Sticky Web", "Glacial Lance",
}

# Spread moves: target hits both foes (looked up from moves.json at build time)
_SPREAD_TARGETS = {"allAdjacent", "allAdjacentFoes"}

# ── Normalisation constants ──────────────────────────────────────────────────

_BASE_STAT_NORM = 255.0
_EV_NORM        = 32.0   # MAX_EV in this format
_ACTUAL_NORM    = {"hp": 400.0, "atk": 350.0, "def": 350.0,
                   "spa": 350.0, "spd": 350.0, "spe": 350.0}

# ── Vocab ────────────────────────────────────────────────────────────────────

@dataclass
class Vocab:
    """Integer-index vocabularies for embedding tables. Index 0 = <unk>."""
    species:   list[str]
    abilities: list[str]
    items:     list[str]
    moves:     list[str]

    species_idx:   dict[str, int] = field(default_factory=dict, repr=False)
    ability_idx:   dict[str, int] = field(default_factory=dict, repr=False)
    item_idx:      dict[str, int] = field(default_factory=dict, repr=False)
    move_idx:      dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.species_idx   = {s: i for i, s in enumerate(self.species)}
        self.ability_idx   = {a: i for i, a in enumerate(self.abilities)}
        self.item_idx      = {t: i for i, t in enumerate(self.items)}
        self.move_idx      = {m: i for i, m in enumerate(self.moves)}

    def save(self, path: str | Path) -> None:
        """Serialize vocabulary lists to a JSON file."""
        data = {
            "species":   self.species,
            "abilities": self.abilities,
            "items":     self.items,
            "moves":     self.moves,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        """Load a previously saved vocab from JSON."""
        data = json.loads(Path(path).read_text())
        return cls(**data)

    @classmethod
    def build(
        cls,
        details: dict[str, dict],
        usage: list[dict],
        all_items: list[str],
        all_moves_meta: dict[str, dict],
        teams: list[Team],
    ) -> "Vocab":
        """Build vocabularies from all data sources. Index 0 = <unk>."""
        # Species: all keys in details dict
        species_set: list[str] = ["<unk>"] + sorted(details.keys())

        # Abilities: from details + teams
        ability_set: set[str] = set()
        for entry in details.values():
            ability_set.update(entry.get("abilities", []))
        for team in teams:
            for mon in team:
                if mon.ability:
                    ability_set.add(mon.ability)
        abilities = ["<unk>"] + sorted(ability_set)

        # Items: from champions_items.json + usage + teams
        item_set: set[str] = set(all_items)
        for entry in usage:
            for it in entry.get("items", []):
                item_set.add(it["name"])
        for team in teams:
            for mon in team:
                if mon.item:
                    item_set.add(mon.item)
        items = ["<unk>"] + sorted(item_set)

        # Moves: all moves from moves.json + usage + teams
        move_set: set[str] = {m["name"] for m in all_moves_meta.values() if "name" in m}
        for entry in usage:
            for mv in entry.get("moves", []):
                move_set.add(mv["name"])
        for team in teams:
            for mon in team:
                move_set.update(mon.moves)
        moves = ["<unk>"] + sorted(move_set)

        return cls(
            species=species_set,
            abilities=abilities,
            items=items,
            moves=moves,
        )


# ── Encoder ──────────────────────────────────────────────────────────────────

class PokemonEncoder:
    """
    Converts Pokemon objects / species names to fixed-length feature vectors.

    Args:
        details:      Merged dict from champions_pokemon_details.json
                      (keys: Showdown-format species names, e.g. "Rotom-Heat")
        usage:        List of usage entries from champions_usage.json
        moves_meta:   Dict from moves.json (move-id → move dict)
        vocab:        Vocabulary for integer IDs
    """

    YOUR_FLOAT_DIM: int = 90   # 36+6+6+6+25+10+1
    OPP_FLOAT_DIM:  int = 148  # 36+6+16+16+64+10
    YOUR_ID_DIM:    int = 7    # species, ability, item, 4×move
    OPP_ID_DIM:     int = 1    # species only

    def __init__(
        self,
        details:    dict[str, dict],
        usage:      list[dict],
        moves_meta: dict[str, dict],
        vocab:      Vocab,
    ) -> None:
        self._details    = details
        self._moves_meta = moves_meta
        self.vocab       = vocab

        # Usage indexed by pokebase slug (lowercase-hyphenated species key)
        self._usage_by_slug: dict[str, dict] = {e["slug"]: e for e in usage}

        # Set of spread-move names (target hits both foes)
        self._spread_moves: set[str] = {
            m["name"]
            for m in moves_meta.values()
            if m.get("target") in _SPREAD_TARGETS and (m.get("basePower") or 0) >= 40
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def encode_your_pokemon(self, mon: Pokemon, is_lead: bool) -> dict:
        """
        Returns:
            ids:    list[int] of length 7  (species, ability, item, move×4)
            floats: list[float] of length 90
        """
        d = self._get_details(mon.species)
        base = d["base_stats"]
        vi   = self.vocab

        ids = [
            vi.species_idx.get(mon.species, 0),
            vi.ability_idx.get(mon.ability, 0),
            vi.item_idx.get(mon.item, 0),
        ] + [vi.move_idx.get(m, 0) for m in (mon.moves + ["", "", "", ""])[:4]]

        floats = (
            _enc_types(d["types"])                  # 36
            + _enc_base_stats(base)                 # 6
            + _enc_evs(mon.evs)                     # 6
            + _enc_actual_stats(mon)                # 6
            + _enc_nature(mon.nature)               # 25
            + self._role_tags(mon.moves, base, mon.ability)  # 10
            + [1.0 if is_lead else 0.0]             # 1
        )
        assert len(floats) == self.YOUR_FLOAT_DIM, len(floats)
        return {"ids": ids, "floats": floats}

    def encode_opp_pokemon(self, species: str) -> dict:
        """
        Returns:
            ids:    list[int] of length 1  (species)
            floats: list[float] of length 148
        """
        d     = self._get_details(species)
        base  = d["base_stats"]
        usage = self._get_usage(species)

        abilities_u = [
            a for a in usage.get("abilities", []) if a["name"] != "(No ability)"
        ] if usage else []
        items_u     = usage.get("items", []) if usage else []
        moves_u     = usage.get("moves", []) if usage else []

        # Top-4 usage moves used for role-tag inference on opponent
        top_moves = [m["name"] for m in moves_u[:4]]
        top_ability = abilities_u[0]["name"] if abilities_u else ""

        floats = (
            _enc_types(d["types"])                              # 36
            + _enc_base_stats(base)                            # 6
            + _enc_prior([a["name"] for a in abilities_u],
                         [a["usage_pct"] / 100 for a in abilities_u], 16)  # 16
            + _enc_prior([i["name"] for i in items_u],
                         [i["usage_pct"] / 100 for i in items_u], 16)     # 16
            + _enc_prior([m["name"] for m in moves_u],
                         [m["usage_pct"] / 100 for m in moves_u], 64)     # 64
            + self._role_tags(top_moves, base, top_ability)               # 10
        )
        assert len(floats) == self.OPP_FLOAT_DIM, len(floats)
        return {
            "ids":    [self.vocab.species_idx.get(species, 0)],
            "floats": floats,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_details(self, species: str) -> dict:
        """Return the details dict for a species, with case-insensitive fallback and safe default."""
        d = self._details.get(species)
        if d is None:
            # Case-insensitive fallback
            key_low = species.lower()
            d = next(
                (v for k, v in self._details.items() if k.lower() == key_low),
                None,
            )
        if d is None:
            return {
                "types": [],
                "base_stats": {s: 0 for s in STAT_NAMES},
                "abilities": [],
            }
        return d

    def _get_usage(self, species: str) -> Optional[dict]:
        """Return the usage entry for a species by lowercase-hyphenated slug, or None."""
        slug = species.lower().replace(" ", "-")
        return self._usage_by_slug.get(slug)

    def _role_tags(
        self,
        moves:   list[str],
        base:    dict[str, int],
        ability: str = "",
    ) -> list[float]:
        move_set = set(moves)
        return [
            float(bool(move_set & _TR_SETTER_MOVES)),                        # 1 TR setter
            float(base.get("spe", 999) <= 60),                              # 2 TR abuser
            float(bool(move_set & _WEATHER_MOVES)
                  or ability in _WEATHER_ABILITIES),                         # 3 weather
            float(bool(move_set & _TERRAIN_MOVES)
                  or ability in _TERRAIN_ABILITIES),                         # 4 terrain
            float(bool(move_set & _REDIRECTOR_MOVES)),                       # 5 redirector
            float(bool(move_set & self._spread_moves)),                      # 6 spread attacker
            float(bool(move_set & _PROTECT_MOVES)),                          # 7 has protect
            float(base.get("hp", 0) + base.get("def", 0)
                  + base.get("spd", 0) > 220),                              # 8 bulky
            float(base.get("spe", 0) > 100),                                # 9 fast
            float(bool(move_set & _SPEED_CTRL_MOVES)),                       # 10 speed control
        ]

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_data_dir(
        cls,
        data_dir:   str | Path,
        teams_dir:  str | Path | None = None,
        vocab_path: str | Path | None = None,
    ) -> "PokemonEncoder":
        """
        Load all data from standard directory layout and return an encoder.
        Builds (or loads) vocab automatically.
        """
        from stage2_selection.pokemon import load_all_teams

        data_dir = Path(data_dir)

        # champions_pokemon_details.json: merge pokemon + megas into one dict
        with open(data_dir / "champions_pokemon_details.json") as f:
            raw = json.load(f)
        details: dict[str, dict] = {**raw["pokemon"], **raw["megas"]}

        # champions_usage.json
        with open(data_dir / "champions_usage.json") as f:
            usage_data = json.load(f)
        usage: list[dict] = usage_data["pokemon"]

        # moves.json  — {"moves": [{id, name, type, category, basePower, target, ...}, ...]}
        with open(data_dir / "moves.json") as f:
            moves_raw = json.load(f)
        moves_list = moves_raw.get("moves", []) if isinstance(moves_raw, dict) else moves_raw
        moves_meta: dict[str, dict] = {
            m["id"]: m for m in moves_list if isinstance(m, dict) and "id" in m
        }

        # All legal items (flat list)
        with open(data_dir / "champions_items.json") as f:
            items_raw = json.load(f)
        all_items: list[str] = [
            item
            for cat in items_raw["categories"].values()
            for item in cat
        ]

        # Teams for vocab coverage
        teams: list[Team] = []
        if teams_dir is None:
            teams_dir = data_dir / "teams"
        td = Path(teams_dir)
        if td.exists():
            teams = load_all_teams(td)

        # Vocab: load from cache or build fresh
        if vocab_path and Path(vocab_path).exists():
            vocab = Vocab.load(vocab_path)
        else:
            vocab = Vocab.build(details, usage, all_items, moves_meta, teams)
            if vocab_path:
                vocab.save(vocab_path)

        return cls(details=details, usage=usage, moves_meta=moves_meta, vocab=vocab)


# ── Pure encoding helpers (module-level for speed) ───────────────────────────

def _enc_types(types: list[str]) -> list[float]:
    """Two type slots → 18+18 = 36 floats (one-hot each)."""
    v = [0.0] * (N_TYPES * 2)
    for slot, t in enumerate(types[:2]):
        idx = _TYPE_IDX.get(t)
        if idx is not None:
            v[slot * N_TYPES + idx] = 1.0
    return v


def _enc_base_stats(base: dict[str, int]) -> list[float]:
    """6 base stats normalised by 255."""
    return [base.get(s, 0) / _BASE_STAT_NORM for s in STAT_NAMES]


def _enc_evs(evs: dict[str, int]) -> list[float]:
    """6 EV values normalised by MAX_EV (32)."""
    return [evs.get(s, 0) / _EV_NORM for s in STAT_NAMES]


def _enc_actual_stats(mon: Pokemon) -> list[float]:
    """Computed actual stats at level 50, normalised."""
    return [mon.actual_stat(s) / _ACTUAL_NORM[s] for s in STAT_NAMES]


def _enc_nature(nature: str) -> list[float]:
    """25-dim one-hot over nature index."""
    v = [0.0] * len(NATURES)
    idx = NATURE_INDEX.get(nature)
    if idx is not None:
        v[idx] = 1.0
    return v


def _enc_prior(
    names: list[str],
    probs: list[float],
    dim:   int,
) -> list[float]:
    """
    Ranked prior vector of length `dim`.
    Entries 0..len(names)-1 are the usage probabilities in descending order.
    Remaining entries are 0.
    """
    v = [0.0] * dim
    for i, p in enumerate(probs[:dim]):
        v[i] = p
    return v
