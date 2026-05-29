"""
Mega Evolution lookup table for EncoderV2.

Loaded at runtime from data/champions_pokemon_details.json["megas"].

Usage:
    table = MegaTable.from_data_dir("data/")
    mega_data = table.get("Kangaskhan")   # → MegaFormData or None
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MegaFormData:
    """Post-Mega form data for one species."""
    mega_species: str              # e.g. "Kangaskhan-Mega"
    base_pokemon:  str             # e.g. "Kangaskhan"
    types: list[str]               # post-Mega types
    ability: str                   # post-Mega ability (first listed)
    base_stats: dict[str, int]     # {hp, atk, def, spa, spd, spe}


class MegaTable:
    """
    Maps base Pokémon names → MegaFormData.

    For species with two Mega forms (Charizard X/Y) both are stored;
    get() returns the first. Use get_all() to retrieve both.
    """

    def __init__(self, entries: list[MegaFormData]) -> None:
        self._by_base:   dict[str, list[MegaFormData]] = {}
        self._by_species: dict[str, MegaFormData] = {}
        for e in entries:
            self._by_base.setdefault(e.base_pokemon, []).append(e)
            self._by_species[e.mega_species] = e

    def get(self, base_pokemon: str) -> Optional[MegaFormData]:
        """Return the first (or only) Mega form for a base Pokémon."""
        forms = self._by_base.get(base_pokemon, [])
        return forms[0] if forms else None

    def get_all(self, base_pokemon: str) -> list[MegaFormData]:
        """Return all Mega forms for a base Pokémon (e.g. both Charizard X/Y)."""
        return self._by_base.get(base_pokemon, [])

    def get_by_mega_species(self, mega_species: str) -> Optional[MegaFormData]:
        """Look up by post-Mega species name (e.g. 'Charizard-Mega-X')."""
        return self._by_species.get(mega_species)

    def has_mega(self, species: str) -> bool:
        """Return True if the species has any legal Mega form in M-A."""
        return species in self._by_base

    @classmethod
    def from_data_dir(cls, data_dir: str | Path) -> "MegaTable":
        """Load from champions_pokemon_details.json in the given directory."""
        path = Path(data_dir) / "champions_pokemon_details.json"
        with open(path) as f:
            raw = json.load(f)

        megas_dict: dict[str, dict] = raw.get("megas", {})
        entries: list[MegaFormData] = []

        # champions_megas.json has the base_pokemon mapping; derive from
        # the Showdown hyphenated form in the details dict.
        # Key format: "Kangaskhan-Mega", "Charizard-Mega-X", etc.
        # Base pokemon is everything before "-Mega" (or the whole string).
        for mega_species, data in megas_dict.items():
            base = _derive_base(mega_species)
            abilities = data.get("abilities", [])
            ability = abilities[0] if abilities else ""
            entries.append(MegaFormData(
                mega_species=mega_species,
                base_pokemon=base,
                types=data.get("types", []),
                ability=ability,
                base_stats=data.get("base_stats", {}),
            ))

        return cls(entries)


def _derive_base(mega_species: str) -> str:
    """
    Derive the base Pokémon name from the Showdown mega key.

    Examples:
      "Kangaskhan-Mega"   → "Kangaskhan"
      "Charizard-Mega-X"  → "Charizard"
      "Lopunny-Mega"      → "Lopunny"
      "Meowstic-M-Mega"   → "Meowstic-M"   (retain form suffix before -Mega)
    """
    # Find the first occurrence of "-Mega" and take everything before it
    idx = mega_species.find("-Mega")
    if idx >= 0:
        return mega_species[:idx]
    return mega_species
