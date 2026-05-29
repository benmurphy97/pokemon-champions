from dataclasses import dataclass, field


@dataclass
class PokemonUsage:
    name: str
    usage_pct: float
    rank: int = 0
    moves: list = field(default_factory=list)
    items: list = field(default_factory=list)
    abilities: list = field(default_factory=list)
    spreads: list = field(default_factory=list)
    teammates: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rank": self.rank,
            "usage_pct": self.usage_pct,
            "moves": self.moves,
            "items": self.items,
            "abilities": self.abilities,
            "spreads": self.spreads,
            "teammates": self.teammates,
        }


@dataclass
class CorePair:
    pokemon: list   # [str, str]
    win_rate: float
    usage_pct: float = 0.0


@dataclass
class Archetype:
    name: str
    win_rate: float
    core_pokemon: list = field(default_factory=list)
    example_teams: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "win_rate": self.win_rate,
            "core_pokemon": self.core_pokemon,
            "example_teams": self.example_teams,
        }
