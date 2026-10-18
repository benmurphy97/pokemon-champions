"""Work out a player's full sets, for training the policy on replays.

In a real game a player knows everything about their own six: moves, item, ability. A replay
log only shows what came up during the match. So each player's sets are reconstructed from the
whole log (moves used, items shown or consumed, abilities triggered, the Mega Stone when a
Pokémon Mega Evolves), and any slot the game never revealed is filled with the most common ladder
choice for that Pokémon. In live play the policy is given the real sets instead.

    info = complete_sets(final_state, "p1", usage)
    info[team_index] == {"moves": [...4 names], "item": ..., "ability": ..., "brought": bool, "stats": {hp, atk, ...}}
"""

import json

from champions.model.battle_state import BattleState
from champions.teams.generate import usage_key
from champions.teams.stats import STAT_KEYS, compute_stats
from champions.model.vocab import REFERENCE

_BASE_STATS = {name: mon["baseStats"] for name, mon in json.loads((REFERENCE / "pokemon.json").read_text()).items()}
_POINTS_KEY = {"HP": "hp", "Atk": "atk", "Def": "def", "SpA": "spa", "SpD": "spd", "Spe": "spe"}


def complete_sets(state: BattleState, side: str, usage: dict) -> dict:
    """Best estimate of each of `side`'s six sets, given the whole game's `state`."""
    info = {}
    for index, mon in enumerate(state.sides[side].team[:6]):
        key = usage_key(mon.species, mon.mega, usage)
        entry = usage.get(key) if key else None

        moves = list(mon.moves[:4])
        if entry:
            moves += [name for name, _ in entry["moves"] if name not in moves][: 4 - len(moves)]

        item = mon.item or (entry["items"][0][0] if entry else None)

        ability = mon.ability or (entry["abilities"][0][0] if entry else None)

        stats = None   # the most common ladder spread for this Pokémon, on its pre-Mega form
        base = _BASE_STATS.get(mon.species)
        if entry and base:
            nature, spread, _ = entry["spreads"][0]
            stats = compute_stats(base, nature, {_POINTS_KEY[k]: v for k, v in spread.items()})
        info[index] = {"moves": moves, "item": item, "ability": ability, "brought": mon.revealed, "stats": stats}
    return info
