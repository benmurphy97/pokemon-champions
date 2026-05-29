# genetic_algorithm/

DEAP-based genetic algorithm for evolving Showdown team compositions.

> **Phase 2 — not yet wired into the main training pipeline.** The mutation operators are used by the PBT league (`battle_decisions/league/`); the full GA island model is a future phase.

## Files

| File | Purpose |
|---|---|
| `chromosome.py` | Parse/serialise Showdown team strings; four mutation operators (EV, item, move, Pokémon swap) |
| `pool.py` | Curated Pokémon pool for `mutate_pokemon` — meta-relevant mons with default sets |
| `fitness.py` | `evaluate_team()` — runs N battles vs each of 8 archetypes, returns meta-weighted win rate |
| `evolve.py` | DEAP toolbox setup, `run_ga()` single-island loop, `island_model()` stub |

## Mutation operators (`chromosome.py`)

| Operator | Weight | Description |
|---|---|---|
| `mutate_evs` | 30% | Shift EVs between two stats on one Pokémon |
| `mutate_item` | 25% | Swap one item for another from the legal pool |
| `mutate_move` | 15% | Replace one move with a candidate from the curated list |
| `mutate_pokemon` | 30% | Swap one non-Mega holder for a random pool Pokémon |

Constraints enforced: EV total ≤ 66, max 32 per stat, Item Clause, at most one Mega Stone holder.

## Usage

```python
from genetic_algorithm.chromosome import mutate
new_team = mutate(team_str, tenure={"Incineroar": 5, "Farigiraf": 2})
```
