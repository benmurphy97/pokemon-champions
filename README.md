# pokemon-champions-rl

Reinforcement learning agent for the **Champions VGC 2026 Reg M-A** format on Pokémon Showdown. The agent learns to play doubles battles end-to-end — from team selection and lead choice through in-battle move decisions.

## Architecture

```
Stage 1 — Data collection       src/scraper/           Pikalytics usage scraper
Stage 2 — Team selection        src/selection/         Heuristic experiment + SelectionMLP
Stage 3 — In-battle decisions   src/battle/            PPO agent with BattleExtractor
Stage 4 — Analytics             src/battle_decisions/analytics/   Battle log metrics
Stage 5 — Team evolution        src/genetic_algorithm/ DEAP-based GA over team chromosomes
Stage 6 — League training       src/battle_decisions/league/      PBT multi-agent league
```

## Quick start

```bash
# Start Showdown server + trainer
docker-compose up

# Run a short PPO training pass
docker-compose run --rm trainer python src/train.py train --steps 10000

# Evaluate against all 8 opponent archetypes
docker-compose run --rm trainer python src/train.py evaluate --battles 20

# Run team-selection experiment (generates data for SelectionMLP)
docker-compose run --rm trainer python src/selection/experiment.py --battles 5

# Train SelectionMLP on experiment results
docker-compose run --rm trainer python src/selection/model.py --epochs 20
```

## Key files

| File | Purpose |
|---|---|
| `src/train.py` | Entry point — train / evaluate / mutate / league modes |
| `src/battle/env.py` | poke-env `DoublesEnv` wrapper |
| `src/battle/encoder.py` | 222-dim observation encoder |
| `src/battle_decisions/extractor.py` | `BattleExtractor` — structured per-mon feature encoder for PPO |
| `src/selection/experiment.py` | Heuristic battle runner that generates (bring, lead) training data |
| `src/selection/model.py` | `SelectionMLP` — scores all 90 bring/lead combinations |
| `src/genetic_algorithm/chromosome.py` | Team mutation operators (EV / item / move / Pokémon swap) |
| `docker-compose.yml` | Showdown + trainer + TensorBoard services |

## Format

- **Format**: `gen9championsvgc2026regma` — bring 4 of 6, lead 2
- **EV cap**: 32 per stat, 66 total
- **Observation**: 222-dim flat vector (HP, type matchups, boosts, moves, field state)
- **Action space**: `MultiDiscrete([107, 107])` — one action per active slot

## Model outputs

Checkpoints and cached model weights are written to `model_outputs/` (git-ignored).
