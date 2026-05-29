# battle_decisions/

In-battle decision models — neural architecture, training pipelines, analytics, and live inference.

## Structure

| Folder | Purpose |
|---|---|
| `architecture/` | Neural net class definitions — `BattleStateEncoder`, future `CoordinatedPolicy` |
| `training/` | Training entry point (`train.py`) plus imitation and league sub-pipelines |
| `training/imitation/` | Behavioural cloning pretrain from Showdown replays |
| `training/league/` | Population-based training (PBT) co-evolution league |
| `serving/` | Live inference utilities (checkpoint loading, agent wrappers) |
| `analytics/` | Battle log parser and per-mon scoring metrics |

## Entry point

```bash
docker-compose run --rm trainer python src/battle_decisions/training/train.py train
# Subcommands: baseline | train | bc | evaluate | mutate | league
```

## BattleStateEncoder (`architecture/extractor.py`)

Replaces the default flat MLP with a per-mon encoder that gives each action head slot-aware context.

```
Input:  222-dim flat obs
Output: 288-dim feature vector = cat(slot0_ctx[144], slot1_ctx[144])

slot0_ctx = (our0[32], our1[32], opp0[32], opp1[32], field[16])
slot1_ctx = (our1[32], our0[32], opp0[32], opp1[32], field[16])  ← our slots swapped
```

Each active mon is encoded by a shared MLP (44→64→32). The field state (bench HP, weather, terrain, bench status, turn counts) is encoded separately (46→32→16). Swapping the slot order gives each action head its own mon's encoding first.

## Upcoming: CoordinatedPolicy (`architecture/policy.py`)

Cross-attention between slot0_ctx and slot1_ctx so both action heads coordinate before committing to moves. Subclasses `MaskableActorCriticPolicy`.
