# battle_decisions/

In-battle decision models — structured feature extraction, imitation learning, analytics, and league training.

## Submodules

| Folder | Purpose |
|---|---|
| `extractor.py` | `BattleExtractor` — structured `BaseFeaturesExtractor` for MaskablePPO |
| `analytics/` | Battle log parser and per-mon scoring metrics |
| `imitation/` | Behavioural cloning pretrain from Showdown replays |
| `league/` | Population-based training (PBT) multi-agent league |

## BattleExtractor

Replaces the default flat MLP with a per-mon encoder that gives each action head slot-aware context.

```
Input:  222-dim flat obs
Output: 288-dim feature vector = cat(slot0_ctx[144], slot1_ctx[144])

slot0_ctx = (our0[32], our1[32], opp0[32], opp1[32], field[16])
slot1_ctx = (our1[32], our0[32], opp0[32], opp1[32], field[16])  ← our slots swapped
```

Each active mon is encoded by a shared MLP (44→64→32). The field state (bench HP, weather, terrain, bench status, turn counts) is encoded separately (46→32→16). Swapping the slot order gives each action head its own mon's encoding first, enabling dual-slot coordination without an autoregressive policy.
