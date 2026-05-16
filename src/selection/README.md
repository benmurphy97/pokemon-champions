# selection/

Team and lead selection pipeline — generates training data via heuristic battles and trains a neural model to score all 90 bring/lead combinations.

## Files

| File | Purpose |
|---|---|
| `pokemon.py` | `Pokemon` and `Team` dataclasses; Showdown team string parser |
| `encoder.py` | `PokemonEncoder` — encodes each mon into ID + float feature vectors for the MLP |
| `heuristic_agent.py` | `SelectionAgent` — forced bring/lead teampreview + turn-1 HP snapshot |
| `client.py` | Battle runner — runs a single (our_team, opp_team, bring, lead) combination and returns metrics |
| `experiment.py` | Enumerates all team matchups × 90 bring/lead combos, runs battles, writes JSONL |
| `dataset.py` | `SelectionDataset` — PyTorch Dataset over the experiment JSONL |
| `model.py` | `SelectionMLP` + `predict_best()` inference + `_train()` CLI |

## Pipeline

```
experiment.py  →  data/experiment_results.jsonl  →  model.py  →  model_outputs/cache/selection_model.pt
```

### Run the experiment

```bash
docker-compose run --rm trainer python src/selection/experiment.py --battles 5
```

Produces one JSONL record per `(our_roster, opp_roster, bring_indices, lead_indices)` with `win_rate` and `avg_turn1_hp_delta`.

### Train the model

```bash
docker-compose run --rm trainer python src/selection/model.py --epochs 20
```

### Inference

```python
from selection.model import SelectionMLP, predict_best
bring, lead = predict_best(model, our_team, opp_team, encoder)
```

## Label

`score = win_rate + 0.3 × avg_turn1_hp_delta`

The HP delta component rewards aggressive leads that deal early damage, not just leads that happen to win through attrition.

## Feature encoding

- **Our mons** (4 brought): 7 IDs (species, ability, item, 4 moves) × embed_dim + 90 floats (stats, type flags, move features, lead flag)
- **Opp mons** (full team of 6): 1 ID (species) + 148 floats (revealed stats, type flags)
- **Architecture**: per-mon encoder → mean pool → MLP head → scalar score
