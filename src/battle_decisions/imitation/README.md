# battle_decisions/imitation/

Behavioural cloning (BC) pretrain — initialise PPO policy weights from human replay data before RL fine-tuning.

## Files

| File | Purpose |
|---|---|
| `fetch_replays.py` | Downloads `.log` files from the Showdown replay API |
| `parse_replay.py` | Converts replay logs into `(obs, action)` pairs — a `BCDataset` |
| `bc_train.py` | Trains the MaskablePPO policy head with cross-entropy loss on expert actions |

## Usage

```bash
# 1. Fetch replays
docker-compose run --rm trainer python src/battle_decisions/imitation/fetch_replays.py --out-dir data/replays --pages 20

# 2. Pretrain via train.py
docker-compose run --rm trainer python src/train.py pretrain \
    --replay-dir data/replays \
    --dataset data/bc_dataset.npz \
    --epochs 20

# 3. Fine-tune with PPO
docker-compose run --rm trainer python src/train.py train --steps 50000 --selfplay
```

BC only touches the policy action head — the feature extractor (`BattleExtractor`) is randomly initialised and trained end-to-end during PPO.
