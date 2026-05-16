# battle_decisions/league/

Population-based training (PBT) league — multiple agents co-evolve against each other across generations.

## Files

| File | Purpose |
|---|---|
| `agent.py` | `Agent` dataclass — owns a team string, checkpoint path, fitness history, and mon tenure |
| `runner.py` | `League` — orchestrates train → evaluate → exploit/explore each generation |

## How it works

Each generation:
1. **Train** — every agent trains for `round_steps` steps against a rotating `HeuristicAgent` pool
2. **Evaluate** — each agent's fitness is measured as a meta-weighted win rate across all 8 opponent archetypes
3. **Exploit** — the bottom agent copies weights from the top agent
4. **Explore** — the exploited agent mutates its team via `genetic_algorithm/chromosome.py`
5. **Snapshot** — every N generations, checkpoints are archived

## Usage

```bash
docker-compose run --rm trainer python src/train.py league \
    --generations 20 --agents 5 --round-steps 3000 --eval-battles 8
```

Checkpoints are saved to `model_outputs/checkpoints/league/agent_N/`.
