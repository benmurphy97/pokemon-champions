# battle_decisions/analytics/

Post-game evaluation — parse Showdown battle logs into structured records and score them.

## Files

| File | Purpose |
|---|---|
| `log_parser.py` | Parses a Showdown `.log` file into a `BattleRecord` (actions, damage events, faints, weather) |
| `metrics.py` | Scores a `BattleRecord` into a `BattleScore` with per-mon contributions and a GA fitness value |

## Public API

```python
from battle_decisions.analytics.log_parser import parse_log_file
from battle_decisions.analytics.metrics import score_battle, aggregate_fitness

record = parse_log_file("data/replays/battle.log")
score  = score_battle(record, our_player=1)
print(score.to_ga_fitness())   # float in [0, 1]
```

`BattleScore.to_ga_fitness()` is consumed by `genetic_algorithm/fitness.py` to evaluate team quality across multiple battles.
