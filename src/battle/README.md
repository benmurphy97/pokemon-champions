# battle/

Core battle infrastructure — environment, observation encoding, and agents.

## Files

| File | Purpose |
|---|---|
| `env.py` | `DoublesEnv` — poke-env gym wrapper; handles action masking, reward shaping, switch conflict resolution |
| `encoder.py` | Encodes a live `DoubleBattle` into a 222-dim `np.ndarray` for the PPO policy |
| `teams.py` | Loads Showdown team strings from `data/teams/*.txt`; `RotatingTeambuilder` for opponent pool |
| `agents/base.py` | `BaseAgent` — thin poke-env `Player` subclass shared by all agents |
| `agents/heuristic.py` | `HeuristicAgent` — protect-aware, focus-fire rule-based agent used as baseline and opponent |
| `agents/random.py` | `RandomAgent` — uniform random moves, used for smoke tests |
| `agents/self_play.py` | `SelfPlayAgent` — wraps a frozen MaskablePPO checkpoint as an opponent |

## Observation layout (222 dims)

```
[0:4]      active HP fractions         (our0, our1, opp0, opp1)
[4:12]     type vulnerability pairs    (4 slots × 2 opponents)
[12:40]    stat boosts                 (4 slots × 7 stats)
[40:96]    our moves                   (2 slots × 4 moves × 7 features)
[96:100]   bench HP                    (4 bench mons)
[100:109]  weather + field flags       (4 + 5 binary)
[109:133]  active status               (4 slots × 6 conditions)
[133:157]  bench status                (4 bench × 6 conditions)
[157:213]  opponent revealed moves     (2 slots × 4 moves × 7 features)
[213:222]  weather + field turn counts (4 + 5)
```

## Action space

`MultiDiscrete([107, 107])` — one integer per active slot.

- `0` — pass (forced)
- `1–6` — switch to bench slot 1–6
- `7–10` — move 1–4
- `27–30` — move 1–4 + mega evolve
