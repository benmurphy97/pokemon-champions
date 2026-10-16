# Pokémon Champions — team discovery

Find the strongest teams for the current Pokémon Champions regulation (VGC doubles), explain why they work, and show the results on a small static website.

**Current regulation:** M-C (2026-09-09 to 2026-12-02). Change it in [src/champions/config.py](src/champions/config.py).

## How it works

```
data  →  simulate battles  →  search for teams  →  analyse  →  website
```

1. **Data** — reference tables (Pokémon, moves, items) plus current meta teams and usage.
2. **Simulate** — run Pokémon Showdown's battle engine locally, with no network round-trips.
3. **Search** — try team variations against the meta, keep what wins, with enough battles to trust the result.
4. **Analyse** — matchup tables, key Pokémon, how to pilot each team.
5. **Website** — a static page that reads the results as JSON.

No paid APIs. No GPU needed.

## Layout

```
src/champions/      Python package: team format, config, (later) search and analysis
  config.py         Regulation, format ID, stat-point rules — the one place to edit per regulation
  teams.py          Read/write Showdown team text, check legality
data/reference/     Pokémon, moves, items (M-A era; refreshed in Phase 1)
tests/              Unit tests; tests/fixtures holds sample teams
docs/               Notes and source links
```

Planned, not yet created: `sim/` (Node, Showdown engine), `web/` (static site).

## Run the tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Roadmap

| Phase | What | Status |
|---|---|---|
| 0 | Clean out the old RL code, new layout | in progress |
| 1 | Local Showdown simulator, M-C data refresh | next |
| 2 | Scripted battle agent (damage calc + shallow search) | |
| 3 | Team search with confidence intervals | |
| 4 | Static website | |
| 5 | Learned play from replays (optional) | |

## History

The first version (PPO + genetic algorithm over a websocket Showdown server) is preserved in git under the tag `archive/rl-v1`. It never produced results; the reasons are in the Phase 0 notes in `docs/`.
