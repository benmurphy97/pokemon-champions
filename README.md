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
  teams.py          Read/write Showdown team text, check structural rules
  simulate.py       Python wrapper: run_battles(), validate_team()
  import_teams.py   Collect real tournament teams (MetaVGC + pokepast.es) into data/teams/
sim/run.js          Node runner: plays battles on Showdown's engine in-process
sim/agents/greedy.js   Scripted player: damage estimates, support-move rules, bring/lead choice
sim/export_reference.js   Writes data/reference/ from the engine's own data
scripts/            setup_showdown.sh clones and builds the pinned engine into vendor/
data/reference/     Pokémon, moves, items for M-C, generated from Showdown
data/teams/m-c/     Real tournament teams as Showdown pastes, plus index.json with sources
tests/              Unit tests; tests/fixtures holds sample teams
docs/               Notes and source links
```

Planned, not yet created: `web/` (static site).

## Setup and tests

Needs Node and Python 3.

```bash
./scripts/setup_showdown.sh                      # once: fetch and build the engine
PYTHONPATH=src python3 -m unittest discover -s tests -v

# play 1,000 battles between two teams (greedy agent by default; --agent-a/--agent-b random|greedy)
PYTHONPATH=src python3 -m champions.simulate tests/fixtures/team_tr_alpha.txt tests/fixtures/team_cybertron.txt --battles 1000
```

Speed on the Mac mini M5: about 5,800 battles per minute on one core.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 0 | Clean out the old RL code, new layout | done |
| 1 | Local Showdown simulator, M-C data refresh | simulator and reference data done; team import written, full run + usage stats to revisit |
| 2 | Scripted battle agent (damage calc + shallow search) | greedy agent done (beats random 95%+); lookahead and calibration to do |
| 3 | Team search with confidence intervals | |
| 4 | Static website | |
| 5 | Learned play from replays (optional) | |

## History

The first version (PPO + genetic algorithm over a websocket Showdown server) is preserved in git under the tag `archive/rl-v1`. It never produced results; the reasons are in the Phase 0 notes in `docs/`.
