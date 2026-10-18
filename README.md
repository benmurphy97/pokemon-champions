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

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # numpy + PyTorch (training only; most tools need just the standard library)
bash scripts/setup_showdown.sh                                        # builds the battle engine into vendor/
```

## Layout

```
src/champions/      Python package: team format, config, (later) search and analysis
  config.py         Regulation, format ID, stat-point rules — the one place to edit per regulation
  teams.py          Read/write Showdown team text, check structural rules
  simulate.py       Python wrapper: run_battles(), validate_team()
  battle_state.py   Reads battle protocol lines (replays or live sim) into what a player can see
  vocab.py / encoding.py   Names -> ids, and a battle state -> arrays for the network
  decisions.py      Pulls each player's choices out of a replay log (training examples)
  policy.py         The policy network (small transformer; scores moves and switches)
  model_io.py       Save/load policies; old checkpoints keep working when the model grows (see docs/extending.md)
  train_il.py       Imitation learning: train the policy on replay decisions
  import_stats.py   Download Smogon ladder stats into data/meta/ (usage, sets, spreads, teammates)
  meta.py           Build realistic opponent teams from those stats
  calibrate.py      Compare simulated strength with ladder popularity
  import_teams.py   Collect tournament teams from MetaVGC (parked: site is slow to crawl)
sim/run.js          Node runner: plays battles on Showdown's engine in-process
sim/agents/greedy.js   Scripted player: damage estimates, support-move rules, bring/lead choice
sim/tournament.js   Round-robin between many teams, in-process (~9,700 games/min)
sim/export_reference.js   Writes data/reference/ from the engine's own data
scripts/            setup_showdown.sh clones and builds the pinned engine into vendor/
data/reference/     Pokémon, moves, items for M-C, generated from Showdown
data/meta/m-c/      Ladder usage summary (Smogon stats): sets, spreads, teammates per Pokémon
results/            Generated pools and tournament output (gitignored)
tests/              Unit tests; tests/fixtures holds sample teams
docs/               Notes and source links
```

`web/` is the static React site (see [web/README.md](web/README.md)); `src/champions/export_site.py` writes the JSON it reads, and `src/champions/ops_server.py` feeds its live operating-room tab.

## Setup and tests

Needs Node and Python 3.

```bash
./scripts/setup_showdown.sh                      # once: fetch and build the engine
python3 -m venv .venv && .venv/bin/pip install torch numpy   # once: for the learned agent
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

# play 1,000 battles between two teams (greedy agent by default; --agent-a/--agent-b random|greedy)
PYTHONPATH=src python3 -m champions.simulate tests/fixtures/team_tr_alpha.txt tests/fixtures/team_cybertron.txt --battles 1000
```

Speed on the Mac mini M5: about 5,800 battles per minute on one core.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 0 | Clean out the old RL code, new layout | done |
| 1 | Local Showdown simulator, M-C data refresh | done (ladder stats from Smogon; MetaVGC tournament import parked) |
| 2 | Baseline agent | greedy rule-based agent kept only as a yardstick to beat; its predictions did not match real results (docs/calibration.md) |
| 2b | Learned agent, imitation learning | in progress: tracker, encoding, network and trainer built; training on replays next |
| 3 | Team search with confidence intervals | |
| 4 | Static website | |
| 5 | Learned play from replays (optional) | |

## History

The first version (PPO + genetic algorithm over a websocket Showdown server) is preserved in git under the tag `archive/rl-v1`. It never produced results; the reasons are in the Phase 0 notes in `docs/`.
