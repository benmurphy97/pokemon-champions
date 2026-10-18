# Pokémon Champions — team discovery

Find the strongest teams for the current Pokémon Champions regulation (VGC doubles), explain why they work, and show the results on a small static website.

**Current regulation:** M-C (2026-09-09 to 2026-12-02). Change it in [src/champions/config.py](src/champions/config.py).

## How it works

```
1 pick 50 teams  →  2 imitation learning  →  3 self-play (100,000s of games)  →  4 evaluate as we go
        ▲                                                                              │
        └──────────── 5 swap out teams that keep losing ◄──────────────────────────────┘
```

1. **Teams** — scrape usage and tournament teams, then pick 50 strong training teams plus held-out test teams.
2. **Imitation** — the policy network first copies human players from Showdown replays.
3. **Self-play** — it then plays those teams against each other in Showdown's engine, learning from results (PPO).
4. **Evaluate** — paired games against fixed yardsticks (random, greedy-blind) on teams it never trained on.
5. **Replace** — teams that keep losing are swapped for new ones.

No paid APIs. No GPU needed.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # numpy + PyTorch (training only; most tools need just the standard library)
bash scripts/setup_showdown.sh                                        # builds the battle engine into vendor/
```

## Layout

```
src/champions/
  config.py            Regulation, format ID, paths — the one place to edit per regulation
  scrapers/            Download from the internet
    battledata.py        In-game Ranked Battle usage incl. stat spreads (championsbattledata.com)
    showdown_stats.py    Showdown ladder usage (Smogon stats)
    replays.py           Showdown ladder replays (for imitation learning)
    metavgc.py           Tournament teams with exact pastes (MetaVGC + pokepast.es)
    labmaus.py           Top online-tournament teams (LabMaus)
    limitless.py         Regional teams (Limitless VGC; spreads estimated)
  teams/               Team files, legality, pools and ratings
    parse.py             Read/write Showdown team text, check structural rules
    stats.py             Battle stats from base stats, nature and stat points
    spreads.py           Estimate a missing nature and spread
    tournament_pool.py   Imported tournament teams -> results/web_teams.json
    ladder_pool.py       Real ladder teams rebuilt from replays
    generate.py          Build teams from usage stats
    pools.py             Pool file paths and loading
    rate.py              Rate teams against a fixed gauntlet
  sim/                 Python side of the simulator
    simulate.py          Run battles / validate teams via sim/run.js
    arena.py             Many games with the learned player in the loop; paired-game scoring
  model/               The learned player
    battle_state.py      What a player can see, from protocol lines
    vocab.py, encoding.py   Names -> ids, battle state -> arrays
    damage.py            Damage maths used as network inputs
    policy.py            The policy network (transformer; joint choice for both active Pokémon)
    agent.py             Plays a live battle with a policy (incl. the repeat-Protect block)
    preview.py           Learned team preview (bring four, lead two)
    model_io.py          Save/load; old checkpoints keep working as the model grows
  training/
    decisions.py, hindsight.py, preview_labels.py   Training examples from replays
    imitation.py         Imitation learning on replays
    ppo.py               The PPO update
    selfplay.py          Self-play league training
  evaluation/
    gates.py             The agreed gates (vs random, vs greedy-blind on held-out teams)
  site/
    ops_server.py        Live API for the operating room
    match_viewer.py      One match with every decision recorded
    export_site.py       JSON for the static site
sim/                   Node: Showdown engine runners (run.js, arena_worker.js, matchups.js, agents/)
vendor/                The pinned Showdown engine (scripts/setup_showdown.sh builds it)
data/                  reference/ (Pokémon, moves, items), meta/ (usage), teams/ (pastes + index), ratings/
models/  results/  cache/   Trained weights, logs and pools, downloads (all gitignored)
web/                   React site: operating room and team browser (see web/README.md)
tests/  docs/
```

Run any module with `PYTHONPATH=src .venv/bin/python -m champions.<folder>.<module>`, e.g. `champions.training.selfplay`.

## Setup and tests

Needs Node and Python 3.

```bash
./scripts/setup_showdown.sh                      # once: fetch and build the engine
python3 -m venv .venv && .venv/bin/pip install torch numpy   # once: for the learned agent
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

# play 1,000 battles between two teams (greedy agent by default; --agent-a/--agent-b random|greedy)
PYTHONPATH=src python3 -m champions.sim.simulate tests/fixtures/team_tr_alpha.txt tests/fixtures/team_cybertron.txt --battles 1000
```

Speed on the Mac mini M5: about 5,800 battles per minute on one core.

## Plan

The current plan, rules and gates are in [docs/plan.md](docs/plan.md); results of every run are in [docs/learned-agent.md](docs/learned-agent.md).

## History

The first version (PPO + genetic algorithm over a websocket Showdown server) is preserved in git under the tag `archive/rl-v1`. It never produced results; the reasons are in the Phase 0 notes in `docs/`.
