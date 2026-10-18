# Website

A small static React site (Vite) that lists tournament teams with their strategies and shows ladder usage per Pokémon.

```bash
PYTHONPATH=src python3 -m champions.site.export_site   # from the repo root: writes web/public/data/*.json
cd web && npm install && npm run dev               # develop at http://localhost:5173
npm run build                                      # static site in web/dist
```

The site reads only the JSON files in `public/data/`, so any static host works. Sprites load from Pokémon Showdown's public sprite server.

## Operating room

The first tab is a live monitor for training. It polls a small feed that reads the logs in `results/` and the running processes:

```bash
PYTHONPATH=src .venv/bin/python -m champions.site.ops_server   # serves http://localhost:8765/api/ops (venv: the match viewer uses PyTorch)
cd web && npm run dev                            # the dev server forwards /api to it
```

It shows the training pulse (a heartbeat that goes flat when nothing is running), iteration and time left, win rates by opponent, the held-out checks against the rule-based player, drift and randomness of the policy, every past run side by side, the known-good teams in training, and alerts (stalled run, ahead of or behind the baseline, drifting too far).

**Live surgery** (second view inside the operating room) plays one match with the learned player against the rule-based player and lets you step through it: the board, what just happened, and for each Pokémon's decision the probability the policy gave every legal action, the damage numbers it was shown for each move, its target preference, the threat it saw, and its estimated chance of winning. Arrow keys step through.
