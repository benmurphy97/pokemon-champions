# Website

A small static React site (Vite) that lists tournament teams with their strategies and shows ladder usage per Pokémon.

```bash
PYTHONPATH=src python3 -m champions.export_site   # from the repo root: writes web/public/data/*.json
cd web && npm install && npm run dev               # develop at http://localhost:5173
npm run build                                      # static site in web/dist
```

The site reads only the JSON files in `public/data/`, so any static host works. Sprites load from Pokémon Showdown's public sprite server.

## Operating room

The first tab is a live monitor for training. It polls a small feed that reads the logs in `results/` and the running processes:

```bash
PYTHONPATH=src python3 -m champions.ops_server   # serves http://localhost:8765/api/ops
cd web && npm run dev                            # the dev server forwards /api to it
```

It shows the training pulse (a heartbeat that goes flat when nothing is running), iteration and time left, win rates by opponent, the held-out checks against the rule-based player, drift and randomness of the policy, every past run side by side, the known-good teams in training, and alerts (stalled run, ahead of or behind the baseline, drifting too far).
