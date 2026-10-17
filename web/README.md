# Website

A small static React site (Vite) that lists tournament teams with their strategies and shows ladder usage per Pokémon.

```bash
PYTHONPATH=src python3 -m champions.export_site   # from the repo root: writes web/public/data/*.json
cd web && npm install && npm run dev               # develop at http://localhost:5173
npm run build                                      # static site in web/dist
```

The site reads only the JSON files in `public/data/`, so any static host works. Sprites load from Pokémon Showdown's public sprite server.
