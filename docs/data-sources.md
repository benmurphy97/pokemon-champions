# Data sources

| Data | Source | How |
|---|---|---|
| Pokémon, moves, items, learnsets | Showdown's `champions` mod | `node sim/export_reference.js` (pinned engine commit in `scripts/setup_showdown.sh`) |
| Ladder usage, sets, spreads, teammates (M-C) | [Smogon monthly stats](https://www.smogon.com/stats/) (official Showdown ladder, ~1.6M battles in 2026-09) | `python3 -m champions.import_stats` |
| Ladder results by team (not yet used) | Showdown replay search, `replay.pokemonshowdown.com/search.json?format=gen9championsvgc2026regmc` | planned: win rates by species, lead and bring |
| Tournament teams (parked) | [MetaVGC](https://metavgc.com) team pages, which link to pokepast.es | `python3 -m champions.import_teams` |

## Notes

- Smogon stats are published monthly with a lag (October's are not out yet), at rating cutoffs 0/1500/1630/1760. We use 1760 by default. Replays show species, the bring and the result but not full sets.
- The MetaVGC importer hangs on the site's large sitemaps from Python (cause not found; `curl` fetches them). It is parked, not deleted.
- MetaVGC's robots.txt allows crawling. The importer sends 1 request per second with an honest user agent, and caches the (slow, ~14 MB) sitemap URL list in `data/teams/m-c/urls.json`. Use `--refresh-urls` to re-download it.
- Teams are kept only if the page's regulation is M-C and the tournament date is on or after 2026-09-09. Each team is checked against Showdown's M-C validator; failures are recorded in `index.json`, not deleted.
- Pikalytics (`pikalytics.com/ai/pokedex/gen9championsvgc2026regmc/<Pokemon>`) publishes usage, win rate, move/item/ability percentages and teammates per Pokémon, but its pages are labelled "Data Date: 2026-05", which predates M-C. Treat it as a rough prior until its freshness is confirmed.
- Pokémon Zone returns 403 to scripted requests, so it is not used.
