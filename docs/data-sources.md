# Data sources

| Data | Source | How |
|---|---|---|
| Pokémon, moves, items, learnsets | Showdown's `champions` mod | `node sim/export_reference.js` (pinned engine commit in `scripts/setup_showdown.sh`) |
| Tournament teams (M-C) | [MetaVGC](https://metavgc.com) team pages, which link to pokepast.es | `python3 -m champions.import_teams` |
| Exact team sets | pokepast.es `/raw` | same importer |

## Notes

- MetaVGC's robots.txt allows crawling. The importer sends 1 request per second with an honest user agent, and caches the (slow, ~14 MB) sitemap URL list in `data/teams/m-c/urls.json`. Use `--refresh-urls` to re-download it.
- Teams are kept only if the page's regulation is M-C and the tournament date is on or after 2026-09-09. Each team is checked against Showdown's M-C validator; failures are recorded in `index.json`, not deleted.
- Pikalytics (`pikalytics.com/ai/pokedex/gen9championsvgc2026regmc/<Pokemon>`) publishes usage, win rate, move/item/ability percentages and teammates per Pokémon, but its pages are labelled "Data Date: 2026-05", which predates M-C. Treat it as a rough prior until its freshness is confirmed.
- Pokémon Zone returns 403 to scripted requests, so it is not used.
