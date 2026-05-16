# scraper/

Pikalytics usage scraper — fetches move, item, ability, spread, and teammate usage data for the Champions VGC meta.

## Files

| File | Purpose |
|---|---|
| `scrape.py` | Entry point — scrapes all Pokémon and writes `data/usage.json` |
| `pikalytics.py` | Fetches and parses individual Pokémon detail pages |
| `models.py` | `PokemonUsage` dataclass |
| `session.py` | Polite HTTP session with rate limiting |
| `fetch_usage.py` | Lightweight fetch script for quick usage-list-only updates |

## Usage

```bash
docker-compose run --rm scraper
# writes data/usage.json — consumed by src/selection/encoder.py
```

The scraper reads from `data/pikalytics_roster.json` (a curated list of 214 format-legal Pokémon) rather than scraping the full Pikalytics listing, to stay within the format's roster constraint.
