import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from stage1_scraper.pikalytics import scrape_usage_list, scrape_pokemon_detail

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)

# Default to data/ relative to project root; override via DATA_DIR env var
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = os.getenv("DATA_DIR", str(_PROJECT_ROOT / "data"))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    log.info("=== Pikalytics ===")
    usage_entries = scrape_usage_list()

    log.info("Fetching detail for all %d Pokemon...", len(usage_entries))
    for mon, slug in usage_entries:
        if not slug:
            log.warning("No slug for %s, skipping detail", mon.name)
            continue
        detail = scrape_pokemon_detail(slug)
        if detail.get("usage_pct") is not None:
            mon.usage_pct = detail["usage_pct"]
        mon.moves = detail.get("moves", [])
        mon.items = detail.get("items", [])
        mon.abilities = detail.get("abilities", [])
        mon.spreads = detail.get("spreads", [])
        mon.teammates = detail.get("teammates", [])

    usage_list = [mon for mon, _ in usage_entries]
    usage_path = os.path.join(DATA_DIR, "usage.json")
    with open(usage_path, "w") as f:
        json.dump(
            {
                "format": "gen9championsvgc2026regma",
                "scraped_at": now,
                "pokemon": [p.to_dict() for p in usage_list],
            },
            f,
            indent=2,
        )
    log.info("Saved %d Pokemon to %s", len(usage_list), usage_path)
    log.info("Scrape complete.")


if __name__ == "__main__":
    main()
