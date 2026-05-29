#!/usr/bin/env python3
"""
Scrape tournament usage stats from https://pokebase.app/pokemon-champions/pokemon
and save to data/usage.json.

Only keeps Pokemon that appear in data/champions_pokemon.json or data/champions_megas.json.
Fetches 3 list pages, then a detail page per matched Pokemon.
"""

import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent
OUT_FILE = DATA_DIR / "usage.json"
POKEMON_FILE = DATA_DIR / "champions_pokemon.json"
MEGAS_FILE = DATA_DIR / "champions_megas.json"

BASE_URL = "https://pokebase.app/pokemon-champions/pokemon"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
DELAY = 0.5


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8")


def name_to_slugs(name: str) -> list[str]:
    """Return candidate pokebase.app slugs for a Pokemon name (may return multiple)."""
    # Megas: "Mega Venusaur" -> "venusaur-mega", "Mega Charizard X" -> "charizard-mega-x"
    if name.startswith("Mega "):
        rest = name[5:]
        parts = rest.lower().split()
        if len(parts) == 1:
            return [f"{parts[0]}-mega"]
        return [f"{parts[0]}-mega-{'-'.join(parts[1:])}"]

    # Regional forms (may include sub-form in parens)
    for prefix, region in [
        ("Alolan ", "alola"), ("Hisuian ", "hisui"),
        ("Galarian ", "galar"), ("Paldean ", "paldea"),
    ]:
        if name.startswith(prefix):
            rest = name[len(prefix):]
            if "(" in rest:
                species = rest[:rest.index("(")].strip().lower().replace(" ", "-")
                form_raw = rest[rest.index("(")+1:rest.index(")")].lower()
                form_word = form_raw.split()[0]
                return [f"{species}-{region}-{form_word}", f"{species}-{form_word}"]
            return [rest.lower().replace(" ", "-") + f"-{region}"]

    # Rotom forms: "Heat Rotom" -> "rotom-heat"
    for form in ["Heat", "Wash", "Frost", "Fan", "Mow"]:
        if name == f"{form} Rotom":
            return [f"rotom-{form.lower()}"]

    # Forms in parentheses
    if "(" in name:
        base = name[:name.index("(")].strip().lower().replace(" ", "-").replace("'", "")
        form = name[name.index("(")+1:name.index(")")].lower().replace(" ", "-")
        if form in ("female", "f"):
            return [f"{base}-f", f"{base}-female", base]
        if form in ("male", "m"):
            return [base, f"{base}-m", f"{base}-male"]
        form_word = form.split("-")[0]
        # e.g. "Small Variety" -> "small", "Midday Form" -> "midday"
        return [f"{base}-{form_word}", f"{base}-{form}"]

    # Standard: strip special chars, spaces to hyphens
    clean = name.lower().replace(" ", "-").replace("'", "").replace(".", "").replace(":", "")
    return [clean]


# Typos in our data files and special slug cases that auto-conversion misses.
# Maps our stored name -> correct pokebase slug(s).
_MANUAL: dict[str, list[str]] = {
    # Typos in champions_pokemon.json
    "Abomnasnow":          ["abomasnow"],
    "Manetric":            ["manectric"],
    "Hisuian Typhloshion": ["typhlosion-hisui"],
    # Typo in champions_megas.json
    "Mega Abomnasnow":     ["abomasnow-mega"],
    "Mega Manetric":       ["manectric-mega"],
    # Floette: stored with long form name, slug is just "floette"
    "Floette (Eternal Flower, by transfer only)": ["floette"],
    # Lycanroc: pokebase uses one slug per forme; also map to bare "lycanroc"
    "Lycanroc (Midday Form)":   ["lycanroc-midday", "lycanroc"],
    "Lycanroc (Midnight Form)": ["lycanroc-midnight"],
    "Lycanroc (Dusk Form)":     ["lycanroc-dusk"],
}


def build_slug_map(pokemon_file: Path, megas_file: Path) -> dict[str, str]:
    """Map pokebase slug -> canonical display name. First match wins on conflict."""
    slug_map: dict[str, str] = {}

    def _add(name: str) -> None:
        slugs = _MANUAL.get(name) or name_to_slugs(name)
        for slug in slugs:
            if slug not in slug_map:
                slug_map[slug] = name

    with open(pokemon_file) as f:
        for entry in json.load(f)["pokemon"]:
            _add(entry["name"])

    with open(megas_file) as f:
        for entry in json.load(f)["megas"]:
            _add(entry["mega"])

    return slug_map


def parse_list_page(html: str) -> list[tuple[str, float]]:
    """Return (slug, usage_pct) pairs from a list page."""
    # Usage% is in <span class="text-xs tabular-nums"> which is the first such span
    # after each Pokemon link — stat spans use just "tabular-nums" (no "text-xs")
    pattern = (
        r'href="/pokemon-champions/pokemon/([^"]+)"'
        r'.*?<span class="text-xs tabular-nums">([\d.]+)<!-- -->%'
    )
    return [(slug, float(pct)) for slug, pct in re.findall(pattern, html, re.DOTALL)]


def parse_section(html: str, section_name: str) -> list[dict]:
    """Extract [{name, usage_pct}] from a named <ul aria-label="..."> section."""
    m = re.search(
        rf'aria-label="{re.escape(section_name)}">(.*?)</ul>',
        html, re.DOTALL
    )
    if not m:
        return []
    section_html = m.group(1)

    items = []
    for item_m in re.finditer(
        r'font-medium text-zinc-800[^>]*>([^<]+)</span>'
        r'.*?tabular-nums[^>]*>([\d.]+)<!-- -->%',
        section_html, re.DOTALL
    ):
        items.append({"name": item_m.group(1).strip(), "usage_pct": float(item_m.group(2))})
    return items


def main():
    print("Loading allowed Pokemon lists...")
    slug_map = build_slug_map(POKEMON_FILE, MEGAS_FILE)
    print(f"  {len(slug_map)} candidate slugs from allowed lists")

    # Phase 1: Collect slugs + usage% from all 3 list pages
    print("\nFetching list pages...")
    all_entries: list[tuple[str, float]] = []
    seen: set[str] = set()

    for page in range(1, 4):
        url = f"{BASE_URL}?page={page}"
        print(f"  Page {page}: {url}")
        html = fetch(url)
        pairs = parse_list_page(html)
        print(f"    Found {len(pairs)} entries")
        for slug, pct in pairs:
            if slug not in seen:
                seen.add(slug)
                all_entries.append((slug, pct))
        time.sleep(DELAY)

    print(f"\nTotal unique slugs: {len(all_entries)}")

    allowed = [(s, p) for s, p in all_entries if s in slug_map]
    skipped = [s for s, _ in all_entries if s not in slug_map]
    print(f"In our allowed list: {len(allowed)}")
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(sorted(skipped))}")

    # Phase 2: Fetch detail pages
    print(f"\nFetching {len(allowed)} detail pages...")
    results = []

    for rank, (slug, usage_pct) in enumerate(allowed, start=1):
        name = slug_map[slug]
        url = f"{BASE_URL}/{slug}"
        print(f"  [{rank:3d}/{len(allowed)}] {name}")

        try:
            detail_html = fetch(url)
            abilities = parse_section(detail_html, "Abilities")
            items = parse_section(detail_html, "Items")
            moves = parse_section(detail_html, "Moves")
            print(f"           abilities={len(abilities)}, items={len(items)}, moves={len(moves)}")
        except Exception as exc:
            print(f"           ERROR: {exc}", file=sys.stderr)
            abilities, items, moves = [], [], []

        results.append({
            "name": name,
            "slug": slug,
            "rank": rank,
            "usage_pct": usage_pct,
            "moves": moves,
            "items": items,
            "abilities": abilities,
        })
        time.sleep(DELAY)

    output = {
        "format": "gen9championsvgc2026regma",
        "source": BASE_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "pokemon": results,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(results)} Pokemon to {OUT_FILE}")


if __name__ == "__main__":
    main()
