import json
import os
import re
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from stage1_scraper.models import PokemonUsage
from stage1_scraper.session import make_session, polite_get

log = logging.getLogger(__name__)

BASE = "https://www.pikalytics.com"
FORMAT = "gen9championsvgc2026regma"

_PCT = re.compile(r"(\d+\.?\d*)%")

_ROSTER_PATH = Path(__file__).parent.parent.parent / "data" / "pikalytics_roster.json"


def scrape_usage_list() -> list[tuple["PokemonUsage", str]]:
    """Load the full Pokemon roster from pikalytics_roster.json. Returns (PokemonUsage, slug) pairs."""
    roster_path = os.path.normpath(_ROSTER_PATH)
    try:
        with open(roster_path) as f:
            roster = json.load(f)
    except FileNotFoundError:
        log.error("pikalytics_roster.json not found at %s", roster_path)
        return []

    results = []
    for rank, entry in enumerate(roster["pokemon"], start=1):
        slug = entry["slug"]
        name = entry["name"]
        seed_usage = entry.get("seed_usage_pct", 0.0)
        results.append((PokemonUsage(name=name, usage_pct=seed_usage, rank=rank), slug))

    log.info("Loaded %d Pokemon from roster (%d with guessed slugs)",
             len(results), sum(1 for e in roster["pokemon"] if e.get("guess")))
    return results


def scrape_pokemon_detail(slug: str) -> dict:
    """Fetch move/item/ability/spread/teammate data for one Pokemon by its Pikalytics slug."""
    session = make_session()
    url = f"{BASE}/pokedex/{FORMAT}/{slug}?l=en"
    log.info("Fetching detail: %s", url)

    try:
        resp = polite_get(session, url)
    except Exception as e:
        log.warning("Failed detail %s: %s", slug, e)
        return {}

    soup = BeautifulSoup(resp.text, "lxml")

    usage_pct = None
    gold = soup.find("div", class_="pokemon-ind-summary-text gold-font")
    if gold:
        raw = gold.get_text(strip=True).replace("%", "").strip()
        try:
            usage_pct = float(raw)
        except ValueError:
            pass

    return {
        "usage_pct": usage_pct,
        "moves": _parse_section(soup, "moves_wrapper", name_class="pokedex-inline-text-offset"),
        "items": _parse_section(soup, "items_wrapper", name_class="pokedex-inline-text"),
        "abilities": _parse_section(soup, "abilities_wrapper", name_class="pokedex-inline-text-offset"),
        "spreads": _parse_spreads(soup),
        "teammates": _parse_teammates(soup),
    }


def _parse_section(soup: BeautifulSoup, wrapper_id: str, name_class: str) -> list:
    wrapper = soup.find("div", id=wrapper_id)
    if not wrapper:
        return []
    results = []
    for entry in wrapper.find_all("div", class_="pokedex-move-entry-new"):
        name_div = entry.find("div", class_=name_class)
        pct_div = entry.find("div", class_="pokedex-inline-right")
        if not name_div or not pct_div:
            continue
        entry_name = name_div.get_text(strip=True)
        m = _PCT.search(pct_div.get_text(strip=True))
        if entry_name and m:
            results.append({"name": entry_name, "usage_pct": float(m.group(1))})
    return results


def _parse_spreads(soup: BeautifulSoup) -> list:
    wrapper = soup.find("div", id="dex_spreads_wrapper")
    if not wrapper:
        return []
    results = []
    for entry in wrapper.find_all("div", class_="pokedex-move-entry-new"):
        nature_div = entry.find("div", class_="pokedex-inline-text-offset")
        stat_divs = entry.find_all("div", class_="pokedex-inline-text")
        pct_div = entry.find("div", class_="pokedex-inline-right")
        if not nature_div or len(stat_divs) < 6 or not pct_div:
            continue
        nature = nature_div.get_text(strip=True)
        stats = [s.get_text(strip=True).rstrip("/") for s in stat_divs[:6]]
        m = _PCT.search(pct_div.get_text(strip=True))
        if not m:
            continue
        try:
            results.append({
                "nature": nature,
                "hp": int(stats[0]), "atk": int(stats[1]), "def": int(stats[2]),
                "spa": int(stats[3]), "spd": int(stats[4]), "spe": int(stats[5]),
                "usage_pct": float(m.group(1)),
            })
        except (ValueError, IndexError):
            continue
    return results


def _parse_teammates(soup: BeautifulSoup) -> list:
    wrapper = soup.find("div", id="dex_team_wrapper")
    if not wrapper:
        return []
    results = []
    for a in wrapper.find_all("a", class_="teammate_entry"):
        name = a.get("data-name", "").strip()
        pct_div = a.find("div", class_="pokedex-inline-right")
        if not name or not pct_div:
            continue
        m = _PCT.search(pct_div.get_text(strip=True))
        if m:
            results.append({"name": name, "usage_pct": float(m.group(1))})
    return results
