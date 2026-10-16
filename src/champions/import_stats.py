"""Download Smogon's monthly ladder stats for the current format and keep a compact summary.

    PYTHONPATH=src python3 -m champions.import_stats                 # latest month, 1760 cutoff
    PYTHONPATH=src python3 -m champions.import_stats --month 2026-09 --cutoff 1630

Writes data/meta/<regulation>/usage.json: per Pokémon, its usage and the share of its users
choosing each ability, item, move, stat-point spread and teammate. The 16 MB raw download is
cached under cache/ (gitignored).
"""

import argparse
import json
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from champions.config import FORMAT_ID, REGULATION

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache"
OUT_DIR = ROOT / "data" / "meta" / REGULATION.lower()
USER_AGENT = "champions-team-research/0.1 (personal project)"

MIN_USAGE = 0.01    # drop Pokémon used by under 1% of teams
KEEP = {"abilities": 5, "items": 8, "moves": 14, "spreads": 5, "teammates": 14}
STAT_ORDER = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]


def download(month: str, cutoff: int) -> Path:
    """Fetch the chaos JSON for one month (cached), returning the local path."""
    path = CACHE_DIR / f"{FORMAT_ID}-{month}-{cutoff}.json"
    if path.exists():
        return path
    url = f"https://www.smogon.com/stats/{month}/chaos/{FORMAT_ID}-{cutoff}.json"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_bytes(data)
    return path


def latest_month(cutoff: int) -> str:
    """The most recent month (searching back up to a year) that has stats for this format."""
    year, month = date.today().year, date.today().month
    for _ in range(12):
        label = f"{year}-{month:02d}"
        try:
            download(label, cutoff)
            return label
        except urllib.error.HTTPError:
            month -= 1
            if month == 0:
                year, month = year - 1, 12
    raise SystemExit(f"no stats found for {FORMAT_ID} in the last 12 months")


def display_names() -> dict[str, str]:
    """Map Showdown IDs (e.g. 'grassysurge') to display names ('Grassy Surge') using data/reference."""
    ref = ROOT / "data" / "reference"
    names = list(json.loads((ref / "moves.json").read_text())) + list(json.loads((ref / "items.json").read_text()))
    for mon in json.loads((ref / "pokemon.json").read_text()).values():
        names += mon["abilities"]
    return {"".join(c for c in n.lower() if c.isalnum()): n for n in names}


NAMES = display_names()


def top(counts: dict, total: float, n: int, rename: bool = True) -> list:
    """The n largest entries as [name, share of total], dropping empty and 'Nothing' keys."""
    ranked = sorted(((k, v) for k, v in counts.items() if k and k.lower() != "nothing"), key=lambda kv: -kv[1])
    return [[NAMES.get(k, k) if rename else k, round(v / total, 4)] for k, v in ranked[:n]]


def summarise_pokemon(entry: dict) -> dict:
    weight = sum(entry["Abilities"].values())  # this Pokémon's total weighted appearances
    spreads = []
    for key, value in sorted(entry["Spreads"].items(), key=lambda kv: -kv[1])[:KEEP["spreads"]]:
        nature, points = key.split(":")
        spreads.append([nature, dict(zip(STAT_ORDER, map(int, points.split("/")))), round(value / weight, 4)])
    return {
        "abilities": top(entry["Abilities"], weight, KEEP["abilities"]),
        "items": top(entry["Items"], weight, KEEP["items"]),
        "moves": top(entry["Moves"], weight, KEEP["moves"]),  # share of users, so these sum to ~4
        "spreads": spreads,
        "teammates": top(entry["Teammates"], weight, KEEP["teammates"], rename=False),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--month", help="YYYY-MM (default: latest available)")
    parser.add_argument("--cutoff", type=int, default=1760, choices=[0, 1500, 1630, 1760])
    args = parser.parse_args()

    month = args.month or latest_month(args.cutoff)
    raw = json.loads(download(month, args.cutoff).read_text())

    # 'usage' is the share of teams containing the Pokémon, so across all Pokémon it sums to about 6.
    pokemon = {}
    for name, entry in sorted(raw["data"].items(), key=lambda kv: -kv[1]["usage"]):
        if entry["usage"] >= MIN_USAGE and entry["Abilities"]:
            pokemon[name] = {"usage": round(entry["usage"], 4), **summarise_pokemon(entry)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "usage.json"
    out.write_text(json.dumps({
        "format": FORMAT_ID, "month": month, "cutoff": args.cutoff,
        "battles": raw["info"].get("number of battles"), "pokemon": pokemon,
    }, indent=1) + "\n")
    print(f"{len(pokemon)} Pokémon (usage ≥ {MIN_USAGE:.0%}) from {month}, cutoff {args.cutoff} → {out}")


if __name__ == "__main__":
    main()
