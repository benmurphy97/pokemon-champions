"""Download in-game Ranked Battle usage for doubles from Pokemon Champions Battle Data (championsbattledata.com).

    PYTHONPATH=src python3 -m champions.import_battledata

This is the in-game ladder, not Showdown, and it includes stat-point spreads. Per Pokémon it lists the
top moves, items, abilities, natures, stat-point spreads (each with its share) and teammates, plus
its usage rank. Natures and spreads are listed separately, not as pairs. Writes
data/meta/<regulation>/battle_data.json from the site's "Current" doubles files (the running season;
the season name is recorded). robots.txt allows crawling; requests are spaced out.
"""

import csv
import io
import json
import subprocess
import time
from pathlib import Path

from champions.config import REGULATION

SITE = "https://championsbattledata.com"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "meta" / REGULATION.lower() / "battle_data.json"
USER_AGENT = "champions-team-research/0.1 (personal project)"
STATS = {"hp_points": "HP", "attack_points": "Atk", "defense_points": "Def", "sp_atk_points": "SpA",
         "sp_def_points": "SpD", "speed_points": "Spe"}
LISTS = {"move": "moves", "held_item": "items", "ability": "abilities", "stat_alignment": "natures"}


def fetch(url: str) -> str:
    run = subprocess.run(["curl", "-s", "--fail", "--max-time", "60", "-A", USER_AGENT, url], capture_output=True, text=True)
    if run.returncode:
        raise OSError(f"curl failed ({run.returncode}) for {url}")
    return run.stdout


def share(text: str) -> float:
    return round(float(text.rstrip("%")) / 100, 4) if text else 0.0


def parse(text: str) -> dict | None:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return None
    entry = {"rank": int(rows[0]["column_position"]), "moves": [], "items": [], "abilities": [], "natures": [],
             "spreads": [], "teammates": []}
    for row in rows:
        kind = row["category"]
        if kind in LISTS:
            entry[LISTS[kind]].append([row["name"], share(row["percentage"])])
        elif kind == "stat_points":
            entry["spreads"].append([{stat: int(row[col] or 0) for col, stat in STATS.items()}, share(row["percentage"])])
        elif kind == "teammate":
            entry["teammates"].append(row["name"])
    return entry


def main():
    index = json.loads(fetch(f"{SITE}/data/pokemon-index.json"))
    season = next((s for s in index["seasons"] if s != "Current"), None)
    pokemon = {}
    for mon in index["pokemon"]:
        path = next((c["path"] for c in mon["battleDataCsvs"] if c["format"] == "Doubles" and not c.get("daily")), None)
        if not path:
            continue
        try:
            entry = parse(fetch(f"{SITE}/{path}"))
        except OSError:
            entry = None
        time.sleep(0.3)
        if entry:
            pokemon[mon["showdownName"]] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": SITE, "generatedAt": index["generatedAt"], "season": season, "format": "Doubles",
                               "pokemon": dict(sorted(pokemon.items(), key=lambda kv: kv[1]["rank"]))}, indent=0) + "\n")
    print(f"{len(pokemon)} Pokémon (season {season}, data of {index['generatedAt'][:10]}) → {OUT}")


if __name__ == "__main__":
    main()
