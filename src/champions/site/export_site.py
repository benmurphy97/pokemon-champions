"""Write the JSON the website reads (web/public/data/).

    PYTHONPATH=src python3 -m champions.site.export_site

teams.json  - every distinct legal tournament team: sets, where it placed, strategy tags, and its
              simulated record against the gauntlet (rate_teams.py) if it has been rated.
strategies.json - per strategy tag: how many teams, average simulated win rate, the Pokémon and Trick Room setters it uses.
model.json  - how the learned player has tested (reinforcement-learning checks over time).
usage.json  - ladder usage per Pokémon: top moves, items, abilities and partners.
"""

import json
from collections import Counter
from pathlib import Path

from champions.config import FORMAT_ID, REGULATION
from champions.teams.tournament_pool import build_pool
from champions.teams.parse import parse_team
from champions.teams.rate import RATINGS

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "web" / "public" / "data"
USAGE = ROOT / "data" / "meta" / REGULATION.lower() / "usage.json"
ITEMS = ROOT / "data" / "reference" / "items.json"

WEATHER_ABILITIES = {"Drought": "Sun", "Drizzle": "Rain", "Sand Stream": "Sand", "Snow Warning": "Snow"}
TERRAIN_ABILITIES = {"Electric Surge": "Electric Terrain", "Grassy Surge": "Grassy Terrain",
                     "Misty Surge": "Misty Terrain", "Psychic Surge": "Psychic Terrain"}
SUPPORT_MOVES = ["Trick Room", "Tailwind", "Fake Out", "Helping Hand", "Follow Me", "Rage Powder"]


def strategy_tags(mons: list[dict], megas: set[str]) -> list[str]:
    """Plain-language labels for how the team plays, read off its moves, abilities and items."""
    moves = {m for mon in mons for m in mon["moves"]}
    tags = [move for move in ("Trick Room", "Tailwind") if move in moves]
    for mon in mons:
        for table in (WEATHER_ABILITIES, TERRAIN_ABILITIES):
            label = table.get(mon.get("ability"))
            if label and label not in tags:
                tags.append(label)
    if any(mon["item"] in megas for mon in mons):
        tags.append("Mega")
    return tags


def teams_json() -> list[dict]:
    megas = {name for name, item in json.loads(ITEMS.read_text()).items() if item.get("megaStone")}
    ratings = json.loads(RATINGS.read_text())["teams"] if RATINGS.exists() else {}
    teams = []
    for team in build_pool():
        mons = parse_team(team["text"])
        record = ratings.get(team["id"])
        teams.append({"id": team["id"].removeprefix("web-"), "rating": record and {"wins": record["wins"], "games": record["games"]},
                      "tournament": team["tournament"], "date": team["date"],
                      "placement": team["placement"], "good": team["good"], "trickRoom": team["trick_room"],
                      "tags": strategy_tags(mons, megas),
                      "mons": [{"name": m["name"], "item": m["item"], "ability": m.get("ability"),
                                "nature": m.get("nature"), "points": {k: v for k, v in m["evs"].items() if v},
                                "moves": m["moves"]} for m in mons]})
    return sorted(teams, key=lambda t: (t["date"], t["id"]), reverse=True)


def strategies_json(teams: list[dict]) -> list[dict]:
    out = []
    for tag in sorted({t for team in teams for t in team["tags"]}):
        group = [team for team in teams if tag in team["tags"]]
        rated = [team["rating"] for team in group if team["rating"]]
        count = lambda items: [[n, round(c / len(group), 3)] for n, c in Counter(items).most_common(8)]
        entry = {"tag": tag, "teams": len(group), "top8": sum(team["good"] for team in group),
                 "simulatedWinRate": round(sum(r["wins"] for r in rated) / max(sum(r["games"] for r in rated), 1), 3),
                 "pokemon": count(m["name"] for team in group for m in team["mons"])}
        if tag == "Trick Room":
            entry["setters"] = count(m["name"] for team in group for m in team["mons"] if "Trick Room" in m["moves"])
        out.append(entry)
    return sorted(out, key=lambda e: -e["teams"])


def model_json() -> list[dict]:
    """Held-out checks of the learned player on Trick Room teams, from the latest reinforcement-learning log."""
    log = ROOT / "results" / "rl_log.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    return [{"iteration": r["iteration"], "policy": r["eval"]["policy"], "baseline": r["eval"]["baseline"],
             "interval": r["eval"]["interval"]} for r in rows if "eval" in r]


def usage_json() -> dict:
    data = json.loads(USAGE.read_text())
    top = lambda pairs, n: [[name, round(share, 3)] for name, share in pairs[:n]]
    pokemon = [{"name": name, "usage": entry["usage"], "moves": top(entry["moves"], 6), "items": top(entry["items"], 3),
                "abilities": top(entry["abilities"], 2), "partners": top(entry["teammates"], 5)}
               for name, entry in data["pokemon"].items()]
    return {"format": data["format"], "month": data["month"], "battles": data["battles"],
            "pokemon": sorted(pokemon, key=lambda p: -p["usage"])}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    teams, usage = teams_json(), usage_json()
    meta = {"regulation": REGULATION, "format": FORMAT_ID, "teams": len(teams)}
    (OUT_DIR / "teams.json").write_text(json.dumps(teams, separators=(",", ":")))
    (OUT_DIR / "usage.json").write_text(json.dumps(usage, separators=(",", ":")))
    (OUT_DIR / "strategies.json").write_text(json.dumps(strategies_json(teams), separators=(",", ":")))
    (OUT_DIR / "model.json").write_text(json.dumps(model_json()))
    (OUT_DIR / "meta.json").write_text(json.dumps(meta))
    print(f"{len(teams)} teams, {len(usage['pokemon'])} Pokémon → {OUT_DIR}")


if __name__ == "__main__":
    main()
