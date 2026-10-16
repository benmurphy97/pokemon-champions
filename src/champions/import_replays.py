"""Download recent ladder replays for the current format and keep what matters about each game.

    PYTHONPATH=src python3 -m champions.import_replays --days 7 --min-rating 1300

Showdown's replay search returns the newest 51 games before a timestamp, which covers only ~17
minutes of play, so we sample pages spread evenly over the last N days rather than crawling
backwards from now. For each game we keep the six Pokémon on each side, the leads, which were
brought, Mega Evolutions, moves seen and the winner. Player names are not stored.

Output: cache/replays-<regulation>.jsonl (one game per line; re-runs skip games already saved).
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from champions.config import FORMAT_ID, REGULATION

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cache" / f"replays-{REGULATION.lower()}.jsonl"
SITE = "https://replay.pokemonshowdown.com"
USER_AGENT = "champions-team-research/0.1 (personal project; 1 request/second)"
DELAY_SECONDS = 1.0

MEGA_SUFFIX = re.compile(r"-Mega(-[XYZ])?$")


def fetch_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def base_species(details: str) -> str:
    """'Raichu-Mega-Y, L50, F, shiny' -> 'Raichu'."""
    return MEGA_SUFFIX.sub("", details.split(",")[0].strip())


def parse_replay(replay: dict) -> dict | None:
    """Reduce a replay (as served by <id>.json) to a compact record, or None if unfinished."""
    sides = {side: {"team": [], "lead": [], "brought": [], "mega": [], "moves": {}} for side in ("p1", "p2")}
    names, ratings, nick_to_species = {}, {}, {}
    winner_name, turns = None, 0

    for line in replay["log"].split("\n"):
        parts = line.split("|")
        if len(parts) < 3:
            continue
        kind = parts[1]
        if kind == "player" and len(parts) > 3 and parts[3]:  # the log ends with blank player lines
            names[parts[2]] = parts[3]
            ratings[parts[2]] = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else None
        elif kind == "poke":
            sides[parts[2]]["team"].append(base_species(parts[3]))
        elif kind in ("switch", "drag"):
            side, nickname, species = parts[2][:2], parts[2].split(": ", 1)[-1], base_species(parts[3])
            nick_to_species[(side, nickname)] = species
            if species not in sides[side]["brought"]:
                sides[side]["brought"].append(species)
            if turns == 0 and len(sides[side]["lead"]) < 2:
                sides[side]["lead"].append(species)
        elif kind == "turn":
            turns = int(parts[2])
        elif kind == "-mega":
            side, nickname = parts[2][:2], parts[2].split(": ", 1)[-1]
            sides[side]["mega"].append(nick_to_species.get((side, nickname), base_species(parts[3])))
        elif kind == "move" and len(parts) > 3:
            side, nickname = parts[2][:2], parts[2].split(": ", 1)[-1]
            species = nick_to_species.get((side, nickname), nickname)
            used = sides[side]["moves"].setdefault(species, [])
            if parts[3] not in used:
                used.append(parts[3])
        elif kind == "win":
            winner_name = parts[2]

    winner = next((side for side, name in names.items() if name == winner_name), None)
    if winner is None:
        return None
    for side, record in sides.items():
        record["rating"] = ratings.get(side)
    return {"id": replay["id"], "time": replay["uploadtime"], "rating": replay.get("rating"),
            "turns": turns, "winner": winner, **sides}


def list_games(days: int, step_hours: float, min_rating: int) -> list[dict]:
    """Search pages spread evenly over the last `days` days; keep public games above the rating floor."""
    now = int(time.time())
    games = {}
    t = now
    while t > now - days * 86400:
        url = f"{SITE}/search.json?format={FORMAT_ID}&before={t}"
        for game in fetch_json(url):
            if not game["private"] and (game["rating"] or 0) >= min_rating:
                games[game["id"]] = game
        time.sleep(DELAY_SECONDS)
        t -= int(step_hours * 3600)
    return sorted(games.values(), key=lambda g: -g["uploadtime"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--step-hours", type=float, default=2.0, help="gap between sampled search pages")
    parser.add_argument("--min-rating", type=int, default=1300)
    parser.add_argument("--max-games", type=int, default=1500)
    args = parser.parse_args()

    OUT.parent.mkdir(exist_ok=True)
    have = {json.loads(line)["id"] for line in OUT.read_text().splitlines()} if OUT.exists() else set()

    games = [g for g in list_games(args.days, args.step_hours, args.min_rating) if g["id"] not in have]
    games = games[: args.max_games]
    print(f"{len(games)} new games to download (rating ≥ {args.min_rating}, {len(have)} already saved)", flush=True)

    for i, game in enumerate(games, 1):
        try:
            record = parse_replay(fetch_json(f"{SITE}/{game['id']}.json"))
        except (OSError, ValueError, KeyError) as err:
            print(f"  skip {game['id']}: {err}", flush=True)
            continue
        finally:
            time.sleep(DELAY_SECONDS)
        if record:
            with OUT.open("a") as f:
                f.write(json.dumps(record) + "\n")
        if i % 100 == 0:
            print(f"  {i}/{len(games)}", flush=True)
    print(f"done → {OUT}")


if __name__ == "__main__":
    main()
