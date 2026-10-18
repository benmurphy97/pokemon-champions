"""Collect well-placed teams for the current regulation from LabMaus's top teams (labmaus.net/teams/top-teams).

    PYTHONPATH=src python3 -m champions.import_labmaus

LabMaus groups results from online tournaments by core and links each team's Showdown paste on
pokepast.es. Many of those pastes are open team sheets with no stat points or nature: those get an
estimated nature and spread (champions.spreads) and are marked "estimated_spreads". Keeps teams that finished in the top 8 or won
at least 70% of at least three rounds, adds them to data/teams/<regulation>/index.json beside the
MetaVGC and Limitless imports, and checks each against the format's rules. Player names are not kept.
Re-running skips pastes already imported. Requests are about one a second (robots.txt allows crawling).
"""

import argparse
import json
import re
import subprocess
import time
from urllib.parse import quote

from champions.config import REGULATION
from champions.import_teams import INDEX, OUT_DIR
from champions.spreads import estimate
from champions.teams import parse_team, serialise_team

API = "https://labmaus.net/api"
REGULATION_NAME = f"Regulation Set {REGULATION}"     # how LabMaus names it, e.g. "Regulation Set M-C"
USER_AGENT = "champions-team-research/0.1 (personal project; 1 request/second)"


def fetch(url: str) -> str:
    # The API answers only requests that say they come from its own site, as its pages do.
    run = subprocess.run(["curl", "-s", "--fail", "--max-time", "120", "-A", USER_AGENT, "-H", "Origin: https://labmaus.net",
                          "-H", "Referer: https://labmaus.net/", url], capture_output=True, text=True)
    if run.returncode:
        raise OSError(f"curl failed ({run.returncode}) for {url}")
    return run.stdout


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def placement(team: dict) -> str | None:
    """'3rd' for a top-8 finish, else the record ('6-2-0'); None if neither counts as good."""
    if team["placement"] <= 8:
        return ordinal(team["placement"])
    wins, losses, draws = map(int, team["record"].split("-"))
    rounds = wins + losses + draws
    return team["record"] if rounds >= 3 and wins / rounds >= 0.7 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-players", type=int, default=0, help="skip tournaments smaller than this")
    parser.add_argument("--limit", type=int, default=2000, help="max new teams to import this run")
    args = parser.parse_args()

    regulation = quote(REGULATION_NAME)
    dates = {t["id"]: t for t in json.loads(fetch(f"{API}/completed_tournaments?regulation={regulation}&language=en"))}
    teams = {}
    for group in json.loads(fetch(f"{API}/top_teams?regulation={regulation}&language=en")):
        for core in group["teams"]:
            for team in core["teams"]:
                teams.setdefault(team["team_url"], team)

    index = json.loads(INDEX.read_text()) if INDEX.exists() else []
    have = {entry.get("id") for entry in index}
    todo = []
    for url, team in teams.items():
        paste_id = url.rstrip("/").rsplit("/", 1)[-1]
        event = dates.get(team["tournament_id"], {})
        if ("pokepast.es" in url and paste_id not in have and placement(team)
                and event.get("num_players", 0) >= args.min_players):
            todo.append((paste_id, url, team, event))
    print(f"{len(teams)} teams listed; {len(todo)} new good ones to import", flush=True)

    fresh = []
    for paste_id, url, team, event in todo[: args.limit]:
        try:
            text = fetch(f"{url.rstrip('/')}/raw").replace("\r\n", "\n")
        except OSError as err:
            print(f"  skip {url}: {err}", flush=True)
            continue
        finally:
            time.sleep(0.8)
        if not re.search(r"^- ", text, flags=re.M):
            continue
        estimated = "EVs:" not in text
        if estimated:
            mons = parse_team(text)
            for mon in mons:
                mon["nature"], mon["evs"] = estimate(mon["name"], mon.get("nature"), mon["moves"])
            text = serialise_team(mons)
        (OUT_DIR / f"{paste_id}.txt").write_text(text)
        entry = {"id": paste_id, "source": f"labmaus:{team['tournament_id']}", "pokepaste": url,
                 "tournament": team["tournament_name"], "date": event.get("date"), "players": event.get("num_players"),
                 "placement": placement(team), "trick_room": "- Trick Room" in text, "estimated_spreads": estimated}
        index.append(entry); fresh.append(entry); have.add(paste_id)
        if len(fresh) % 50 == 0:
            INDEX.write_text(json.dumps(index, indent=1) + "\n")
            print(f"  {len(fresh)} imported", flush=True)

    from champions.simulate import validate_many
    for entry, issues in zip(fresh, validate_many([(OUT_DIR / f"{e['id']}.txt").read_text() for e in fresh])):
        entry["problems"] = issues
    INDEX.write_text(json.dumps(index, indent=1) + "\n")
    legal = [e for e in fresh if not e["problems"]]
    print(f"done: {len(fresh)} imported ({sum(e['estimated_spreads'] for e in fresh)} with estimated spreads), {len(legal)} legal, {sum(e['trick_room'] for e in legal)} legal with Trick Room → {INDEX}")


if __name__ == "__main__":
    main()
