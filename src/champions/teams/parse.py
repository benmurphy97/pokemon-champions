"""Read and write teams in Showdown's text format, and check they are legal.

A team is a list of dicts, one per Pokémon:
    {"name", "item", "ability", "nature", "evs": {stat: points}, "moves": [..]}
"""

import re

from champions.config import LEVEL, STAT_POINT_MAX_PER_STAT, STAT_POINT_TOTAL, STATS, TEAM_SIZE


def parse_team(text: str) -> list[dict]:
    """Parse a Showdown team export into a list of Pokémon dicts."""
    mons = []
    for block in (b.strip() for b in text.strip().split("\n\n")):
        if not block:
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]

        name, _, item = lines[0].partition(" @ ")
        mon = {"name": name.strip(), "item": item.strip() or None, "moves": []}
        points = {}

        for line in lines[1:]:
            if line.startswith("Ability:"):
                mon["ability"] = line[len("Ability:"):].strip()
            elif line.startswith("EVs:"):
                for part in line[len("EVs:"):].split("/"):
                    match = re.match(r"\s*(\d+)\s+(\w+)", part)
                    if match:
                        points[match.group(2)] = int(match.group(1))
            elif line.endswith("Nature"):
                mon["nature"] = line[: -len("Nature")].strip()
            elif line.startswith("- "):
                mon["moves"].append(line[2:].strip())

        mon["evs"] = {stat: points.get(stat, 0) for stat in STATS}
        mons.append(mon)
    return mons


def serialise_team(mons: list[dict]) -> str:
    """Write a list of Pokémon dicts back out as a Showdown team export."""
    blocks = []
    for mon in mons:
        lines = [f"{mon['name']} @ {mon['item']}" if mon.get("item") else mon["name"]]
        if "ability" in mon:
            lines.append(f"Ability: {mon['ability']}")
        lines.append(f"Level: {LEVEL}")

        spent = [f"{points} {stat}" for stat, points in mon["evs"].items() if points > 0]
        if spent:
            lines.append(f"EVs: {' / '.join(spent)}")
        if "nature" in mon:
            lines.append(f"{mon['nature']} Nature")
        lines.extend(f"- {move}" for move in mon["moves"])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def check_team(mons: list[dict]) -> list[str]:
    """Return a list of rule violations (empty means the team is legal).

    Species, move and item legality for the regulation is checked against
    data/reference in a later phase; this covers the structural rules.
    """
    problems = []
    if len(mons) != TEAM_SIZE:
        problems.append(f"team has {len(mons)} Pokémon, expected {TEAM_SIZE}")

    items = [m["item"] for m in mons if m.get("item")]
    for item in sorted({i for i in items if items.count(i) > 1}):
        problems.append(f"item clause: {item} used more than once")

    for mon in mons:
        evs = mon["evs"]
        if sum(evs.values()) > STAT_POINT_TOTAL:
            problems.append(f"{mon['name']}: {sum(evs.values())} stat points, max {STAT_POINT_TOTAL}")
        if any(v > STAT_POINT_MAX_PER_STAT for v in evs.values()):
            problems.append(f"{mon['name']}: over {STAT_POINT_MAX_PER_STAT} points in one stat")
        if len(mon["moves"]) > 4:
            problems.append(f"{mon['name']}: more than 4 moves")
    return problems
