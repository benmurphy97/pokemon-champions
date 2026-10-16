"""Compare simulated strength with real ladder popularity.

    PYTHONPATH=src python3 -m champions.calibrate results/tournament_greedy.json

For each Pokémon: the average win rate of simulated teams containing it, next to its ladder
usage. Popular Pokémon tend to be strong, so a simulator whose picture of strength is sane should
show a positive rank correlation. It is a sanity check, not proof: popularity also reflects
fashion, and the pool is itself generated from usage.
"""

import argparse
import json
from collections import defaultdict
from statistics import mean

from champions.meta import load_usage

MIN_TEAMS = 8  # ignore Pokémon seen in fewer simulated teams than this


def ranks(values: list[float]) -> list[float]:
    """Average ranks (1 = smallest), with ties sharing the mean rank."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return result


def spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def species_strength(rows: list[dict]) -> dict[str, tuple[float, int]]:
    """{species: (mean win rate of teams containing it, number of teams)}."""
    by_species = defaultdict(list)
    for row in rows:
        for species in row["species"]:
            by_species[species].append(row["winRate"])
    return {s: (mean(rates), len(rates)) for s, rates in by_species.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tournament")
    args = parser.parse_args()

    rows = json.load(open(args.tournament))["rows"]
    usage = load_usage()
    strength = {s: v for s, v in species_strength(rows).items() if v[1] >= MIN_TEAMS and s in usage}

    names = sorted(strength)
    rho = spearman([usage[n]["usage"] for n in names], [strength[n][0] for n in names])
    print(f"{len(rows)} teams, {len(names)} Pokémon seen in ≥{MIN_TEAMS} teams")
    print(f"Spearman correlation, ladder usage vs simulated win rate: {rho:+.2f}\n")

    ranked = sorted(names, key=lambda n: -strength[n][0])
    print(f"{'Pokémon':<18}{'sim win%':>9}{'teams':>7}{'ladder usage':>14}")
    for name in ranked[:8] + ["…"] + ranked[-8:]:
        if name == "…":
            print("  …")
            continue
        rate, count = strength[name]
        print(f"{name:<18}{rate:>8.1%}{count:>7}{usage[name]['usage']:>13.1%}")


if __name__ == "__main__":
    main()
