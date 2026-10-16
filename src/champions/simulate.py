"""Run battles on Showdown's engine via sim/run.js.

    PYTHONPATH=src python3 -m champions.simulate teamA.txt teamB.txt --battles 1000
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from champions.config import FORMAT_ID

RUNNER = Path(__file__).resolve().parents[2] / "sim" / "run.js"


class SimError(RuntimeError):
    pass


def _run(args: list[str]) -> dict:
    proc = subprocess.run(["node", str(RUNNER), *args], capture_output=True, text=True)
    if proc.returncode == 1:  # illegal team: runner prints the problems as JSON
        raise SimError(proc.stdout.strip())
    if proc.returncode != 0:
        raise SimError(proc.stderr.strip() or "simulator failed")
    return json.loads(proc.stdout)


def _with_team_files(team_a: str, team_b: str, fn):
    """Call fn(path_a, path_b) with the two team texts written to temporary files."""
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp, "a.txt"), Path(tmp, "b.txt")
        a.write_text(team_a)
        b.write_text(team_b)
        return fn(str(a), str(b))


def run_battles(team_a: str, team_b: str, battles: int = 100, seed: int = 1,
                format_id: str = FORMAT_ID) -> dict:
    """Play `battles` games between two Showdown team texts (random move choices).

    Returns {"winsA", "winsB", "ties", "meanTurns", "seconds", "battlesPerMinute", ...}.
    Raises SimError if either team is illegal in the format.
    """
    return _with_team_files(team_a, team_b, lambda a, b: _run(
        [a, b, "--battles", str(battles), "--seed", str(seed), "--format", format_id]))


def validate_team(team: str, format_id: str = FORMAT_ID) -> list[str]:
    """Return Showdown's list of rule violations for a team (empty means legal)."""
    try:
        _with_team_files(team, team, lambda a, b: _run([a, b, "--validate", "--format", format_id]))
    except SimError as err:
        return json.loads(str(err))["problems"]["A"]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    result = run_battles(Path(args.team_a).read_text(), Path(args.team_b).read_text(),
                         battles=args.battles, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
