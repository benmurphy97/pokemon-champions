# Phase 0 notes — why the first version was retired

Snapshot: git tag `archive/rl-v1` (2026-05-29 state, 8,765 lines of Python).

## Problems with v1

- **No results.** No checkpoints, win rates or evolution logs were ever recorded.
- **Weak fitness signal.** A team was judged on 10 battles per opponent against 8 hand-pasted teams, accepted on any improvement. Noise is about ±5–6 points, so accept/reject was close to a random walk.
- **Biased judge.** The PPO policy was trained on one team and then used to rate other teams.
- **Slow simulator path.** Every decision went through a Showdown websocket.
- **Dead weight.** About 35% of the code was unreachable from the entry point: the GA loop (stub), the selection model (never wired in), analytics (not wired to fitness), two empty training files.
- **Broken infrastructure.** `docker/showdown` was excluded by `.gitignore` and never committed.

## Kept from v1

- Showdown team parsing/serialising (now [src/champions/teams.py](../src/champions/teams.py), with tests).
- Pokémon, move and item reference data (M-A era, to be refreshed).
- Two sample teams as test fixtures.

## Recoverable from the tag, not carried over

Scraper (`stage1_scraper`), selection model (`stage2_selection`), 4,104-dim encoder (`state_encoder`), PPO/league code, DEAP GA, M-A usage data, the eight M-A opponent teams, and the old design docs. Retrieve with `git show archive/rl-v1:<path>`.

## Source links

`docs/team_refs.md` keeps the community team-source links from v1.
