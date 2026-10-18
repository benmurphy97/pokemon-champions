# Plan and rules for the learned player

Agreed 2026-10-18 after the joint-choice change forced a retrain. Written so we stop redoing runs.

## What is frozen before the next full training run

Changing the network's inputs or outputs forces a retrain from scratch, so these go in together, once:

1. **Joint choice for both active Pokémon** (done): the second slot is scored knowing the first slot's move, target or switch.
2. **Team preview** (to build): learn which four of six to bring and which two lead. Copy players first (replays show who was brought and who led), then refine with reinforcement learning. The player sees both teams' six species and nothing more, as in a real match.
3. **Bigger network**: 192 wide x 4 layers (2.5M parameters, 2.5x), measured at about 7 minutes per epoch, so about 1.2 hours for a 10-epoch imitation run. The 256 x 4 option (4.2M) is about 1.7 hours. Per-decision latency is about 5-6 ms at every size, since the damage features dominate.
4. **Search at play time** (no training change): sample plausible hidden opponent sets, simulate ahead with the learned policy choosing moves, pick the best-scoring plan. Measured with and without; not used to make training data for now.

## Data

Imitation on every public ladder game rated 1300+ (13,900 games), weighted by rating so stronger players count more. No filtering of games.

## Update 2026-10-18: general player trained by self-play

The imitation rebuild gave no gain and reinforcement learning on 10 teams only tunes to those teams, so the direction is a general player trained against itself:

- **Teams of every kind** (Trick Room included): both sides draw from the pool of real tournament and ladder teams. The 10 known-good teams become a sub-score.
- **A league, not pure self-play:** latest self 40%, past snapshots 30%, greedy-blind 20%, the imitation start model 10%. The fixed opponents stop it drifting into a private style.
- **Judged against fixed yardsticks:** gate 1 (90% vs random) stays. Gate 2 is now at least 55% against greedy-blind on held-out teams of every kind (paired games, so 50% is parity). Trick Room and the known teams are reported as sub-scores.
- **Preview:** rule-based during self-play; the learned preview is retrained afterwards, starting from the rule-based choices (the imitation preview was 11 points worse).

Order of work now: 5 self-play league trainer (pilot, then full run), 6 search at play time, 7 learn preview, 8 team discovery.

## Update 2026-10-18 (later): train on 50 strong teams, faster games first

Run 6 played only about 6,000 games in 3 hours (about 14 model decisions a second across all workers), and
the league stayed flat. Learning from experience needs far more games, so speed comes before the next run.
The repeat-Protect block (a play-time rule) is now on by default: 45.8% to 48.8% against greedy-blind, level within noise.

Decided:

- **Speed first:** profile a game, then score all running games' decisions in one batch. Target about 10x more
  games per hour. Results must match before and after (same choices on the same seeds).
- **50 training teams with real spreads only:** from the good tournament teams with exact pastes, ranked by
  gauntlet rating, spread across styles (Trick Room, Tailwind, weather, Mega, balance). Limitless teams have
  estimated spreads and only fill gaps. 15 more strong teams are held out for testing.
- **The model flies all 50:** each game gives it a random team from the 50 and an opponent team from the 50.
- **Opponent pilot: greedy-blind only.** Simpler and faster. Watch for it learning greedy-blind's quirks: the
  held-out test and the gates are the check.
- **Updates after batches of games** (as now, about every 200 games); reward stays result plus health margin
  with the idle penalty.

Order of work now: 6 speed, 7 pick the 50 (+15 held out), 8 50-team run (pilot, then full), 9 search at play
time, 10 learn preview, 11 team discovery.

## Update 2026-10-18 (later still): teams built from in-game usage

Decided: the 50 training teams are built from in-game Ranked Battle usage (championsbattledata.com),
not picked from tournaments. teams/build.py drew 1,000 teams; all were rated against the fixed gauntlet
(greedy-blind on both sides, 120 games each), the best 300 re-rated (480 games each), and the top 65 kept,
with no Pokémon on more than 28 of them. Every 13th-ish team by rank goes to the test split, so test teams
are as strong as training teams. Result: data/teams/m-c/training_teams.json, 50 train (mean 62.4%, range
53.8-73.8%) and 15 test (mean 62.3%); 106 species, 23 with Trick Room, 23 with a weather setter. For
comparison: all built teams average 46.7%, all tournament teams 50.3%, the best 50 tournament teams about 67-70%.

Flow from here: imitation start -> self-play on the 50 -> evaluate as we go -> swap out teams that keep losing.

## Gates (original wording, superseded above for gate 2)

1. **Playable:** wins at least 90% against the random player (400 paired games, any pair of teams).
2. **Good:** wins at least 60% piloting the 10 known-good Trick Room teams against held-out strong tournament teams (1,200 paired games). For reference the rule-based greedy-blind pilot scored 57.0% on that test, so 60% is only just above it; read results against that baseline and the ±4 point noise.

Both are checked with `champions.sim.arena` (gate 1) and the held-out check in `train_rl.evaluate` with `--known` (gate 2).

## Process rules

- **Pilot before every full run:** a 5-minute run on a tenth of the data to catch bugs, then one full run.
- **One change at a time**, so a result can be attributed. No input or output changes in the middle of a run.
- **Budget:** up to about 2 hours per run. Longer runs only with a reason written down first.
- Keep each run's log and best model under a run name (`rl_log_runN.jsonl`, `rl_runN_best.pt`) and write the result into `docs/learned-agent.md` the same day.
- Results inside about ±4 points are noise: report them as level, not as wins or losses.
- Tests pass before any full run (`python -m unittest discover -s tests`).

## Order of work

1. Team-preview labels from replays: pilot extraction on a few hundred games, check they look right.
2. Network size: time one training step at the candidate sizes.
3. Preview head and loss added to the policy; tests.
4. Pilot imitation run (10% data), then the full imitation run. Check gate 1 and gate 2.
5. Short reinforcement learning pass only if gate 2 is not met.
6. Search at play time; measure with and without.
7. Team discovery with the finished player; add results to the site.
