# Calibration log

How far can we trust the simulator's idea of team strength? Each entry records a check and what it showed.

## 2026-10-16 — usage vs simulated win rate (greedy agent)

- Pool: 60 teams generated from September M-C ladder stats (`champions.teams.generate`, seed 7). Round robin, 20 games per pair, 35,400 games, greedy agent on both sides.
- Spearman correlation between a Pokémon's ladder usage and the average simulated win rate of teams containing it: **-0.16**, over only 14 Pokémon that appeared in 8+ teams.
- Rillaboom, Sneasler and Salamence-Mega (the three most-used) came out near the bottom.
- **Reading:** inconclusive, with a warning. 14 points is too few for a correlation of this size to mean much, and the pool is built from usage, so species are tangled up with their teammates. But the most popular Pokémon not looking strong is what a weak agent would produce. The greedy agent ignores most ability effects (Aerilate, Grassy Surge, Tough Claws and similar), has no lookahead, and never predicts Protect.
- Team spread was wide: best 72%, median 53%, worst 24%, so the simulator does separate teams. Whether it separates them *correctly* is the open question.

## 2026-10-17 — does the simulator predict real ladder games?

- Data: 1,294 public M-C ladder replays (rating ≥ 1300, sampled across 7 days). 1,029 had all 12 species in the ladder data and were rebuilt with the most common set for each Pokémon. Each matchup was simulated 20 times (seats alternating) and compared with the real winner. Method: `champions.replay_check`.
- Result (AUC, 0.5 = coin flip, 95% bootstrap interval):

| Predictor | AUC |
|---|---|
| Greedy agent simulation | 0.526 (0.493–0.562) |
| Random agent simulation | 0.528 (0.489–0.564) |
| Players' rating difference | 0.535 |

- **Reading:** no detectable signal. The greedy agent is no better than random agents, and both intervals include 0.5. The test also has a low ceiling: rating difference, a real skill signal, only reaches 0.535, so ladder outcomes in this rating band are mostly noise (skill, play, hidden sets). A simulator would need to be much better than anything here to show up clearly with ~1,000 games.
- **Overconfidence:** greedy play is nearly deterministic, so 74% of matchups came out lopsided (simulated win probability under 30% or over 70%); real games in those buckets went 48.9% and 53.9% for player 1. Direction is right, size is far too small (and not significant).

| Sim says player 1 wins | Games | Real player 1 win rate |
|---|---|---|
| under 30% | 374 | 48.9% |
| 30–45% | 90 | 45.6% |
| 45–55% | 67 | 47.8% |
| 55–70% | 73 | 56.2% |
| over 70% | 425 | 53.9% |

- **Consequence:** simulated win rate is not yet validated as a measure of team quality. Treat any team ranking it produces as a hypothesis.

## Next checks

1. ~~Real results~~ done above (win/loss prediction). Still to do: win rates by species, lead pair and core from a much larger replay set.
2. Behaviour check: how often does the agent pick the same bring and lead as human players in the same matchup?
3. Improve the agent (ability and item damage effects, Protect prediction, one-turn lookahead) and repeat check 1 to see whether the correlation moves.
