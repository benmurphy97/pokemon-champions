# Calibration log

How far can we trust the simulator's idea of team strength? Each entry records a check and what it showed.

## 2026-10-16 — usage vs simulated win rate (greedy agent)

- Pool: 60 teams generated from September M-C ladder stats (`champions.meta`, seed 7). Round robin, 20 games per pair, 35,400 games, greedy agent on both sides.
- Spearman correlation between a Pokémon's ladder usage and the average simulated win rate of teams containing it: **-0.16**, over only 14 Pokémon that appeared in 8+ teams.
- Rillaboom, Sneasler and Salamence-Mega (the three most-used) came out near the bottom.
- **Reading:** inconclusive, with a warning. 14 points is too few for a correlation of this size to mean much, and the pool is built from usage, so species are tangled up with their teammates. But the most popular Pokémon not looking strong is what a weak agent would produce. The greedy agent ignores most ability effects (Aerilate, Grassy Surge, Tough Claws and similar), has no lookahead, and never predicts Protect.
- Team spread was wide: best 72%, median 53%, worst 24%, so the simulator does separate teams. Whether it separates them *correctly* is the open question.

## Next checks

1. Real results: pull recent M-C replays and compute win rates by species, lead pair and bring. Compare with simulated rates.
2. Behaviour check: how often does the agent pick the same bring and lead as human players in the same matchup?
3. Improve the agent (ability and item damage effects, Protect prediction, one-turn lookahead) and repeat check 1 to see whether the correlation moves.
