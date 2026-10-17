# The learned agent

A neural policy trained to imitate human players on the M-C ladder, then measured in the simulator against the greedy rule-based agent (kept only as a yardstick).

## Pipeline

```
Showdown replays ──import_replays──► cache/raw/*.json.gz        full logs, local only
                                         │
                         decisions.extract (uses battle_state.BattleState + encoding.encode)
                                         ▼
                              (what the player could see, what they did)   ~30 per game
                                         │
                                    train_il ──► models/il_policy.pt
                                         │
              arena ◄── agent.NeuralAgent ◄─┘      plays in the simulator via sim/arena_worker.js
```

## Information model (closed team sheets)

- **Your own six: everything.** Moves, item, ability, stats. In live play these come from the simulator's request. Replays do not contain them, so in training each player's sets are reconstructed from the whole match (`hindsight.py`): moves used, items shown or consumed, abilities triggered, the Mega Stone, with unrevealed slots filled from the most common ladder choice and the stats computed from the most common ladder spread (`stats.py`, checked against the engine).
- **The opponent: their six species** from team preview (so types and base stats), then only what is revealed: HP percentage, status, boosts, moves/items/abilities once shown. Not their stats or sets, and not which four they bring.
- The simulator's `Open Team Sheets` rule only adds an opt-in prompt and reveals nothing by default, so the in-game information is already closed. Human ladder players may have opted in, which the logs do not show.
- The greedy yardstick comes in two forms: `greedy` reads the engine and sees everything, `greedy-blind` plays under the same closed rules (opponent stats guessed from base stats, opponent moves assumed to be strong attacks of their types). Use `greedy-blind` for fair comparisons.

## What the policy sees and does

- **Input:** both teams (species, types, base stats), HP, status, stat boosts, Mega state, weather, terrain, Trick Room, Tailwind and similar side conditions, the turn number; your own full sets and real stats; the opponent's revealed moves/items/abilities. Plus per-move damage features against each foe and a speed comparison.
- **Same code for training and play.** `BattleState` reads Showdown protocol lines. Replay logs and the live simulator stream use the same format, so the policy sees identical inputs in both.
- **Output:** a score for every move (a learned embedding per move, plus its type/power/accuracy) and for switching to each of the six own Pokémon, restricted at play time to the actions the simulator says are legal. A second head picks the target for single-target moves.
- **Decisions are made one Pokémon at a time.** The two slots are not modelled jointly, and the second slot only avoids switching to the same Pokémon as the first.
- **Not yet learned:** team preview (bring 4, lead 2). In the arena every agent uses the greedy agent's preview, so differences in win rate come from in-battle play. Mega Evolution is always used when available.

## What cannot be recovered from a log

A Pokémon that flinched, slept or was fully paralysed shows no move, so that choice is skipped. Moves triggered by other effects (Sleep Talk, Dancer, a locked-in Outrage) were not chosen that turn. Pivot switches (U-turn) and replacements after a faint are recorded as separate "replacement" decisions.

## Results so far

Trained on 1,486 ladder games (41,000 decisions, rating 1300+), 8 epochs, about 1 minute on CPU. Win rates are from the arena over 600 paired games.

| Model | Top-1 accuracy | Top-3 | vs random | vs greedy |
|---|---|---|---|---|
| Most-common-action baseline | 15.0% | | | |
| 300 games | 26.0% | 52% | ~50% | 11% |
| 1,486 games | 37.0% | 73% | 51.5% (47.5–55.5) | 14.8% (12.2–17.9) |
| 1,486 games + damage features | 39.5% | 74% | 63.2% (59.2–66.9) | 18.5% (15.6–21.8) |

- **Accuracy at copying humans is not the same as playing well.** The policy copies the ladder's mix (about 52% attacks, 36% status and Protect, 4% switches) but picks attacks worse than the greedy agent, which beats random 95%+.
- **Damage features** (type effectiveness, same-type bonus, attack/defence ratio of each move against each foe) are inputs, not rules: the network still learns what to do with them. They gave the largest single gain.
### Later runs (2,900–3,200 games)

| Change | Games | Top-1 | vs random | vs greedy (full info) | vs greedy-blind |
|---|---|---|---|---|---|
| Know own moves/items/abilities (hindsight); training moves restricted to the real four | 2,894 | 49.0% | 76.2% | 26.5% (23.1–30.2) | |
| + own real stats and speed comparison | 3,220 | 49.8% | 76.2% (72.6–79.4) | 25.7% (22.3–29.3) | 28.3% (24.9–32.1) |

- Top-1 accuracy is not comparable with the first table: the candidate moves are now the player's real four, not every move the species could learn.
- Knowing its own sets was the biggest step, but it is mixed up with doubling the data, so the two effects are not separated.
- Own stats made no measurable difference yet. In training they are the typical ladder spread, in play the real ones.
- The blind yardstick is only slightly weaker than the all-seeing one, so the information gap is not what separates the agents; play quality is.
- The 3–4 invalid choices per 600 games (down from dozens) have not been traced.

- **Imitation alone is unlikely to beat greedy at this data size.** More data helps gradually; reinforcement learning in the simulator, using the win/loss result, is the route to going beyond what the replays show.

## Reinforcement learning (Trick Room teams)

PPO in `train_rl.py`, starting from the imitation policy, with a value estimate, a penalty for drifting from the starting policy, and the dense detail in the module docstring. Our teams each carry exactly two Trick Room setters (`meta.py --trick-room-setters 2`); the opponents are real ladder teams of every kind plus tournament teams from MetaVGC.

Test: our Trick Room teams against held-out real teams, most-likely-move play, 400 to 1,200 games. The baseline is greedy-blind piloting the same teams against the same opponents.

| Run | What changed | Best policy | Baseline | Result |
|---|---|---|---|---|
| Imitation only (6.7k games) | | 31.0% | 37.5% | below baseline |
| RL run 1 (60 iterations, 256 games each) | ladder opponents, generated Trick Room teams | 39.5% (iterations 20 and 35), ended 33.5% | 37.5% | roughly level at best, unstable; interval about ±7 |
| RL run 2 (80 iterations) | + tournament teams as opponents and as our teams, KL penalty 0.1, lr 5e-5, 1,200-game checks | 42–46% throughout | 48.2% | flat, about 4 points under the baseline |

- **Training-game win rate against greedy-blind rose slowly**: 38.8% to 41.6% over run 2. Held-out results did not follow.
- **By team type (run 2):** on good tournament Trick Room teams the policy and the baseline are level (about 65% each, 8 teams, noisy). On our generated Trick Room teams the policy is consistently below the baseline (19–26% vs 30%).
- **Team quality dominates.** Good tournament Trick Room teams win about 66% even with the rule-based pilot; our generated ones win 30%.
- **Not yet tried:** more precise damage inputs (the greedy agent's edge is likely an accurate damage calculation), a denser reward than win/loss, self-play against earlier versions, a larger network.

## Running it

```bash
PYTHONPATH=src python3 -u -m champions.import_replays --days 21 --step-hours 0.5 --min-rating 1300 --max-games 15000 --delay 0.6
PYTHONPATH=src .venv/bin/python -m champions.train_il --epochs 8 --rebuild
PYTHONPATH=src .venv/bin/python -m champions.arena --vs greedy --games 400
```

The arena plays every sampled pair of teams twice, so each agent pilots both teams; team strength cancels out. It stops with an error if any game is lost, rather than reporting a smaller sample.

## Bugs found while building (worth remembering)

- A request for a Pokémon locked into a move (Recharge, Struggle) has no `target` field. Treat any request with no usable move as "move 1".
- When the engine says "[Unavailable choice] ... trapped" it re-sends a corrected request. Answering both the error and the new request leaves every later answer one request behind, and produces a cascade of "invalid choice" errors.
- Replay HP strings carry a colour letter (`45/100y`), and some form names differ between team preview and battle (Floette-Eternal/Floette, Mimikyu-Busted).
