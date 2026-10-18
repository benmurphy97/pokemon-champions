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
### Run 3: damage calculator inputs, 13,900 games (best model: iteration 20)

New inputs (`damage.py`, checked against Showdown: median error 2%): damage as a share of each foe's health, sure/possible knockout flags, and incoming threat to our two active Pokémon. Imitation retrained on all 13,900 games: validation top-1 57.8% (was 53.1%), and 53.0% (48.1–57.8) against greedy-blind in the arena (was about 27%). Reinforcement learning from it (60 iterations, no reward shaping):

| Iteration | Learned player | Rule-based baseline |
|---|---|---|
| 10 / 20 / 30 / 40 / 50 / 60 | 46.5 / 49.8 / 47.5 / 48.5 / 43.5 / 47.7% | 49.3% |

On good tournament Trick Room teams it matches the baseline (64–71% vs 69%); on generated ones it is level (23–29% vs 30%). Reinforcement learning added nothing measurable over the imitation start; the gain came from the inputs and data. `--shaping` and `--gae-lambda` (health-lead reward) exist but have not been run.

- **Not yet tried (before run 3):** more precise damage inputs (the greedy agent's edge is likely an accurate damage calculation), a denser reward than win/loss, self-play against earlier versions, a larger network.

## Running it

```bash
PYTHONPATH=src python3 -u -m champions.scrapers.replays --days 21 --step-hours 0.5 --min-rating 1300 --max-games 15000 --delay 0.6
PYTHONPATH=src .venv/bin/python -m champions.training.imitation --epochs 8 --rebuild
PYTHONPATH=src .venv/bin/python -m champions.sim.arena --vs greedy --games 400
```

The arena plays every sampled pair of teams twice, so each agent pilots both teams; team strength cancels out. It stops with an error if any game is lost, rather than reporting a smaller sample.

## Bugs found while building (worth remembering)

- A request for a Pokémon locked into a move (Recharge, Struggle) has no `target` field. Treat any request with no usable move as "move 1".
- When the engine says "[Unavailable choice] ... trapped" it re-sends a corrected request. Answering both the error and the new request leaves every later answer one request behind, and produces a cascade of "invalid choice" errors.
- Replay HP strings carry a colour letter (`45/100y`), and some form names differ between team preview and battle (Floette-Eternal/Floette, Mimikyu-Busted).

### Run 4: health-lead reward (`--shaping 0.25 --gae-lambda 0.9`), same start as run 3

Held-out checks at iterations 10 to 60: 51.3 / 49.0 / 50.0 / 48.5 / 48.3 / 48.3% against the 49.3% baseline (interval about ±4). On good tournament Trick Room teams 70.7–72.3% (baseline 69%); on generated ones 25–31% (baseline 30%). Training-game win rate against greedy-blind stayed at 39–51%. The denser reward made no measurable difference: reinforcement learning from this start is at parity with the rule-based player however it is rewarded.

### Run 5: 10 known-good Trick Room teams against strong tournament teams (stopped at iteration 51)

Own teams: only the 10 known-good Trick Room teams (`known_teams.py`); opponents 80% the 317 best-rated tournament teams, 20% ladder. Held-out checks against held-out strong teams (baseline greedy-blind piloting the same teams, 57.0%): 51.8 / 50.7 / 47.8 / 59.2 / 57.8% at iterations 10 to 50 (interval about ±4). Mixed: ahead at iterations 40 and 50, behind at 20 and 30, so about level. Stopped early because the network inputs changed (joint two-Pokémon choice); this run used the older inputs.

### Step 4 (2026-10-18): joint choice, 192x4, team preview, rating-weighted imitation

Full imitation run on 13,900 games (log `results/il_v3.log`, model `il_policy_v3.pt`). Gates (`champions.evaluation.gates`):

| Model | Preview | vs random (gate 1: 90%) | Known 10 teams (gate 2: 60%; rule-based pilot 57.0%) |
|---|---|---|---|
| il_policy_v2 (128x3, no joint choice) | rule-based | 88.8% | 50.0% |
| il_policy_v3 (192x4, joint choice) | rule-based | 86.2% | 49.3% |
| il_policy_v3 | learned | 75.8% | 38.0% |
| rl_run5_best (older inputs, RL on the known teams) | rule-based | 85.8% | 59.2% |

- Joint choice, the bigger network and rating-weighted sampling gave no measurable gain at the imitation stage (49.3% vs 50.0%, level).
- The model overfits: validation loss was lowest at epoch 3 (1.11) and rose to 1.20 by epoch 10 while training loss fell to 0.90.
- The learned preview (copying players' picks, 46% bring-four and lead accuracy) plays about 11 points worse than the rule-based preview. Copying what players pick does not copy why; the rule-based preview scores matchups directly.
- Reinforcement learning on the known teams is what moved gate 2 (59.2%), by training on those teams.

### Run 6: self-play league, teams of every kind (2026-10-18)

30 iterations x 128 games from il_policy_v3; league latest self 40% / past snapshots 30% / greedy-blind 20% / imitation start 10%; reward half result, half health margin, minus 0.03 per idle turn, -0.5 for a game past 30 turns. Held-out checks (paired games, 600 vs greedy-blind):

| Iteration | win vs greedy-blind | health margin | win vs imitation start |
|---|---|---|---|
| 10 | 41.0% | -0.098 | 52% |
| 20 | 45.8% | -0.062 | 54% |
| 30 | 41.8% | -0.080 | 54% |

Gates on the best model (iteration 20), rule-based preview: vs random 82.2% (82.5% for the imitation start), general 43.5% (40.7-46.3, margin -0.078 +- 0.013; gate 55%), known Trick Room teams 50.3% (rule-based 57.0%). Neither gate met.

- Stalling guards worked: protect/heal share 12-14% (humans 11%), 6.3 turns per game, no timeouts. A first pilot hung because some games never ended (and idle workers waited forever when there were fewer games than workers x 4); both fixed.
- Self-play moved the model slightly past its imitation start (54% against it) but not towards the rule-based player, and it lost a few points against random. Four different training routes (imitation, RL on Trick Room teams, self-play) now all land between 41% and 50% on the held-out checks against greedy-blind.


### Step 8a (2026-10-18): joint action head, opponent head, 90-plan preview (il_joint_v4)

Encoding v3, one example per side per turn (243,398 decisions from 13,900 games; 27,722 previews). Warm start
from il_policy_v3, 6 epochs, rating-weighted; best validation loss at epoch 3 (1.824). Validation: exact pair
41.2%, per slot 60.4%, opponent top-1 42.1% / top-3 76.6%, preview exact plan (four fully seen) 24.0% (chance 1.1%),
value MSE 0.842. 0.2% of labels (moves outside a learnset, e.g. Struggle) dropped from the loss.

Gates (first time both met):

| Model | vs random (90%) | vs greedy-blind, held-out (55%) | margin |
|---|---|---|---|
| il_joint_v4, rule preview | 92.2% (89.2-94.5) | 57.2% (54.3-59.9) | +0.050 +/- 0.013 |
| il_joint_v4, learned preview | 88.5% | 51.4% | +0.005 |
| il_policy_v3 (sequential slots) | 86.2% | ~49-50% | |

The joint choice is the change that moved play: same data, same trunk. The learned preview is still about 6 points
worse than the rule-based one (was 11), so self-play keeps the rule-based preview.
