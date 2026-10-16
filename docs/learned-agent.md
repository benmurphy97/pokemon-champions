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

## What the policy sees and does

- **Input:** public information only. Both teams (species, types, base stats), HP, status, stat boosts, revealed moves/items/abilities, Mega state, weather, terrain, Trick Room, Tailwind and similar side conditions, the turn number. It never sees its own unrevealed moves, because replays do not contain them.
- **Same code for training and play.** `BattleState` reads Showdown protocol lines. Replay logs and the live simulator stream use the same format, so the policy sees identical inputs in both.
- **Output:** a score for every move (a learned embedding per move, plus its type/power/accuracy) and for switching to each of the six own Pokémon, restricted at play time to the actions the simulator says are legal. A second head picks the target for single-target moves.
- **Decisions are made one Pokémon at a time.** The two slots are not modelled jointly, and the second slot only avoids switching to the same Pokémon as the first.
- **Not yet learned:** team preview (bring 4, lead 2). In the arena every agent uses the greedy agent's preview, so differences in win rate come from in-battle play. Mega Evolution is always used when available.

## What cannot be recovered from a log

A Pokémon that flinched, slept or was fully paralysed shows no move, so that choice is skipped. Moves triggered by other effects (Sleep Talk, Dancer, a locked-in Outrage) were not chosen that turn. Pivot switches (U-turn) and replacements after a faint are recorded as separate "replacement" decisions.

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
