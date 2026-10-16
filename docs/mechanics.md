# Mechanics confirmed in the simulator

Facts checked against Showdown's M-C engine, so later phases don't have to rediscover them.

## Mega Evolution is triggered by the held item

- A team slot is a base species holding a Mega Stone. It switches in as the base form and Mega Evolves in battle.
- Writing the slot as `Salamence-Mega @ Salamencite` or `Salamence @ Salamencite` gives the same Pokémon: Showdown converts the first to base Salamence and replaces the ability with the base form's (Intimidate), not the Mega ability (Aerilate). Checked 2026-10-17.
- Only one Mega Evolution can happen per battle, but a team can hold several different stones (real teams carry two). So "has a Mega" is a property of the item choice, not of the team as a whole.
- Consequences:
  - The ladder stats list `Salamence-Mega` separately because they count post-evolution. When we build a set from that entry, the base ability falls back to the engine default. This is an approximation: the real base ability choice (e.g. Intimidate vs Moxie) is not in the stats.
  - In the team search (Phase 3), swapping an item can add or remove a Mega. Mutations should treat the stone as the item, and the one-Mega-per-battle limit is handled by the engine at battle time.
  - Replays record the base species in `|poke|` lines; a Mega only shows up through `|-mega|` and `|detailschange|` once it happens. A Pokémon that never Mega Evolved may or may not have been holding a stone. `replay_check.py` assumes it was when the Mega form is the more common one on the ladder.
