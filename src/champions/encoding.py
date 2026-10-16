"""Turn a BattleState into the fixed-size arrays the policy network reads.

One example describes one decision: "the Pokémon in slot S of side X must act". The 12 Pokémon
tokens are ordered own team (as listed at team preview) then the opponent's, so a switch is
just "point at own token i".

    arrays = encode(state, side="p1", slot=0, kind=MOVE_PHASE)
"""

import numpy as np

from champions.battle_state import BOOSTS, SIDE_CONDITIONS, STATUSES, BattleState, Mon
from champions.vocab import (ABILITY_ID, ITEM_ID, LEARNABLE, MOVE_ID, NUM_MOVES, SPECIES_ID)

MOVE_PHASE, REPLACEMENT = 0, 1   # kinds of decision: normal turn action, or forced switch-in
N_TOKENS, N_MOVE_SLOTS = 12, 4
WEATHERS = ["sun", "rain", "sand", "snow"]
TERRAINS = ["electric", "grassy", "misty", "psychic"]

MON_NUM = 1 + 1 + 2 + 1 + 1 + 1 + len(STATUSES) + len(BOOSTS)    # see _mon_numbers
GLOBAL_NUM = 1 + len(WEATHERS) + len(TERRAINS) + 2 + 2 * len(SIDE_CONDITIONS) + 2

# Action indices: 1..NUM_MOVES-1 are moves (the move's vocabulary id); then six switch targets.
SWITCH_BASE = NUM_MOVES
N_ACTIONS = NUM_MOVES + 6
# Target classes for a move: foe slot a, foe slot b, the partner, or none/spread/self.
TARGET_FOE_A, TARGET_FOE_B, TARGET_ALLY, TARGET_NONE = range(4)


def _mon_numbers(mon: Mon, slot_a: bool, slot_b: bool) -> list[float]:
    status = [1.0 if mon.status == s else 0.0 for s in STATUSES]
    boosts = [mon.boosts[b] / 6 for b in BOOSTS]
    return [mon.hp, float(mon.fainted), float(slot_a), float(slot_b), float(mon.revealed),
            float(mon.mega), min(mon.turns_active, 5) / 5, *status, *boosts]


def _species_id(mon: Mon) -> int:
    return SPECIES_ID.get(mon.form or mon.species, SPECIES_ID.get(mon.species, 0))


def learnable_moves(mon: Mon) -> np.ndarray:
    """Moves this Pokémon could plausibly know: its learnset plus anything it has been seen using."""
    mask = LEARNABLE[_species_id(mon)].copy()
    for move in mon.moves:
        mask[MOVE_ID.get(move, 0)] = True
    mask[0] = False
    return mask


def encode(state: BattleState, side: str, slot: int, kind: int) -> dict:
    """Arrays for the decision of `side`'s Pokémon in `slot`, from that side's point of view."""
    own, foe = state.sides[side], state.sides["p2" if side == "p1" else "p1"]
    ordered = [(own, True), (foe, False)]

    species = np.zeros(N_TOKENS, dtype=np.int64)
    items = np.zeros(N_TOKENS, dtype=np.int64)
    abilities = np.zeros(N_TOKENS, dtype=np.int64)
    moves = np.zeros((N_TOKENS, N_MOVE_SLOTS), dtype=np.int64)
    numbers = np.zeros((N_TOKENS, MON_NUM), dtype=np.float32)
    present = np.zeros(N_TOKENS, dtype=bool)

    for half, (s, _is_own) in enumerate(ordered):
        for i, mon in enumerate(s.team[:6]):
            t = half * 6 + i
            present[t] = True
            species[t] = _species_id(mon)
            items[t] = 1 if mon.item_gone and not mon.item else ITEM_ID.get(mon.item or "", 0)
            abilities[t] = ABILITY_ID.get(mon.ability or "", 0)
            for k, move in enumerate(mon.moves[:N_MOVE_SLOTS]):
                moves[t, k] = MOVE_ID.get(move, 0)
            numbers[t] = _mon_numbers(mon, s.active[0] == i, s.active[1] == i)

    globals_ = np.zeros(GLOBAL_NUM, dtype=np.float32)
    g = [min(state.turn, 20) / 20]
    g += [float(state.weather == w) for w in WEATHERS] + [float(state.terrain == t) for t in TERRAINS]
    g += [float(state.trick_room), float(state.gravity)]
    g += [float(c in own.conditions) for c in SIDE_CONDITIONS] + [float(c in foe.conditions) for c in SIDE_CONDITIONS]
    g += [float(own.mega_used), float(foe.mega_used)]
    globals_[:] = g

    # Which actions are plausible. Switches: own Pokémon that are alive and not on the field.
    move_mask = np.zeros(N_ACTIONS, dtype=bool)
    actor_index = own.active[slot]
    if kind == MOVE_PHASE and actor_index is not None:
        move_mask[:NUM_MOVES] = learnable_moves(own.team[actor_index])
    for i, mon in enumerate(own.team[:6]):
        if not mon.fainted and i not in own.active:
            move_mask[SWITCH_BASE + i] = True

    return {
        "species": species, "items": items, "abilities": abilities, "moves": moves, "numbers": numbers,
        "present": present, "globals": globals_, "slot": np.int64(slot), "kind": np.int64(kind),
        "actor": np.int64(-1 if actor_index is None else actor_index), "mask": move_mask,
    }
