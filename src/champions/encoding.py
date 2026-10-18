"""Turn a BattleState into the fixed-size arrays the policy network reads.

One example describes one decision: "the Pokémon in slot S of side X must act". The 12 Pokémon
tokens are ordered own team (as listed at team preview) then the opponent's, so a switch is
just "point at own token i".

    arrays = encode(state, side="p1", slot=0, kind=MOVE_PHASE)
"""

import numpy as np

from champions.battle_state import BOOSTS, SIDE_CONDITIONS, STATUSES, BattleState, Mon
from champions.vocab import (ABILITY_ID, ITEM_ID, LEARNABLE, MOVE_ID, NUM_MOVES, SPECIES_ID)

# Bump when the arrays from encode() change meaning or shape: cached training examples are stamped with it.
ENCODING_VERSION = 2
# Kinds of decision. Room is kept for team preview (bring four, pick two leads) and one spare, so adding
# them later does not change the network's shape.
MOVE_PHASE, REPLACEMENT, PREVIEW_BRING, PREVIEW_LEAD, SPARE_KIND_A, SPARE_KIND_B = range(6)
N_KINDS = 6
# Every array a network input needs, in one place (training batches, the live agent and the cache all use it).
ARRAY_KEYS = ["species", "items", "abilities", "moves", "numbers", "present", "stats", "globals", "slot", "kind",
              "actor", "mask", "partner_action", "partner_target"]
N_TOKENS, N_MOVE_SLOTS = 12, 4
STAT_SCALE = 250.0   # own stats are given to the network divided by this
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


def encode(state: BattleState, side: str, slot: int, kind: int, own_info: dict | None = None,
           partner: tuple[int, int] | None = None) -> dict:
    """Arrays for the decision of `side`'s Pokémon in `slot`, from that side's point of view.

    `partner` is what the other active Pokémon has already been told to do this turn, as
    (action index, target class), so the two choices are made together: the second slot is scored
    knowing the first's plan. None for the first slot, and for replacements.

    `own_info` is what the player knows about their own six ({team index: moves/item/ability/brought}):
    the real sets in live play, or sets reconstructed from the whole replay in training
    (see hindsight.py). The opponent's Pokémon only ever show what has been revealed so far.
    """
    own, foe = state.sides[side], state.sides["p2" if side == "p1" else "p1"]
    ordered = [(own, True), (foe, False)]

    species = np.zeros(N_TOKENS, dtype=np.int64)
    items = np.zeros(N_TOKENS, dtype=np.int64)
    abilities = np.zeros(N_TOKENS, dtype=np.int64)
    moves = np.zeros((N_TOKENS, N_MOVE_SLOTS), dtype=np.int64)
    numbers = np.zeros((N_TOKENS, MON_NUM), dtype=np.float32)
    present = np.zeros(N_TOKENS, dtype=bool)
    stats = np.zeros((N_TOKENS, 6), dtype=np.float32)   # your own Pokémon's real stats; zeros for the opponent's

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
            if own_info is not None and half == 0 and i in own_info:   # you know your own team completely
                known = own_info[i]
                items[t] = ITEM_ID.get(known["item"] or "", items[t])
                abilities[t] = ABILITY_ID.get(known["ability"] or "", abilities[t])
                moves[t] = 0
                for k, name in enumerate(known["moves"][:N_MOVE_SLOTS]):
                    moves[t, k] = MOVE_ID.get(name, 0)
                if known.get("stats"):
                    stats[t] = [known["stats"][key] / STAT_SCALE for key in ("hp", "atk", "def", "spa", "spd", "spe")]

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
        if own_info is not None and actor_index in own_info:
            for name in own_info[actor_index]["moves"]:
                move_mask[MOVE_ID.get(name, 0)] = True
            move_mask[0] = False
        else:
            move_mask[:NUM_MOVES] = learnable_moves(own.team[actor_index])
    for i, mon in enumerate(own.team[:6]):
        brought = own_info[i]["brought"] if own_info is not None and i in own_info else True
        if brought and not mon.fainted and i not in own.active:
            move_mask[SWITCH_BASE + i] = True

    partner_action, partner_target = (0, 0) if partner is None else (partner[0], partner[1] + 1)
    if partner_action >= SWITCH_BASE:       # the partner is bringing that teammate in, so this slot cannot
        move_mask[partner_action] = False

    return {
        "partner_action": np.int64(partner_action), "partner_target": np.int64(partner_target),
        "species": species, "items": items, "abilities": abilities, "moves": moves, "numbers": numbers,
        "present": present, "stats": stats, "globals": globals_, "slot": np.int64(slot), "kind": np.int64(kind),
        "actor": np.int64(-1 if actor_index is None else actor_index), "mask": move_mask,
    }
