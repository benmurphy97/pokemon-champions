"""Turn a BattleState into the fixed-size arrays the policy network reads.

One example describes one side's whole decision for a turn: both active Pokémon choose together.
The 12 Pokémon tokens are ordered own team (as listed at team preview) then the opponent's.

Each slot's options are laid out as N_ATOMS "atoms" (see `atom_move` and the constants below):
a move slot (0-3) with a target class, a switch to own team index 0-5, pass, or a locked move the
engine plays by itself. The network scores every legal (atom for slot a, atom for slot b) pair.

    arrays = encode(state, side="p1", kind=MOVE_PHASE, own_info=...)
    arrays["atom_mask"]   # [2, N_ATOMS] legal atoms per slot (the live agent overwrites it from the request)
"""

import numpy as np

from champions.model.battle_state import BOOSTS, SIDE_CONDITIONS, STATUSES, BattleState, Mon
from champions.model.vocab import (ABILITY_ID, ITEM_ID, LEARNABLE, MOVE_CATS, MOVE_ID, NUM_MOVES, SPECIES_ID,
                                   TARGETS)

# Bump when the arrays from encode() change meaning or shape: cached training examples are stamped with it.
ENCODING_VERSION = 3
# Kinds of decision. Two spares, so adding a kind later does not change the network's shape.
MOVE_PHASE, REPLACEMENT, PREVIEW, SPARE_KIND_A, SPARE_KIND_B, SPARE_KIND_C = range(6)
N_KINDS = 6
# Every array a network input needs, in one place (training batches, the live agent and the cache all use it).
ARRAY_KEYS = ["species", "items", "abilities", "moves", "numbers", "present", "stats", "globals", "kind",
              "actors", "actor_moves", "atom_mask", "foes"]
N_TOKENS, N_MOVE_SLOTS = 12, 4
STAT_SCALE = 250.0   # own stats are given to the network divided by this
WEATHERS = ["sun", "rain", "sand", "snow"]
TERRAINS = ["electric", "grassy", "misty", "psychic"]

MON_NUM = 1 + 1 + 2 + 1 + 1 + 1 + len(STATUSES) + len(BOOSTS)    # see _mon_numbers
GLOBAL_NUM = 1 + len(WEATHERS) + len(TERRAINS) + 2 + 2 * len(SIDE_CONDITIONS) + 2

# Target classes for a move: foe slot a, foe slot b, the partner, or none (spread, self, field).
TARGET_FOE_A, TARGET_FOE_B, TARGET_ALLY, TARGET_NONE = range(4)
N_TARGETS = 4
# Atoms: move slot k with target class t is k * N_TARGETS + t; then six switches, pass, locked.
ATOM_SWITCH = N_MOVE_SLOTS * N_TARGETS          # 16: switch to own team index i is ATOM_SWITCH + i
ATOM_PASS = ATOM_SWITCH + 6                     # 22: nothing to choose (fainted, empty, not needed)
ATOM_LOCKED = ATOM_PASS + 1                     # 23: the engine plays the move (recharge, lock-in, Struggle)
N_ATOMS = ATOM_LOCKED + 1
N_PAIRS = N_ATOMS * N_ATOMS

# The opponent's action, for the auxiliary prediction head: a move by vocabulary id, or a switch to their team index.
SWITCH_BASE = NUM_MOVES
N_ACTIONS = NUM_MOVES + 6

SINGLE_TARGET = {"normal", "any", "adjacentFoe"}
ALLY_TARGET = {"adjacentAlly", "adjacentAllyOrSelf"}


def atom_move(k: int, target: int) -> int:
    return k * N_TARGETS + target


def target_classes(target_type: str | None, foes_present: tuple[bool, bool]) -> list[int]:
    """Which target classes a move with this Showdown target type can take."""
    if target_type in SINGLE_TARGET:
        present = [t for t, here in ((TARGET_FOE_A, foes_present[0]), (TARGET_FOE_B, foes_present[1])) if here]
        return present or [TARGET_FOE_A]
    if target_type in ALLY_TARGET:
        return [TARGET_ALLY]
    return [TARGET_NONE]


def move_target_type(move_id: int) -> str | None:
    t = int(MOVE_CATS[move_id, 2])
    return TARGETS[t - 1] if t else None


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


def encode(state: BattleState, side: str, kind: int, own_info: dict | None = None) -> dict:
    """Arrays for `side`'s decision this turn (both active Pokémon), from that side's point of view.

    `own_info` is what the player knows about their own six ({team index: moves/item/ability/brought/stats}):
    the real sets in live play, or sets reconstructed from the whole replay in training (see hindsight.py).
    The opponent's Pokémon only ever show what has been revealed so far.

    `atom_mask` is filled from the state (what replays allow): each active Pokémon's known moves with the
    targets their type allows, and switches to brought Pokémon that are alive and on the bench. The live
    agent replaces it with the request's exact options.
    """
    own, foe = state.sides[side], state.sides["p2" if side == "p1" else "p1"]

    species = np.zeros(N_TOKENS, dtype=np.int64)
    items = np.zeros(N_TOKENS, dtype=np.int64)
    abilities = np.zeros(N_TOKENS, dtype=np.int64)
    moves = np.zeros((N_TOKENS, N_MOVE_SLOTS), dtype=np.int64)
    numbers = np.zeros((N_TOKENS, MON_NUM), dtype=np.float32)
    present = np.zeros(N_TOKENS, dtype=bool)
    stats = np.zeros((N_TOKENS, 6), dtype=np.float32)   # your own Pokémon's real stats; zeros for the opponent's

    for half, s in enumerate((own, foe)):
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

    actors = np.array([i if i is not None and not own.team[i].fainted else -1 for i in own.active[:2]], dtype=np.int64)
    foes = np.array([i if i is not None and not foe.team[i].fainted else -1 for i in foe.active[:2]], dtype=np.int64)
    actor_moves = np.zeros((2, N_MOVE_SLOTS), dtype=np.int64)
    for slot, index in enumerate(actors):
        if index >= 0:
            actor_moves[slot] = moves[index]

    atom_mask = np.zeros((2, N_ATOMS), dtype=bool)
    foes_present = (bool(foes[0] >= 0), bool(foes[1] >= 0))
    bench = bench_of(state, side, own_info)
    for slot, index in enumerate(actors):
        if index < 0:
            atom_mask[slot, ATOM_PASS] = True
            continue
        if kind == MOVE_PHASE:
            for k, move_id in enumerate(actor_moves[slot]):
                if move_id:
                    for t in target_classes(move_target_type(int(move_id)), foes_present):
                        atom_mask[slot, atom_move(k, t)] = True
        for i in bench:
            atom_mask[slot, ATOM_SWITCH + i] = True
        if not atom_mask[slot].any():
            atom_mask[slot, ATOM_PASS] = True

    return {"species": species, "items": items, "abilities": abilities, "moves": moves, "numbers": numbers,
            "present": present, "stats": stats, "globals": globals_, "kind": np.int64(kind),
            "actors": actors, "actor_moves": actor_moves, "atom_mask": atom_mask, "foes": foes}


def bench_of(state: BattleState, side: str, own_info: dict | None = None) -> list[int]:
    """Own team indices that could be switched in: brought, alive, not on the field."""
    own = state.sides[side]
    return [i for i, mon in enumerate(own.team[:6])
            if not mon.fainted and i not in own.active
            and (own_info is None or i not in own_info or own_info[i].get("brought", True))]


def replacement_mask(bench: list[int], slots: list[int]) -> np.ndarray:
    """[2, N_ATOMS] for a forced switch-in: each slot in `slots` picks from the bench, the other passes."""
    mask = np.zeros((2, N_ATOMS), dtype=bool)
    for slot in (0, 1):
        if slot in slots and bench:
            mask[slot, [ATOM_SWITCH + i for i in bench]] = True
        else:
            mask[slot, ATOM_PASS] = True
    return mask


def pair_mask(atom_mask: np.ndarray) -> np.ndarray:
    """[N_PAIRS] legal (slot a, slot b) pairs: both atoms legal, and not both switching to the same Pokémon."""
    legal = atom_mask[0][:, None] & atom_mask[1][None, :]
    for i in range(6):
        legal[ATOM_SWITCH + i, ATOM_SWITCH + i] = False
    return legal.reshape(-1)
