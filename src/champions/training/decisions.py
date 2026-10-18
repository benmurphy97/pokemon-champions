"""Pull the players' decisions out of a replay log, as (state, joint action) training examples.

One example per side per turn: what the player could see at the start of the turn, and what each of
their two active Pokémon chose, as atoms (see encoding). Also recorded: what the opponent's two
Pokémon did that turn, the label for the opponent-prediction head.

Logs only show what happened, so some choices cannot be recovered: a Pokémon that flinched, was asleep
or fully paralysed shows no move, and moves triggered by other effects (Sleep Talk, Dancer, a locked-in
Outrage) were not chosen that turn. Such a slot's label is -1 and training marginalises it out. A turn
where neither slot's choice is known is skipped.

Forced switch-ins (after a faint, or mid-turn after U-turn / Eject Button) are REPLACEMENT examples:
the slot that switches picks from the bench, the other slot passes.

    for example in extract(log_lines, usage):
        example["arrays"], example["label"], example["foe_label"]
"""

import copy

import numpy as np

from champions.model.battle_state import BattleState, base_species, parse_position
from champions.model.encoding import (ATOM_PASS, ATOM_SWITCH, MOVE_PHASE, REPLACEMENT, SWITCH_BASE, TARGET_ALLY,
                                      TARGET_FOE_A, TARGET_FOE_B, TARGET_NONE, atom_move, bench_of, encode,
                                      replacement_mask)
from champions.training.hindsight import complete_sets
from champions.model.vocab import MOVE_ID


def target_class(mover_side: str, mover_slot: int, target_text: str | None, tags: list[str]) -> int:
    if not target_text or target_text.startswith("[") or any(t.startswith("[spread]") for t in tags):
        return TARGET_NONE
    side, slot, _ = parse_position(target_text)
    if slot is None:
        return TARGET_NONE
    if side != mover_side:
        return TARGET_FOE_A if slot == 0 else TARGET_FOE_B
    return TARGET_ALLY if slot != mover_slot else TARGET_NONE


def extract(lines: list[str], usage: dict | None = None) -> list[dict]:
    """All recoverable decisions in a log, in order.

    With `usage` (ladder stats), each player's own six are reconstructed from the whole game and given to
    the encoder, as a real player knows their own team (see hindsight.py).
    """
    own_sets = {}
    if usage is not None:
        final = BattleState()
        for line in lines:
            final.update(line)
        own_sets = {side: complete_sets(final, side, usage) for side in ("p1", "p2")}

    state = BattleState()
    examples: list[dict] = []
    turn_state: BattleState | None = None    # snapshot at the start of the current turn
    replace_state: BattleState | None = None  # snapshot after upkeep, for forced switch-ins
    chosen: dict = {}                        # (side, slot) -> ("move", id, target) or ("switch", team index)
    blocked: set = set()                     # (side, slot) whose choice is hidden this turn
    upkeep_seen = move_seen = False
    turn_no = 0

    def finish_turn():
        """Turn the choices seen this turn into one MOVE_PHASE example per side."""
        if turn_state is None:
            return
        for side in ("p1", "p2"):
            other = "p2" if side == "p1" else "p1"
            arrays = encode(turn_state, side, MOVE_PHASE, own_info=own_sets.get(side))
            label = [-1, -1]
            for slot in (0, 1):
                if arrays["actors"][slot] < 0:
                    label[slot] = ATOM_PASS
                    continue
                what = chosen.get((side, slot))
                if what is None or (side, slot) in blocked:
                    continue
                if what[0] == "switch":
                    atom = ATOM_SWITCH + what[1]
                else:
                    known = list(arrays["actor_moves"][slot])
                    if what[1] not in known:
                        continue
                    atom = atom_move(known.index(what[1]), what[2])
                arrays["atom_mask"][slot, atom] = True     # what the player did is always allowed
                label[slot] = atom
            if label[0] < 0 and label[1] < 0:
                continue
            foe_label = [-1, -1]
            for slot in (0, 1):
                what = chosen.get((other, slot))
                if what is not None and (other, slot) not in blocked and arrays["foes"][slot] >= 0:
                    foe_label[slot] = what[1] if what[0] == "move" else SWITCH_BASE + what[1]
            examples.append({"arrays": arrays, "label": label, "foe_label": foe_label, "side": side,
                             "kind": MOVE_PHASE, "turn": turn_no})

    def add_replacement(snapshot, side, slot, incoming):
        info = own_sets.get(side)
        bench = bench_of(snapshot, side, info)
        if incoming not in bench:
            bench = sorted(set(bench) | {incoming})
        arrays = encode(snapshot, side, REPLACEMENT, own_info=info)
        arrays["atom_mask"] = replacement_mask(bench, [slot])
        label = [ATOM_PASS, ATOM_PASS]
        label[slot] = ATOM_SWITCH + incoming
        examples.append({"arrays": arrays, "label": label, "foe_label": [-1, -1], "side": side,
                         "kind": REPLACEMENT, "turn": turn_no})

    for line in lines:
        parts = line.split("|")
        kind = parts[1] if len(parts) > 1 else ""

        if kind in ("turn", "win", "tie"):
            finish_turn()
            if kind != "turn":
                break
            state.update(line)
            turn_no += 1
            turn_state, replace_state = copy.deepcopy(state), None
            chosen.clear(); blocked.clear()
            upkeep_seen = move_seen = False
            continue

        if turn_state is not None:
            if kind == "move" and len(parts) > 3 and not any(p.startswith("[from]") for p in parts[4:]):
                side, slot, _ = parse_position(parts[2])
                move_seen = True
                if (side, slot) not in chosen and parts[3] in MOVE_ID:
                    target_text = parts[4] if len(parts) > 4 else None
                    chosen[(side, slot)] = ("move", MOVE_ID[parts[3]], target_class(side, slot, target_text, parts[4:]))
            elif kind == "cant":
                side, slot, _ = parse_position(parts[2])
                blocked.add((side, slot))
            elif kind == "upkeep":
                state.update(line)
                upkeep_seen = True
                replace_state = copy.deepcopy(state)
                continue
            elif kind == "switch":
                side, slot, _ = parse_position(parts[2])
                incoming = state._index_of(side, base_species(parts[3]))
                if incoming >= 6:   # a Pokémon we cannot place in the team of six
                    state.update(line)
                    continue
                if upkeep_seen:                      # forced replacement after a faint
                    add_replacement(replace_state, side, slot, incoming)
                elif move_seen:                      # pivot move or Eject Button mid-turn
                    add_replacement(copy.deepcopy(state), side, slot, incoming)
                elif (side, slot) not in chosen and state.sides[side].active[slot] is not None:   # chosen switch, before any move
                    chosen[(side, slot)] = ("switch", incoming)

        state.update(line)
    return examples


def stack(examples: list[dict]) -> dict:
    """Examples as arrays: every encoder array, plus `label` [N, 2] and `foe_label` [N, 2]."""
    from champions.model.encoding import ARRAY_KEYS
    out = {k: np.stack([e["arrays"][k] for e in examples]) for k in ARRAY_KEYS}
    out["label"] = np.array([e["label"] for e in examples], dtype=np.int64)
    out["foe_label"] = np.array([e["foe_label"] for e in examples], dtype=np.int64)
    return out
