"""Pull the players' decisions out of a replay log, as (state, action) training examples.

Each example is one Pokémon's choice: what the player could see at that moment, and what they did.
Logs only show what happened, so some choices cannot be recovered and are skipped:
a Pokémon that flinched, was asleep or was fully paralysed shows no move, and moves triggered
by other effects (Sleep Talk, Dancer, a locked-in Outrage) were not chosen that turn.

    for example in extract(log_lines):
        example["arrays"], example["action"], example["target"]
"""

import copy

from champions.battle_state import BattleState, base_species, parse_position
from champions.encoding import (MOVE_PHASE, REPLACEMENT, SWITCH_BASE, TARGET_ALLY, TARGET_FOE_A,
                                TARGET_FOE_B, TARGET_NONE, encode)
from champions.hindsight import complete_sets
from champions.vocab import MOVE_ID


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

    With `usage` (ladder stats), each player's own six are reconstructed from the whole game
    and given to the encoder, as a real player knows their own team (see hindsight.py).
    """
    own_sets = {}
    if usage is not None:
        final = BattleState()
        for line in lines:
            final.update(line)
        own_sets = {side: complete_sets(final, side, usage) for side in ("p1", "p2")}

    state = BattleState()
    examples: list[dict] = []

    turn_state: BattleState | None = None   # snapshot at the start of the current turn
    replace_state: BattleState | None = None  # snapshot after upkeep, for forced switch-ins
    acted: set = set()                       # (side, slot) already labelled this turn
    blocked: set = set()                     # (side, slot) whose choice is hidden this turn
    upkeep_seen = move_seen = False

    def add(snapshot, side, slot, kind, action, target):
        arrays = encode(snapshot, side, slot, kind, own_info=own_sets.get(side))
        if arrays["actor"] < 0 and kind == MOVE_PHASE:
            return
        examples.append({"arrays": arrays, "action": action, "target": target, "side": side, "kind": kind})

    for line in lines:
        parts = line.split("|")
        kind = parts[1] if len(parts) > 1 else ""

        if kind == "turn":
            state.update(line)
            turn_state, replace_state = copy.deepcopy(state), None
            acted.clear(); blocked.clear()
            upkeep_seen = move_seen = False
            continue

        if turn_state is not None:
            if kind == "move" and len(parts) > 3 and not any(p.startswith("[from]") for p in parts[4:]):
                side, slot, _ = parse_position(parts[2])
                key = (side, slot)
                move_seen = True
                if key not in acted and key not in blocked and parts[3] in MOVE_ID:
                    acted.add(key)
                    target_text = parts[4] if len(parts) > 4 else None
                    add(turn_state, side, slot, MOVE_PHASE, MOVE_ID[parts[3]],
                        target_class(side, slot, target_text, parts[4:]))
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
                own = state.sides[side]
                incoming = state._index_of(side, base_species(parts[3]))
                if incoming >= 6:   # a Pokémon we cannot place in the team of six
                    state.update(line)
                    continue
                key = (side, slot)
                if upkeep_seen:                      # forced replacement after a faint
                    add(replace_state, side, slot, REPLACEMENT, SWITCH_BASE + incoming, TARGET_NONE)
                elif move_seen:                      # pivot move or Eject Button mid-turn
                    add(copy.deepcopy(state), side, slot, REPLACEMENT, SWITCH_BASE + incoming, TARGET_NONE)
                elif key not in acted and own.active[slot] is not None:  # chosen switch, before any move
                    acted.add(key)
                    add(turn_state, side, slot, MOVE_PHASE, SWITCH_BASE + incoming, TARGET_NONE)

        state.update(line)

    return examples
