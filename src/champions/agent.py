"""Use a trained policy to choose actions in a live battle.

    agent = NeuralAgent(net)
    choice = agent.decide(state, "p1", request)    # e.g. "move 2 1, switch 4"

`state` is a BattleState fed with every log line the player has received; `request` is the
simulator's decision request. Only legal actions are considered.
"""

import numpy as np
import torch

from champions.battle_state import BattleState, base_species
from champions.encoding import (MOVE_PHASE, N_ACTIONS, REPLACEMENT, SWITCH_BASE, TARGET_FOE_A, TARGET_FOE_B,
                                encode)
from champions.vocab import MOVE_ID

KEYS = ["species", "items", "abilities", "moves", "numbers", "present", "globals", "slot", "kind", "actor", "mask"]
SINGLE_TARGET = {"normal", "any", "adjacentFoe"}


def to_tensors(arrays: dict) -> dict:
    return {k: torch.as_tensor(np.asarray(arrays[k]))[None] for k in KEYS}


class NeuralAgent:
    def __init__(self, net, temperature: float = 0.0, seed: int = 0):
        self.net = net.eval()
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def pick(self, logits: torch.Tensor, legal: np.ndarray) -> int:
        scores = logits[0].clone()
        scores[~torch.as_tensor(legal)] = -1e9
        if self.temperature <= 0:
            return int(scores.argmax())
        probs = torch.softmax(scores / self.temperature, dim=-1).numpy().astype(np.float64)
        return int(self.rng.choice(len(probs), p=probs / probs.sum()))

    def decide(self, state: BattleState, side: str, request: dict) -> str:
        pokemon = request["side"]["pokemon"]
        team_index = [state._index_of(side, base_species(p["details"])) for p in pokemon]
        position_of = {ti: j + 1 for j, ti in enumerate(team_index)}
        bench = [ti for ti, p in zip(team_index, pokemon) if not p["active"] and not p["condition"].endswith(" fnt")]
        taken: set[int] = set()
        choices = []

        with torch.no_grad():
            if request.get("forceSwitch"):
                for slot, needed in enumerate(request["forceSwitch"]):
                    options = [ti for ti in bench if ti not in taken]
                    if not needed or not options:
                        choices.append("pass")
                        continue
                    arrays = encode(state, side, slot, REPLACEMENT)
                    legal = np.zeros(N_ACTIONS, dtype=bool)
                    legal[[SWITCH_BASE + ti for ti in options]] = True
                    arrays["mask"] = legal
                    logits, _ = self.net(to_tensors(arrays))
                    chosen = self.pick(logits, legal) - SWITCH_BASE
                    taken.add(chosen)
                    choices.append(f"switch {position_of[chosen]}")
                return ", ".join(choices)

            mega_used = False
            for slot, active in enumerate(request["active"]):
                mon = pokemon[slot]
                if mon["condition"].endswith(" fnt") or mon.get("commanding"):
                    choices.append("pass")
                    continue

                arrays = encode(state, side, slot, MOVE_PHASE)
                legal = np.zeros(N_ACTIONS, dtype=bool)
                slot_of_move = {}
                for k, move in enumerate(active["moves"]):
                    move_id = MOVE_ID.get(move["move"], 0)
                    if not move.get("disabled") and move_id:
                        legal[move_id] = True
                        slot_of_move[move_id] = (k + 1, move["target"])
                if not active.get("trapped"):
                    for ti in bench:
                        if ti not in taken:
                            legal[SWITCH_BASE + ti] = True
                if not legal.any():
                    choices.append("move 1")
                    continue

                arrays["mask"] = legal
                logits, _ = self.net(to_tensors(arrays))
                action = self.pick(logits, legal)
                if action >= SWITCH_BASE:
                    taken.add(action - SWITCH_BASE)
                    choices.append(f"switch {position_of[action - SWITCH_BASE]}")
                    continue

                move_slot, target_type = slot_of_move[action]
                choice = f"move {move_slot}"
                if target_type in SINGLE_TARGET:
                    _, target_logits = self.net(to_tensors(arrays), target_for_move=torch.tensor([action]))
                    foes = state.sides["p2" if side == "p1" else "p1"].active
                    alive = [(cls, foe_slot + 1) for cls, foe_slot in ((TARGET_FOE_A, 0), (TARGET_FOE_B, 1))
                             if foes[foe_slot] is not None] or [(TARGET_FOE_A, 1)]
                    choice += f" {max(alive, key=lambda c: float(target_logits[0, c[0]]))[1]}"
                elif target_type == "adjacentAlly":
                    choice += f" -{(slot ^ 1) + 1}"
                elif target_type == "adjacentAllyOrSelf":
                    choice += f" -{slot + 1}"
                if active.get("canMegaEvo") and not mega_used:
                    choice += " mega"
                    mega_used = True
                choices.append(choice)
        return ", ".join(choices)
