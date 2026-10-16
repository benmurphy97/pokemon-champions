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
from champions.hindsight import complete_sets
from champions.meta import load_usage
from champions.vocab import MOVE_ID, NAME_BY_ID

KEYS = ["species", "items", "abilities", "moves", "numbers", "present", "stats", "globals", "slot", "kind", "actor", "mask"]
SINGLE_TARGET = {"normal", "any", "adjacentFoe"}


def to_tensors(arrays: dict) -> dict:
    return {k: torch.as_tensor(np.asarray(arrays[k]))[None] for k in KEYS}


def own_team_info(state: BattleState, side: str, request: dict, team_index: list[int], usage: dict) -> dict:
    """What the player knows about their own six: real sets from the request for the four brought,
    and reconstructed ones for the two left behind (which can never be switched in)."""
    info = complete_sets(state, side, usage)
    for ti in info:
        info[ti]["brought"] = False
    for ti, mon in zip(team_index, request["side"]["pokemon"]):
        stats = dict(mon.get("stats") or {})
        max_hp = mon["condition"].split(" ")[0].split("/")
        stats["hp"] = int(max_hp[1]) if len(max_hp) == 2 else (info[ti].get("stats") or {}).get("hp", 0)
        info[ti] = {
            "stats": stats if stats.get("atk") else info[ti].get("stats"),
            "moves": [NAME_BY_ID.get(m, m) for m in mon.get("moves", [])],
            "item": NAME_BY_ID.get(mon.get("item", ""), None),
            "ability": NAME_BY_ID.get(mon.get("baseAbility") or mon.get("ability", ""), None),
            "brought": True,
        }
    return info


class NeuralAgent:
    """Chooses actions with a trained policy.

    temperature 0 plays the single most likely legal action (for evaluation); temperature 1 samples
    from the policy (for reinforcement learning). Pass `record=[]` to `decide` to keep every
    choice that had alternatives, with the log-probability it was taken with, for training.
    """

    def __init__(self, net, temperature: float = 0.0, seed: int = 0):
        self.usage = load_usage()
        self.net = net.eval()
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def pick(self, logits: torch.Tensor, legal: np.ndarray) -> tuple[int, float]:
        """(chosen action, its log-probability under the policy at temperature 1)."""
        scores = logits[0].clone()
        scores[~torch.as_tensor(legal)] = -1e9
        log_probs = torch.log_softmax(scores, dim=-1)
        if self.temperature <= 0:
            action = int(scores.argmax())
        else:
            probs = torch.softmax(scores / self.temperature, dim=-1).numpy().astype(np.float64)
            action = int(self.rng.choice(len(probs), p=probs / probs.sum()))
        return action, float(log_probs[action])

    def pick_target(self, target_logits: torch.Tensor, allowed: np.ndarray) -> tuple[int, float]:
        scores = target_logits[0].clone()
        scores[~torch.as_tensor(allowed)] = -1e9
        log_probs = torch.log_softmax(scores, dim=-1)
        if self.temperature <= 0:
            target = int(scores.argmax())
        else:
            probs = torch.softmax(scores / self.temperature, dim=-1).numpy().astype(np.float64)
            target = int(self.rng.choice(len(probs), p=probs / probs.sum()))
        return target, float(log_probs[target])

    @staticmethod
    def note(record, arrays, legal, action, logp, target=-1, target_mask=None):
        record.append({"arrays": dict(arrays), "action": action, "logp": logp, "target": target,
                       "target_mask": np.zeros(4, dtype=bool) if target_mask is None else target_mask})

    def decide(self, state: BattleState, side: str, request: dict, record: list | None = None) -> str:
        pokemon = request["side"]["pokemon"]
        team_index = [state._index_of(side, base_species(p["details"])) for p in pokemon]
        position_of = {ti: j + 1 for j, ti in enumerate(team_index)}
        bench = [ti for ti, p in zip(team_index, pokemon) if not p["active"] and not p["condition"].endswith(" fnt")]
        own = own_team_info(state, side, request, team_index, self.usage)
        taken: set[int] = set()
        choices = []

        with torch.no_grad():
            if request.get("forceSwitch"):
                for slot, needed in enumerate(request["forceSwitch"]):
                    options = [ti for ti in bench if ti not in taken]
                    if not needed or not options:
                        choices.append("pass")
                        continue
                    arrays = encode(state, side, slot, REPLACEMENT, own_info=own)
                    legal = np.zeros(N_ACTIONS, dtype=bool)
                    legal[[SWITCH_BASE + ti for ti in options]] = True
                    arrays["mask"] = legal
                    logits, _ = self.net(to_tensors(arrays))
                    action, logp = self.pick(logits, legal)
                    if record is not None and len(options) > 1:
                        self.note(record, arrays, legal, action, logp)
                    chosen = action - SWITCH_BASE
                    taken.add(chosen)
                    choices.append(f"switch {position_of[chosen]}")
                return ", ".join(choices)

            mega_used = False
            for slot, active in enumerate(request["active"]):
                mon = pokemon[slot]
                if mon["condition"].endswith(" fnt") or mon.get("commanding"):
                    choices.append("pass")
                    continue

                arrays = encode(state, side, slot, MOVE_PHASE, own_info=own)
                legal = np.zeros(N_ACTIONS, dtype=bool)
                slot_of_move = {}
                for k, move in enumerate(active["moves"]):
                    move_id = MOVE_ID.get(move["move"], 0)
                    if not move.get("disabled") and move_id:
                        legal[move_id] = True
                        slot_of_move[move_id] = (k + 1, move.get("target"))
                if not slot_of_move:
                    # Forced: recharging, locked into a move, or Struggle. The engine decides the details.
                    choices.append("move 1")
                    continue
                if not active.get("trapped"):
                    for ti in bench:
                        if ti not in taken:
                            legal[SWITCH_BASE + ti] = True

                arrays["mask"] = legal
                logits, _ = self.net(to_tensors(arrays))
                action, logp = self.pick(logits, legal)
                if action >= SWITCH_BASE:
                    if record is not None and legal.sum() > 1:
                        self.note(record, arrays, legal, action, logp)
                    taken.add(action - SWITCH_BASE)
                    choices.append(f"switch {position_of[action - SWITCH_BASE]}")
                    continue

                move_slot, target_type = slot_of_move[action]
                choice = f"move {move_slot}"
                target, target_mask = -1, None
                if target_type in SINGLE_TARGET:
                    _, target_logits = self.net(to_tensors(arrays), target_for_move=torch.tensor([action]))
                    foes = state.sides["p2" if side == "p1" else "p1"].active
                    target_mask = np.zeros(4, dtype=bool)
                    for cls, foe_slot in ((TARGET_FOE_A, 0), (TARGET_FOE_B, 1)):
                        target_mask[cls] = foes[foe_slot] is not None
                    if not target_mask.any():
                        target_mask[TARGET_FOE_A] = True
                    target, target_logp = self.pick_target(target_logits, target_mask)
                    logp += target_logp
                    choice += f" {target + 1}"
                elif target_type == "adjacentAlly":
                    choice += f" -{(slot ^ 1) + 1}"
                elif target_type == "adjacentAllyOrSelf":
                    choice += f" -{slot + 1}"
                if record is not None and (legal.sum() > 1 or (target_mask is not None and target_mask.sum() > 1)):
                    self.note(record, arrays, legal, action, logp, target, target_mask)
                if active.get("canMegaEvo") and not mega_used:
                    choice += " mega"
                    mega_used = True
                choices.append(choice)
        return ", ".join(choices)
