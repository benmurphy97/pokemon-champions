"""Use a trained policy to choose actions in a live battle.

    agent = NeuralAgent(net)
    choice = agent.decide(state, "p1", request)    # e.g. "move 2 1 mega, switch 4"

`state` is a BattleState fed with every log line the player has received; `request` is the simulator's
decision request. Both active Pokémon are chosen together: the network scores every legal pair of atoms
(see encoding) and the agent takes the best pair (temperature 0) or samples one (temperature > 0).
"""

import numpy as np
import torch

from champions.model.battle_state import BattleState, base_species
from champions.model.encoding import ARRAY_KEYS as KEYS
from champions.model.encoding import (ATOM_LOCKED, ATOM_PASS, ATOM_SWITCH, MOVE_PHASE, N_ATOMS, N_MOVE_SLOTS,
                                      N_TARGETS, REPLACEMENT, SWITCH_BASE, TARGET_ALLY, TARGET_FOE_A, TARGET_FOE_B,
                                      atom_move, encode, pair_mask, target_classes)
import random

from champions.model.belief import observed_from, sample_assignments, shared_corpus
from champions.training.hindsight import complete_sets
from champions.teams.generate import load_usage
from champions.model.vocab import MOVE_ID, MOVES, NAME_BY_ID

PROTECT_MOVES = {"Protect", "Detect", "Spiky Shield", "King's Shield", "Baneful Bunker", "Silk Trap", "Burning Bulwark",
                 "Obstruct", "Max Guard", "Wide Guard", "Quick Guard", "Mat Block", "Crafty Shield"}


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


def lead(state: BattleState, side: str) -> float:
    """How far ahead `side` is on health, from -1 to 1: its remaining health minus the foe's, over six Pokémon each."""
    own, foe = state.sides[side], state.sides["p2" if side == "p1" else "p1"]
    return (sum(m.hp for m in own.team[:6]) - sum(m.hp for m in foe.team[:6])) / 6


class NeuralAgent:
    """Chooses both active Pokémon's actions together with a trained policy.

    temperature 0 plays the most likely legal pair (for evaluation); temperature 1 samples from the policy
    (for reinforcement learning). Pass `record=[]` to `decide` to keep every decision that had alternatives,
    with the log-probability it was taken with, for training.
    """

    def __init__(self, net, temperature: float = 0.0, seed: int = 0, preview=None, block_repeat_protect: bool = True,
                 search_k: int = 0, search_fair: bool = True, belief_samples: int = 0):
        # search_k > 0: in the move phase, hand the top-k joint choices to the worker's one-turn look-ahead
        # (sim/search.js) through `last_candidates`; the worker picks among them.
        # search_fair: the copies get our estimate of the opponent's sets (`last_foe_sets`), not their real ones.
        # belief_samples > 0: the copies get that many sets drawn from the posterior over their sets (belief.py)
        # instead of the single best guess, and the search averages over them.
        self.search_k, self.search_fair, self.belief_samples = search_k, search_fair, belief_samples
        self.belief_rng = random.Random(seed)
        self.last_candidates = self.last_foe_sets = None
        # A Protect-type move used on the previous turn almost always fails, so it is left out of the legal set.
        self.block_repeat_protect = block_repeat_protect
        # `preview(state, side, request) -> "team 1234"` plugs in a team-preview policy (bring four,
        # first two lead); None keeps the greedy agent's rule-based preview.
        self.preview = preview
        self.usage = load_usage()
        self.net = net.eval()
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def repeat_block(self, state: BattleState, side: str, pokemon: dict) -> bool:
        """True when this Pokémon used a Protect-type move last turn."""
        if not self.block_repeat_protect:
            return False
        index = state._index_of(side, base_species(pokemon["details"]))
        mon = state.sides[side].team[index]
        return mon.last_move in PROTECT_MOVES and mon.last_move_turn == state.turn - 1

    def options(self, state: BattleState, side: str, request: dict, arrays: dict) -> list[dict]:
        """For each slot, {atom: choice string}; also writes the request's exact options into `arrays`.

        `arrays["actor_moves"]` becomes the request's move list for each active Pokémon, so move atom k is
        "move k+1" (this matters when locked into one move, where the request lists only that move).
        """
        pokemon = request["side"]["pokemon"]
        team_index = [state._index_of(side, base_species(p["details"])) for p in pokemon]
        position_of = {ti: j + 1 for j, ti in enumerate(team_index)}
        bench = [ti for ti, p in zip(team_index, pokemon) if not p["active"] and not p["condition"].endswith(" fnt")]
        foe_side = state.sides["p2" if side == "p1" else "p1"]
        foes_present = tuple(i is not None and not foe_side.team[i].fainted for i in (foe_side.active + [None, None])[:2])
        force = request.get("forceSwitch")
        slots = [{}, {}]
        mask = np.zeros((2, N_ATOMS), dtype=bool)
        for slot in (0, 1):
            mon = pokemon[slot] if slot < len(pokemon) else None
            fainted = mon is None or mon["condition"].endswith(" fnt") or mon.get("commanding")
            if force is not None:
                if slot < len(force) and force[slot] and bench:
                    slots[slot] = {ATOM_SWITCH + ti: f"switch {position_of[ti]}" for ti in bench}
                    # Fewer Pokémon left than slots to fill: the second slot may pass (never both, so one switch is made).
                    if slot == 1 and len(bench) < sum(bool(f) for f in force[:2]):
                        slots[slot][ATOM_PASS] = "pass"
                else:
                    slots[slot] = {ATOM_PASS: "pass"}
            elif fainted or slot >= len(request.get("active", [])):
                slots[slot] = {ATOM_PASS: "pass"}
            else:
                active = request["active"][slot]
                ids = np.zeros(N_MOVE_SLOTS, dtype=np.int64)
                blocked = self.repeat_block(state, side, mon)
                usable = [m for m in active["moves"] if not m.get("disabled")]
                only_protect = all(m["move"] in PROTECT_MOVES for m in usable)
                for k, move in enumerate(active["moves"][:N_MOVE_SLOTS]):
                    move_id = MOVE_ID.get(move["move"], 0)
                    ids[k] = move_id
                    if not move_id or move.get("disabled") or (blocked and move["move"] in PROTECT_MOVES and not only_protect):
                        continue
                    for t in target_classes(move.get("target"), foes_present):
                        suffix = {TARGET_FOE_A: " 1", TARGET_FOE_B: " 2", TARGET_ALLY: f" -{(slot ^ 1) + 1}"}.get(t, "")
                        slots[slot][atom_move(k, t)] = f"move {k + 1}{suffix}"
                arrays["actor_moves"][slot] = ids
                if not slots[slot]:
                    slots[slot] = {ATOM_LOCKED: "move 1"}       # recharging, locked in, or Struggle
                elif not active.get("trapped"):
                    slots[slot].update({ATOM_SWITCH + ti: f"switch {position_of[ti]}" for ti in bench})
                if active.get("canMegaEvo"):
                    slots[slot]["mega"] = True
            for atom in slots[slot]:
                if atom != "mega":
                    mask[slot, atom] = True
        arrays["atom_mask"] = mask
        return slots

    def decide(self, state: BattleState, side: str, request: dict, record: list | None = None,
               thoughts: list | None = None) -> str:
        self.last_candidates = self.last_foe_sets = None
        if request.get("teamPreview"):
            return self.preview(state, side, request) if self.preview else "default"
        pokemon = request["side"]["pokemon"]
        team_index = [state._index_of(side, base_species(p["details"])) for p in pokemon]
        own = own_team_info(state, side, request, team_index, self.usage)
        kind = REPLACEMENT if request.get("forceSwitch") else MOVE_PHASE
        arrays = encode(state, side, kind, own_info=own)
        slots = self.options(state, side, request, arrays)
        legal = pair_mask(arrays["atom_mask"])

        with torch.no_grad():
            batch = to_tensors(arrays)
            q, mons = self.net.trunk(batch)
            logits = self.net.pair_logits(batch, q, mons)[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        if self.temperature <= 0:
            pair = int(logits.argmax())
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1).numpy().astype(np.float64)
            pair = int(self.rng.choice(len(probs), p=probs / probs.sum()))
        atoms = divmod(pair, N_ATOMS)
        n = len(request.get("forceSwitch") or request.get("active") or [0, 0])
        if self.search_k and kind == MOVE_PHASE and legal.sum() > 1:
            top = torch.argsort(logits, descending=True)[:self.search_k].tolist()
            self.last_candidates = [{"choice": self.pair_choice(slots, divmod(p, N_ATOMS), n), "prior": float(log_probs[p].exp())}
                                    for p in top if legal.flatten()[p]]
            if self.search_fair:     # what we believe the opponent has: revealed details, the rest from usage
                foe_side = "p2" if side == "p1" else "p1"
                if self.belief_samples:
                    six, observed = observed_from(state, foe_side)
                    self.last_foe_sets = sample_assignments(shared_corpus(), self.usage, six, observed, self.belief_samples, self.belief_rng)
                else:
                    foe = complete_sets(state, foe_side, self.usage)
                    self.last_foe_sets = [{"species": state.sides[foe_side].team[i].species, **{k: foe[i][k] for k in
                                           ("moves", "item", "ability", "nature", "points")}} for i in foe]
        if record is not None and legal.sum() > 1:
            used = [int(arrays["actor_moves"][s][a // N_TARGETS]) for s, a in enumerate(atoms) if a < ATOM_SWITCH]
            record.append({"arrays": arrays, "action": pair, "logp": float(log_probs[pair]), "lead": lead(state, side),
                           "moves_used": used})
        if thoughts is not None:
            self.think(state, side, arrays, batch, q, mons, logits, atoms, thoughts)
        return self.pair_choice(slots, atoms, n)

    @staticmethod
    def pair_choice(slots: list[dict], atoms: tuple, n: int) -> str:
        """The simulator choice string for a pair of atoms; the first move that can Mega Evolve does."""
        choices, mega_used = [], False
        for slot, atom in enumerate(atoms):
            choice = slots[slot][atom]
            if choice.startswith("move") and slots[slot].get("mega") and not mega_used and atom != ATOM_LOCKED:
                choice += " mega"
                mega_used = True
            choices.append(choice)
        return ", ".join(choices[:max(n, 1)])

    # ------------------------------------------------------------------ for the match viewer

    def atom_label(self, state: BattleState, side: str, arrays: dict, slot: int, atom: int) -> str:
        mine, foe = state.sides[side], state.sides["p2" if side == "p1" else "p1"]
        name = lambda s, i: (s.team[i].form or s.team[i].species) if i is not None and i >= 0 else None
        if atom >= ATOM_SWITCH and atom < ATOM_PASS:
            return f"Switch to {name(mine, atom - ATOM_SWITCH)}"
        if atom == ATOM_PASS:
            return "Pass"
        if atom == ATOM_LOCKED:
            return "Locked move"
        k, t = divmod(atom, N_TARGETS)
        move = MOVES[int(arrays["actor_moves"][slot][k])]
        target = {TARGET_FOE_A: name(foe, int(arrays["foes"][0])), TARGET_FOE_B: name(foe, int(arrays["foes"][1])),
                  TARGET_ALLY: "ally"}.get(t)
        return f"{move} → {target}" if target else move

    def think(self, state, side, arrays, batch, q, mons, logits, atoms, thoughts: list) -> None:
        """What the network saw and scored for this decision: per-slot marginals, top joint plans, the
        opponent prediction, value and threat. Appends one entry per slot that had a choice to `thoughts`
        and keeps the plans in `self.last_plans`."""
        net = self.net
        with torch.no_grad():
            value = float(net.value_head(q))
            threat = net.threat_features(batch)[0]
            damage = [net.damage_features(batch, s)[0] for s in (0, 1)]
            foe_probs = torch.softmax(net.foe_logits(batch, q, mons)[0], dim=-1)
        probs = torch.softmax(logits, dim=-1).reshape(N_ATOMS, N_ATOMS)
        marginal = [probs.sum(1), probs.sum(0)]
        mine, foe = state.sides[side], state.sides["p2" if side == "p1" else "p1"]
        name = lambda s, i: (s.team[i].form or s.team[i].species) if i is not None and i >= 0 else None
        foes = [name(foe, int(i)) for i in arrays["foes"]]

        top = torch.argsort(probs.flatten(), descending=True)[:8].tolist()
        self.last_plans = [{"first": self.atom_label(state, side, arrays, 0, p // N_ATOMS),
                            "second": self.atom_label(state, side, arrays, 1, p % N_ATOMS),
                            "prob": float(probs.flatten()[p]), "chosen": divmod(p, N_ATOMS) == tuple(atoms)}
                           for p in top if probs.flatten()[p] > 0.005]

        prediction = []
        for f in (0, 1):
            if foes[f] is None:
                continue
            best = torch.argsort(foe_probs[f], descending=True)[:4].tolist()
            label = lambda a: MOVES[a] if a < SWITCH_BASE else f"Switch to {name(foe, a - SWITCH_BASE)}"
            prediction.append({"foe": foes[f], "top": [{"label": label(a), "prob": float(foe_probs[f][a])} for a in best]})

        legal = arrays["atom_mask"]
        for slot in (0, 1):
            if legal[slot].sum() <= 1:
                continue
            options = []
            for atom in torch.argsort(marginal[slot], descending=True).tolist():
                if not legal[slot][atom] or len(options) >= 8:
                    continue
                entry = {"label": self.atom_label(state, side, arrays, slot, atom),
                         "kind": "switch" if ATOM_SWITCH <= atom < ATOM_PASS else "move", "prob": float(marginal[slot][atom])}
                if atom < ATOM_SWITCH:
                    f = damage[slot][int(arrays["actor_moves"][slot][atom // N_TARGETS])].tolist()
                    entry["damage"] = [{"foe": foes[i], "effectiveness": f[i * 5], "pctOfFullHp": f[i * 5 + 1],
                                        "pctOfCurrentHp": f[i * 5 + 2] * 3, "sureKo": bool(f[i * 5 + 3]), "maybeKo": bool(f[i * 5 + 4])}
                                       for i in (0, 1) if foes[i]]
                options.append(entry)
            thoughts.append({
                "kind": "replacement" if int(arrays["kind"]) == REPLACEMENT else "move", "slot": slot,
                "actor": name(mine, int(arrays["actors"][slot])), "foes": foes, "winChance": (value + 1) / 2,
                "options": options, "chosen": self.atom_label(state, side, arrays, slot, atoms[slot]), "target": None,
                "foePrediction": prediction if slot == 0 else [],
                "threat": [{"foe": foes[fi], "target": name(mine, int(arrays["actors"][ti])),
                            "revealedMove": threat[fi * 4 + ti * 2].item() * 3, "guess": threat[fi * 4 + ti * 2 + 1].item() * 3}
                           for fi in (0, 1) for ti in (0, 1) if foes[fi] and arrays["actors"][ti] >= 0]})
