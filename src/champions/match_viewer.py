"""Play one match with the learned player and record what it was thinking at every step.

    PYTHONPATH=src .venv/bin/python -m champions.match_viewer          # prints a summary

Used by the operating room's "live surgery" panel through ops_server. The learned player pilots
team A against a rule-based opponent (greedy-blind); each decision is stored with the board, the
scores it gave every option, the damage numbers it was shown, and which target it preferred.
"""

import json
import re
from pathlib import Path

from champions.agent import NeuralAgent
from champions.arena import load_policy, run_jobs
from champions.battle_state import BattleState, base_species

ROOT = Path(__file__).resolve().parents[2]
MODEL_CANDIDATES = [ROOT / "models" / "rl_policy.pt", ROOT / "models" / "il_policy_v2.pt"]


def mon_name(token: str) -> str:
    return token.split(": ", 1)[-1]


def describe(lines: list[str], me: str = "p1") -> list[dict]:
    """Protocol lines as short readable events: {"text", "tone": ours | foe | field | turn}."""
    out = []
    tone = lambda token: "ours" if token.startswith(me) else "foe"
    for line in lines:
        p = line.split("|")
        kind = p[1] if len(p) > 1 else ""
        try:
            if kind == "turn":
                out.append({"text": f"Turn {p[2]}", "tone": "turn"})
            elif kind == "move":
                target = f" on {mon_name(p[4])}" if len(p) > 4 and p[4] and not p[4].startswith("[") else ""
                out.append({"text": f"{mon_name(p[2])} used {p[3]}{target}", "tone": tone(p[2])})
            elif kind in ("switch", "drag"):
                out.append({"text": f"{'We' if p[2].startswith(me) else 'Foe'} sent out {p[3].split(',')[0]}", "tone": tone(p[2])})
            elif kind == "-damage":
                hp = p[3].split(" ")[0]
                num, _, den = hp.partition("/")
                pct = f"{round(100 * int(num) / int(den))}%" if den else "0%"
                out.append({"text": f"{mon_name(p[2])} → {pct} HP" + (f" ({p[4].strip('[]').replace('from: ', '')})" if len(p) > 4 and p[4] else ""), "tone": tone(p[2])})
            elif kind == "-heal":
                out.append({"text": f"{mon_name(p[2])} healed", "tone": tone(p[2])})
            elif kind == "faint":
                out.append({"text": f"{mon_name(p[2])} fainted", "tone": tone(p[2])})
            elif kind == "-crit":
                out.append({"text": "A critical hit!", "tone": "field"})
            elif kind == "-supereffective":
                out.append({"text": "It's super effective", "tone": "field"})
            elif kind == "-resisted":
                out.append({"text": "It's not very effective", "tone": "field"})
            elif kind == "-immune":
                out.append({"text": f"{mon_name(p[2])} is immune", "tone": tone(p[2])})
            elif kind == "-miss":
                out.append({"text": f"{mon_name(p[2])}'s attack missed", "tone": tone(p[2])})
            elif kind in ("-boost", "-unboost"):
                out.append({"text": f"{mon_name(p[2])} {p[3]} {'+' if kind == '-boost' else '-'}{p[4]}", "tone": tone(p[2])})
            elif kind == "-status":
                out.append({"text": f"{mon_name(p[2])} is now {p[3]}", "tone": tone(p[2])})
            elif kind == "-weather" and p[2] != "none":
                out.append({"text": f"Weather: {p[2]}", "tone": "field"})
            elif kind == "-fieldstart":
                out.append({"text": f"{p[2].replace('move: ', '')} began", "tone": "field"})
            elif kind == "-fieldend":
                out.append({"text": f"{p[2].replace('move: ', '')} ended", "tone": "field"})
            elif kind == "-sidestart":
                out.append({"text": f"{p[3].replace('move: ', '')} on {'our' if p[2].startswith(me) else 'their'} side", "tone": "field"})
            elif kind in ("-mega", "detailschange"):
                out.append({"text": f"{mon_name(p[2])} Mega Evolved", "tone": tone(p[2])})
            elif kind in ("-ability", "-item", "-enditem"):
                out.append({"text": f"{mon_name(p[2])}: {p[3]}", "tone": tone(p[2])})
            elif kind == "win":
                out.append({"text": f"{p[2]} wins", "tone": "turn"})
        except (IndexError, ValueError):
            continue
    return out


def snapshot(state: BattleState, side: str, brought: list[int]) -> dict:
    """The board. Our side is only the four we brought; theirs is their six, with `seen` marking who has appeared."""
    other = "p2" if side == "p1" else "p1"

    def team(s, only=None):
        return [{"name": m.form or m.species, "hp": round(m.hp, 3), "fainted": m.fainted, "status": m.status, "seen": m.revealed,
                 "active": s.active.index(i) if i in s.active else None,
                 "boosts": {k: v for k, v in m.boosts.items() if v}, "item": m.item, "ability": m.ability,
                 "moves": m.moves} for i, m in enumerate(s.team[:6]) if only is None or i in only]
    return {"turn": state.turn, "weather": state.weather, "terrain": state.terrain, "trickRoom": state.trick_room,
            "ours": team(state.sides[side], set(brought)), "foe": team(state.sides[other]),
            "ourConditions": sorted(state.sides[side].conditions), "foeConditions": sorted(state.sides[other].conditions)}


class Viewer(NeuralAgent):
    def __init__(self, net, temperature: float = 0.0, seed: int = 0):
        super().__init__(net, temperature, seed)
        self.steps, self.lines = [], []

    def observe(self, job_id, side, lines):
        self.lines += lines

    def decide(self, state, side, request, record=None, thoughts=None):
        if request.get("teamPreview"):
            return super().decide(state, side, request, record)
        thoughts = []
        plans = self.plan(state, side, request)
        choice = super().decide(state, side, request, record, thoughts=thoughts)
        brought = [state._index_of(side, base_species(p["details"])) for p in request["side"]["pokemon"]]
        self.steps.append({"turn": state.turn, "events": describe(self.lines), "board": snapshot(state, side, brought),
                           "thoughts": thoughts, "plans": plans, "choice": choice})
        self.lines = []
        return choice


def find_model() -> Path:
    return next(p for p in MODEL_CANDIDATES if p.exists())


def run_match(ours: str, theirs: str, seed: int = 1, temperature: float = 0.0, opponent: str = "greedy-blind") -> dict:
    """Play one game (team texts, our side piloted by the learned player) and return every step."""
    model = find_model()
    viewer = Viewer(load_policy(model), temperature=temperature, seed=seed)
    job = {"id": "viewer", "a": ours, "b": theirs, "seed": seed, "agentA": "neural", "agentB": opponent}
    result = run_jobs([job], viewer, workers=1, observer=viewer.observe)[0]
    final = describe(result.get("tail", []))
    winner = {"A": "us", "B": "foe"}.get(result["winner"], "draw")
    return {"model": model.name, "seed": seed, "temperature": temperature, "opponent": opponent, "winner": winner,
            "turns": result["turns"], "steps": viewer.steps, "finalEvents": final}


if __name__ == "__main__":
    known = json.loads((ROOT / "data" / "teams" / "m-c" / "known_good_tr.json").read_text())
    match = run_match(known[0]["text"], known[1]["text"], seed=3)
    print(match["model"], match["winner"], match["turns"], "turns,", len(match["steps"]), "steps")
    step = match["steps"][0]
    print(json.dumps(step["thoughts"][0], indent=1)[:1500])
