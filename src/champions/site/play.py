"""Play a match yourself, with the model as a coach.

You pilot team A against a bot (greedy-blind by default). Before every decision the coach shows what the model
would do (its joint plans, its prediction of the opponent, and the one-turn search table), then waits for your
choice. The game runs in a background thread; ops_server drives it through `Session`.

    session = Session(ours_text, theirs_text, seed=1)
    state = session.wait()               # the first decision: board, coaching and your menu
    state = session.move("move 1 2, move 3")
"""

import queue
import threading
from pathlib import Path

from champions.sim.arena import load_policy, run_jobs
from champions.site.match_viewer import SEARCH_K, Viewer, describe, find_model

TIMEOUT = 3600          # seconds a decision may wait for you before the coach's choice is played


class Coach(Viewer):
    """Shows its analysis for each decision, then plays the human's choice."""

    def __init__(self, net, session, search_k: int = SEARCH_K):
        super().__init__(net, search_k=search_k)
        self.session = session

    def menu(self, state, side, request) -> list[dict]:
        """Per slot: the legal options in words, and whether that slot may Mega Evolve."""
        arrays, slots = self.last_options
        team = state.sides[side].team
        out = []
        for slot, options in enumerate(slots):
            actor = int(arrays["actors"][slot])
            out.append({"actor": (team[actor].form or team[actor].species) if actor >= 0 else None,
                        "canMega": bool(options.get("mega")),
                        "options": [{"choice": c, "label": self.atom_label(state, side, arrays, slot, atom)}
                                    for atom, c in options.items() if atom != "mega"]})
        return out

    def decide(self, state, side, request, record=None, thoughts=None):
        if request.get("teamPreview"):
            return "default"          # the rule-based preview brings four and picks the leads
        step = self.look(state, side, request)
        step["menu"] = self.menu(state, side, request)
        step["slots"] = len(request.get("forceSwitch") or request.get("active") or [0, 0])
        step["forced"] = bool(request.get("forceSwitch"))
        self.steps.append(step)
        return self.session.ask(step)


class Session:
    def __init__(self, ours: str, theirs: str, seed: int = 1, opponent: str = "greedy-blind", model: Path | None = None):
        self.model = (model or find_model()).name
        self.coach = Coach(load_policy(model or find_model()), self)
        self.pending, self.result, self.error = None, None, None
        self.changed = threading.Condition()
        self.answers: queue.Queue = queue.Queue()
        self.abandoned = False
        job = {"id": "play", "a": ours, "b": theirs, "seed": seed, "agentA": "neural", "agentB": opponent}
        threading.Thread(target=self.run, args=(job,), daemon=True).start()

    def run(self, job):
        try:
            result = run_jobs([job], self.coach, workers=1, observer=self.coach.observe)[0]
            self.result = {"winner": {"A": "you", "B": "bot"}.get(result["winner"], "draw"), "turns": result["turns"],
                           "finalEvents": describe(result.get("tail", []))}
        except Exception as err:
            self.error = repr(err)
        with self.changed:
            self.pending = None
            self.changed.notify_all()

    def ask(self, step: dict) -> str:
        """Called from the game thread: publish the decision and wait for the human's answer."""
        if self.abandoned:
            return step["choice"]
        with self.changed:
            self.pending = step
            self.changed.notify_all()
        try:
            choice = self.answers.get(timeout=TIMEOUT)
        except queue.Empty:
            choice = None
        step["played"] = choice or step["choice"]
        return step["played"]

    def state(self) -> dict:
        done = self.coach.steps[:-1] if self.pending else self.coach.steps
        return {"model": self.model, "pending": self.pending, "result": self.result, "error": self.error,
                "history": [{"turn": s["turn"], "events": s["events"], "suggested": s["choice"], "played": s.get("played")} for s in done]}

    def wait(self, timeout: float = 120) -> dict:
        """Block until a decision is pending or the game is over."""
        with self.changed:
            self.changed.wait_for(lambda: self.pending is not None or self.result or self.error, timeout=timeout)
        return self.state()

    def move(self, choice: str, timeout: float = 120) -> dict:
        with self.changed:
            self.pending = None
        self.answers.put(choice)
        return self.wait(timeout)

    def abandon(self):
        """Let the game finish on the coach's own choices (a new game replaces this one)."""
        self.abandoned = True
        self.answers.put(None)
