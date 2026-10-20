"""Live status for the website's operating room.

    PYTHONPATH=src .venv/bin/python -m champions.site.ops_server   # http://localhost:8765/api/ops

Also runs single matches for the viewer: /api/match/options and /api/match?ours=ID&opp=ID&seed=N
(needs PyTorch, so start it with the .venv python).

Reads the training logs in results/ and the running processes, and serves them as JSON. Nothing
here changes anything: it only looks. The website polls it every few seconds (web/vite.config.js
forwards /api to this port during development).
"""

import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
KNOWN = ROOT / "data" / "teams" / "m-c" / "known_good_tr.json"
RATINGS = ROOT / "data" / "ratings" / "m-c" / "teams.json"
PORT = 8765
WATCHED = {"training.selfplay": "self-play training", "training.imitation": "imitation learning", "scrapers.metavgc": "team importer",
           "scrapers.labmaus": "team importer", "scrapers.limitless": "team importer", "teams.rate": "team rating",
           "scrapers.replays": "replay download", "sim.arena": "arena test", "evaluation.gates": "gate check"}


def read_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:      # a line still being written
            pass
    return rows


def processes() -> list[dict]:
    """Running project jobs: what they are, how long they have run, how hard they work the CPU."""
    out = subprocess.run(["ps", "-axo", "pid=,etime=,pcpu=,rss=,command="], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        match = re.match(r"\s*(\d+)\s+(\S+)\s+([\d.]+)\s+(\d+)\s+(.*)", line)
        if not match or "champions." not in match.group(5):
            continue
        pid, elapsed, cpu, rss, command = match.groups()
        for key, label in WATCHED.items():
            if f"champions.{key} " in command + " ":
                found.append({"pid": int(pid), "kind": key, "label": label, "elapsed": elapsed, "cpu": float(cpu),
                              "memoryMb": round(int(rss) / 1024), "command": command.split("champions.", 1)[1]})
    return found


def run_summary(name: str, path: Path, live: bool) -> dict:
    rows = read_log(path)
    evals = [{"iteration": r["iteration"], **r["eval"]} for r in rows if "eval" in r]
    return {"name": name, "live": live, "updated": path.stat().st_mtime, "iterations": rows,
            "evals": evals}


def status() -> dict:
    runs = []
    archived = sorted(RESULTS.glob("rl_log_run*.jsonl"), key=lambda p: int(re.search(r"run(\d+)", p.name).group(1)))
    for path in archived:
        runs.append(run_summary(re.search(r"run\d+", path.name).group(0), path, live=False))
    current = RESULTS / "rl_log.jsonl"
    if current.exists():
        runs.append(run_summary("current", current, live=True))

    jobs = processes()
    training = next((p for p in jobs if p["kind"] == "training.selfplay"), None)
    total = 60
    if training:
        match = re.search(r"--iterations (\d+)", training["command"])
        total = int(match.group(1)) if match else 60
    league = []
    for path in sorted((MODELS / "league").glob("snap*.pt"), key=lambda p: int(re.search(r"\d+", p.name).group(0))):
        league.append({"name": path.stem, "iteration": int(re.search(r"\d+", path.name).group(0)), "modified": path.stat().st_mtime})
    known = json.loads(KNOWN.read_text()) if KNOWN.exists() else []
    ratings = json.loads(RATINGS.read_text())["teams"] if RATINGS.exists() else {}
    roster = [{"id": t["id"], "species": t["species"], "placement": t["placement"], "tournament": t["tournament"],
               "rating": ratings.get(t["id"])} for t in known]
    return {"now": time.time(), "load": os.getloadavg(), "cores": os.cpu_count(), "jobs": jobs,
            "training": {"running": training is not None, "totalIterations": total,
                         "command": training["command"] if training else None},
            "runs": runs, "roster": roster, "league": league}


MODELS = ROOT / "models"
PLAN = ROOT / "docs" / "plan_status.json"
GATES = RESULTS / "gates.json"
_model_cache: dict = {}


def models_info() -> list[dict]:
    """Every saved model: its settings, what it was trained with, and its recorded scores."""
    import torch      # only the venv Python has it
    out = []
    for path in sorted(MODELS.glob("*.pt"), key=lambda p: -p.stat().st_mtime):
        key = (path.name, path.stat().st_mtime)
        if key not in _model_cache:
            saved = torch.load(path, map_location="cpu")
            state = saved["state"]
            config = saved.get("config") or {"dim": saved.get("dim"), "layers": saved.get("layers")}
            partner = state.get("partner_move.weight")
            metrics = saved.get("metrics") or {}
            check = saved.get("eval") or {}
            _model_cache[key] = {
                "name": path.name, "sizeMb": round(path.stat().st_size / 1e6, 1), "modified": path.stat().st_mtime,
                "config": config, "parameters": sum(t.numel() for t in state.values() if t.dtype.is_floating_point),
                "encodingVersion": saved.get("encoding_version"),
                "jointChoice": bool(partner is not None and float(partner.abs().sum()) > 0),
                "imitationTop1": metrics.get("top1"), "iteration": saved.get("iteration"),
                "heldOut": {"policy": check.get("policy"), "baseline": check.get("baseline")} if check else None}
        out.append(_model_cache[key])
    return out


def plan_status() -> dict:
    plan = json.loads(PLAN.read_text()) if PLAN.exists() else {"frozen": [], "steps": [], "gates": [], "rules": []}
    plan["results"] = json.loads(GATES.read_text()) if GATES.exists() else []
    return plan


WEB_TEAMS = RESULTS / "web_teams.json"
match_lock = threading.Lock()      # one simulated match at a time


def match_teams() -> tuple[list[dict], list[dict]]:
    """(our known-good teams, opponents to pick from: held-out tournament teams with the best simulated records)."""
    ours = json.loads(KNOWN.read_text()) if KNOWN.exists() else []
    ratings = json.loads(RATINGS.read_text())["teams"] if RATINGS.exists() else {}
    pool = json.loads(WEB_TEAMS.read_text()) if WEB_TEAMS.exists() else []
    rate = lambda t: ratings.get(t["id"], {"wins": 0, "games": 1})
    opponents = sorted((t for t in pool if t["split"] == "eval"), key=lambda t: -rate(t)["wins"] / rate(t)["games"])[:60]
    return ours, opponents


def match_options() -> dict:
    ours, opponents = match_teams()
    brief = lambda t: {"id": t["id"], "species": t["species"], "label": f"{t['placement'] or '–'} · {t['tournament']}"}
    models = [m["name"] for m in models_info()]
    return {"ours": [brief(t) for t in ours], "opponents": [brief(t) for t in opponents], "models": models,
            "previews": [{"id": "rule", "label": "Rule-based (greedy)"}]}


def match(query: dict) -> dict:
    from champions.site.match_viewer import run_match      # imports PyTorch, so only when a match is asked for
    ours, opponents = match_teams()
    by_id = {t["id"]: t for t in ours + opponents}
    a, b = by_id[query["ours"][0]], by_id[query["opp"][0]]
    seed = int(query.get("seed", ["1"])[0])
    with match_lock:
        result = run_match(a["text"], b["text"], seed=seed, temperature=float(query.get("temp", ["0"])[0]),
                           model=(MODELS / query["model"][0]) if "model" in query else None,
                           search_k=int(query.get("search", ["8"])[0]))
    return {**result, "ours": a["species"], "theirs": b["species"]}


session = None      # the one game being played by hand (site/play.py)


def play(path: str, query: dict) -> dict:
    """/api/play/start?ours&opp&seed&model starts a game (replacing any other), /api/play/move?choice plays
    your choice, /api/play/state reads where things are. Each returns the session state."""
    global session
    from champions.site.play import Session        # imports PyTorch, so only when asked
    if path == "/api/play/start":
        ours, opponents = match_teams()
        by_id = {t["id"]: t for t in ours + opponents}
        a, b = by_id[query["ours"][0]], by_id[query["opp"][0]]
        if session:
            session.abandon()
        session = Session(a["text"], b["text"], seed=int(query.get("seed", ["1"])[0]),
                          model=(MODELS / query["model"][0]) if "model" in query else None)
        session.teams = {"ours": a["species"], "theirs": b["species"]}
        return {**session.wait(), **session.teams}
    if session is None:
        return {"pending": None, "result": None, "error": None, "history": []}
    if path == "/api/play/move":
        return {**session.move(query["choice"][0]), **session.teams}
    return {**session.state(), **session.teams}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/ops":
                payload = status()
            elif url.path == "/api/plan":
                payload = {**plan_status(), "models": models_info()}
            elif url.path == "/api/match/options":
                payload = match_options()
            elif url.path == "/api/match":
                payload = match(parse_qs(url.query))
            elif url.path.startswith("/api/play/"):
                payload = play(url.path, parse_qs(url.query))
            else:
                self.send_error(404)
                return
        except Exception as err:      # show the problem in the page instead of a dead connection
            payload = {"error": repr(err)}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    print(f"operating room feed on http://localhost:{PORT}/api/ops", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
