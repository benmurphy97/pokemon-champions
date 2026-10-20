"""What we believe about the opponent's hidden sets, updated as the battle reveals them.

    corpus = Corpus.load()                                  # every tournament team we have (about 1,800)
    beliefs = infer(corpus, usage, species_six, observed)   # one dict per opponent Pokémon

`observed[i]` is what has been revealed about their i-th Pokémon: {"moves": [...], "item": str | None,
"ability": str | None}. The estimate is a posterior over whole sets, not a table of independent shares:

  * Hypotheses for a Pokémon are the real sets that tournament players ran on it.
  * The team context sets the prior. Teams that share more of the opponent's six species count for more
    (weight exp(beta * overlap)), so a Farigiraf team's Pokémon look like Trick Room team Pokémon.
  * Revealed facts are the likelihood: a set that lacks a revealed move, item or ability is nearly ruled out
    (EPSILON, not zero, so a set that is not in the corpus cannot zero everything).
  * Item Clause: an item revealed on one of their Pokémon is removed from the others' hypotheses.
  * Where the evidence is thin (small effective sample), the estimate is blended with the ladder usage shares.

Marginals come out per move, item, ability and spread; revealed facts are 1.0 and, once four moves are known,
every other move is 0. Not used yet: speed order and damage dealt, which also narrow the spread.
"""

import json
import math
from collections import defaultdict
from functools import cache
from pathlib import Path

from champions.config import REGULATION
from champions.teams.generate import usage_key
from champions.teams.parse import parse_team
from champions.teams.stats import compute_stats
from champions.training.hindsight import _BASE_STATS

ROOT = Path(__file__).resolve().parents[3]
TEAM_DIR = ROOT / "data" / "teams" / REGULATION.lower()
EPSILON = 0.01          # likelihood of a set that contradicts something revealed
BETA = 1.0              # prior weight per species shared with the opponent's six
BACKOFF = 3.0           # pseudo-count of ladder usage blended in (effective sample size is the other weight)
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_POINT_KEY = dict(zip(["HP", "Atk", "Def", "SpA", "SpD", "Spe"], STAT_KEYS))


@cache
def _mega_stones() -> frozenset:
    items = json.loads((ROOT / "data" / "reference" / "items.json").read_text())
    return frozenset(name for name, v in items.items() if v.get("megaStone"))


def family(name: str) -> str:
    """The species a Pokémon is listed under before Mega Evolving: 'Garchomp-Mega-Z' -> 'Garchomp'."""
    return name.split("-Mega")[0]


def signature(s: dict) -> tuple:
    return (tuple(sorted(s["moves"])), s["item"], s.get("ability"), s.get("nature"), tuple(sorted(s["evs"].items())))


class Corpus:
    """Real tournament sets, indexed by species, each remembering which team it came from."""

    def __init__(self, teams: list[list[dict]]):
        self.teams = teams
        self.species = [{family(s["name"]) for s in team} for team in teams]
        self.by_species: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for index, team in enumerate(teams):
            for s in team:
                self.by_species[family(s["name"])].append((index, s))
        tokens: dict[str, set] = defaultdict(set)
        for key in self.by_species:
            tokens[key.split("-")[0]].add(key)
        self._first_token = {t: next(iter(keys)) for t, keys in tokens.items() if len(keys) == 1}

    def key(self, species: str) -> str:
        """The corpus species for a battle name: 'Floette-Eternal' -> 'Floette' when that is the only Floette."""
        base = family(species)
        return base if base in self.by_species else self._first_token.get(base.split("-")[0], base)

    @classmethod
    def load(cls, exclude: set[str] = frozenset(), like: list[str] = ()) -> "Corpus":
        """All parsed teams in data/teams. `exclude` holds paste ids (file names) to leave out. `like` holds team texts:
        every corpus team sharing 3 or more identical sets with one of them is left out too, so those teams are
        genuinely unseen (tournament builds recur, so most teams have near-copies)."""
        near = [{signature(s) for s in parse_team(text)} for text in like]
        teams = []
        for path in sorted(TEAM_DIR.glob("*.txt")):
            if path.stem in exclude:
                continue
            try:
                team = parse_team(path.read_text())
            except Exception:
                continue
            if len(team) == 6 and all(s["moves"] for s in team):
                mine = {signature(s) for s in team}
                if any(len(mine & other) >= 3 for other in near):
                    continue
                teams.append(team)
        return cls(teams)


def _blend(post: dict[str, float], ladder: dict[str, float], n_eff: float, known: list[str], slots: int) -> dict[str, float]:
    """Posterior shares blended with ladder shares (both rescaled to `slots` expected members), known facts pinned to 1."""
    def scaled(shares):
        pool = {k: v for k, v in shares.items() if k not in known}
        total = sum(pool.values())
        return {k: v * slots / total for k, v in pool.items()} if total > 0 and slots > 0 else {}
    a, b = scaled(post), scaled(ladder)
    if not b:
        return a
    out = {k: (n_eff * a.get(k, 0.0) + BACKOFF * b.get(k, 0.0)) / (n_eff + BACKOFF) for k in set(a) | set(b)}
    return out


def _weights(corpus: Corpus, base: str, six: list[str], observed: dict, taken_items: set, beta: float):
    """Candidate sets for a species and how much each is worth given the team context and what has been revealed."""
    others = {corpus.key(x) for x in six} - {base}
    known_moves = list(dict.fromkeys(observed.get("moves") or []))
    item, ability = observed.get("item"), observed.get("ability")
    weights, sets = [], []
    for team_index, s in corpus.by_species.get(base, []):
        w = math.exp(beta * len(corpus.species[team_index] & others))
        if not all(m in s["moves"] for m in known_moves):
            w *= EPSILON
        if item:
            w *= 1.0 if s["item"] == item else EPSILON
        elif s["item"] in taken_items:
            w *= EPSILON
        if ability and s.get("ability") != ability:
            w *= EPSILON
        weights.append(w)
        sets.append(s)
    return sets, weights


def infer_one(corpus: Corpus, usage: dict, species: str, six: list[str], observed: dict, taken_items: set, beta: float,
              backoff: bool = True) -> dict:
    base = corpus.key(species)
    known_moves = list(dict.fromkeys(observed.get("moves") or []))
    item, ability = observed.get("item"), observed.get("ability")

    sets, weights = _weights(corpus, base, six, observed, taken_items, beta)
    total = sum(weights)
    n_eff = total ** 2 / sum(w * w for w in weights) if total > 0 else 0.0
    # n_eff counts the hypotheses that still fit; a set that contradicts the evidence barely counts
    post = [w / total for w in weights] if total > 0 else []

    def marginal(values: list[list[str]]) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for p, vs in zip(post, values):
            for v in set(vs):
                out[v] += p
        return out

    entry = usage.get(usage_key(base, False, usage) or "")
    ladder = {k: (dict(entry[k]) if entry else {}) for k in ("moves", "items", "abilities")}
    use_ladder = backoff

    def finish(shares, ladder_shares, known, slots, rows):
        mixed = _blend(shares, ladder_shares if use_ladder else {}, n_eff if use_ladder else 1.0, known, slots)
        out = [{"name": k, "p": 1.0, "known": True} for k in known]
        out += [{"name": k, "p": round(min(v, 0.999), 3), "known": False} for k, v in sorted(mixed.items(), key=lambda kv: -kv[1])]
        return out[:rows]

    move_slots = max(0, 4 - len(known_moves))
    moves = finish(marginal([s["moves"] for s in sets]), ladder["moves"], known_moves, move_slots, 8) if move_slots else \
        [{"name": k, "p": 1.0, "known": True} for k in known_moves]
    items = [{"name": item, "p": 1.0, "known": True}] if item else \
        finish(marginal([[s["item"]] for s in sets if s["item"] not in taken_items]), ladder["items"], [], 1, 3)
    abilities = [{"name": ability, "p": 1.0, "known": True}] if ability else \
        finish(marginal([[s.get("ability")] for s in sets if s.get("ability")]), ladder["abilities"], [], 1, 2)

    spreads: dict[tuple, float] = defaultdict(float)
    for p, s in zip(post, sets):
        spreads[(s.get("nature"), tuple(sorted((k, v) for k, v in s["evs"].items())))] += p
    base_stats = _BASE_STATS.get(base)
    top, expected = [], None
    if base_stats and spreads:
        ranked = sorted(spreads.items(), key=lambda kv: -kv[1])
        for (nature, evs), p in ranked[:3]:
            points = {_POINT_KEY[k]: v for k, v in evs}
            top.append({"nature": nature, "points": points, "p": round(p, 3), "stats": compute_stats(base_stats, nature, points)})
        expected = {k: 0.0 for k in STAT_KEYS}
        for (nature, evs), p in spreads.items():
            stats = compute_stats(base_stats, nature or "Hardy", {_POINT_KEY[k]: v for k, v in evs})
            for k in STAT_KEYS:
                expected[k] += p * stats[k]
        expected = {k: round(v) for k, v in expected.items()}
    return {"moves": moves, "items": items, "abilities": abilities, "spread": top[0] if top else None, "spreads": top,
            "expectedStats": expected, "evidence": round(n_eff, 1), "hypotheses": len(sets)}


def with_mega_items(corpus: Corpus, observed: list[dict]) -> list[dict]:
    """A Pokémon seen in its Mega form is holding its Mega Stone: fill that in as a revealed item (the log does not say)."""
    out = []
    for o in observed:
        form = o.get("form") or ""
        if "-Mega" in form and not o.get("item"):
            stones = defaultdict(int)
            for _, s in corpus.by_species.get(corpus.key(form), []):
                if s["name"] == form and s["item"] in _mega_stones():
                    stones[s["item"]] += 1
            if stones:
                o = {**o, "item": max(stones, key=stones.get)}
        out.append(o)
    return out


def infer(corpus: Corpus, usage: dict, six: list[str], observed: list[dict], beta: float = BETA, backoff: bool = True) -> list[dict]:
    """Beliefs for each of their six, given everything revealed so far (Item Clause across the six)."""
    observed = with_mega_items(corpus, observed)
    revealed_items = [o.get("item") for o in observed if o.get("item")]
    out = []
    for species, obs in zip(six, observed):
        taken = {i for i in revealed_items if i != obs.get("item")}
        out.append(infer_one(corpus, usage, species, six, obs, taken, beta, backoff))
    return out


@cache
def shared_corpus() -> Corpus:
    """The corpus agents use. Left out: near-copies of every team an agent is ever tested against (the held-out
    split of the web teams, and the training pool's own teams), so a test opponent is never one the belief has seen.
    Tournament builds recur, so this removes most teams that look like them as well."""
    web = json.loads((ROOT / "results" / "web_teams.json").read_text())
    pool = json.loads((TEAM_DIR / "training_teams.json").read_text())
    return Corpus.load(like=[t["text"] for t in web if t.get("split") == "eval"] + [t["text"] for t in pool])


def observed_from(state, side: str) -> tuple[list[str], list[dict]]:
    """Their six species and what the battle has revealed about each (moves, item, ability)."""
    team = state.sides[side].team[:6]
    return ([m.species for m in team],
            [{"moves": list(m.moves[:4]), "item": m.item or None, "ability": m.ability or None, "form": m.form or m.species} for m in team])


def sample_assignments(corpus: Corpus, usage: dict, six: list[str], observed: list[dict], n: int, rng, beta: float = BETA) -> list[list[dict]]:
    """`n` plausible complete assignments of sets to their six, drawn from the posterior (for the search to play
    against). Each set is {"species", "moves", "item", "ability", "nature", "points"}; revealed facts always hold.
    A Pokémon with no matching real sets gets its ladder-usage set instead."""
    observed = with_mega_items(corpus, observed)
    revealed = [o.get("item") for o in observed if o.get("item")]
    pools = []
    for species, obs in zip(six, observed):
        taken = {i for i in revealed if i != obs.get("item")}
        sets, weights = _weights(corpus, corpus.key(species), six, obs, taken, beta)
        pools.append((sets, weights))
    out = []
    for _ in range(n):
        used, assignment = set(revealed), []
        for species, obs, (sets, weights) in zip(six, observed, pools):
            known = list(dict.fromkeys(obs.get("moves") or []))
            chosen = None
            for _try in range(5):
                if not sets:
                    break
                candidate = rng.choices(sets, weights=weights)[0]
                if obs.get("item") or candidate["item"] not in used:
                    chosen = candidate
                    break
            entry = usage.get(usage_key(family(species), False, usage) or "")
            if chosen is None and entry is None:
                assignment.append({"species": species, "moves": known, "item": obs.get("item"), "ability": obs.get("ability")})
                continue
            if chosen is not None:
                moves = known + [m for m in chosen["moves"] if m not in known]
                item = obs.get("item") or chosen["item"]
                ability = obs.get("ability") or chosen.get("ability")
                nature, points = chosen.get("nature"), {_POINT_KEY[k]: v for k, v in chosen["evs"].items()}
            else:                # no real sets for this species: the most common ladder choices
                moves = known + [m for m, _ in entry["moves"] if m not in known]
                item = obs.get("item") or entry["items"][0][0]
                ability = obs.get("ability") or entry["abilities"][0][0]
                nature, spread, _ = entry["spreads"][0]
                points = {_POINT_KEY[k]: v for k, v in spread.items()}
            used.add(item)
            assignment.append({"species": species, "moves": moves[:4], "item": item, "ability": ability, "nature": nature, "points": points})
        out.append(assignment)
    return out
