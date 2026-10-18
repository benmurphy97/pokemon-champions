"""Track what a player can see of a battle by reading Showdown protocol lines.

The same code feeds the learning pipeline (replay logs) and live play (the simulator's player
stream), so the policy sees identical inputs in both. It records only public information: HP
(as a fraction), status, stat boosts, and moves/items/abilities once they have been revealed.

    state = BattleState()
    for line in log_lines:
        state.update(line)
"""

import re
from dataclasses import dataclass, field

BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
STATUSES = ["par", "brn", "psn", "tox", "slp", "frz"]
MEGA_SUFFIX = re.compile(r"-Mega(-[XYZ])?$")

WEATHERS = {"sunnyday": "sun", "desolateland": "sun", "raindance": "rain", "primordialsea": "rain",
            "sandstorm": "sand", "snowscape": "snow", "snow": "snow", "hail": "snow"}
TERRAINS = {"electric terrain": "electric", "grassy terrain": "grassy", "misty terrain": "misty",
            "psychic terrain": "psychic"}
SIDE_CONDITIONS = ["tailwind", "reflect", "light screen", "aurora veil", "stealth rock", "spikes",
                   "toxic spikes", "sticky web"]


def base_species(details: str) -> str:
    """'Raichu-Mega-Y, L50, F, shiny' -> 'Raichu'."""
    return MEGA_SUFFIX.sub("", details.split(",")[0].strip())


def parse_position(text: str) -> tuple[str, int | None, str]:
    """'p1a: Nick' -> ('p1', 0, 'Nick'); 'p1: Player' -> ('p1', None, 'Player')."""
    head, _, nickname = text.partition(": ")
    side = head[:2]
    slot = {"a": 0, "b": 1}.get(head[2:3])
    return side, slot, nickname


HP_PATTERN = re.compile(r"^(\d+)(?:/(\d+))?[a-z]*$")  # replays add a colour letter: '45/100y'


def parse_hp(text: str) -> tuple[float, str | None]:
    """'77/164 par' -> (0.47, 'par'); '0 fnt' -> (0.0, 'fnt'). Works for exact HP and percentages."""
    hp, _, status = text.partition(" ")
    current, maximum = HP_PATTERN.match(hp).groups()
    return int(current) / max(int(maximum or 100), 1), status or None


@dataclass
class Mon:
    species: str                       # base species, as listed at team preview
    form: str = ""                     # current form (e.g. after Mega Evolving)
    hp: float = 1.0
    status: str | None = None
    boosts: dict = field(default_factory=lambda: {b: 0 for b in BOOSTS})
    moves: list = field(default_factory=list)   # moves seen so far
    item: str | None = None            # revealed item
    item_gone: bool = False            # eaten, knocked off, etc.
    ability: str | None = None
    fainted: bool = False
    revealed: bool = False             # has been on the field
    mega: bool = False
    turns_active: int = 0
    last_move: str | None = None
    last_move_turn: int = -1    # the turn counter when it was used


@dataclass
class Side:
    team: list = field(default_factory=list)
    active: list = field(default_factory=lambda: [None, None])   # team index per slot
    conditions: set = field(default_factory=set)
    mega_used: bool = False


class BattleState:
    def __init__(self):
        self.sides = {"p1": Side(), "p2": Side()}
        self.turn = 0
        self.weather: str | None = None
        self.terrain: str | None = None
        self.trick_room = False
        self.gravity = False
        self.winner: str | None = None
        self._names: dict = {}   # (side, nickname) -> team index
        self._players: dict = {}

    # ---- lookups -------------------------------------------------------------------------

    def mon_at(self, position: str) -> Mon | None:
        side, slot, nickname = parse_position(position)
        index = self._names.get((side, nickname))
        if index is None and slot is not None:
            index = self.sides[side].active[slot]
        return self.sides[side].team[index] if index is not None else None

    def _index_of(self, side: str, species: str) -> int:
        """Team index for a species, tolerating form changes (Floette-Eternal -> Floette, Mimikyu-Busted)."""
        team = self.sides[side].team
        for i, mon in enumerate(team):
            if mon.species == species:
                return i
        family = [i for i, mon in enumerate(team) if mon.species.split("-")[0] == species.split("-")[0]]
        if len(family) == 1:
            return family[0]
        team.append(Mon(species=species))   # not announced at preview (e.g. Illusion): add it
        return len(team) - 1

    # ---- protocol ------------------------------------------------------------------------

    def update(self, line: str) -> None:
        parts = line.split("|")
        if len(parts) < 2:
            return
        handler = getattr(self, "_on_" + parts[1].lstrip("-").replace("-", "_"), None)
        if handler:
            handler(parts)

    def _on_player(self, p):
        if len(p) > 3 and p[3]:
            self._players[p[2]] = p[3]

    def _on_poke(self, p):
        self.sides[p[2]].team.append(Mon(species=base_species(p[3])))

    def _on_switch(self, p):
        side, slot, nickname = parse_position(p[2])
        index = self._index_of(side, base_species(p[3]))
        mon = self.sides[side].team[index]
        self._names[(side, nickname)] = index
        self.sides[side].active[slot] = index
        mon.form = p[3].split(",")[0].strip()
        mon.hp, status = parse_hp(p[4]) if len(p) > 4 else (1.0, None)
        mon.status = status if status in STATUSES else None
        mon.revealed, mon.turns_active, mon.boosts = True, 0, {b: 0 for b in BOOSTS}

    _on_drag = _on_switch

    def _on_detailschange(self, p):
        mon = self.mon_at(p[2])
        if mon:
            mon.form = p[3].split(",")[0].strip()

    _on_formechange = _on_detailschange

    def _set_hp(self, p):
        mon = self.mon_at(p[2])
        if mon and len(p) > 3:
            mon.hp, status = parse_hp(p[3])
            if status == "fnt":
                self._faint(p[2])
            elif status in STATUSES:
                mon.status = status

    _on_damage = _on_heal = _on_sethp = _set_hp

    def _faint(self, position):
        side, slot, _ = parse_position(position)
        mon = self.mon_at(position)
        if mon:
            mon.hp, mon.fainted, mon.status = 0.0, True, None
        if slot is not None and mon is not None and self.sides[side].active[slot] is not None:
            if self.sides[side].team[self.sides[side].active[slot]] is mon:
                self.sides[side].active[slot] = None

    def _on_faint(self, p):
        self._faint(p[2])

    def _on_status(self, p):
        mon = self.mon_at(p[2])
        if mon and p[3] in STATUSES:
            mon.status = p[3]

    def _on_curestatus(self, p):
        mon = self.mon_at(p[2])
        if mon:
            mon.status = None

    def _on_cureteam(self, p):
        side = p[2][:2]
        for mon in self.sides[side].team:
            mon.status = None

    def _boost(self, p, sign):
        mon = self.mon_at(p[2])
        if mon and p[3] in mon.boosts:
            mon.boosts[p[3]] = max(-6, min(6, mon.boosts[p[3]] + sign * int(p[4])))

    def _on_boost(self, p):
        self._boost(p, +1)

    def _on_unboost(self, p):
        self._boost(p, -1)

    def _on_setboost(self, p):
        mon = self.mon_at(p[2])
        if mon and p[3] in mon.boosts:
            mon.boosts[p[3]] = int(p[4])

    def _on_clearboost(self, p):
        mon = self.mon_at(p[2])
        if mon:
            mon.boosts = {b: 0 for b in BOOSTS}

    def _on_clearnegativeboost(self, p):
        mon = self.mon_at(p[2])
        if mon:
            mon.boosts = {b: max(v, 0) for b, v in mon.boosts.items()}

    def _on_clearallboost(self, p):
        for side in self.sides.values():
            for mon in side.team:
                mon.boosts = {b: 0 for b in BOOSTS}

    def _on_invertboost(self, p):
        mon = self.mon_at(p[2])
        if mon:
            mon.boosts = {b: -v for b, v in mon.boosts.items()}

    def _on_weather(self, p):
        key = p[2].lower().replace(" ", "")
        self.weather = WEATHERS.get(key)

    def _on_fieldstart(self, p):
        name = p[2].replace("move: ", "").lower()
        if name == "trick room":
            self.trick_room = True
        elif name == "gravity":
            self.gravity = True
        elif name in TERRAINS:
            self.terrain = TERRAINS[name]

    def _on_fieldend(self, p):
        name = p[2].replace("move: ", "").lower()
        if name == "trick room":
            self.trick_room = False
        elif name == "gravity":
            self.gravity = False
        elif name in TERRAINS and self.terrain == TERRAINS[name]:
            self.terrain = None

    def _on_sidestart(self, p):
        name = p[3].replace("move: ", "").lower()
        if name in SIDE_CONDITIONS:
            self.sides[p[2][:2]].conditions.add(name)

    def _on_sideend(self, p):
        self.sides[p[2][:2]].conditions.discard(p[3].replace("move: ", "").lower())

    def _on_item(self, p):
        mon = self.mon_at(p[2])
        if mon:
            mon.item, mon.item_gone = p[3], False

    def _on_enditem(self, p):
        mon = self.mon_at(p[2])
        if mon:
            mon.item, mon.item_gone = p[3], True

    def _on_ability(self, p):
        mon = self.mon_at(p[2])
        if mon:
            mon.ability = p[3]

    def _on_mega(self, p):
        mon = self.mon_at(p[2])
        if mon:
            mon.mega = True
            if len(p) > 4:
                mon.item = p[4]
        self.sides[p[2][:2]].mega_used = True

    def _on_move(self, p):
        if len(p) < 4 or any(part.startswith("[from]") for part in p[4:]):
            return
        mon = self.mon_at(p[2])
        if mon:
            if p[3] not in mon.moves:
                mon.moves.append(p[3])
            mon.last_move = p[3]
            mon.last_move_turn = self.turn

    def _on_turn(self, p):
        self.turn = int(p[2])
        for side in self.sides.values():
            for index in side.active:
                if index is not None:
                    side.team[index].turns_active += 1

    def _on_win(self, p):
        self.winner = next((side for side, name in self._players.items() if name == p[2]), None)
