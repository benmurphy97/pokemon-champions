"""
Stage 4: Post-Game Evaluation — Battle Log Parser

Parses a Showdown battle log (text format) into a structured record that the
metrics module can score.  The log format used by our local Showdown server
matches the public replay format at replay.pokemonshowdown.com.

Public API
----------
parse_battle_log(log_text)  → BattleRecord
parse_log_file(path)        → BattleRecord

All downstream analytics (metrics.py, GA fitness, Claude API commentary) consume
BattleRecord — keep this parser as the single source of truth for log parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnAction:
    """One action taken by one Pokémon on one turn."""
    turn: int
    player: int           # 1 or 2
    pokemon: str          # species name
    action_type: str      # "move" | "switch" | "mega"
    detail: str           # move name or incoming Pokémon name


@dataclass
class DamageEvent:
    """A single damage or heal event."""
    turn: int
    target_player: int
    target_pokemon: str
    hp_before: float      # fraction 0–1
    hp_after: float       # fraction 0–1
    source: Optional[str] = None   # move or effect that caused it


@dataclass
class FaintEvent:
    turn: int
    player: int
    pokemon: str


@dataclass
class MonRecord:
    """Per-Pokémon aggregated stats across one battle."""
    species: str
    player: int
    turns_active: int = 0
    damage_dealt: float = 0.0    # sum of HP fractions removed from opponents
    damage_taken: float = 0.0    # sum of HP fractions lost
    kos_scored: int = 0
    fainted: bool = False
    turn_fainted: Optional[int] = None


@dataclass
class BattleRecord:
    """Fully parsed representation of one Showdown battle."""
    battle_id: str
    format: str
    winner: Optional[int]        # 1 or 2, None if no |win| line found
    total_turns: int
    player1: str                 # username
    player2: str
    actions: list[TurnAction] = field(default_factory=list)
    damage_events: list[DamageEvent] = field(default_factory=list)
    faint_events: list[FaintEvent] = field(default_factory=list)
    weather_turns: dict[str, int] = field(default_factory=dict)   # weather → turns active
    field_turns: dict[str, int] = field(default_factory=dict)     # field → turns active
    mons: dict[str, MonRecord] = field(default_factory=dict)      # species → MonRecord

    @property
    def mons_p1(self) -> list[MonRecord]:
        return [m for m in self.mons.values() if m.player == 1]

    @property
    def mons_p2(self) -> list[MonRecord]:
        return [m for m in self.mons.values() if m.player == 2]


# ---------------------------------------------------------------------------
# Regex patterns (Showdown log format)
# ---------------------------------------------------------------------------

_RE_PLAYER   = re.compile(r"^\|player\|p(\d)\|([^|]*)")
_RE_TURN     = re.compile(r"^\|turn\|(\d+)$")
_RE_WIN      = re.compile(r"^\|win\|(.+)$")
_RE_MOVE     = re.compile(r"^\|move\|p(\d)[ab]?: ?([^|]+)\|([^|]+)")
_RE_SWITCH   = re.compile(r"^\|switch\|p(\d)[ab]?: ?([^|]+)\|([^|]+)\|(\d+)/(\d+)")
_RE_DRAG     = re.compile(r"^\|drag\|p(\d)[ab]?: ?([^|]+)\|([^|]+)\|(\d+)/(\d+)")
_RE_MEGA     = re.compile(r"^\|-mega\|p(\d)[ab]?: ?([^|]+)\|")
_RE_FAINT    = re.compile(r"^\|faint\|p(\d)[ab]?: ?(.+)$")
_RE_DAMAGE   = re.compile(r"^\|-damage\|p(\d)[ab]?: ?([^|]+)\|(\d+)/(\d+)")
_RE_HEAL     = re.compile(r"^\|-heal\|p(\d)[ab]?: ?([^|]+)\|(\d+)/(\d+)")
_RE_WEATHER  = re.compile(r"^\|-weather\|(\w+)")
_RE_FIELD_S  = re.compile(r"^\|-fieldstart\|move: ?([^|]+)")
_RE_FIELD_E  = re.compile(r"^\|-fieldend\|move: ?([^|]+)")
_RE_FORMAT   = re.compile(r"^\|j\|☆?(.+)$")


def _species(raw: str) -> str:
    """Normalise species name from log lines (strip level, gender, forme suffixes)."""
    return raw.strip().rstrip("*").split(",")[0].strip()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_battle_log(log_text: str, battle_id: str = "") -> BattleRecord:
    """
    Parse a complete Showdown battle log string into a BattleRecord.

    TODO: Expand tracking for:
      - Stat boost/unboost events (for opponent model in Stage 3)
      - Terrain turn counting
      - Item consumption events (for GA fitness signals)
      - Z-move / dynamax detection (not in current format but planned)
    """
    record = BattleRecord(
        battle_id=battle_id,
        format="",
        winner=None,
        total_turns=0,
        player1="p1",
        player2="p2",
    )

    # Tracks current HP for each mon (player, species) → fraction
    current_hp: dict[tuple[int, str], float] = {}
    # Max HP per mon (player, species) → int
    max_hp_map: dict[tuple[int, str], int] = {}
    # Name → player mapping built from |switch| lines
    mon_player: dict[str, int] = {}

    current_turn = 0
    weather_start: dict[str, int] = {}
    field_start: dict[str, int] = {}
    player_names: dict[int, str] = {}

    def _mon_key(player: int, species: str) -> tuple[int, str]:
        return (player, _species(species))

    def _ensure_mon(player: int, species: str):
        key = _species(species)
        full_key = (player, key)
        if key not in record.mons:
            record.mons[key] = MonRecord(species=key, player=player)
        current_hp.setdefault(full_key, 1.0)

    def _update_hp(player: int, species_raw: str, hp_cur: int, hp_max: int):
        key = _mon_key(player, species_raw)
        if hp_max > 0:
            max_hp_map[key] = hp_max
        stored_max = max_hp_map.get(key, hp_max if hp_max > 0 else 100)
        new_frac = hp_cur / stored_max if stored_max > 0 else 0.0
        old_frac = current_hp.get(key, new_frac)
        current_hp[key] = new_frac
        return old_frac, new_frac

    for raw_line in log_text.splitlines():
        line = raw_line.strip()

        m = _RE_PLAYER.match(line)
        if m:
            pid, uname = int(m.group(1)), m.group(2).strip()
            player_names[pid] = uname
            if pid == 1:
                record.player1 = uname
            else:
                record.player2 = uname
            continue

        m = _RE_TURN.match(line)
        if m:
            current_turn = int(m.group(1))
            record.total_turns = current_turn
            continue

        m = _RE_WIN.match(line)
        if m:
            winner_name = m.group(1).strip()
            for pid, uname in player_names.items():
                if uname == winner_name:
                    record.winner = pid
                    break
            continue

        # Switch / drag
        for pat in (_RE_SWITCH, _RE_DRAG):
            m = pat.match(line)
            if m:
                player = int(m.group(1))
                species_raw = m.group(2)
                hp_cur, hp_max = int(m.group(4)), int(m.group(5))
                _ensure_mon(player, species_raw)
                _update_hp(player, species_raw, hp_cur, hp_max)
                if pat is _RE_SWITCH:
                    record.actions.append(TurnAction(
                        turn=current_turn,
                        player=player,
                        pokemon=_species(species_raw),
                        action_type="switch",
                        detail=_species(species_raw),
                    ))
                break

        m = _RE_MOVE.match(line)
        if m:
            player, species_raw, move_name = int(m.group(1)), m.group(2), m.group(3)
            _ensure_mon(player, species_raw)
            record.actions.append(TurnAction(
                turn=current_turn,
                player=player,
                pokemon=_species(species_raw),
                action_type="move",
                detail=move_name.strip(),
            ))
            continue

        m = _RE_MEGA.match(line)
        if m:
            player, species_raw = int(m.group(1)), m.group(2)
            record.actions.append(TurnAction(
                turn=current_turn,
                player=player,
                pokemon=_species(species_raw),
                action_type="mega",
                detail="",
            ))
            continue

        for pat in (_RE_DAMAGE, _RE_HEAL):
            m = pat.match(line)
            if m:
                player, species_raw = int(m.group(1)), m.group(2)
                hp_cur, hp_max = int(m.group(3)), int(m.group(4))
                _ensure_mon(player, species_raw)
                old_frac, new_frac = _update_hp(player, species_raw, hp_cur, hp_max)
                record.damage_events.append(DamageEvent(
                    turn=current_turn,
                    target_player=player,
                    target_pokemon=_species(species_raw),
                    hp_before=old_frac,
                    hp_after=new_frac,
                ))
                break

        m = _RE_FAINT.match(line)
        if m:
            player, species_raw = int(m.group(1)), m.group(2)
            species = _species(species_raw)
            _ensure_mon(player, species_raw)
            record.damage_events.append(DamageEvent(
                turn=current_turn,
                target_player=player,
                target_pokemon=species,
                hp_before=current_hp.get(_mon_key(player, species_raw), 0.0),
                hp_after=0.0,
            ))
            current_hp[_mon_key(player, species_raw)] = 0.0
            record.faint_events.append(FaintEvent(
                turn=current_turn,
                player=player,
                pokemon=species,
            ))
            if species in record.mons:
                record.mons[species].fainted = True
                record.mons[species].turn_fainted = current_turn
            continue

        m = _RE_WEATHER.match(line)
        if m:
            w = m.group(1)
            if w != "none":
                weather_start.setdefault(w, current_turn)
            else:
                # All active weather ended — close open entries
                for wkey, wstart in list(weather_start.items()):
                    record.weather_turns[wkey] = (
                        record.weather_turns.get(wkey, 0) + current_turn - wstart
                    )
                    del weather_start[wkey]
            continue

        m = _RE_FIELD_S.match(line)
        if m:
            fname = m.group(1).strip()
            field_start.setdefault(fname, current_turn)
            continue

        m = _RE_FIELD_E.match(line)
        if m:
            fname = m.group(1).strip()
            if fname in field_start:
                record.field_turns[fname] = (
                    record.field_turns.get(fname, 0) + current_turn - field_start.pop(fname)
                )
            continue

    # Close any still-active weather / fields at end of battle
    for wkey, wstart in weather_start.items():
        record.weather_turns[wkey] = (
            record.weather_turns.get(wkey, 0) + record.total_turns - wstart
        )
    for fname, fstart in field_start.items():
        record.field_turns[fname] = (
            record.field_turns.get(fname, 0) + record.total_turns - fstart
        )

    return record


def parse_log_file(path: str | Path) -> BattleRecord:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_battle_log(text, battle_id=path.stem)
