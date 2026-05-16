"""
Stage 4: Post-Game Evaluation — Per-Pokémon and Team Metrics

Consumes BattleRecord (from log_parser.py) and produces:
  - Per-Pokémon performance scores used as the GA fitness signal (Stage 5)
  - Team-level metrics used for opponent pool updates (Stage 6)
  - Structural weakness flags used by the Claude API commentary hook

Public API
----------
score_battle(record, our_player)  → BattleScore
score_replay_dir(path, our_player)  → list[BattleScore]

BattleScore.to_ga_fitness()  → float in [0, 1] for use in Stage 5 DEAP evaluate()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from battle_decisions.analytics.log_parser import BattleRecord, MonRecord, parse_log_file


# ---------------------------------------------------------------------------
# Score data structures
# ---------------------------------------------------------------------------

@dataclass
class MonScore:
    """Derived performance score for one Pokémon in one battle."""
    species: str
    turns_active: int
    damage_contribution: float  # fraction of total damage dealt by our side
    kos_scored: int
    survived: bool
    role_completion: float      # TODO: 0–1 score for role-specific goals (see below)

    @property
    def raw_score(self) -> float:
        """Simple weighted composite. Tune weights as more data accumulates."""
        return (
            self.damage_contribution * 0.4
            + self.kos_scored * 0.25
            + (1.0 if self.survived else 0.0) * 0.2
            + self.role_completion * 0.15
        )


@dataclass
class BattleScore:
    """Full evaluation of one battle from our team's perspective."""
    battle_id: str
    our_player: int
    won: bool
    total_turns: int
    win_margin: float            # (our remaining HP sum) - (opp remaining HP sum), normalised
    mon_scores: list[MonScore] = field(default_factory=list)
    threat_pokemon: list[str] = field(default_factory=list)   # opp mons that dealt most damage
    underperformers: list[str] = field(default_factory=list)  # our mons with low contribution

    def to_ga_fitness(self) -> float:
        """
        Convert to a scalar fitness value for DEAP's evaluate() function.
        Range: [0, 1] where 1 = perfect win with all mons surviving.

        TODO: Add momentum-shift bonus (turns we held lead vs. turned it around).
        TODO: Penalise wins that relied on a single Pokémon carrying.
        """
        base = 1.0 if self.won else 0.0
        margin_bonus = max(0.0, self.win_margin) * 0.2
        # Cap total at 1.0 for DEAP compatibility
        return min(1.0, base * 0.7 + margin_bonus + sum(
            ms.raw_score for ms in self.mon_scores
        ) / max(1, len(self.mon_scores)) * 0.1)


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def score_battle(record: BattleRecord, our_player: int = 1) -> BattleScore:
    """
    Derive BattleScore from a BattleRecord.

    Args:
        record:     Parsed battle log.
        our_player: Which player number is our agent (1 or 2).

    TODO: Implement per-Pokémon role_completion scoring:
      - Trick Room setter: did it successfully set TR ≥ 1 time?
      - Weather setter: did weather activate at least once?
      - Fake Out lead: did it land Fake Out turn 1?
      - Sweeper: did it score ≥ 1 KO?
    TODO: Implement win_margin from final HP states (need last HP events per mon).
    TODO: Flag underperformers (turns_active < 2 AND damage_contribution < 0.05).
    TODO: Identify threat_pokemon — opponent mons with highest damage_dealt to us.
    """
    opp_player = 3 - our_player
    won = record.winner == our_player

    our_mons = record.mons_p1 if our_player == 1 else record.mons_p2
    opp_mons = record.mons_p2 if our_player == 1 else record.mons_p1

    # Compute total damage dealt by our side from damage events
    our_damage_events = [
        e for e in record.damage_events
        if e.target_player == opp_player and e.hp_before > e.hp_after
    ]
    total_our_damage = sum(e.hp_before - e.hp_after for e in our_damage_events)

    # Per-mon damage contribution (approximate — log doesn't always record source)
    # TODO: track source Pokémon per damage event for accurate attribution
    n_our_mons = max(1, len(our_mons))
    per_mon_damage = total_our_damage / n_our_mons  # placeholder equal split

    mon_scores: list[MonScore] = []
    for mon in our_mons:
        kos = sum(1 for f in record.faint_events if f.player == opp_player)  # TODO: attribute per mon
        mon_scores.append(MonScore(
            species=mon.species,
            turns_active=mon.turns_active,
            damage_contribution=per_mon_damage / max(0.001, total_our_damage),
            kos_scored=0,         # TODO: attribute KOs per mon
            survived=not mon.fainted,
            role_completion=0.0,  # TODO: implement role checks
        ))

    # TODO: compute actual win_margin from final HP states
    win_margin = 0.3 if won else -0.3

    # TODO: identify real underperformers and threats
    underperformers: list[str] = []
    threats: list[str] = []

    return BattleScore(
        battle_id=record.battle_id,
        our_player=our_player,
        won=won,
        total_turns=record.total_turns,
        win_margin=win_margin,
        mon_scores=mon_scores,
        threat_pokemon=threats,
        underperformers=underperformers,
    )


def score_replay_dir(
    replay_dir: str | Path,
    our_player: int = 1,
) -> list[BattleScore]:
    """
    Parse and score all .log files in a directory.

    Returns one BattleScore per successfully parsed replay.
    """
    replay_dir = Path(replay_dir)
    scores: list[BattleScore] = []

    for log_path in sorted(replay_dir.glob("*.log")):
        try:
            record = parse_log_file(log_path)
            scores.append(score_battle(record, our_player=our_player))
        except Exception as exc:
            print(f"WARNING: could not score {log_path.name}: {exc}")

    return scores


def aggregate_fitness(scores: list[BattleScore]) -> float:
    """
    Aggregate multiple BattleScore objects into a single GA fitness value.

    Used by Stage 5 (ga/evolve.py) when evaluating a team across N battles.
    Returns a float in [0, 1].

    TODO: Weight recent battles more heavily than older ones (recency bias).
    TODO: Add diversity bonus — reward teams that win via different mechanisms.
    """
    if not scores:
        return 0.0
    return sum(s.to_ga_fitness() for s in scores) / len(scores)
