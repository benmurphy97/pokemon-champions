"""
SelectionAgent — HeuristicAgent extended for the selection experiment pipeline.

Adds:
  - configurable teampreview(): forced (bring/lead indices) or auto (BST/speed heuristic)
  - turn-1 HP snapshot stored in self._t1_hp for use by experiment.py
"""
from __future__ import annotations

from battle.agents.heuristic import HeuristicAgent


class SelectionAgent(HeuristicAgent):
    """
    HeuristicAgent extended for bring/lead experiment control.

    teampreview() modes:
      forced  — bring_indices + lead_indices set via set_selection() before each battle
      auto    — top-4 by BST, lead fastest 2 (used for the opponent side)
    """

    def __init__(
        self,
        bring_indices: list[int] | None = None,
        lead_indices:  list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._bring: list[int] | None = bring_indices
        self._lead:  list[int] | None = lead_indices
        # Keyed by battle_tag; set on turn 2 — read by client after battle ends
        self._t1_hp: dict[str, tuple[float, float]] = {}

    def set_selection(self, bring: list[int], lead: list[int]) -> None:
        """Fix the bring/lead indices used by teampreview() for the next battle."""
        self._bring = bring
        self._lead  = lead

    def clear_selection(self) -> None:
        """Revert to auto heuristic teampreview (top-4 BST, lead fastest 2)."""
        self._bring = None
        self._lead  = None

    # ── teampreview ──────────────────────────────────────────────────────────

    async def teampreview(self, battle) -> str:
        """Return a Showdown /team string using forced indices if set, else the BST heuristic."""
        if self._bring is not None and self._lead is not None:
            return self._forced_teampreview()
        return self._heuristic_teampreview(battle)

    def _forced_teampreview(self) -> str:
        """Convert 0-based bring/lead indices to a Showdown /team string."""
        bring_slots = [b + 1 for b in self._bring]           # 1-indexed positions 1–6
        lead_set    = {bring_slots[i] for i in self._lead}
        lead_slots  = [bring_slots[i] for i in self._lead]   # ordered
        bench_slots = [s for s in bring_slots if s not in lead_set]
        return f"/team {''.join(str(p) for p in lead_slots + bench_slots)}"

    def _heuristic_teampreview(self, battle) -> str:
        """Bring top-4 by BST; lead the fastest 2 of those 4."""
        mons = list(battle.team.values())

        def bst(mon) -> int:
            return sum((mon.base_stats or {}).values())

        def spe(mon) -> int:
            return (mon.base_stats or {}).get("spe", 0)

        ranked = sorted(enumerate(mons, start=1), key=lambda x: -bst(x[1]))
        top4   = ranked[:4]
        leads  = sorted(top4, key=lambda x: -spe(x[1]))[:2]
        leads_set = set(id(x) for x in leads)
        bench  = [x for x in top4 if id(x) not in leads_set]

        order = [pos for pos, _ in leads] + [pos for pos, _ in bench]
        return f"/team {''.join(str(p) for p in order)}"

    # ── choose_move ──────────────────────────────────────────────────────────

    def choose_move(self, battle):
        """Snapshot turn-1 HP on turn 2, then delegate move selection to HeuristicAgent."""
        if battle.turn == 2 and battle.battle_tag not in self._t1_hp:
            our_active  = [m for m in (battle.active_pokemon or []) if m]
            opp_active_ = [m for m in (battle.opponent_active_pokemon or []) if m]
            our_hp = sum(m.current_hp_fraction for m in our_active)  / 2
            opp_hp = sum(m.current_hp_fraction for m in opp_active_) / 2
            self._t1_hp[battle.battle_tag] = (our_hp, opp_hp)
        return super().choose_move(battle)
