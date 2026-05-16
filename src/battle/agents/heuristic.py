import logging
from poke_env.player.battle_order import DoubleBattleOrder

from battle.agents.base import BaseAgent
from battle.encoder import _damage_estimate, _max_effectiveness, _live_types

log = logging.getLogger(__name__)

_PROTECT_IDS = {
    "protect", "wideguard", "quickguard", "detect",
    "spikyshield", "banefulbunker", "silktrap",
    "burningbulwark", "kingsshield",
}

_LOW_HP_THRESHOLD     = 0.30
_AGGRO_TURN_THRESHOLD = 15


class HeuristicAgent(BaseAgent):
    """
    Protect-aware, focus-fire heuristic agent for VGC doubles.

    Scoring priority:
      low-HP protect (turns ≤ 15) > attacks (focus on lowest-HP opponent) >
      status moves > voluntary switches > pass
    Falls back to random doubles move if valid_orders is empty.
    """

    def choose_move(self, battle):
        if not battle.valid_orders or not battle.valid_orders[0]:
            return self.choose_random_doubles_move(battle)

        first_orders, *rest = battle.valid_orders
        second_orders = rest[0] if rest else []

        active = list(battle.active_pokemon or [None, None])
        hp0 = active[0].current_hp_fraction if (active[0] and not active[0].fainted) else 1.0
        hp1 = (active[1].current_hp_fraction
               if (len(active) > 1 and active[1] and not active[1].fainted) else 1.0)

        opp_active = [m for m in (battle.opponent_active_pokemon or []) if m and not m.fainted]

        best_first = max(first_orders,
                         key=lambda o: self._score_order(o, battle, hp0, opp_active))

        if second_orders:
            best_second = max(second_orders,
                              key=lambda o: self._score_order(o, battle, hp1, opp_active))
            combos = DoubleBattleOrder.join_orders([best_first], [best_second])
            if combos:
                return combos[0]
            combos = DoubleBattleOrder.join_orders([best_first], second_orders)
            if combos:
                valid_seconds = [c.second_order for c in combos]
                best_second_valid = max(valid_seconds,
                                        key=lambda o: self._score_order(o, battle, hp1, opp_active))
                final = DoubleBattleOrder.join_orders([best_first], [best_second_valid])
                return final[0] if final else combos[0]

        return self.choose_random_doubles_move(battle)

    def _score_order(self, order, battle, hp_fraction: float = 1.0,
                     opp_active: list | None = None) -> float:
        from poke_env.battle.pokemon import Pokemon as PokemonObj
        from poke_env.player.battle_order import PassBattleOrder

        if isinstance(order, PassBattleOrder):
            return -1.0

        if isinstance(order.order, PokemonObj):
            return 0.0  # voluntary switch — only if all moves score ≤ 0

        move = order.order
        is_protect = getattr(move, "id", "") in _PROTECT_IDS

        if (hp_fraction < _LOW_HP_THRESHOLD and is_protect
                and battle.turn <= _AGGRO_TURN_THRESHOLD):
            return 10.0

        if not move.base_power:
            return 0.01

        if opp_active is None:
            opp_active = [m for m in (battle.opponent_active_pokemon or [])
                          if m and not m.fainted]

        if not opp_active:
            return move.base_power / 250.0

        focus = min(opp_active, key=lambda m: m.current_hp_fraction)
        focus_dmg = _damage_estimate(move, focus)
        if focus_dmg == 0.0 and len(opp_active) > 1:
            return max(_damage_estimate(move, opp) for opp in opp_active)
        return focus_dmg
