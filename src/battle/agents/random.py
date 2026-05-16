import logging
from battle.agents.base import BaseAgent

log = logging.getLogger(__name__)


class RandomAgent(BaseAgent):

    def choose_move(self, battle):
        move = self.choose_random_doubles_move(battle)
        log.debug("[T%03d] %s → %s", battle.turn, battle.battle_tag, move)
        return move
