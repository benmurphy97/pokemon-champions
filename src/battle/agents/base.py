from abc import abstractmethod
import logging
import numpy as np
from poke_env import Player

from battle.encoder import encode_battle

log = logging.getLogger(__name__)


class BaseAgent(Player):

    def embed_battle(self, battle) -> np.ndarray:
        return encode_battle(battle)

    @abstractmethod
    def choose_move(self, battle):
        raise NotImplementedError

    def _battle_finished_callback(self, battle):
        if battle.won is True:
            result = "WIN"
        elif battle.won is False:
            result = "LOSS"
        else:
            result = "DRAW"
        remaining = sum(1 for m in battle.team.values() if not m.fainted)
        opp_remaining = sum(1 for m in battle.opponent_team.values() if not m.fainted)
        log.info(
            "[%s] %s | turn %d | us: %d/6 | them: %d/6",
            result,
            battle.battle_tag,
            battle.turn,
            remaining,
            opp_remaining,
        )
