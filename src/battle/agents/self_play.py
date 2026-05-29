import logging
import numpy as np
from poke_env.environment.doubles_env import DoublesEnv

from battle.agents.base import BaseAgent
from battle.env import _resolve_switch_conflicts
from battle.encoder import encode_battle
from stage3_battle_decisions.architecture.extractor import BattleStateEncoder  # noqa: F401 — must be importable for SB3 checkpoint load

log = logging.getLogger(__name__)


class SelfPlayAgent(BaseAgent):
    """
    Opponent that uses a frozen MaskablePPO checkpoint to choose moves.
    Brain improves each time the checkpoint is updated between training rounds.

    Note: relies on DoublesEnv.get_action_mask() and DoublesEnv.action_to_order()
    being available as class-level utilities in poke-env 0.15.0. If those methods
    are not present (poke-env API change), choose_move falls back to random legal
    moves and logs a warning once.
    """

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self._warned_fallback = False

    def choose_move(self, battle):
        try:
            obs = encode_battle(battle).reshape(1, -1)
            mask = np.array(DoublesEnv.get_action_mask(battle), dtype=bool)
            action, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
            act = _resolve_switch_conflicts(action[0].copy(), mask, len(mask) // 2)
            return DoublesEnv.action_to_order(act, battle, strict=False)
        except Exception as exc:
            if not self._warned_fallback:
                log.warning(
                    "SelfPlayAgent: model-based move failed (%s) — falling back to random. "
                    "Check that poke-env 0.15.0 exposes DoublesEnv.get_action_mask / "
                    "action_to_order as class-level utilities.",
                    exc,
                )
                self._warned_fallback = True
            return self.choose_random_doubles_move(battle)
