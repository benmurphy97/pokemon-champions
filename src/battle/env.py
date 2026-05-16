import numpy as np
import gymnasium as gym
from gymnasium import spaces
from poke_env import ServerConfiguration
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from config import BATTLE_FORMAT
from battle.encoder import encode_battle, STATE_SIZE


# Pokemon that should prioritise leading — Fake Out users and weather setters
_FAKE_OUT_LEADS = {"Incineroar", "Sneasler", "Blastoise", "Hitmontop", "Kangaskhan"}
_WEATHER_SETTERS = {"Charizard-Mega-Y", "Torkoal", "Pelipper", "Ninetales-Alola",
                    "Froslass-Mega", "Tyranitar", "Tyranitar-Mega"}
_AVOID_LEADING = {"Farigiraf", "Sinistcha", "Hatterene", "Oranguru"}  # TR setters — keep safe


class PokemonEnv(DoublesEnv):
    """VGC doubles environment. Subclasses DoublesEnv with our observation and reward."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        obs_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_SIZE,), dtype=np.float32)
        # Setting observation_spaces triggers PokeEnv.__setattr__ which wraps each
        # raw space into Dict({"observation": raw, "action_mask": Box(...)})
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}
        self._prev_hp: dict = {}
        self._stall_turns: dict = {}

    async def teampreview(self, battle) -> str:
        our_mons = list(battle.team.values())
        opp_mons = list(battle.opponent_team.values()) if battle.opponent_team else []

        # Score each of our 6 Pokemon for overall "bring value"
        bring_scores = [(self._bring_score(mon, opp_mons), i + 1, mon)
                        for i, mon in enumerate(our_mons)]
        bring_scores.sort(key=lambda x: -x[0])
        top4 = bring_scores[:4]

        # From the top 4, score for lead suitability — highest leads first
        lead_scores = [(self._lead_score(mon, opp_mons), pos)
                       for _, pos, mon in top4]
        lead_scores.sort(key=lambda x: -x[0])

        order = [pos for _, pos in lead_scores]
        return f"/team {''.join(str(p) for p in order)}"

    def _bring_score(self, mon, opp_mons: list) -> float:
        """Score how valuable this Pokemon is to bring based on type matchups."""
        score = 0.0
        our_types = [t for t in (mon.types or []) if t is not None]

        for opp in opp_mons:
            opp_types = [t for t in (opp.types or []) if t is not None]
            if not opp_types:
                continue
            for our_type in our_types:
                try:
                    mult = our_type.damage_multiplier(*opp_types)
                    if mult > 1.0:
                        score += mult - 1.0
                except Exception:
                    pass
            for opp_type in opp_types:
                try:
                    mult = opp_type.damage_multiplier(*our_types)
                    if mult > 1.0:
                        score -= (mult - 1.0) * 0.3
                except Exception:
                    pass

        species = mon.species or ""
        if species in _FAKE_OUT_LEADS:
            score += 3.0
        if species in _WEATHER_SETTERS:
            score += 2.0
        if species in _AVOID_LEADING:
            score += 1.0  # TR setters are valuable to bring but not to lead

        return score

    def _lead_score(self, mon, opp_mons: list) -> float:
        """Score how good this Pokemon is as a turn-1 lead."""
        species = mon.species or ""
        score = self._bring_score(mon, opp_mons)

        if species in _FAKE_OUT_LEADS:
            score += 4.0
        if species in _WEATHER_SETTERS:
            score += 2.0
        if species in _AVOID_LEADING:
            score -= 5.0  # TR setters die if they lead — keep them safe

        return score

    def embed_battle(self, battle) -> np.ndarray:
        return encode_battle(battle)

    def calc_reward(self, battle) -> float:
        if battle.finished:
            return 5.0 if battle.won else -5.0

        our_hp = sum(m.current_hp_fraction for m in battle.team.values() if not m.fainted)
        opp_hp = sum(
            m.current_hp_fraction for m in battle.opponent_team.values() if not m.fainted
        ) if battle.opponent_team else 3.0

        opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted) if battle.opponent_team else 0
        our_fainted = sum(1 for m in battle.team.values() if m.fainted)

        tag = battle.battle_tag
        prev_our, prev_opp = self._prev_hp.get(tag, (6.0, 6.0))
        self._prev_hp[tag] = (our_hp, opp_hp)

        delta_our = our_hp - prev_our   # negative = we took damage
        delta_opp = opp_hp - prev_opp   # negative = we dealt damage

        # Stall penalty: discourages turns with zero meaningful progress.
        # Kicks in after 1 consecutive no-progress turn; capped at -0.75.
        our_boosts = sum(
            sum(abs(v) for v in m.boosts.values())
            for m in battle.team.values() if not m.fainted
        )
        opp_boosts = sum(
            sum(abs(v) for v in m.boosts.values())
            for m in battle.opponent_team.values() if not m.fainted
        ) if battle.opponent_team else 0
        field_state = (
            frozenset(battle.weather.keys()) |
            frozenset(battle.fields.keys())
        )
        prev_boosts, prev_field = self._stall_turns.get(tag + "_state", (0, frozenset()))
        boost_changed = abs(our_boosts + opp_boosts - prev_boosts) > 0
        field_changed = field_state != prev_field
        self._stall_turns[tag + "_state"] = (our_boosts + opp_boosts, field_state)

        no_progress = (
            abs(delta_our) < 0.01
            and abs(delta_opp) < 0.01
            and not boost_changed
            and not field_changed
        )
        stall = self._stall_turns.get(tag, 0)
        stall = stall + 1 if no_progress else 0
        self._stall_turns[tag] = stall
        stall_penalty = -0.15 * min(stall, 5) if stall > 1 else 0.0

        return (
            -delta_our * 0.2
            + -delta_opp * 0.3
            + (opp_fainted - our_fainted) * 0.05
            + stall_penalty
        )


def _resolve_switch_conflicts(action: np.ndarray, mask: np.ndarray, slot_size: int) -> np.ndarray:
    """
    Fix joint-action conflicts that poke-env's join_orders rejects:
      1. Both slots switching to the same bench slot (e.g. [1,1]).
      2. Both slots passing when at least one has a non-pass action available
         (forced-switch state with one remaining Pokémon — one slot must switch in).
    Pure wait-state double-pass (mask has only action 0 valid) is left untouched.
    """
    a0, a1 = int(action[0]), int(action[1])
    mask0 = mask[:slot_size].astype(bool)
    mask1 = mask[slot_size:].astype(bool)

    # Case 1: both slots switching to the same bench slot
    if 1 <= a0 <= 6 and a0 == a1:
        for alt in range(slot_size):
            if mask1[alt] and alt != a0:
                return np.array([a0, alt], dtype=action.dtype)
        for alt in range(slot_size):
            if mask0[alt] and alt != a1:
                return np.array([alt, a1], dtype=action.dtype)
        return action  # no valid resolution; let poke-env random-fallback

    # Case 2: both passing when a switch is available (forced-switch with one bench mon)
    if a0 == 0 and a1 == 0:
        non_pass0 = next((alt for alt in range(1, slot_size) if mask0[alt]), None)
        non_pass1 = next((alt for alt in range(1, slot_size) if mask1[alt]), None)
        if non_pass0 is not None:
            return np.array([non_pass0, a1], dtype=action.dtype)
        if non_pass1 is not None:
            return np.array([a0, non_pass1], dtype=action.dtype)

    return action


class FlatObsWrapper(gym.Wrapper):
    """Strips poke-env's Dict obs down to flat Box and exposes action_masks() for MaskablePPO."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space["observation"]
        self._action_mask: np.ndarray | None = None

    def step(self, action):
        action = self._resolve_conflicts(np.asarray(action))
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._action_mask = obs["action_mask"]
        return obs["observation"], reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._action_mask = obs["action_mask"]
        return obs["observation"], info

    def action_masks(self) -> np.ndarray:
        if self._action_mask is None:
            return np.ones(sum(self.action_space.nvec), dtype=bool)
        return self._action_mask.astype(bool)

    def _resolve_conflicts(self, action: np.ndarray) -> np.ndarray:
        slot_size = int(self.action_space.nvec[0])
        mask = self._action_mask if self._action_mask is not None else np.ones(slot_size * 2, dtype=np.int64)
        return _resolve_switch_conflicts(action, mask, slot_size)


def make_env(server_cfg: ServerConfiguration, team: str, opponent) -> gym.Env:
    """
    Build a single-agent, flat-obs VGC doubles gym environment.

    Args:
        server_cfg: Showdown server connection config.
        team:       Showdown team string for the learning agent.
        opponent:   A poke-env Player instance that acts as the second agent.

    Returns:
        A gymnasium.Env with Box(STATE_SIZE,) observations and
        MultiDiscrete([107, 107]) actions.
    """
    env = PokemonEnv(
        battle_format=BATTLE_FORMAT,
        server_configuration=server_cfg,
        team=team,
        log_level=30,                  # WARNING — suppress poke-env chatter
        choose_on_teampreview=False,   # teampreview() override handles selection
        strict=False,                  # invalid actions fall back to random legal move
    )
    wrapped = SingleAgentWrapper(env, opponent)
    return FlatObsWrapper(wrapped)
