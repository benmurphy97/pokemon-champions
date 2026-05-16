"""
parse_replay.py — Parse Showdown .log files into (obs, action) pairs for BC.

Action space: MultiDiscrete([107, 107])
  Slot encoding (per slot):
    0        = pass / forced (no real choice)
    1–6      = switch to bench slot 1–6
    7–10     = move 1–4 (no mega)
    11–26    = unused (padding to match env; we only emit 7–10 and 27–30)
    27–30    = move 1–4 + mega evolve

State: encode_battle() → np.ndarray of shape (STATE_SIZE,)

Usage:
    from battle_decisions.imitation.parse_replay import parse_replay_dir
    dataset = parse_replay_dir(Path("data/replays"))
    # dataset.obs  → np.ndarray (N, STATE_SIZE)
    # dataset.acts → np.ndarray (N, 2)   MultiDiscrete actions

The parser replays the log through a lightweight state machine turn-by-turn,
calling _encode_state() at each decision point (right before a |move| or
|switch| line from the chosen player's perspective).
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action space constants — must match battle/env.py
# ---------------------------------------------------------------------------
ACTION_PASS        = 0
ACTION_SWITCH_BASE = 1   # switch to bench slot n → n  (1-indexed)
ACTION_MOVE_BASE   = 7   # move n (0-indexed internally) → 7+n
ACTION_MEGA_BASE   = 27  # move n + mega              → 27+n
N_ACTIONS = 107          # per slot (matches MultiDiscrete nvec)


# ---------------------------------------------------------------------------
# Lightweight log-line parser
# ---------------------------------------------------------------------------

_RE_PLAYER     = re.compile(r"^\|player\|p(\d)\|(.+)$")
_RE_TURN       = re.compile(r"^\|turn\|(\d+)$")
_RE_POKE       = re.compile(r"^\|poke\|p(\d)\|([^|]+)\|")
_RE_SWITCH     = re.compile(r"^\|switch\|p(\d)a?:? ?(.*?)\|([^|]+)\|(\d+)/(\d+)")
_RE_DRAG       = re.compile(r"^\|drag\|p(\d)a?:? ?(.*?)\|([^|]+)\|(\d+)/(\d+)")
_RE_MOVE       = re.compile(r"^\|move\|p(\d)a?:? ?(.*?)\|(.*?)(\||\[)")
_RE_MEGA       = re.compile(r"^\|-mega\|p(\d)a?:? ?(.*?)\|")
_RE_FAINT      = re.compile(r"^\|faint\|p(\d)a?:? ?(.*?)$")
_RE_HP         = re.compile(r"^\|-damage\|p(\d)a?:? ?(.*?)\|(\d+)/(\d+)")
_RE_HP_HEAL    = re.compile(r"^\|-heal\|p(\d)a?:? ?(.*?)\|(\d+)/(\d+)")
_RE_BOOST      = re.compile(r"^\|-boost\|p(\d)a?:? ?(.*?)\|(\w+)\|(-?\d+)")
_RE_UNBOOST    = re.compile(r"^\|-unboost\|p(\d)a?:? ?(.*?)\|(\w+)\|(-?\d+)")
_RE_WEATHER    = re.compile(r"^\|-weather\|([\w]+)")
_RE_FIELD      = re.compile(r"^\|-fieldstart\|([\w: ]+)")
_RE_FIELDEND   = re.compile(r"^\|-fieldend\|([\w: ]+)")
_RE_STATUS     = re.compile(r"^\|-status\|p(\d)a?:? ?(.*?)\|(\w+)")
_RE_CURESTATUS = re.compile(r"^\|-curestatus\|p(\d)a?:? ?(.*?)\|")
_RE_WIN        = re.compile(r"^\|win\|")
_RE_TEAMSIZE   = re.compile(r"^\|teamsize\|p(\d)\|(\d+)")


# ---------------------------------------------------------------------------
# Minimal battle state tracker (pure log parsing — no poke-env dependency)
# ---------------------------------------------------------------------------

@dataclass
class MonState:
    species: str
    hp: float = 1.0          # fraction 0–1
    max_hp: int = 100
    fainted: bool = False
    boosts: dict = field(default_factory=dict)  # stat → int
    mega: bool = False
    status: str = "none"     # brn / frz / par / psn / tox / slp / none


@dataclass
class BattleState:
    """Minimal replicated state of one side of a doubles battle."""
    team: list[MonState] = field(default_factory=list)
    active: list[MonState | None] = field(default_factory=lambda: [None, None])
    bench_order: list[int] = field(default_factory=list)
    weather: str = "none"
    weather_since_turn: int = 0   # turn number when current weather was set
    fields: set[str] = field(default_factory=set)
    field_since_turn: dict = field(default_factory=dict)  # field_name → turn set
    turn: int = 0


def _species_key(raw: str) -> str:
    """Normalise species name from log (strip *, M, F suffixes, forme info)."""
    s = raw.strip().rstrip("*").split(",")[0].strip()
    return s


def _slot_of(state: BattleState, nickname_or_species: str) -> int | None:
    """Return team index [0-5] for the mon with this name in the team."""
    key = _species_key(nickname_or_species)
    for i, mon in enumerate(state.team):
        if _species_key(mon.species) == key:
            return i
    return None


# ---------------------------------------------------------------------------
# Feature vector — mirrors battle/encoder.py but built from BattleState
# ---------------------------------------------------------------------------

_SUN_NAMES   = {"sunnyday", "desolateland", "harshsunshine"}
_RAIN_NAMES  = {"raindance", "primordialsea"}
_SNOW_NAMES  = {"snowscape", "hail"}
_SAND_NAMES  = {"sandstorm"}
_FIELD_NAMES = ["electricterrain", "grassyterrain", "mistyterrain",
                "psychicterrain", "trickroom"]
_STATUS_NAMES = ["brn", "frz", "par", "psn", "tox", "slp"]

_BOOST_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
_MOVE_FEATURES_LOCAL = 7
_WEATHER_MAX_TURNS   = 8.0
_FIELD_MAX_TURNS     = 8.0
STATE_SIZE_LOCAL = 222  # must match battle/encoder.py STATE_SIZE


def _hp_frac(mon: MonState | None) -> float:
    if mon is None or mon.fainted:
        return 0.0
    return max(0.0, min(1.0, mon.hp))


def _boosts_vec(mon: MonState | None) -> list[float]:
    if mon is None or mon.fainted:
        return [0.0] * len(_BOOST_STATS)
    b = mon.boosts
    return [b.get(s, 0) / 6.0 for s in _BOOST_STATS]


def _status_vec(mon: MonState | None) -> list[float]:
    if mon is None or mon.fainted:
        return [0.0] * len(_STATUS_NAMES)
    s = (mon.status or "none").lower()
    return [1.0 if s == name else 0.0 for name in _STATUS_NAMES]


def _encode_state(us: BattleState, them: BattleState) -> np.ndarray:
    """
    Produce a STATE_SIZE=222 feature vector from two BattleState objects.
    Approximates battle/encoder.py without a live poke-env battle object.
    Move power and type effectiveness are zeroed — a BC model still benefits
    from HP, boosts, status, weather, and field state.
    """
    feats: list[float] = []

    our_active  = us.active[:2]   + [None] * max(0, 2 - len(us.active))
    opp_active  = them.active[:2] + [None] * max(0, 2 - len(them.active))
    all_active  = our_active + opp_active

    # HP fractions — 4 slots
    for mon in all_active:
        feats.append(_hp_frac(mon))

    # Vulnerability — 8 features (zeroed; no type data available in log)
    feats.extend([0.25] * 8)

    # Stat boosts — 4 × 7
    for mon in all_active:
        feats.extend(_boosts_vec(mon))

    # Our move features — 2 slots × 4 moves × 7 (zeroed; no power/type in log)
    feats.extend([0.0] * (2 * 4 * _MOVE_FEATURES_LOCAL))

    # Bench HP — up to 4 non-active, non-fainted team members
    bench = [m for m in us.team if m not in our_active and not m.fainted]
    for i in range(4):
        feats.append(_hp_frac(bench[i]) if i < len(bench) else 0.0)

    # Weather binary flags
    w = us.weather.lower().replace(" ", "").replace("_", "")
    feats.append(1.0 if w in _SUN_NAMES  else 0.0)
    feats.append(1.0 if w in _RAIN_NAMES else 0.0)
    feats.append(1.0 if w in _SNOW_NAMES else 0.0)
    feats.append(1.0 if w in _SAND_NAMES else 0.0)

    # Field binary flags
    norm_fields = {f.lower().replace(" ", "").replace("_", "") for f in us.fields}
    for fname in _FIELD_NAMES:
        feats.append(1.0 if fname in norm_fields else 0.0)

    # Status conditions for all 4 active slots
    for mon in all_active:
        feats.extend(_status_vec(mon))

    # Status conditions for up to 4 bench mons
    for i in range(4):
        feats.extend(_status_vec(bench[i]) if i < len(bench) else [0.0] * len(_STATUS_NAMES))

    # Opponent revealed moves — zeroed (no power/type data in log)
    feats.extend([0.0] * (2 * 4 * _MOVE_FEATURES_LOCAL))

    # Weather turn counts (normalised by max extended duration)
    turns_elapsed = us.turn - us.weather_since_turn if w != "none" else 0
    for wset in (_SUN_NAMES, _RAIN_NAMES, _SNOW_NAMES, _SAND_NAMES):
        feats.append(min(1.0, turns_elapsed / _WEATHER_MAX_TURNS) if w in wset else 0.0)

    # Field turn counts (normalised)
    for fname in _FIELD_NAMES:
        if fname in norm_fields:
            since = us.field_since_turn.get(fname, us.turn)
            feats.append(min(1.0, (us.turn - since) / _FIELD_MAX_TURNS))
        else:
            feats.append(0.0)

    arr = np.array(feats, dtype=np.float32)
    assert len(arr) == STATE_SIZE_LOCAL, f"Expected {STATE_SIZE_LOCAL}, got {len(arr)}"
    return arr


# ---------------------------------------------------------------------------
# Turn-action collector
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """Actions taken by p1 and p2 on a single turn, plus state snapshot."""
    turn: int
    obs: np.ndarray        # state just before decisions
    p1_acts: list[int]     # 0 or 2 entries (one per active slot)
    p2_acts: list[int]


def _action_for_move(move_idx: int, is_mega: bool) -> int:
    base = ACTION_MEGA_BASE if is_mega else ACTION_MOVE_BASE
    return base + move_idx


def _action_for_switch(bench_slot: int) -> int:
    return ACTION_SWITCH_BASE + (bench_slot - 1)


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

class ReplayParser:
    """
    Parses a single Showdown .log file and yields (obs, action) tuples for
    the chosen player (1 or 2).
    """

    def __init__(self, player: int = 1):
        assert player in (1, 2)
        self.player = player
        self._opp = 3 - player

    def parse(self, log_text: str) -> list[tuple[np.ndarray, np.ndarray]]:
        p1 = BattleState()
        p2 = BattleState()
        samples: list[tuple[np.ndarray, np.ndarray]] = []

        turn_actions: dict[tuple[int, int], int] = {}
        pending_mega: set[int] = set()
        current_turn = 0
        obs_this_turn: np.ndarray | None = None

        def snap():
            nonlocal obs_this_turn
            us   = p1 if self.player == 1 else p2
            them = p2 if self.player == 1 else p1
            obs_this_turn = _encode_state(us, them)

        def flush_turn(new_turn: int):
            nonlocal turn_actions, pending_mega, obs_this_turn, current_turn
            if obs_this_turn is not None and turn_actions:
                us_player = self.player
                a0 = turn_actions.get((us_player, 0), ACTION_PASS)
                a1 = turn_actions.get((us_player, 1), ACTION_PASS)
                if a0 != ACTION_PASS or a1 != ACTION_PASS:
                    samples.append((
                        obs_this_turn,
                        np.array([a0, a1], dtype=np.int64),
                    ))
            turn_actions.clear()
            pending_mega.clear()
            obs_this_turn = None
            current_turn = new_turn
            p1.turn = new_turn
            p2.turn = new_turn

        def active_slot(state: BattleState, species_raw: str) -> int:
            key = _species_key(species_raw)
            for i, mon in enumerate(state.active[:2]):
                if mon and _species_key(mon.species) == key:
                    return i
            return 0

        def state_of(player_num: int) -> BattleState:
            return p1 if player_num == 1 else p2

        def update_hp(state: BattleState, species_raw: str, hp: int, max_hp: int):
            key = _species_key(species_raw)
            for mon in state.team:
                if _species_key(mon.species) == key:
                    mon.max_hp = max_hp if max_hp > 0 else mon.max_hp
                    mon.hp = hp / mon.max_hp if mon.max_hp > 0 else 0.0
                    return

        def get_bench_slot(state: BattleState, species_raw: str) -> int:
            key = _species_key(species_raw)
            bench = [m for m in state.team if m not in state.active and not m.fainted]
            for i, mon in enumerate(bench):
                if _species_key(mon.species) == key:
                    return i + 1
            return 1

        # Move name → move index within a mon's moveset (inferred from appearance order)
        move_order: dict[str, list[str]] = {}

        def move_idx(species_raw: str, move_name: str) -> int:
            key = _species_key(species_raw)
            if key not in move_order:
                move_order[key] = []
            ml = move_order[key]
            if move_name not in ml:
                if len(ml) < 4:
                    ml.append(move_name)
            try:
                return ml.index(move_name)
            except ValueError:
                return 0

        # ---- main parse loop ----
        for raw_line in log_text.splitlines():
            line = raw_line.strip()

            m = _RE_POKE.match(line)
            if m:
                player_num = int(m.group(1))
                species = m.group(2).split(",")[0].strip()
                state_of(player_num).team.append(MonState(species=species))
                continue

            m = _RE_TURN.match(line)
            if m:
                new_turn = int(m.group(1))
                flush_turn(new_turn)
                snap()
                continue

            for pat in (_RE_SWITCH, _RE_DRAG):
                m = pat.match(line)
                if m:
                    player_num = int(m.group(1))
                    species_raw = m.group(2)
                    hp_cur = int(m.group(4))
                    hp_max = int(m.group(5))
                    st = state_of(player_num)
                    update_hp(st, species_raw, hp_cur, hp_max)

                    key = _species_key(species_raw)
                    mon = next((x for x in st.team if _species_key(x.species) == key), None)
                    if mon is None:
                        mon = MonState(species=species_raw)
                        st.team.append(mon)

                    if len(st.active) < 2:
                        slot = len(st.active)
                        st.active.append(mon)
                    else:
                        replaced = False
                        for i in range(2):
                            if st.active[i] is None or st.active[i].fainted:
                                slot = i
                                st.active[i] = mon
                                replaced = True
                                break
                        if not replaced:
                            slot = 0
                            st.active[0] = mon

                    if pat is _RE_SWITCH and player_num == self.player:
                        bench_n = get_bench_slot(st, species_raw)
                        turn_actions[(player_num, slot)] = _action_for_switch(bench_n)
                    break

            m = _RE_MEGA.match(line)
            if m:
                pending_mega.add(int(m.group(1)))
                continue

            m = _RE_MOVE.match(line)
            if m:
                player_num = int(m.group(1))
                species_raw = m.group(2)
                move_name = m.group(3).strip()
                st = state_of(player_num)
                slot = active_slot(st, species_raw)
                midx = move_idx(species_raw, move_name)
                is_mega = player_num in pending_mega
                if player_num == self.player:
                    turn_actions[(player_num, slot)] = _action_for_move(midx, is_mega)
                pending_mega.discard(player_num)
                continue

            for pat in (_RE_HP, _RE_HP_HEAL):
                m = pat.match(line)
                if m:
                    player_num = int(m.group(1))
                    species_raw = m.group(2)
                    hp_cur = int(m.group(3))
                    hp_max = int(m.group(4))
                    update_hp(state_of(player_num), species_raw, hp_cur, hp_max)
                    break

            m = _RE_BOOST.match(line)
            if m:
                player_num = int(m.group(1))
                species_raw = m.group(2)
                stat = m.group(3).lower()
                val = int(m.group(4))
                st = state_of(player_num)
                key = _species_key(species_raw)
                for mon in st.team:
                    if _species_key(mon.species) == key:
                        mon.boosts[stat] = mon.boosts.get(stat, 0) + val
                continue

            m = _RE_UNBOOST.match(line)
            if m:
                player_num = int(m.group(1))
                species_raw = m.group(2)
                stat = m.group(3).lower()
                val = int(m.group(4))
                st = state_of(player_num)
                key = _species_key(species_raw)
                for mon in st.team:
                    if _species_key(mon.species) == key:
                        mon.boosts[stat] = mon.boosts.get(stat, 0) - val
                continue

            m = _RE_FAINT.match(line)
            if m:
                player_num = int(m.group(1))
                species_raw = m.group(2)
                st = state_of(player_num)
                key = _species_key(species_raw)
                for mon in st.team:
                    if _species_key(mon.species) == key:
                        mon.fainted = True
                        mon.hp = 0.0
                for i in range(2):
                    if st.active[i] and _species_key(st.active[i].species) == key:
                        st.active[i] = None
                continue

            m = _RE_STATUS.match(line)
            if m:
                player_num  = int(m.group(1))
                species_raw = m.group(2)
                status_name = m.group(3).lower()
                key = _species_key(species_raw)
                for mon in state_of(player_num).team:
                    if _species_key(mon.species) == key:
                        mon.status = status_name
                continue

            m = _RE_CURESTATUS.match(line)
            if m:
                player_num  = int(m.group(1))
                species_raw = m.group(2)
                key = _species_key(species_raw)
                for mon in state_of(player_num).team:
                    if _species_key(mon.species) == key:
                        mon.status = "none"
                continue

            m = _RE_WEATHER.match(line)
            if m:
                weather_name = m.group(1)
                for state in (p1, p2):
                    if state.weather.lower() != weather_name.lower():
                        state.weather_since_turn = current_turn
                    state.weather = weather_name
                continue

            m = _RE_FIELD.match(line)
            if m:
                fname = m.group(1).strip()
                fname_norm = fname.lower().replace(" ", "").replace("_", "")
                for state in (p1, p2):
                    state.fields.add(fname)
                    if fname_norm not in state.field_since_turn:
                        state.field_since_turn[fname_norm] = current_turn
                continue

            m = _RE_FIELDEND.match(line)
            if m:
                fname = m.group(1).strip()
                fname_norm = fname.lower().replace(" ", "").replace("_", "")
                for state in (p1, p2):
                    state.fields.discard(fname)
                    state.field_since_turn.pop(fname_norm, None)
                continue

        flush_turn(-1)
        return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class BCDataset:
    obs:  np.ndarray   # (N, STATE_SIZE)
    acts: np.ndarray   # (N, 2)  — slot0, slot1 actions

    def __len__(self):
        return len(self.obs)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), obs=self.obs, acts=self.acts)
        print(f"Saved dataset: {len(self)} samples → {path}")

    @classmethod
    def load(cls, path: Path) -> "BCDataset":
        data = np.load(str(path))
        return cls(obs=data["obs"], acts=data["acts"])

    @classmethod
    def merge(cls, datasets: list["BCDataset"]) -> "BCDataset":
        return cls(
            obs=np.concatenate([d.obs for d in datasets], axis=0),
            acts=np.concatenate([d.acts for d in datasets], axis=0),
        )


def parse_replay_file(path: Path, player: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    parser = ReplayParser(player=player)
    return parser.parse(text)


def parse_replay_dir(
    replay_dir: Path,
    player: int = 1,
    save_path: Path | None = None,
    checkpoint_every: int = 10,
) -> BCDataset:
    """
    Parse all .log files in replay_dir, checkpointing every N replays.

    Returns an empty BCDataset (obs/acts shape (0, STATE_SIZE) / (0, 2)) if no
    valid samples are found — callers must check len(dataset) before using it.
    """
    replay_dir = Path(replay_dir)
    log_files = sorted(replay_dir.glob("*.log"))
    print(f"Found {len(log_files)} replay log files in {replay_dir}")

    all_obs:  list[np.ndarray] = []
    all_acts: list[np.ndarray] = []
    skipped = 0
    parsed_since_ckpt = 0

    def _flush(label: str):
        if save_path is not None and all_obs:
            tmp = BCDataset(
                obs=np.stack(all_obs,  axis=0).astype(np.float32),
                acts=np.stack(all_acts, axis=0).astype(np.int64),
            )
            tmp.save(Path(save_path))
            print(f"  [{label}] checkpoint → {save_path}  ({len(tmp)} samples total)")

    for i, path in enumerate(log_files):
        try:
            samples = parse_replay_file(path, player=player)
            if not samples:
                skipped += 1
                continue
            for obs, act in samples:
                all_obs.append(obs)
                all_acts.append(act)
            parsed_since_ckpt += 1

            if parsed_since_ckpt >= checkpoint_every:
                _flush(f"{i + 1}/{len(log_files)} replays")
                parsed_since_ckpt = 0

        except Exception as e:
            log.warning("Failed to parse %s: %s", path.name, e)
            skipped += 1

    print(f"Done. {len(all_obs)} samples from {len(log_files) - skipped} replays "
          f"({skipped} skipped).")

    # Guard against empty dataset — np.stack([]) raises ValueError
    if not all_obs:
        log.warning("parse_replay_dir: no samples found in %s", replay_dir)
        empty_obs  = np.empty((0, STATE_SIZE_LOCAL), dtype=np.float32)
        empty_acts = np.empty((0, 2), dtype=np.int64)
        return BCDataset(obs=empty_obs, acts=empty_acts)

    dataset = BCDataset(
        obs=np.stack(all_obs,  axis=0).astype(np.float32),
        acts=np.stack(all_acts, axis=0).astype(np.int64),
    )

    if save_path is not None:
        dataset.save(save_path)

    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse Showdown replay logs into BC dataset")
    parser.add_argument("--replay-dir", default="data/replays")
    parser.add_argument("--out",        default="data/bc_dataset.npz")
    parser.add_argument("--player",     type=int, default=1, choices=[1, 2])
    args = parser.parse_args()

    ds = parse_replay_dir(Path(args.replay_dir), player=args.player, save_path=Path(args.out))
    print(f"obs shape:  {ds.obs.shape}")
    print(f"acts shape: {ds.acts.shape}")
    if len(ds) > 0:
        print(f"Action distribution slot 0: {np.bincount(ds.acts[:, 0], minlength=N_ACTIONS)}")
        print(f"Action distribution slot 1: {np.bincount(ds.acts[:, 1], minlength=N_ACTIONS)}")
