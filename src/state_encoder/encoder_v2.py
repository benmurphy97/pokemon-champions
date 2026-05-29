"""
EncoderV2 — main encoder for Pokémon Champions Regulation M-A.

Interface:
    state = GameState(...)
    encoder = EncoderV2.from_data_dir("data/")
    tensor = encoder.encode(state)   # np.ndarray, shape (STATE_DIM,)

Block layout and dimensions (exact):
  your_active  × 2   YOUR_ACTIVE_DIM  = 418  → total  836
  your_bench   × 2   YOUR_BENCH_DIM   = 393  → total  786
  opp_active   × 2   OPP_ACTIVE_DIM   = 400  → total  800
  opp_bench    × 4   OPP_BENCH_DIM    = 381  → total 1524
  field                FIELD_DIM       =  38
  context             CONTEXT_DIM      = 120
  ─────────────────────────────────────────────
  STATE_DIM                            = 4104

The brief approximates 3450; the exact count is 4104 because the brief
undercounts some sub-vectors (move feat=64 not 55, ability=35 not 25,
item=20 not 19). The projection layer after the encoder maps this to a
fixed model input size regardless.

Encoding changelog:
  v1 — 222-dim flat encoding (src/battle/encoder.py)
  v2 — this file; full M-A spec with SP, Stat Alignment, Mega, no IVs, no Tera
  v3 — triggered by: new Tera/Z-Move mechanic, new Mega ability outside current
       flags, or Regulation M-B adding Pokémon with novel mechanics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .game_state import GameState
from .move_encoder import MoveDB
from .item_encoder import load_item_categories
from .pokemon_encoder import (
    encode_your_active, encode_your_bench,
    encode_opp_active, encode_opp_bench_slot,
    YOUR_ACTIVE_DIM, YOUR_BENCH_DIM, OPP_ACTIVE_DIM, OPP_BENCH_DIM,
)
from .field_encoder import encode_field, FIELD_DIM
from .battle_context import encode_context, CONTEXT_DIM

STATE_DIM = (
    2 * YOUR_ACTIVE_DIM   # your active ×2
    + 2 * YOUR_BENCH_DIM  # your bench  ×2
    + 2 * OPP_ACTIVE_DIM  # opp active  ×2
    + 4 * OPP_BENCH_DIM   # opp bench   ×4
    + FIELD_DIM
    + CONTEXT_DIM
)


class EncoderV2:
    """
    Converts a GameState snapshot to a fixed-length numpy observation vector.

    Construct with from_data_dir() to load move DB and item categories.
    The encoder is stateless after construction — safe to share across threads.
    """

    VERSION = "v2"

    def __init__(self, move_db: MoveDB) -> None:
        self._db = move_db

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_data_dir(cls, data_dir: str | Path) -> "EncoderV2":
        """
        Build EncoderV2 from the standard data directory layout:
          data/moves.json
          data/champions_items.json
        """
        data_dir = Path(data_dir)
        move_db = MoveDB(data_dir / "moves.json")
        load_item_categories(data_dir / "champions_items.json")
        return cls(move_db)

    # ── Encoding ─────────────────────────────────────────────────────────────

    def encode(self, state: GameState) -> np.ndarray:
        """
        Encode a GameState to a (STATE_DIM,) float32 numpy array.

        Caller must supply exactly:
          state.your_active  — 2 YourActivePokemon
          state.your_bench   — 2 YourBenchPokemon  (is_fainted=True for absent)
          state.opp_active   — 2 OppActivePokemon
          state.opp_bench    — 4 OppBenchSlot
          state.field        — FieldState
          state.context      — BattleContext
        """
        parts: list[list[float]] = []

        # Your active (×2) — slot 0 first, then slot 1
        for mon in state.your_active[:2]:
            parts.append(encode_your_active(mon, self._db))

        # Your bench (×2)
        for mon in state.your_bench[:2]:
            parts.append(encode_your_bench(mon, self._db))

        # Opponent active (×2)
        for mon in state.opp_active[:2]:
            parts.append(encode_opp_active(mon, self._db))

        # Opponent bench (×4)
        for slot in state.opp_bench[:4]:
            parts.append(encode_opp_bench_slot(slot, self._db))

        # Field
        parts.append(encode_field(state.field))

        # Battle context
        parts.append(encode_context(state.context))

        flat: list[float] = []
        for p in parts:
            flat.extend(p)

        arr = np.array(flat, dtype=np.float32)
        assert arr.shape[0] == STATE_DIM, (
            f"EncoderV2 output shape mismatch: {arr.shape[0]} != {STATE_DIM}"
        )
        return arr

    def encode_batch(self, states: list[GameState]) -> np.ndarray:
        """Encode a list of states → (N, STATE_DIM) float32 array."""
        return np.stack([self.encode(s) for s in states])
