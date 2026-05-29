"""
BattleStateEncoder — structured BaseFeaturesExtractor for VGC doubles.

Parses the 4104-dim EncoderV2 obs into per-mon blocks, encodes each
with a shared MLP, then produces two slot-perspective context vectors
for dual-slot coordination in SB3's MaskablePPO.

Block layout (from EncoderV2 / STATE_DIM=4104):
  your_active[0]:  [0   : 418]
  your_active[1]:  [418 : 836]
  your_bench[0]:   [836 : 1229]
  your_bench[1]:   [1229: 1622]
  opp_active[0]:   [1622: 2022]
  opp_active[1]:   [2022: 2422]
  opp_bench[0-3]:  [2422: 3946]
  field:           [3946: 3984]
  context:         [3984: 4104]

Architecture:
  1. Slice 4 active-mon blocks + field+context block
  2. Encode each active mon with a shared MLP → 32-dim
     (separate encoders for your mons vs opp mons)
  3. Encode field+context → 16-dim
  4. Build two slot-perspective contexts (own mon listed first per slot)
  5. Output: cat(slot0_ctx, slot1_ctx) → 288-dim features

Slot coordination:
  slot0_ctx = cat(your0, your1, opp0, opp1, field_ctx)  = (B, 144)
  slot1_ctx = cat(your1, your0, opp0, opp1, field_ctx)  = (B, 144)
  ← your slots swapped so each actor sees its own mon first

Usage (in training script):
    from stage3_battle_decisions.architecture.extractor import BattleStateEncoder
    policy_kwargs = dict(
        features_extractor_class=BattleStateEncoder,
        features_extractor_kwargs=dict(features_dim=288),
        net_arch=[128],
    )
"""
from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from state_encoder.pokemon_encoder import YOUR_ACTIVE_DIM, YOUR_BENCH_DIM, OPP_ACTIVE_DIM
from state_encoder.field_encoder import FIELD_DIM
from state_encoder.battle_context import CONTEXT_DIM

# ── Block offsets (derived from EncoderV2 layout) ─────────────────────────────

_YOUR0_START = 0
_YOUR0_END   = YOUR_ACTIVE_DIM
_YOUR1_START = YOUR_ACTIVE_DIM
_YOUR1_END   = 2 * YOUR_ACTIVE_DIM

_OPP0_START  = 2 * YOUR_ACTIVE_DIM + 2 * YOUR_BENCH_DIM
_OPP0_END    = _OPP0_START + OPP_ACTIVE_DIM
_OPP1_START  = _OPP0_END
_OPP1_END    = _OPP1_START + OPP_ACTIVE_DIM

_FIELD_START   = _OPP1_END + 4 * 381        # 4 × OPP_BENCH_DIM
_FIELD_END     = _FIELD_START + FIELD_DIM
_CONTEXT_START = _FIELD_END
_CONTEXT_END   = _CONTEXT_START + CONTEXT_DIM  # = STATE_DIM


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim), nn.ReLU(),
    )


class BattleStateEncoder(BaseFeaturesExtractor):
    """
    Structured feature extractor for VGC doubles.

    Slices the 4104-dim EncoderV2 obs into per-mon blocks and a field+context
    block, encodes each with a shared MLP, then produces two slot-perspective
    context vectors — one per active slot — with the primary mon listed first.

    Output shape: (batch, 288)
    """

    _MON_ENC_DIM:   int = 32   # encoded dim per active mon (your or opp)
    _FIELD_ENC_DIM: int = 16   # encoded dim for field + context combined
    _FEATURES_DIM:  int = 288  # 2 × (4 mons × 32 + field 16)

    def __init__(self, observation_space, features_dim: int = 288) -> None:
        assert features_dim == self._FEATURES_DIM, (
            f"BattleStateEncoder requires features_dim={self._FEATURES_DIM}, got {features_dim}"
        )
        super().__init__(observation_space, features_dim)

        # Shared encoder for your own active mons (rich 418-dim blocks)
        self.your_mon_enc = _mlp(YOUR_ACTIVE_DIM, 128, self._MON_ENC_DIM)

        # Shared encoder for opponent active mons (400-dim blocks)
        self.opp_mon_enc = _mlp(OPP_ACTIVE_DIM, 128, self._MON_ENC_DIM)

        # Field + context encoder (38 + 120 = 158 dims)
        self.field_enc = _mlp(FIELD_DIM + CONTEXT_DIM, 64, self._FIELD_ENC_DIM)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Slice obs into blocks, encode, and return dual-slot context."""
        # ── per-mon blocks ───────────────────────────────────────────────────
        your0 = obs[:, _YOUR0_START:_YOUR0_END]
        your1 = obs[:, _YOUR1_START:_YOUR1_END]
        opp0  = obs[:, _OPP0_START:_OPP0_END]
        opp1  = obs[:, _OPP1_START:_OPP1_END]

        # ── field + context ──────────────────────────────────────────────────
        field_ctx = torch.cat([
            obs[:, _FIELD_START:_FIELD_END],
            obs[:, _CONTEXT_START:_CONTEXT_END],
        ], dim=1)   # (B, 158)

        # ── encode ──────────────────────────────────────────────────────────
        v0 = self.your_mon_enc(your0)   # (B, 32)
        v1 = self.your_mon_enc(your1)
        u0 = self.opp_mon_enc(opp0)
        u1 = self.opp_mon_enc(opp1)
        fv = self.field_enc(field_ctx)  # (B, 16)

        # ── slot-perspective contexts ─────────────────────────────────────────
        # Each slot sees its own mon first so the actor head can attend to self
        slot0 = torch.cat([v0, v1, u0, u1, fv], dim=1)  # (B, 144)
        slot1 = torch.cat([v1, v0, u0, u1, fv], dim=1)  # (B, 144) — our slots swapped

        return torch.cat([slot0, slot1], dim=1)           # (B, 288)
