"""
BattleExtractor — structured BaseFeaturesExtractor for VGC doubles.

Replaces the default flat MLP with a per-mon encoder + slot-perspective
context that enables dual-slot coordination within SB3's MaskablePPO.

Architecture:
  1. Parse 222-dim obs into 4 per-mon feature groups (44 dims each) + field (46 dims)
  2. Encode each active mon with a shared MLP → 32-dim vector
  3. Encode field state → 16-dim vector
  4. Build two slot-perspective contexts (primary mon listed first per slot)
  5. Output: cat(slot0_ctx, slot1_ctx) → 288-dim features for actor + critic

Slot coordination: slot0_ctx = (our0, our1, opp0, opp1, field)
                   slot1_ctx = (our1, our0, opp0, opp1, field)  ← our slots swapped
The actor head for each slot therefore sees its own mon's encoding first,
with the partner slot's context visible in the second position.

Usage (in train.py):
    from battle_decisions.extractor import BattleExtractor
    policy_kwargs = dict(
        features_extractor_class=BattleExtractor,
        features_extractor_kwargs=dict(features_dim=288),
        net_arch=[128],
    )
"""
from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BattleExtractor(BaseFeaturesExtractor):
    """
    Structured feature extractor for VGC doubles.

    Parses the 222-dim obs into per-mon feature groups, encodes each with a
    shared MLP, then produces two slot-perspective context vectors — one per
    active slot — where the primary mon is listed first.

    Output shape: (batch, 288)
    """

    # Per active mon: HP(1) + vuln(2) + boosts(7) + moves(28) + status(6) = 44
    MON_DIM: int = 44
    # Field state: bench_hp(4) + weather_flags(4) + field_flags(5) +
    #              bench_status(24) + weather_counts(4) + field_counts(5) = 46
    FIELD_DIM: int = 46

    # Encoded dimensions
    _MON_ENC_DIM:   int = 32
    _FIELD_ENC_DIM: int = 16

    # features_dim = 2 slots × (4 mons × 32 + field 16) = 2 × 144 = 288
    _FEATURES_DIM: int = 288

    def __init__(self, observation_space, features_dim: int = 288) -> None:
        assert features_dim == self._FEATURES_DIM, (
            f"BattleExtractor requires features_dim={self._FEATURES_DIM}, got {features_dim}"
        )
        super().__init__(observation_space, features_dim)

        # Shared encoder — same weights applied to each of the 4 active mon slots
        self.mon_enc = nn.Sequential(
            nn.Linear(self.MON_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, self._MON_ENC_DIM),
            nn.ReLU(),
        )

        # Field + bench state encoder
        self.field_enc = nn.Sequential(
            nn.Linear(self.FIELD_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, self._FIELD_ENC_DIM),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Parse obs into per-mon groups, encode, and return dual-slot context."""
        # ── per-mon feature groups (44 dims each) ───────────────────────────
        # Obs layout (from battle/encoder.py):
        #   [0:4]      active HP (our0, our1, opp0, opp1)
        #   [4:12]     vulnerability: our0vs(opp0,opp1), our1vs(opp0,opp1),
        #                             opp0vs(our0,our1), opp1vs(our0,our1)
        #   [12:40]    stat boosts: 4 slots × 7 stats
        #   [40:96]    our moves: 2 slots × 4 moves × 7 features
        #   [96:100]   bench HP
        #   [100:109]  weather/field binary flags (4+5)
        #   [109:133]  active status: 4 slots × 6
        #   [133:157]  bench status: 4 bench × 6
        #   [157:213]  opp revealed moves: 2 slots × 4 moves × 7 features
        #   [213:222]  weather/field turn counts (4+5)
        our0 = torch.cat([
            obs[:, 0:1],     # HP
            obs[:, 4:6],     # vuln vs opp0, opp1
            obs[:, 12:19],   # boosts
            obs[:, 40:68],   # 4 moves
            obs[:, 109:115], # status
        ], dim=1)            # (B, 44)

        our1 = torch.cat([
            obs[:, 1:2],
            obs[:, 6:8],
            obs[:, 19:26],
            obs[:, 68:96],
            obs[:, 115:121],
        ], dim=1)

        opp0 = torch.cat([
            obs[:, 2:3],
            obs[:, 8:10],
            obs[:, 26:33],
            obs[:, 157:185],  # revealed opp moves
            obs[:, 121:127],
        ], dim=1)

        opp1 = torch.cat([
            obs[:, 3:4],
            obs[:, 10:12],
            obs[:, 33:40],
            obs[:, 185:213],
            obs[:, 127:133],
        ], dim=1)

        field = torch.cat([
            obs[:, 96:100],   # bench HP
            obs[:, 100:104],  # weather flags
            obs[:, 104:109],  # field flags (terrains + trick room)
            obs[:, 133:157],  # bench status
            obs[:, 213:217],  # weather turn counts
            obs[:, 217:222],  # field turn counts
        ], dim=1)             # (B, 46)

        # ── encode ──────────────────────────────────────────────────────────
        v0 = self.mon_enc(our0)    # (B, 32)
        v1 = self.mon_enc(our1)
        u0 = self.mon_enc(opp0)
        u1 = self.mon_enc(opp1)
        fv = self.field_enc(field)  # (B, 16)

        # ── slot-perspective contexts ────────────────────────────────────────
        # Slot 0: own mon first, then partner, then opponents, then field
        slot0_ctx = torch.cat([v0, v1, u0, u1, fv], dim=1)  # (B, 144)
        # Slot 1: own mon first (our slots swapped vs slot0)
        slot1_ctx = torch.cat([v1, v0, u0, u1, fv], dim=1)  # (B, 144)

        return torch.cat([slot0_ctx, slot1_ctx], dim=1)      # (B, 288)
