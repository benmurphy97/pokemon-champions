"""The policy network: reads one decision's encoded state and scores every possible action.

A small transformer over 14 tokens: a decision (query) token, a field token, and one token for
each of the 12 Pokémon (own six, then the opponent's six). Moves are scored against a learned
embedding of every move, and switches by pointing at one of the own-team tokens, so the same
network handles any team.

    logits, target_logits = net(batch)       # batch: dict of tensors from encoding.encode
"""

import torch
from torch import nn

from champions.encoding import GLOBAL_NUM, MON_NUM, N_MOVE_SLOTS, N_TOKENS, N_ACTIONS, SWITCH_BASE
from champions.vocab import (ABILITIES, ITEMS, MOVE_CATS, MOVE_NUM, MOVES, NUM_ABILITIES, NUM_ITEMS,
                             NUM_MOVES, NUM_SPECIES, SPECIES_STATS, SPECIES_TYPES, TARGETS, TYPES, TYPE_CHART)

N_TARGET_CLASSES = 4
DAMAGE_FEATURES = 9


class PolicyNet(nn.Module):
    def __init__(self, dim: int = 128, layers: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        # static facts about each species / move, fixed
        self.register_buffer("species_types", torch.as_tensor(SPECIES_TYPES))
        self.register_buffer("species_stats", torch.as_tensor(SPECIES_STATS))
        self.register_buffer("move_cats", torch.as_tensor(MOVE_CATS))
        self.register_buffer("move_num", torch.as_tensor(MOVE_NUM))
        self.register_buffer("type_chart", torch.as_tensor(TYPE_CHART))

        self.species_emb = nn.Embedding(NUM_SPECIES, dim)
        self.item_emb = nn.Embedding(NUM_ITEMS, dim)
        self.ability_emb = nn.Embedding(NUM_ABILITIES, dim)
        self.type_emb = nn.Embedding(len(TYPES) + 1, 16, padding_idx=0)
        self.cat_emb = nn.Embedding(4, 8, padding_idx=0)
        self.target_emb = nn.Embedding(len(TARGETS) + 1, 8, padding_idx=0)
        self.species_static = nn.Linear(16 * 2 + 6, dim)
        self.move_static = nn.Linear(16 + 8 + 8 + 3, dim)
        self.move_in = nn.Embedding(NUM_MOVES, dim, padding_idx=0)
        self.mon_numbers = nn.Linear(MON_NUM, dim)
        self.side_emb = nn.Embedding(2, dim)
        self.position_emb = nn.Embedding(6, dim)
        self.actor_emb = nn.Parameter(torch.zeros(dim))
        self.field = nn.Linear(GLOBAL_NUM, dim)
        self.kind_emb = nn.Embedding(2, dim)
        self.slot_emb = nn.Embedding(2, dim)

        block = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(dim)

        self.move_out = nn.Embedding(NUM_MOVES, dim, padding_idx=0)
        self.move_query = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.switch_query = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        # Damage-awareness: facts about each move against each foe (type effectiveness, same-type
        # bonus, attack-vs-defence ratio) are inputs; what to do with them is learned.
        self.damage_scalar = nn.Sequential(nn.Linear(DAMAGE_FEATURES, 32), nn.GELU(), nn.Linear(32, 1))
        self.damage_context = nn.Linear(DAMAGE_FEATURES, dim)
        self.target_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, N_TARGET_CLASSES))

    def move_table(self) -> torch.Tensor:
        """One vector per move for scoring: learned embedding plus its static description."""
        cats = self.move_cats
        static = torch.cat([self.type_emb(cats[:, 0]), self.cat_emb(cats[:, 1]),
                            self.target_emb(cats[:, 2]), self.move_num], dim=-1)
        return self.move_out.weight + self.move_static(static)

    def encode_moves(self, ids: torch.Tensor) -> torch.Tensor:
        cats = self.move_cats[ids]
        static = torch.cat([self.type_emb(cats[..., 0]), self.cat_emb(cats[..., 1]),
                            self.target_emb(cats[..., 2]), self.move_num[ids]], dim=-1)
        return self.move_in(ids) + self.move_static(static)

    def tokens(self, b: dict) -> torch.Tensor:
        species = b["species"]
        types = self.type_emb(self.species_types[species]).flatten(-2)                  # [B,12,32]
        static = self.species_static(torch.cat([types, self.species_stats[species]], dim=-1))
        moves = self.encode_moves(b["moves"])                                           # [B,12,4,D]
        known = (b["moves"] > 0).unsqueeze(-1).float()
        move_bag = (moves * known).sum(2) / known.sum(2).clamp(min=1)
        x = (self.species_emb(species) + static + self.item_emb(b["items"]) + self.ability_emb(b["abilities"])
             + move_bag + self.mon_numbers(b["numbers"]))
        side = torch.arange(N_TOKENS, device=species.device) // 6
        x = x + self.side_emb(side) + self.position_emb(torch.arange(N_TOKENS, device=species.device) % 6)
        actor = b["actor"]
        rows = (actor >= 0).nonzero(as_tuple=True)[0]
        marker = torch.zeros_like(x)
        marker[rows, actor[rows]] = self.actor_emb
        return x + marker

    def damage_features(self, b: dict) -> torch.Tensor:
        """[B, NUM_MOVES, DAMAGE_FEATURES]: how each move would fare against the two foes on the field."""
        species, numbers = b["species"], b["numbers"]
        rows = torch.arange(species.shape[0], device=species.device)
        actor = b["actor"].clamp(min=0)
        a_types, a_stats = self.species_types[species[rows, actor]], self.species_stats[species[rows, actor]]
        has_actor = (b["actor"] >= 0).float()

        move_type, category = self.move_cats[:, 0], self.move_cats[:, 1]      # [M]
        power = self.move_num[:, 0]
        attacking = (category <= 2).float()                                   # physical or special
        special = (category == 2).long()
        stab = ((move_type[None, :] == a_types[:, :1]) | (move_type[None, :] == a_types[:, 1:])) & (move_type[None, :] > 0)

        feats = []
        for flag in (2, 3):                                                   # foe in slot a, then slot b
            flags = numbers[:, 6:, flag] * (1 - numbers[:, 6:, 1])            # on the field and not fainted
            present, which = flags.max(dim=1)
            foe = species[rows, 6 + which]
            f_types, f_stats = self.species_types[foe], self.species_stats[foe]
            eff = (self.type_chart[move_type[None, :, None], f_types[:, None, :]]).prod(-1)       # [B, M]
            atk = torch.where(special[None, :] == 1, a_stats[:, 3:4], a_stats[:, 1:2])           # [B, M]
            dfn = torch.where(special[None, :] == 1, f_stats[:, 4:5], f_stats[:, 2:3])
            ratio = (atk / dfn.clamp(min=0.05)).clamp(0.2, 4.0)
            damage = power[None, :] * (1 + 0.5 * stab.float()) * eff * ratio * attacking[None, :]
            on = (present > 0.5).float()[:, None] * has_actor[:, None]
            feats += [eff * on, ratio * on / 4, damage * on]
        feats += [stab.float() * has_actor[:, None], power[None, :].expand(species.shape[0], -1) * attacking[None, :],
                  (1 - attacking)[None, :].expand(species.shape[0], -1)]
        return torch.stack(feats, dim=-1)

    def forward(self, b: dict, target_for_move: torch.Tensor | None = None):
        """Return (action logits [B, N_ACTIONS], target logits [B, 4]).

        Target logits are conditioned on `target_for_move` (the move taken, or being considered).
        """
        mons = self.tokens(b)
        query = (self.kind_emb(b["kind"]) + self.slot_emb(b["slot"])).unsqueeze(1)
        field = self.field(b["globals"]).unsqueeze(1)
        seq = torch.cat([query, field, mons], dim=1)
        pad = torch.cat([torch.zeros(seq.shape[0], 2, dtype=torch.bool, device=seq.device), ~b["present"]], dim=1)
        out = self.out_norm(self.encoder(seq, src_key_padding_mask=pad))
        q, own = out[:, 0], out[:, 2:8]

        damage = self.damage_features(b)
        move_logits = (self.move_query(q) @ self.move_table().T                           # [B, NUM_MOVES]
                       + self.damage_scalar(damage).squeeze(-1)
                       + torch.einsum("bmd,bd->bm", self.damage_context(damage), q) / self.dim ** 0.5)
        switch_logits = torch.einsum("bd,bkd->bk", self.switch_query(q), own)             # [B, 6]
        logits = torch.cat([move_logits, switch_logits], dim=-1).masked_fill(~b["mask"], -1e9)

        target_logits = None
        if target_for_move is not None:
            moves = target_for_move.clamp(min=0, max=NUM_MOVES - 1)
            target_logits = self.target_head(torch.cat([q, self.move_table()[moves]], dim=-1))
        return logits, target_logits
