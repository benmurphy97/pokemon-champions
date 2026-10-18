"""The policy network: reads one decision's encoded state and scores every possible action.

A small transformer over 14 tokens: a decision (query) token, a field token, and one token for
each of the 12 Pokémon (own six, then the opponent's six). Moves are scored against a learned
embedding of every move, and switches by pointing at one of the own-team tokens, so the same
network handles any team.

    logits, target_logits = net(batch)       # batch: dict of tensors from encoding.encode
"""

import torch
from torch import nn

from champions.damage import DamageCalc, boost_multiplier
from champions.encoding import GLOBAL_NUM, MON_NUM, N_KINDS, N_MOVE_SLOTS, N_TOKENS, N_ACTIONS, STAT_SCALE, SWITCH_BASE
from champions.vocab import (ABILITIES, ITEMS, MOVE_CATS, MOVE_NUM, MOVES, NUM_ABILITIES, NUM_ITEMS,
                             NUM_MOVES, NUM_SPECIES, SPECIES_STATS, SPECIES_TYPES, TARGETS, TYPES, TYPE_CHART)

N_TARGET_CLASSES = 4
DAMAGE_FEATURES = 13
THREAT_FEATURES = 8
# A foe's unknown spread, as stat points added to base stats (HP, atk, def, spa, spd, spe): a typical
# spread, the most bulk it could have, and the least.
FOE_SPREADS = {"mid": [91.0, 36, 36, 36, 36, 36], "bulky": [107.0, 52, 52, 52, 52, 52], "frail": [75.0, 20, 20, 20, 20, 20]}
SURE_KO, MAYBE_KO = 0.85 / 0.925, 1.0 / 0.925   # lowest / highest damage roll relative to the average one
GENERIC_POWER = 90.0   # stand-in for a foe's unrevealed STAB attack


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
        self.calc = DamageCalc()

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
        self.stats_proj = nn.Linear(6, dim)      # your own Pokémon's real stats
        self.speed_proj = nn.Linear(2, dim)      # who outspeeds the two foes on the field
        self.threat_proj = nn.Linear(THREAT_FEATURES, dim)   # how hard the foes can hit our two active Pokémon
        # What the partner Pokémon has already been told to do this turn (see encoding.encode). Starts at
        # zero, so a network trained without it behaves the same.
        self.partner_move = nn.Linear(dim, dim)
        self.partner_switch = nn.Embedding(7, dim)
        self.partner_target = nn.Embedding(5, dim)
        for module in (self.partner_move, self.partner_switch, self.partner_target):
            for weight in module.parameters():
                nn.init.zeros_(weight)
        self.side_emb = nn.Embedding(2, dim)
        self.position_emb = nn.Embedding(6, dim)
        self.actor_emb = nn.Parameter(torch.zeros(dim))
        self.field = nn.Linear(GLOBAL_NUM, dim)
        self.kind_emb = nn.Embedding(N_KINDS, dim)
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
        self.value_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))   # used by reinforcement learning
        self.target_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, N_TARGET_CLASSES))

    def config(self) -> dict:
        """Settings that rebuild this network (stored in checkpoints, see model_io)."""
        layer = self.encoder.layers[0]
        return {"dim": self.dim, "layers": len(self.encoder.layers), "heads": layer.self_attn.num_heads,
                "dropout": layer.dropout.p}

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
             + move_bag + self.mon_numbers(b["numbers"]) + self.stats_proj(b["stats"]))
        side = torch.arange(N_TOKENS, device=species.device) // 6
        x = x + self.side_emb(side) + self.position_emb(torch.arange(N_TOKENS, device=species.device) % 6)
        actor = b["actor"]
        rows = (actor >= 0).nonzero(as_tuple=True)[0]
        marker = torch.zeros_like(x)
        marker[rows, actor[rows]] = self.actor_emb
        return x + marker

    @staticmethod
    def estimate_stats(base: torch.Tensor) -> torch.Tensor:
        """Guess a Pokémon's real stats from its base stats alone (about half the stat points, neutral nature).

        Used for the opponent, whose spread is unknown, and as a fallback for your own team.
        """
        return base + torch.tensor([91.0, 36, 36, 36, 36, 36], device=base.device)

    def actor_and_foes(self, b: dict):
        """Real-or-estimated stats of the acting Pokémon, and the foe in each field slot (a, b)."""
        species, numbers = b["species"], b["numbers"]
        rows = torch.arange(species.shape[0], device=species.device)
        actor = b["actor"].clamp(min=0)
        has_actor = (b["actor"] >= 0).float()
        a_species = species[rows, actor]
        a_true = b["stats"][rows, actor] * STAT_SCALE
        a_stats = torch.where(a_true[:, 1:2] > 0, a_true, self.estimate_stats(self.species_stats[a_species] * 255))
        a_boosts = numbers[rows, actor, 13:20] * 6
        foes = []
        for flag in (2, 3):                                                   # foe in slot a, then slot b
            flags = numbers[:, 6:, flag] * (1 - numbers[:, 6:, 1])            # on the field and not fainted
            present, which = flags.max(dim=1)
            foe = species[rows, 6 + which]
            foes.append({"present": (present > 0.5).float() * has_actor, "species": foe,
                         "token": 6 + which, "stats": self.estimate_stats(self.species_stats[foe] * 255),
                         "boosts": numbers[rows, 6 + which, 13:20] * 6})
        return a_species, a_stats, a_boosts, has_actor, foes

    boost_multiplier = staticmethod(boost_multiplier)

    def speed_features(self, b: dict) -> torch.Tensor:
        """[B, 2]: log of (our speed / their speed) against the foe in slot a and slot b, boosts included."""
        _, a_stats, a_boosts, _, foes = self.actor_and_foes(b)
        ours = a_stats[:, 5] * self.boost_multiplier(a_boosts[:, 4])
        out = []
        for foe in foes:
            theirs = foe["stats"][:, 5] * self.boost_multiplier(foe["boosts"][:, 4])
            out.append(torch.log(ours / theirs) * foe["present"])
        return torch.stack(out, dim=-1)

    def fighter(self, b: dict, rows: torch.Tensor, token: torch.Tensor, stats: torch.Tensor, boosts: torch.Tensor) -> dict:
        """What the damage calculator needs to know about the Pokémon in `token`."""
        status = b["numbers"][rows, token, 7:13]
        return {"types": self.species_types[b["species"][rows, token]], "stats": stats, "boosts": boosts,
                "item": b["items"][rows, token], "ability": b["abilities"][rows, token],
                "burn": status[:, 1], "statused": status.sum(-1)}

    def damage_features(self, b: dict) -> torch.Tensor:
        """[B, NUM_MOVES, DAMAGE_FEATURES]: how each move would fare against the two foes on the field.

        Per foe: type effectiveness, damage as a share of its full and of its current health, and
        whether the move would surely / possibly knock it out given the foe's unknown stat spread.
        """
        numbers, batch = b["numbers"], b["species"].shape[0]
        rows = torch.arange(batch, device=numbers.device)
        a_species, a_stats, a_boosts, has_actor, foes = self.actor_and_foes(b)
        actor = b["actor"].clamp(min=0)
        attacker = self.fighter(b, rows, actor, a_stats, a_boosts)

        move_type, category = self.move_cats[:, 0], self.move_cats[:, 1]      # [M]
        spread_to = lambda t: t.unsqueeze(0).expand(batch, -1)
        m_type, m_cat, m_target = spread_to(move_type), spread_to(category), spread_to(self.move_cats[:, 2])
        m_power = spread_to(self.move_num[:, 0] * 150)
        attacking = (category <= 2).float()
        two_targets = ((foes[0]["present"] > 0.5) & (foes[1]["present"] > 0.5)).float()
        weather, terrain = b["globals"][:, 1:5], b["globals"][:, 5:9]
        a_types = self.species_types[a_species]
        stab = ((move_type[None, :] == a_types[:, :1]) | (move_type[None, :] == a_types[:, 1:])) & (move_type[None, :] > 0)

        feats = []
        for foe in foes:
            base = self.species_stats[foe["species"]] * 255
            defender = self.fighter(b, rows, foe["token"], None, foe["boosts"])
            damage, hp = {}, {}
            for name, spread in FOE_SPREADS.items():
                defender["stats"] = base + torch.tensor(spread, device=base.device)
                damage[name] = self.calc(attacker, defender, m_type, m_cat, m_power, m_target, two_targets, weather, terrain)
                hp[name] = defender["stats"][:, 0:1]
            left = numbers[rows, foe["token"], 0:1]                            # share of its health it still has
            eff = self.type_chart[move_type[None, :, None], defender["types"][:, None, :]].prod(-1)
            on = foe["present"][:, None]
            sure = SURE_KO * damage["bulky"] >= (left * hp["bulky"]).clamp(min=1.0)
            maybe = MAYBE_KO * damage["frail"] >= (left * hp["frail"]).clamp(min=1.0)
            feats += [eff * on, (damage["mid"] / hp["mid"]).clamp(max=1.5) * on,
                      (damage["mid"] / (left * hp["mid"]).clamp(min=1.0)).clamp(max=3.0) / 3 * on,
                      (sure & (damage["bulky"] > 0)).float() * on, (maybe & (damage["frail"] > 0)).float() * on]
        feats += [stab.float() * has_actor[:, None], m_power / 150 * attacking[None, :], (1 - attacking)[None, :].expand(batch, -1)]
        return torch.stack(feats, dim=-1)

    def threat_features(self, b: dict) -> torch.Tensor:
        """[B, THREAT_FEATURES]: for each foe on the field and each of our two active Pokémon, how much of
        its current health the foe's strongest revealed move takes, and a guess at its best unrevealed
        same-type attack. Capped at 3 (three times over) and divided by 3.
        """
        numbers, batch = b["numbers"], b["species"].shape[0]
        rows = torch.arange(batch, device=numbers.device)
        _, _, _, _, foes = self.actor_and_foes(b)
        mine = []                                                              # our active Pokémon, slot a then b
        for slot in (0, 1):
            flags = numbers[:, :6, 2 + slot] * (1 - numbers[:, :6, 1])
            present, token = flags.max(dim=1)
            true = b["stats"][rows, token] * STAT_SCALE
            stats = torch.where(true[:, 1:2] > 0, true, self.estimate_stats(self.species_stats[b["species"][rows, token]] * 255))
            mine.append((token, (present > 0.5).float(), stats, self.fighter(b, rows, token, stats, numbers[rows, token, 13:20] * 6)))
        two_targets = (mine[0][1] * mine[1][1])
        weather, terrain = b["globals"][:, 1:5], b["globals"][:, 5:9]

        out = []
        for foe in foes:
            attacker = self.fighter(b, rows, foe["token"], foe["stats"], foe["boosts"])
            ids = b["moves"][rows, foe["token"]]                               # [B, 4] revealed moves
            real = (self.move_cats[ids, 0], self.move_cats[ids, 1], self.move_num[ids, 0] * 150, self.move_cats[ids, 2])
            generic_cat = torch.where(foe["stats"][:, 3] > foe["stats"][:, 1], 2, 1)[:, None].expand(-1, 2)
            guess = (attacker["types"], generic_cat, torch.full_like(attacker["types"], GENERIC_POWER, dtype=torch.float32),
                     torch.ones_like(attacker["types"]))
            for token, here, stats, defender in mine:
                current = (numbers[rows, token, 0] * stats[:, 0]).clamp(min=1.0)[:, None]
                for moves in (real, guess):
                    hit = self.calc(attacker, defender, *moves, two_targets, weather, terrain)
                    out.append((hit / current).max(dim=1).values.clamp(max=3.0) / 3 * foe["present"] * here)
        return torch.stack(out, dim=-1)

    def partner_plan(self, b: dict) -> torch.Tensor:
        """[B, D]: the partner's already-chosen action (a move and its target, or a switch), zero if none."""
        action = b["partner_action"]
        is_move = ((action > 0) & (action < SWITCH_BASE)).unsqueeze(-1)
        move = self.partner_move(self.encode_moves(action.clamp(0, NUM_MOVES - 1))) * is_move
        switch = self.partner_switch(torch.where(action >= SWITCH_BASE, action - SWITCH_BASE + 1, torch.zeros_like(action)))
        return move + switch + self.partner_target(b["partner_target"])

    def trunk(self, b: dict):
        """Run the transformer; returns the decision token (B, D) and the own-team tokens (B, 6, D)."""
        mons = self.tokens(b)
        query = (self.kind_emb(b["kind"]) + self.slot_emb(b["slot"]) + self.speed_proj(self.speed_features(b))
                 + self.threat_proj(self.threat_features(b)) + self.partner_plan(b)).unsqueeze(1)
        field = self.field(b["globals"]).unsqueeze(1)
        seq = torch.cat([query, field, mons], dim=1)
        pad = torch.cat([torch.zeros(seq.shape[0], 2, dtype=torch.bool, device=seq.device), ~b["present"]], dim=1)
        out = self.out_norm(self.encoder(seq, src_key_padding_mask=pad))
        return out[:, 0], out[:, 2:8]

    def action_logits(self, b: dict, q: torch.Tensor, own: torch.Tensor) -> torch.Tensor:
        damage = self.damage_features(b)
        move_logits = (self.move_query(q) @ self.move_table().T                           # [B, NUM_MOVES]
                       + self.damage_scalar(damage).squeeze(-1)
                       + torch.einsum("bmd,bd->bm", self.damage_context(damage), q) / self.dim ** 0.5)
        switch_logits = torch.einsum("bd,bkd->bk", self.switch_query(q), own)             # [B, 6]
        return torch.cat([move_logits, switch_logits], dim=-1).masked_fill(~b["mask"], -1e9)

    def target_logits_for(self, q: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        moves = moves.clamp(min=0, max=NUM_MOVES - 1)
        return self.target_head(torch.cat([q, self.move_table()[moves]], dim=-1))

    def forward(self, b: dict, target_for_move: torch.Tensor | None = None):
        """Return (action logits [B, N_ACTIONS], target logits [B, 4]).

        Target logits are conditioned on `target_for_move` (the move taken, or being considered).
        """
        q, own = self.trunk(b)
        logits = self.action_logits(b, q, own)
        target_logits = None if target_for_move is None else self.target_logits_for(q, target_for_move)
        return logits, target_logits

    def evaluate(self, b: dict, action: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor):
        """Re-score recorded decisions for reinforcement learning.

        Returns (log-probability of the whole decision, entropy of the action choice, value estimate,
        full action log-distribution). A decision's probability is that of the action times, for
        single-target moves (target >= 0), that of the target among the allowed ones.
        """
        q, own = self.trunk(b)
        log_probs = torch.log_softmax(self.action_logits(b, q, own), dim=-1)
        logp = log_probs.gather(1, action[:, None]).squeeze(1)
        entropy = -(log_probs.exp() * log_probs.clamp(min=-30)).sum(-1)

        has_target = target >= 0
        if has_target.any():
            tl = self.target_logits_for(q, action).masked_fill(~target_mask, -1e9)
            tlp = torch.log_softmax(tl, dim=-1).gather(1, target.clamp(min=0)[:, None]).squeeze(1)
            logp = logp + torch.where(has_target, tlp, torch.zeros_like(tlp))
        return logp, entropy, self.value_head(q).squeeze(-1), log_probs
