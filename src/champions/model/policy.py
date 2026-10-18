"""The policy network: reads one side's encoded turn and scores every joint action for its two Pokémon.

A small transformer over 14 tokens: a decision (query) token, a field token, and one token for each of
the 12 Pokémon (own six, then the opponent's six). Heads on top of it:

  * joint action: every legal (atom for slot a, atom for slot b) pair gets a score
        S(a, b) = u_A(a) + u_B(b) + w(a, b)
    where u is a per-slot score of the atom (move + target, switch, pass, locked) and w is a low-rank
    bilinear interaction gated by the decision token, so the two choices are made together. Softmax over
    the N_PAIRS pairs.
  * value: how well the game is going for this side, in [-1, 1] (reinforcement learning).
  * opponent: for each foe on the field, a distribution over its move (by vocabulary id) or switch
    (an auxiliary task; also gives search its likely opponent actions).
  * preview: a score for each of the 90 plans (four to bring, two of them to lead), from set-level
    embeddings of the four, so members interact.

    pair_logits, value, foe_logits = net(batch)        # batch: dict of tensors from encoding.encode
"""

from itertools import combinations

import torch
from torch import nn

from champions.model.damage import DamageCalc, boost_multiplier
from champions.model.encoding import (ATOM_LOCKED, ATOM_PASS, ATOM_SWITCH, GLOBAL_NUM, MON_NUM, N_ACTIONS, N_ATOMS,
                                      N_KINDS, N_MOVE_SLOTS, N_TARGETS, N_TOKENS, STAT_SCALE)
from champions.model.vocab import (LEARNABLE, MOVE_CATS, MOVE_NUM, NUM_ABILITIES, NUM_ITEMS, NUM_MOVES, NUM_SPECIES,
                                   SPECIES_STATS, SPECIES_TYPES, TARGETS, TYPES, TYPE_CHART)

SUBSETS = list(combinations(range(6), 4))     # the 15 ways to bring four of six
PAIRS = list(combinations(range(6), 2))       # the 15 ways to pick two leads
PLANS = [(s, p) for s, subset in enumerate(SUBSETS) for p, pair in enumerate(PAIRS) if set(pair) <= set(subset)]   # 90
DAMAGE_FEATURES = 13
THREAT_FEATURES = 8
INTERACTION_RANK = 64
# A foe's unknown spread, as stat points added to base stats (HP, atk, def, spa, spd, spe): a typical
# spread, the most bulk it could have, and the least.
FOE_SPREADS = {"mid": [91.0, 36, 36, 36, 36, 36], "bulky": [107.0, 52, 52, 52, 52, 52], "frail": [75.0, 20, 20, 20, 20, 20]}
SURE_KO, MAYBE_KO = 0.85 / 0.925, 1.0 / 0.925   # lowest / highest damage roll relative to the average one
GENERIC_POWER = 90.0   # stand-in for a foe's unrevealed STAB attack
NEG = -1e9


def mlp(d_in: int, d_out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))


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
        self.register_buffer("learnable", torch.as_tensor(LEARNABLE), persistent=False)
        self.calc = DamageCalc()

        # --- tokens ---
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
        self.side_emb = nn.Embedding(2, dim)
        self.position_emb = nn.Embedding(6, dim)
        self.actor_marks = nn.Parameter(torch.zeros(2, dim))   # added to the token of the Pokémon in slot a / slot b
        self.field = nn.Linear(GLOBAL_NUM, dim)
        self.kind_emb = nn.Embedding(N_KINDS, dim)
        self.speed_proj = nn.Linear(4, dim)      # does each of our two active Pokémon outspeed each foe on the field
        self.threat_proj = nn.Linear(THREAT_FEATURES, dim)   # how hard the foes can hit our two active Pokémon

        block = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(dim)
        self.move_out = nn.Embedding(NUM_MOVES, dim, padding_idx=0)

        # --- joint action head ---
        self.slot_emb = nn.Embedding(2, dim)
        self.atom_actor = nn.Linear(dim, dim)
        self.atom_move = nn.Linear(dim, dim)
        self.atom_target = nn.Embedding(N_TARGETS, dim)
        self.atom_switch = nn.Linear(dim, dim)
        self.atom_special = nn.Embedding(2, dim)            # pass, locked
        self.damage_context = nn.Linear(DAMAGE_FEATURES, dim)
        self.atom_mix = mlp(dim, dim, dim)
        self.unary = mlp(2 * dim, 1, dim)
        # Damage-awareness: the move's damage features plus its target, straight to a score.
        self.damage_scalar = mlp(DAMAGE_FEATURES + N_TARGETS, 1, 32)
        self.pair_left = nn.Linear(dim, INTERACTION_RANK)
        self.pair_right = nn.Linear(dim, INTERACTION_RANK)
        self.pair_gate = nn.Linear(dim, INTERACTION_RANK)

        # --- value and opponent heads ---
        self.value_head = mlp(dim, 1, dim)
        self.foe_move_query = mlp(2 * dim, dim, dim)
        self.foe_switch_query = mlp(2 * dim, dim, dim)

        # --- preview head (90 plans) ---
        self.set_embed = mlp(2 * dim, dim, dim)
        self.lead_embed = mlp(2 * dim, dim, dim)
        self.plan_head = mlp(3 * dim, 1, dim)
        self.register_buffer("subset_index", torch.tensor(SUBSETS), persistent=False)
        self.register_buffer("pair_index", torch.tensor(PAIRS), persistent=False)
        self.register_buffer("plan_index", torch.tensor(PLANS), persistent=False)
        self.register_buffer("same_switch", self._same_switch(), persistent=False)

    @staticmethod
    def _same_switch() -> torch.Tensor:
        blocked = torch.zeros(N_ATOMS, N_ATOMS, dtype=torch.bool)
        for i in range(6):
            blocked[ATOM_SWITCH + i, ATOM_SWITCH + i] = True
        return blocked.reshape(-1)

    def config(self) -> dict:
        """Settings that rebuild this network (stored in checkpoints, see model_io)."""
        layer = self.encoder.layers[0]
        return {"dim": self.dim, "layers": len(self.encoder.layers), "heads": layer.self_attn.num_heads,
                "dropout": layer.dropout.p}

    # ------------------------------------------------------------------ tokens and trunk

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
        positions = torch.arange(N_TOKENS, device=species.device)
        x = x + self.side_emb(positions // 6) + self.position_emb(positions % 6)
        marks = torch.zeros_like(x)
        rows = torch.arange(species.shape[0], device=species.device)
        for slot in (0, 1):
            actor = b["actors"][:, slot]
            here = actor >= 0
            marks[rows[here], actor[here]] += self.actor_marks[slot]
        return x + marks

    @staticmethod
    def estimate_stats(base: torch.Tensor) -> torch.Tensor:
        """Guess a Pokémon's real stats from its base stats alone (about half the stat points, neutral nature).

        Used for the opponent, whose spread is unknown, and as a fallback for your own team.
        """
        return base + torch.tensor([91.0, 36, 36, 36, 36, 36], device=base.device)

    def own_stats(self, b: dict, rows: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        true = b["stats"][rows, token] * STAT_SCALE
        return torch.where(true[:, 1:2] > 0, true, self.estimate_stats(self.species_stats[b["species"][rows, token]] * 255))

    def foes(self, b: dict) -> list[dict]:
        """The foe in each field slot (a, b): token, species, estimated stats, boosts, presence."""
        species, numbers = b["species"], b["numbers"]
        rows = torch.arange(species.shape[0], device=species.device)
        out = []
        for slot in (0, 1):
            index = b["foes"][:, slot]
            token = 6 + index.clamp(min=0)
            foe = species[rows, token]
            out.append({"present": (index >= 0).float(), "species": foe, "token": token,
                        "stats": self.estimate_stats(self.species_stats[foe] * 255),
                        "boosts": numbers[rows, token, 13:20] * 6})
        return out

    def actor(self, b: dict, slot: int) -> dict:
        """Our Pokémon in `slot`: token, species, stats (real if known), boosts, presence."""
        rows = torch.arange(b["species"].shape[0], device=b["species"].device)
        index = b["actors"][:, slot]
        token = index.clamp(min=0)
        return {"present": (index >= 0).float(), "token": token, "species": b["species"][rows, token],
                "stats": self.own_stats(b, rows, token), "boosts": b["numbers"][rows, token, 13:20] * 6}

    boost_multiplier = staticmethod(boost_multiplier)

    def speed_features(self, b: dict) -> torch.Tensor:
        """[B, 4]: log of (our speed / their speed) for each of our two active Pokémon against each foe, boosts included."""
        foes = self.foes(b)
        out = []
        for slot in (0, 1):
            me = self.actor(b, slot)
            ours = me["stats"][:, 5] * self.boost_multiplier(me["boosts"][:, 4])
            for foe in foes:
                theirs = foe["stats"][:, 5] * self.boost_multiplier(foe["boosts"][:, 4])
                out.append(torch.log(ours / theirs) * foe["present"] * me["present"])
        return torch.stack(out, dim=-1)

    def fighter(self, b: dict, rows: torch.Tensor, token: torch.Tensor, stats: torch.Tensor, boosts: torch.Tensor) -> dict:
        """What the damage calculator needs to know about the Pokémon in `token`."""
        status = b["numbers"][rows, token, 7:13]
        return {"types": self.species_types[b["species"][rows, token]], "stats": stats, "boosts": boosts,
                "item": b["items"][rows, token], "ability": b["abilities"][rows, token],
                "burn": status[:, 1], "statused": status.sum(-1)}

    def damage_features(self, b: dict, slot: int) -> torch.Tensor:
        """[B, NUM_MOVES, DAMAGE_FEATURES]: how each move used by our Pokémon in `slot` would fare against the two foes.

        Per foe: type effectiveness, damage as a share of its full and of its current health, and
        whether the move would surely / possibly knock it out given the foe's unknown stat spread.
        Then same-type bonus, power, and whether the move is a status move.
        """
        numbers, batch = b["numbers"], b["species"].shape[0]
        rows = torch.arange(batch, device=numbers.device)
        me, foes = self.actor(b, slot), self.foes(b)
        attacker = self.fighter(b, rows, me["token"], me["stats"], me["boosts"])

        move_type, category = self.move_cats[:, 0], self.move_cats[:, 1]      # [M]
        spread_to = lambda t: t.unsqueeze(0).expand(batch, -1)
        m_type, m_cat, m_target = spread_to(move_type), spread_to(category), spread_to(self.move_cats[:, 2])
        m_power = spread_to(self.move_num[:, 0] * 150)
        attacking = (category <= 2).float()
        two_targets = foes[0]["present"] * foes[1]["present"]
        weather, terrain = b["globals"][:, 1:5], b["globals"][:, 5:9]
        a_types = self.species_types[me["species"]]
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
            on = (foe["present"] * me["present"])[:, None]
            sure = SURE_KO * damage["bulky"] >= (left * hp["bulky"]).clamp(min=1.0)
            maybe = MAYBE_KO * damage["frail"] >= (left * hp["frail"]).clamp(min=1.0)
            feats += [eff * on, (damage["mid"] / hp["mid"]).clamp(max=1.5) * on,
                      (damage["mid"] / (left * hp["mid"]).clamp(min=1.0)).clamp(max=3.0) / 3 * on,
                      (sure & (damage["bulky"] > 0)).float() * on, (maybe & (damage["frail"] > 0)).float() * on]
        feats += [stab.float() * me["present"][:, None], m_power / 150 * attacking[None, :], (1 - attacking)[None, :].expand(batch, -1)]
        return torch.stack(feats, dim=-1)

    def threat_features(self, b: dict) -> torch.Tensor:
        """[B, THREAT_FEATURES]: for each foe on the field and each of our two active Pokémon, how much of
        its current health the foe's strongest revealed move takes, and a guess at its best unrevealed
        same-type attack. Capped at 3 (three times over) and divided by 3.
        """
        numbers, batch = b["numbers"], b["species"].shape[0]
        rows = torch.arange(batch, device=numbers.device)
        mine = []
        for slot in (0, 1):
            me = self.actor(b, slot)
            mine.append((me["token"], me["present"], me["stats"], self.fighter(b, rows, me["token"], me["stats"], me["boosts"])))
        two_targets = mine[0][1] * mine[1][1]
        weather, terrain = b["globals"][:, 1:5], b["globals"][:, 5:9]

        out = []
        for foe in self.foes(b):
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

    def trunk(self, b: dict):
        """Run the transformer; returns the decision token [B, D] and all 12 Pokémon tokens [B, 12, D]."""
        mons = self.tokens(b)
        query = (self.kind_emb(b["kind"]) + self.speed_proj(self.speed_features(b))
                 + self.threat_proj(self.threat_features(b))).unsqueeze(1)
        field = self.field(b["globals"]).unsqueeze(1)
        seq = torch.cat([query, field, mons], dim=1)
        pad = torch.cat([torch.zeros(seq.shape[0], 2, dtype=torch.bool, device=seq.device), ~b["present"]], dim=1)
        out = self.out_norm(self.encoder(seq, src_key_padding_mask=pad))
        return out[:, 0], out[:, 2:]

    # ------------------------------------------------------------------ joint action head

    def atoms(self, b: dict, q: torch.Tensor, mons: torch.Tensor):
        """Atom embeddings [B, 2, N_ATOMS, D], per-atom scores u [B, 2, N_ATOMS], and the move damage features [B, 2, 4, F]."""
        batch = q.shape[0]
        rows = torch.arange(batch, device=q.device)
        table = self.move_table()
        own = mons[:, :6]
        targets = self.atom_target.weight                                                # [T, D]
        onehot = torch.eye(N_TARGETS, device=q.device)
        embeds, unaries, damages = [], [], []
        for slot in (0, 1):
            me = self.actor(b, slot)
            base = self.atom_actor(own[rows, me["token"]]) + self.slot_emb.weight[slot]   # [B, D]
            ids = b["actor_moves"][:, slot]                                               # [B, 4]
            damage = self.damage_features(b, slot).gather(1, ids[..., None].expand(-1, -1, DAMAGE_FEATURES))   # [B, 4, F]
            move = self.atom_move(table[ids]) + self.damage_context(damage)              # [B, 4, D]
            e_move = (base[:, None, None] + move[:, :, None] + targets[None, None]).reshape(batch, N_MOVE_SLOTS * N_TARGETS, -1)
            e_switch = base[:, None] + self.atom_switch(own)                              # [B, 6, D]
            e_special = base[:, None] + self.atom_special.weight[None]                    # [B, 2, D]
            e = torch.cat([e_move, e_switch, e_special], dim=1)
            e = e + self.atom_mix(e)
            u = self.unary(torch.cat([q[:, None].expand(-1, N_ATOMS, -1), e], dim=-1)).squeeze(-1)    # [B, A]
            move_scalar = self.damage_scalar(torch.cat([damage[:, :, None].expand(-1, -1, N_TARGETS, -1),
                                                        onehot[None, None].expand(batch, N_MOVE_SLOTS, -1, -1)], dim=-1))
            u = u + torch.cat([move_scalar.reshape(batch, -1), torch.zeros(batch, N_ATOMS - N_MOVE_SLOTS * N_TARGETS, device=q.device)], dim=1)
            embeds.append(e); unaries.append(u); damages.append(damage)
        return torch.stack(embeds, 1), torch.stack(unaries, 1), torch.stack(damages, 1)

    def pair_logits(self, b: dict, q: torch.Tensor, mons: torch.Tensor) -> torch.Tensor:
        """[B, N_PAIRS] scores of every (slot a atom, slot b atom), illegal pairs masked."""
        e, u, _ = self.atoms(b, q, mons)
        interaction = torch.einsum("bir,bjr,br->bij", self.pair_left(e[:, 0]), self.pair_right(e[:, 1]), self.pair_gate(q))
        scores = (u[:, 0, :, None] + u[:, 1, None, :] + interaction).flatten(1)
        legal = (b["atom_mask"][:, 0, :, None] & b["atom_mask"][:, 1, None, :]).flatten(1) & ~self.same_switch
        return scores.masked_fill(~legal, NEG)

    # ------------------------------------------------------------------ opponent head

    def foe_logits(self, b: dict, q: torch.Tensor, mons: torch.Tensor) -> torch.Tensor:
        """[B, 2, N_ACTIONS]: for each foe on the field, its move (vocabulary id) or switch (their team index).

        Moves are limited to its learnset plus what it has shown; switches to their Pokémon that are alive
        and not on the field. A foe slot that is empty has everything masked (and no label).
        """
        rows = torch.arange(q.shape[0], device=q.device)
        table, theirs = self.move_table(), mons[:, 6:]
        alive = b["present"][:, 6:] & (b["numbers"][:, 6:, 1] < 0.5)
        on_field = (b["numbers"][:, 6:, 2] > 0.5) | (b["numbers"][:, 6:, 3] > 0.5)
        out = []
        for foe in self.foes(b):
            h = mons[rows, foe["token"]]
            x = torch.cat([q, h], dim=-1)
            moves = self.foe_move_query(x) @ table.T / self.dim ** 0.5                 # [B, M]
            switches = torch.einsum("bd,bkd->bk", self.foe_switch_query(x), theirs) / self.dim ** 0.5
            move_ok = self.learnable[foe["species"]].clone()
            revealed = b["moves"][rows, foe["token"]]
            move_ok[rows[:, None], revealed] = True
            move_ok[:, 0] = False
            ok = torch.cat([move_ok, alive & ~on_field], dim=1) & (foe["present"] > 0)[:, None]
            out.append(torch.cat([moves, switches], dim=1).masked_fill(~ok, NEG))
        return torch.stack(out, dim=1)

    # ------------------------------------------------------------------ preview head

    def preview_logits(self, b: dict) -> torch.Tensor:
        """[B, 90] scores of every plan (four to bring, two of them to lead); see PLANS.

        The four is embedded as a set (mean and max of its members' tokens, through an MLP), so members
        interact; the lead pair from the product and sum of its two tokens. The decision token has attended
        to the opponent's six.
        """
        q, mons = self.trunk(b)
        own = mons[:, :6]
        members = own[:, self.subset_index]                                               # [B, 15, 4, D]
        sets = self.set_embed(torch.cat([members.mean(2), members.max(2).values], dim=-1))  # [B, 15, D]
        a, c = own[:, self.pair_index[:, 0]], own[:, self.pair_index[:, 1]]
        leads = self.lead_embed(torch.cat([a * c, a + c], dim=-1))                         # [B, 15, D]
        plan_sets, plan_leads = sets[:, self.plan_index[:, 0]], leads[:, self.plan_index[:, 1]]
        x = torch.cat([q[:, None].expand(-1, len(PLANS), -1), plan_sets, plan_leads], dim=-1)
        return self.plan_head(x).squeeze(-1)

    # ------------------------------------------------------------------ entry points

    def forward(self, b: dict):
        """(pair logits [B, N_PAIRS], value [B], opponent logits [B, 2, N_ACTIONS])."""
        q, mons = self.trunk(b)
        return self.pair_logits(b, q, mons), self.value_head(q).squeeze(-1), self.foe_logits(b, q, mons)

    def evaluate(self, b: dict, action: torch.Tensor):
        """Re-score recorded joint decisions for reinforcement learning.

        Returns (log-probability of the pair taken, entropy over pairs, value estimate, full pair log-distribution).
        """
        q, mons = self.trunk(b)
        log_probs = torch.log_softmax(self.pair_logits(b, q, mons), dim=-1)
        logp = log_probs.gather(1, action[:, None]).squeeze(1)
        entropy = -(log_probs.exp() * log_probs.clamp(min=-30)).sum(-1)
        return logp, entropy, self.value_head(q).squeeze(-1), log_probs
