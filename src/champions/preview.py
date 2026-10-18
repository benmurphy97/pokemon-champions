"""Learned team preview: which four to bring and which two lead.

    preview = LearnedPreview(net)
    agent = NeuralAgent(net, preview=preview)       # the arena then asks it at team preview

The network scores the six own Pokémon given both teams' six species (see PolicyNet.preview_logits).
`preview_loss` is the imitation loss used by train_il.
"""

import numpy as np
import torch
import torch.nn.functional as F

from champions.agent import own_team_info, to_tensors
from champions.battle_state import base_species
from champions.encoding import PREVIEW_BRING, encode
from champions.meta import load_usage
from champions.policy import PAIRS, SUBSETS

NEG = -1e9


def lead_mask_for(subset: tuple[int, ...]) -> torch.Tensor:
    """[15] bool: the lead pairs that lie inside a chosen four."""
    return torch.tensor([set(p) <= set(subset) for p in PAIRS])


def preview_loss(net, batch: dict) -> tuple[torch.Tensor, dict]:
    """Imitation loss for a batch of preview examples (keys as in preview_labels.build_examples)."""
    bring_logits, lead_logits = net.preview_logits(batch)
    bring, lead = batch["bring"], batch["lead"]
    known = bring >= 0
    loss = F.cross_entropy(lead_logits.masked_fill(~batch["lead_ok"], NEG), lead)
    if known.any():
        loss = loss + F.cross_entropy(bring_logits[known], bring[known])
    with torch.no_grad():
        metrics = {"bring": (bring_logits[known].argmax(-1) == bring[known]).float().sum().item() if known.any() else 0.0,
                   "bring_n": int(known.sum()),
                   "lead": (lead_logits.masked_fill(~batch["lead_ok"], NEG).argmax(-1) == lead).float().sum().item(),
                   "lead_n": len(lead)}
    return loss, metrics


class LearnedPreview:
    """A callable `(state, side, request) -> "team 1234"` for NeuralAgent(preview=...)."""

    def __init__(self, net, temperature: float = 0.0, seed: int = 0):
        self.net, self.temperature, self.rng = net.eval(), temperature, np.random.default_rng(seed)
        self.usage = load_usage()

    def _pick(self, logits: torch.Tensor) -> int:
        if self.temperature <= 0:
            return int(logits.argmax())
        probs = torch.softmax(logits / self.temperature, dim=-1).numpy().astype(np.float64)
        return int(self.rng.choice(len(probs), p=probs / probs.sum()))

    def scores(self, state, side: str, request: dict):
        """(bring logits [15], lead logits [15], team indices in request order)."""
        pokemon = request["side"]["pokemon"]
        team_index = [state._index_of(side, base_species(p["details"])) for p in pokemon]
        own = own_team_info(state, side, request, team_index, self.usage)
        arrays = encode(state, side, 0, PREVIEW_BRING, own_info=own)
        with torch.no_grad():
            bring, lead = self.net.preview_logits(to_tensors(arrays))
        return bring[0], lead[0], team_index

    def __call__(self, state, side: str, request: dict) -> str:
        bring, lead, team_index = self.scores(state, side, request)
        subset = SUBSETS[self._pick(bring)]
        pair = PAIRS[self._pick(lead.masked_fill(~lead_mask_for(subset), NEG))]
        position = {ti: j + 1 for j, ti in enumerate(team_index)}
        order = list(pair) + [i for i in subset if i not in pair]
        return "team " + "".join(str(position[i]) for i in order)
