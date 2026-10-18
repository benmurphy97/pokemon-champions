"""Learned team preview: which four to bring and which two of them lead, chosen as one plan.

    preview = LearnedPreview(net)
    agent = NeuralAgent(net, preview=preview)       # the arena then asks it at team preview

The network scores all 90 plans given our full six and their six species (see PolicyNet.preview_logits).
`preview_loss` is the imitation loss: the log of the total probability of the plans consistent with the
replay (the brought four is not always fully seen).
"""

import numpy as np
import torch

from champions.model.agent import own_team_info, to_tensors
from champions.model.battle_state import base_species
from champions.model.encoding import PREVIEW, encode
from champions.model.policy import NEG, PAIRS, PLANS, SUBSETS
from champions.teams.generate import load_usage


def preview_loss(net, batch: dict) -> tuple[torch.Tensor, dict]:
    """-log sum over consistent plans of P(plan), averaged; plus top-1 accuracy (the best plan is consistent)."""
    logits = net.preview_logits(batch)
    ok = batch["plan_ok"]
    loss = (torch.logsumexp(logits, dim=-1) - torch.logsumexp(logits.masked_fill(~ok, NEG), dim=-1)).mean()
    with torch.no_grad():
        best = logits.argmax(-1)
        hit = ok.gather(1, best[:, None]).squeeze(1)
        complete = batch["complete"]
        metrics = {"plan": hit.float().sum().item(), "n": len(best),
                   "full": hit[complete].float().sum().item(), "full_n": int(complete.sum())}
    return loss, metrics


class LearnedPreview:
    """A callable `(state, side, request) -> "team 1234"` for NeuralAgent(preview=...)."""

    def __init__(self, net, temperature: float = 0.0, seed: int = 0):
        self.net, self.temperature, self.rng = net.eval(), temperature, np.random.default_rng(seed)
        self.usage = load_usage()

    def scores(self, state, side: str, request: dict):
        """(plan logits [90], team indices in request order)."""
        pokemon = request["side"]["pokemon"]
        team_index = [state._index_of(side, base_species(p["details"])) for p in pokemon]
        own = own_team_info(state, side, request, team_index, self.usage)
        arrays = encode(state, side, PREVIEW, own_info=own)
        with torch.no_grad():
            logits = self.net.preview_logits(to_tensors(arrays))
        return logits[0], team_index

    def __call__(self, state, side: str, request: dict) -> str:
        logits, team_index = self.scores(state, side, request)
        if self.temperature <= 0:
            plan = int(logits.argmax())
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1).numpy().astype(np.float64)
            plan = int(self.rng.choice(len(probs), p=probs / probs.sum()))
        subset, pair = SUBSETS[PLANS[plan][0]], PAIRS[PLANS[plan][1]]
        position = {ti: j + 1 for j, ti in enumerate(team_index)}
        order = list(pair) + [i for i in subset if i not in pair]
        return "team " + "".join(str(position[i]) for i in order)
