"""PPO: nudge the policy towards decisions that came before good results and away from bad ones.

A value estimate says how well a position was already going, so a decision is credited for what it
changed (advantage = result - value), not for the result alone. A penalty for drifting far from a
reference policy (the imitation start) keeps it from forgetting how to play.
"""

import numpy as np
import torch

from champions.model.agent import KEYS


def to_batch(steps: list[dict]) -> dict:
    batch = {k: torch.as_tensor(np.stack([np.asarray(s["arrays"][k]) for s in steps])) for k in KEYS}
    batch["action"] = torch.as_tensor([s["action"] for s in steps])
    batch["old_logp"] = torch.as_tensor([s["logp"] for s in steps], dtype=torch.float32)
    batch["reward"] = torch.as_tensor([s["reward"] for s in steps], dtype=torch.float32)
    batch["last"] = torch.as_tensor([s["last"] for s in steps])
    return batch


def advantages(reward: torch.Tensor, last: torch.Tensor, value: torch.Tensor, lam: float) -> torch.Tensor:
    """Generalised advantage estimate. Steps are in game order and `last` marks each game's final one.

    lam = 1 credits a decision with everything that happened after it (result minus the value
    estimate); smaller values lean more on the value estimate, which is steadier but biased.
    """
    advantage, running = torch.zeros_like(value), 0.0
    for t in reversed(range(len(value))):
        following = 0.0 if last[t] else float(value[t + 1])
        running = float(reward[t]) + following - float(value[t]) + (0.0 if last[t] else lam * running)
        advantage[t] = running
    return advantage


def take(batch: dict, idx: torch.Tensor) -> dict:
    return {k: v[idx] for k, v in batch.items()}


def ppo_update(net, reference, optimiser, batch: dict, args, value_only: bool) -> dict:
    """Several passes over one batch of recorded decisions."""
    n = len(batch["reward"])
    with torch.no_grad():
        _, _, old_value, _ = net.evaluate(batch, batch["action"])
        _, _, _, ref_log_probs = reference.evaluate(batch, batch["action"])
    advantage = advantages(batch["reward"], batch["last"], old_value, args.gae_lambda)
    batch["ret"] = advantage + old_value
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

    stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "clipped": 0.0}
    passes = 0
    for _ in range(args.epochs):
        order = torch.randperm(n)
        for start in range(0, n, args.minibatch):
            idx = order[start:start + args.minibatch]
            b = take(batch, idx)
            logp, entropy, value, log_probs = net.evaluate(b, b["action"])
            ratio = (logp - b["old_logp"]).exp()
            adv = advantage[idx]
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
            value_loss = ((value - b["ret"]) ** 2).mean()
            kl = (log_probs.exp() * (log_probs - ref_log_probs[idx]).clamp(min=-30, max=30)).sum(-1).mean()
            loss = 0.5 * value_loss if value_only else (policy_loss + 0.5 * value_loss
                                                         - args.entropy * entropy.mean() + args.kl * kl)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            passes += 1
            stats["policy"] += policy_loss.item(); stats["value"] += value_loss.item()
            stats["entropy"] += entropy.mean().item(); stats["kl"] += kl.item()
            stats["clipped"] += ((ratio - 1).abs() > args.clip).float().mean().item()
    return {k: v / max(passes, 1) for k, v in stats.items()}
