"""
bc_train.py — Behavioral cloning: pretrain MaskablePPO policy weights
from a BCDataset before PPO fine-tuning.

Strategy:
  - Extract the MlpPolicy's action_net (which maps latent → logits for each slot)
  - Train with cross-entropy loss on the expert actions
  - Save the pretrained policy weights back into a MaskablePPO checkpoint

Usage:
    python src/data/imitation/bc_train.py \
        --dataset data/bc_dataset.npz \
        --checkpoint-dir checkpoints \
        --epochs 20 \
        --lr 1e-3 \
        --batch-size 256

After this, run:
    python src/train.py train --steps 50000 --selfplay
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)

N_ACTIONS = 107   # per slot — must match battle/env.py


def _build_dummy_policy(state_size: int, checkpoint_path: str | None, device: torch.device):
    """
    Create a MaskablePPO with a dummy env (no Showdown needed) to get the
    policy network. If a checkpoint exists we load it so BC fine-tunes existing
    weights rather than starting from scratch.
    """
    import gymnasium as gym
    from gymnasium import spaces
    from sb3_contrib import MaskablePPO

    class DummyEnv(gym.Env):
        metadata = {}
        render_mode = None

        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(state_size,), dtype=np.float32
            )
            self.action_space = spaces.MultiDiscrete([N_ACTIONS, N_ACTIONS])

        def reset(self, **kw):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, True, False, {}

        def close(self):
            pass

    dummy_env = DummyEnv()

    if checkpoint_path and os.path.exists(checkpoint_path + ".zip"):
        log.info("Loading existing checkpoint %s for BC fine-tuning", checkpoint_path)
        model = MaskablePPO.load(checkpoint_path, env=dummy_env, device=str(device))
    else:
        log.info("No checkpoint found — initialising fresh MaskablePPO for BC")
        model = MaskablePPO(
            "MlpPolicy",
            dummy_env,
            policy_kwargs=dict(net_arch=[64, 64]),
            device=str(device),
            verbose=0,
        )

    return model


def run_bc(
    dataset_path: Path,
    checkpoint_dir: str = "model_outputs/checkpoints",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device_str: str = "auto",
):
    from battle_decisions.imitation.parse_replay import BCDataset

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    log.info("Using device: %s", device)

    dataset = BCDataset.load(dataset_path)
    N = len(dataset)
    log.info("Loaded dataset: %d samples, obs shape %s", N, dataset.obs.shape)

    state_size = dataset.obs.shape[1]

    idx = np.random.permutation(N)
    n_val = max(1, int(N * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    obs_t  = torch.tensor(dataset.obs,  dtype=torch.float32)
    acts_t = torch.tensor(dataset.acts, dtype=torch.long)

    train_ds = TensorDataset(obs_t[train_idx], acts_t[train_idx])
    val_ds   = TensorDataset(obs_t[val_idx],   acts_t[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    log.info("Train: %d  Val: %d", len(train_idx), len(val_idx))

    checkpoint_path = os.path.join(checkpoint_dir, "ppo_latest")
    model = _build_dummy_policy(state_size, checkpoint_path, device)
    policy = model.policy.to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        policy.train()
        train_loss = 0.0
        n_train = 0

        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            latent_pi, _ = policy.mlp_extractor(policy.extract_features(obs_batch))
            logits_flat = policy.action_net(latent_pi)
            logits_list = logits_flat.split(N_ACTIONS, dim=1)

            loss = sum(
                criterion(logits_list[i], act_batch[:, i])
                for i in range(len(logits_list))
            ) / len(logits_list)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item() * len(obs_batch)
            n_train    += len(obs_batch)

        train_loss /= n_train

        policy.eval()
        val_loss = 0.0
        val_acc0 = val_acc1 = 0.0
        n_val_total = 0

        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)

                latent_pi, _ = policy.mlp_extractor(policy.extract_features(obs_batch))
                logits_flat = policy.action_net(latent_pi)
                logits_list = logits_flat.split(N_ACTIONS, dim=1)

                loss = sum(
                    criterion(logits_list[i], act_batch[:, i])
                    for i in range(len(logits_list))
                ) / len(logits_list)

                val_loss   += loss.item() * len(obs_batch)
                val_acc0   += (logits_list[0].argmax(dim=1) == act_batch[:, 0]).float().sum().item()
                val_acc1   += (logits_list[1].argmax(dim=1) == act_batch[:, 1]).float().sum().item()
                n_val_total += len(obs_batch)

        val_loss /= n_val_total
        val_acc0 /= n_val_total
        val_acc1 /= n_val_total

        scheduler.step()

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
            "slot0_acc=%.1f%%  slot1_acc=%.1f%%",
            epoch, epochs, train_loss, val_loss,
            val_acc0 * 100, val_acc1 * 100,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
            log.info("  ↑ New best val_loss=%.4f", best_val_loss)

    if best_state is not None:
        policy.load_state_dict(best_state)
        log.info("Restored best policy weights (val_loss=%.4f)", best_val_loss)

    os.makedirs(checkpoint_dir, exist_ok=True)
    model.policy = policy.cpu()
    model.save(checkpoint_path)
    log.info("Saved pretrained checkpoint → %s.zip", checkpoint_path)
    log.info("Done. Now run:  python src/train.py train --steps 50000 --selfplay")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Behavioural cloning pretrain")
    parser.add_argument("--dataset",        default="data/bc_dataset.npz")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--val-split",      type=float, default=0.1)
    parser.add_argument("--device",         default="auto")
    args = parser.parse_args()

    run_bc(
        dataset_path=Path(args.dataset),
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device_str=args.device,
    )
