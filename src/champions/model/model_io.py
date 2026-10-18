"""Save and load policies so a model can grow without being retrained from scratch.

A checkpoint stores the network's settings next to its weights, so loading needs no flags. When the
code has since grown (a new input, a new kind of decision, more layers), `load_policy` carries the
old weights over and starts every new part so that it changes nothing:

  * new input columns and new modules start at zero, so the model plays exactly as before until
    training teaches it to use them;
  * extra transformer layers start as the identity (their output projections are zero);
  * extra embedding rows (e.g. a new kind of decision) start at their normal random values.

Anything that cannot be carried over this way (a wider network) is reported and left at its fresh
start. Training can then continue from the old weights instead of from nothing.
"""

from pathlib import Path

import torch
from torch import nn

from champions.model.encoding import ENCODING_VERSION
from champions.model.policy import PolicyNet


def save_policy(net: PolicyNet, path: Path, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    torch.save({"state": net.state_dict(), "config": net.config(), "encoding_version": ENCODING_VERSION,
                "dim": net.dim, "layers": len(net.encoder.layers), **extra}, path)


def adapt_state(net: nn.Module, saved: dict) -> dict:
    """Weights from `saved`, reshaped to fit `net`. Returns a report of what was carried over, padded, or new."""
    fresh, adapted = net.state_dict(), {}
    report = {"copied": 0, "padded": [], "new": [], "unfit": []}
    for name, target in fresh.items():
        old = saved.get(name)
        if old is None:
            report["new"].append(name)
        elif old.shape == target.shape:
            adapted[name] = old
            report["copied"] += 1
        elif old.ndim == target.ndim and all(o <= t for o, t in zip(old.shape, target.shape)):
            grown = target.clone()
            module = net.get_submodule(name.rsplit(".", 1)[0]) if "." in name else None
            if isinstance(module, nn.Linear) and name.endswith("weight") and old.shape[1] < target.shape[1]:
                grown[:, old.shape[1]:] = 0           # new input columns: no effect until trained
            grown[tuple(slice(0, s) for s in old.shape)] = old
            adapted[name] = grown
            report["padded"].append(name)
        else:
            report["unfit"].append(name)
    # An extra transformer layer must start as the identity: zero what it adds to the residual stream.
    for name in report["new"]:
        if name.startswith("encoder.layers.") and name.rsplit(".", 2)[-2:] in (["out_proj", "weight"], ["out_proj", "bias"], ["linear2", "weight"], ["linear2", "bias"]):
            adapted[name] = torch.zeros_like(fresh[name])
    return adapted, report


def load_policy(path: Path, report: bool = False):
    """Build the network a checkpoint describes and load it, adapting if the code has grown since."""
    saved = torch.load(path, map_location="cpu")
    config = saved.get("config") or {"dim": saved["dim"], "layers": saved["layers"]}
    net = PolicyNet(**config)
    adapted, summary = adapt_state(net, saved["state"])
    net.load_state_dict(adapted, strict=False)
    net.eval()
    return (net, summary) if report else net
