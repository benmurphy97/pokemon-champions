import json
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

FIXTURE = Path(__file__).parent / "fixtures" / "replay_sample.json"


@unittest.skipUnless(HAVE_TORCH, "run with .venv/bin/python (needs torch)")
class PolicyTest(unittest.TestCase):
    def setUp(self):
        from champions.decisions import extract
        from champions.encoding import SWITCH_BASE
        from champions.policy import PolicyNet
        from champions.train_il import KEYS
        torch.manual_seed(0)
        examples = extract(json.loads(FIXTURE.read_text())["log"].split("\n"))
        self.batch = {k: torch.as_tensor(np.stack([e["arrays"][k] for e in examples])) for k in KEYS}
        self.action = torch.as_tensor([e["action"] for e in examples])
        self.target = torch.as_tensor([e["target"] for e in examples])
        self.batch["mask"][torch.arange(len(self.action)), self.action] = True
        self.net = PolicyNet(dim=32, layers=1, heads=2, dropout=0.0)
        self.SWITCH_BASE = SWITCH_BASE

    def test_output_shapes_and_masking(self):
        logits, target_logits = self.net(self.batch, target_for_move=self.action)
        self.assertEqual(logits.shape, (len(self.action), self.batch["mask"].shape[1]))
        self.assertEqual(target_logits.shape, (len(self.action), 4))
        self.assertTrue((logits[~self.batch["mask"]] < -1e8).all())

    def test_can_overfit_one_game(self):
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=3e-3)
        losses = []
        for _ in range(80):
            logits, _ = self.net(self.batch, target_for_move=self.action)
            loss = F.cross_entropy(logits, self.action)
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], 0.3 * losses[0])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_TORCH, "run with .venv/bin/python (needs torch)")
class DamageFeatureTest(unittest.TestCase):
    def test_type_effectiveness_features_match_the_type_chart(self):
        from champions.decisions import extract
        from champions.policy import PolicyNet
        from champions.train_il import KEYS
        from champions.vocab import MOVE_ID, SPECIES
        examples = extract(json.loads(FIXTURE.read_text())["log"].split("\n"))
        batch = {k: torch.as_tensor(np.stack([e["arrays"][k] for e in examples])) for k in KEYS}
        features = PolicyNet(dim=32, layers=1, heads=2).damage_features(batch)
        # Raichu (Electric) facing Indeedee-F (Psychic/Normal) and Meowstic-F (Psychic) on turn 1
        i = next(n for n, e in enumerate(examples)
                 if SPECIES[int(e["arrays"]["species"][int(e["arrays"]["actor"])])] == "Raichu")
        focus_blast, protect = features[i, MOVE_ID["Focus Blast"]], features[i, MOVE_ID["Protect"]]
        self.assertAlmostEqual(float(focus_blast[0]), 1.0)   # Fighting vs Psychic/Normal: 0.5 * 2
        self.assertAlmostEqual(float(focus_blast[3]), 0.5)   # Fighting vs Psychic
        self.assertEqual(float(protect[2]), 0.0)             # a status move deals no damage
