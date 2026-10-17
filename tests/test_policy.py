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
        self.assertAlmostEqual(float(focus_blast[5]), 0.5)   # Fighting vs Psychic
        self.assertEqual(float(protect[1]), 0.0)             # a status move deals no damage


@unittest.skipUnless(HAVE_TORCH, "run with .venv/bin/python (needs torch)")
class DamageCalcTest(unittest.TestCase):
    def hit(self, power=100, move_type="Normal", cat=1, target=1, weather=(0, 0, 0, 0), two=0.0, atk_over=None, dfn_over=None):
        from champions.damage import DamageCalc, type_id
        one = lambda *v, dtype=torch.float32: torch.tensor([list(v)], dtype=dtype)
        attacker = {"types": one(type_id("Normal"), 0, dtype=torch.long), "stats": one(150, 150, 100, 150, 100, 100),
                    "boosts": torch.zeros(1, 7), "item": torch.zeros(1, dtype=torch.long), "ability": torch.zeros(1, dtype=torch.long),
                    "burn": torch.zeros(1), "statused": torch.zeros(1)}
        defender = {"types": one(type_id("Water"), 0, dtype=torch.long), "stats": one(150, 100, 100, 100, 100, 100),
                    "boosts": torch.zeros(1, 7), "item": torch.zeros(1, dtype=torch.long), "ability": torch.zeros(1, dtype=torch.long)}
        attacker.update(atk_over or {}); defender.update(dfn_over or {})
        out = DamageCalc()(attacker, defender, one(type_id(move_type), dtype=torch.long), one(cat, dtype=torch.long),
                           one(power), one(target, dtype=torch.long), torch.tensor([two]), torch.tensor([weather], dtype=torch.float32), torch.zeros(1, 4))
        return float(out[0, 0])

    def test_matches_the_damage_formula(self):
        # level 50, 100 power, attack 150 vs defence 100: (22*100*150/100/50 + 2) = 68, STAB 1.5, average roll 0.925
        self.assertAlmostEqual(self.hit(), 68 * 1.5 * 0.925, places=3)

    def test_modifiers(self):
        base = self.hit()
        self.assertAlmostEqual(self.hit(move_type="Fire", weather=(1, 0, 0, 0)) / self.hit(move_type="Fire"), 1.5, places=5)
        self.assertAlmostEqual(self.hit(target=6, two=1.0) / base, 0.75, places=5)       # spread move, two targets
        self.assertAlmostEqual(self.hit(target=6, two=0.0) / base, 1.0, places=5)
        self.assertAlmostEqual(self.hit(atk_over={"burn": torch.ones(1), "statused": torch.ones(1)}) / base, 0.5, places=5)
        self.assertEqual(self.hit(power=0), 0.0)
        self.assertEqual(self.hit(cat=3), 0.0)                                  # status moves deal nothing
        self.assertEqual(self.hit(dfn_over={"types": torch.tensor([[14, 0]])}), 0.0)   # Ghost takes nothing from Normal

    def test_boosts_and_levitate(self):
        from champions.vocab import ABILITY_ID
        self.assertGreater(self.hit(atk_over={"boosts": torch.tensor([[2.0, 0, 0, 0, 0, 0, 0]])}), 1.9 * self.hit() * 0.9)
        levitate = torch.tensor([ABILITY_ID["Levitate"]])
        self.assertEqual(self.hit(move_type="Ground", dfn_over={"ability": levitate}), 0.0)
