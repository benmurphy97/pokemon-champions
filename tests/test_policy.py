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
        from champions.training.decisions import extract
        from champions.model.encoding import SWITCH_BASE
        from champions.model.policy import PolicyNet
        from champions.training.imitation import KEYS
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
        from champions.training.decisions import extract
        from champions.model.policy import PolicyNet
        from champions.training.imitation import KEYS
        from champions.model.vocab import MOVE_ID, SPECIES
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
        from champions.model.damage import DamageCalc, type_id
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
        from champions.model.vocab import ABILITY_ID
        self.assertGreater(self.hit(atk_over={"boosts": torch.tensor([[2.0, 0, 0, 0, 0, 0, 0]])}), 1.9 * self.hit() * 0.9)
        levitate = torch.tensor([ABILITY_ID["Levitate"]])
        self.assertEqual(self.hit(move_type="Ground", dfn_over={"ability": levitate}), 0.0)


@unittest.skipUnless(HAVE_TORCH, "run with .venv/bin/python (needs torch)")
class PartnerTest(unittest.TestCase):
    def setUp(self):
        from champions.training.decisions import extract
        from champions.training.imitation import KEYS
        torch.manual_seed(0)
        self.examples = extract(json.loads(FIXTURE.read_text())["log"].split("\n"))
        self.batch = {k: torch.as_tensor(np.stack([e["arrays"][k] for e in self.examples])) for k in KEYS}

    def test_second_slot_examples_carry_the_first_slots_action(self):
        from champions.model.encoding import SWITCH_BASE
        linked = [e for e in self.examples if e["arrays"]["partner_action"] > 0]
        self.assertTrue(linked)
        for e in linked:
            self.assertEqual(int(e["arrays"]["slot"]), 1)
            mate = next(m for m in self.examples if m["turn"] == e["turn"] and m["side"] == e["side"] and m["arrays"]["slot"] == 0)
            self.assertEqual(int(e["arrays"]["partner_action"]), mate["action"])
            if mate["action"] >= SWITCH_BASE:
                self.assertFalse(e["arrays"]["mask"][mate["action"]])   # cannot switch in the same teammate

    def test_partner_input_starts_inert_then_matters_once_trained(self):
        from champions.model.policy import PolicyNet
        net = PolicyNet(dim=32, layers=1, heads=2, dropout=0.0).eval()
        other = dict(self.batch)
        other["partner_action"] = torch.zeros_like(self.batch["partner_action"])
        other["partner_target"] = torch.zeros_like(self.batch["partner_target"])
        with torch.no_grad():
            same = torch.allclose(net(self.batch)[0], net(other)[0])
            for p in (net.partner_move, net.partner_switch, net.partner_target):
                for w in p.parameters():
                    w.normal_()
            differs = not torch.allclose(net(self.batch)[0], net(other)[0])
        self.assertTrue(same)
        self.assertTrue(differs)


@unittest.skipUnless(HAVE_TORCH, "run with .venv/bin/python (needs torch)")
class PreviewTest(unittest.TestCase):
    def setUp(self):
        from champions.teams.generate import load_usage
        from champions.training.preview_labels import preview_examples
        from champions.model.encoding import ARRAY_KEYS
        torch.manual_seed(0)
        self.examples = preview_examples(json.loads(FIXTURE.read_text())["log"].split("\n"), load_usage())
        self.batch = {k: torch.as_tensor(np.stack([e["arrays"][k] for e in self.examples])) for k in ARRAY_KEYS}
        self.batch["bring"] = torch.as_tensor([e["bring"] for e in self.examples])
        self.batch["lead"] = torch.as_tensor([e["lead"] for e in self.examples])
        self.batch["lead_ok"] = torch.as_tensor(np.array([e["lead_ok"] for e in self.examples]))

    def test_labels_are_consistent(self):
        from champions.model.policy import PAIRS, SUBSETS
        for e in self.examples:
            self.assertEqual(len(e["lead_ok"]), 15)
            if e["bring"] >= 0:                      # a full four: the lead pair is inside it
                self.assertTrue(set(PAIRS[e["lead"]]) <= set(SUBSETS[e["bring"]]))
                self.assertTrue(e["lead_ok"][e["lead"]])

    def test_a_net_can_learn_one_game_of_previews(self):
        from champions.model.policy import PolicyNet
        from champions.model.preview import preview_loss
        net = PolicyNet(dim=32, layers=1, heads=2, dropout=0.0)
        optimiser = torch.optim.AdamW(net.parameters(), lr=3e-3)
        losses = []
        for _ in range(60):
            loss, _ = preview_loss(net, self.batch)
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], 0.5 * losses[0])
        logits_bring, logits_lead = net.preview_logits(self.batch)
        self.assertEqual(logits_bring.shape, (len(self.examples), 15))
        self.assertEqual(logits_lead.shape, (len(self.examples), 15))

    def test_learned_preview_returns_a_valid_team_string(self):
        from champions.model.battle_state import BattleState
        from champions.model.policy import PolicyNet
        from champions.model.preview import LearnedPreview
        state = BattleState()
        names = ["Rillaboom", "Incineroar", "Sneasler", "Indeedee-F", "Farigiraf", "Torkoal", "Sinistcha"]
        for n in names[:6]:
            state.update(f"|poke|p1|{n}, L50|")
        for n in ("Gholdengo", "Garchomp", "Kingambit", "Whimsicott", "Basculegion", "Pelipper"):
            state.update(f"|poke|p2|{n}, L50|")
        request = {"side": {"pokemon": [{"details": f"{n}, L50", "condition": "100/100", "active": False,
                                         "moves": ["protect"], "item": "leftovers", "baseAbility": "intimidate"} for n in names[:6]]}}
        choice = LearnedPreview(PolicyNet(dim=32, layers=1, heads=2))(state, "p1", request)
        digits = choice.removeprefix("team ")
        self.assertEqual(len(digits), 4)
        self.assertEqual(len(set(digits)), 4)
        self.assertTrue(set(digits) <= set("123456"))
