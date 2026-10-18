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


def sample_batch():
    from champions.model.encoding import ARRAY_KEYS
    from champions.teams.generate import load_usage
    from champions.training.decisions import extract, stack
    examples = extract(json.loads(FIXTURE.read_text())["log"].split("\n"), load_usage())
    return examples, {k: torch.as_tensor(v) for k, v in stack(examples).items()}


@unittest.skipUnless(HAVE_TORCH, "run with .venv/bin/python (needs torch)")
class PolicyTest(unittest.TestCase):
    def setUp(self):
        from champions.model.policy import PolicyNet
        torch.manual_seed(0)
        self.examples, self.batch = sample_batch()
        self.net = PolicyNet(dim=32, layers=1, heads=2, dropout=0.0)

    def test_output_shapes_masking_and_labels(self):
        from champions.model.encoding import N_ACTIONS, N_ATOMS, N_PAIRS
        pair_logits, value, foe_logits = self.net(self.batch)
        n = len(self.examples)
        self.assertEqual(pair_logits.shape, (n, N_PAIRS))
        self.assertEqual(value.shape, (n,))
        self.assertEqual(foe_logits.shape, (n, 2, N_ACTIONS))
        mask = self.batch["atom_mask"]
        legal = (mask[:, 0, :, None] & mask[:, 1, None, :]).flatten(1)
        self.assertTrue((pair_logits[~legal] < -1e8).all())
        label = self.batch["label"]
        both = (label >= 0).all(1)
        idx = label[both, 0] * N_ATOMS + label[both, 1]
        self.assertTrue((pair_logits[both].gather(1, idx[:, None]) > -1e8).all())     # what players did is legal
        foe = self.batch["foe_label"]
        seen = foe >= 0
        self.assertTrue(seen.any())
        self.assertTrue((foe_logits[seen].gather(1, foe[seen][:, None]) > -1e8).all())

    def test_both_slots_never_switch_to_the_same_pokemon(self):
        from champions.model.encoding import ATOM_SWITCH, N_ATOMS
        pair_logits, _, _ = self.net(self.batch)
        for i in range(6):
            self.assertTrue((pair_logits[:, (ATOM_SWITCH + i) * N_ATOMS + ATOM_SWITCH + i] < -1e8).all())

    def test_pair_scores_are_not_additive_once_the_interaction_is_used(self):
        """S(a, b) - S(a, b') must depend on a: the second slot's best answer can change with the first slot's choice."""
        from champions.model.encoding import N_ATOMS
        with torch.no_grad():
            for module in (self.net.pair_left, self.net.pair_right, self.net.pair_gate):
                module.weight.normal_()
            q, mons = self.net.trunk(self.batch)
            scores = self.net.pair_logits(self.batch, q, mons).reshape(-1, N_ATOMS, N_ATOMS)
        mask = self.batch["atom_mask"][0]
        a = torch.nonzero(mask[0]).flatten()[:2]
        b = torch.nonzero(mask[1]).flatten()[:2]
        diff = scores[0, a, b[0]] - scores[0, a, b[1]]
        self.assertGreater(float((diff[0] - diff[1]).abs()), 1e-4)

    def test_a_hidden_slot_is_summed_out(self):
        from champions.model.encoding import N_ATOMS
        from champions.training.imitation import joint_nll
        logits = torch.randn(1, N_ATOMS * N_ATOMS)
        logp = torch.log_softmax(logits, -1).reshape(N_ATOMS, N_ATOMS)
        self.assertAlmostEqual(float(joint_nll(logits, torch.tensor([[3, -1]]))), float(-torch.logsumexp(logp[3], 0)), places=4)
        self.assertAlmostEqual(float(joint_nll(logits, torch.tensor([[-1, 5]]))), float(-torch.logsumexp(logp[:, 5], 0)), places=4)
        self.assertAlmostEqual(float(joint_nll(logits, torch.tensor([[3, 5]]))), float(-logp[3, 5]), places=4)

    def test_can_overfit_one_game(self):
        from champions.training.imitation import joint_nll
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=3e-3)
        losses = []
        for _ in range(80):
            loss = joint_nll(self.net(self.batch)[0], self.batch["label"]).mean()
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], 0.3 * losses[0])


@unittest.skipUnless(HAVE_TORCH, "run with .venv/bin/python (needs torch)")
class DamageFeatureTest(unittest.TestCase):
    def test_type_effectiveness_features_match_the_type_chart(self):
        from champions.model.policy import PolicyNet
        from champions.model.vocab import MOVE_ID, SPECIES
        examples, batch = sample_batch()
        net = PolicyNet(dim=32, layers=1, heads=2)
        # Raichu (Electric) facing Indeedee-F (Psychic/Normal) and Meowstic-F (Psychic) on turn 1
        for n, e in enumerate(examples):
            for slot in (0, 1):
                actor = int(e["arrays"]["actors"][slot])
                if actor >= 0 and SPECIES[int(e["arrays"]["species"][actor])] == "Raichu":
                    features = net.damage_features(batch, slot)
                    focus_blast, protect = features[n, MOVE_ID["Focus Blast"]], features[n, MOVE_ID["Protect"]]
                    self.assertAlmostEqual(float(focus_blast[0]), 1.0)   # Fighting vs Psychic/Normal: 0.5 * 2
                    self.assertAlmostEqual(float(focus_blast[5]), 0.5)   # Fighting vs Psychic
                    self.assertEqual(float(protect[1]), 0.0)             # a status move deals no damage
                    return
        self.fail("no Raichu decision in the sample")


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
class PreviewTest(unittest.TestCase):
    def setUp(self):
        from champions.teams.generate import load_usage
        from champions.training.preview_labels import preview_examples
        from champions.model.encoding import ARRAY_KEYS
        torch.manual_seed(0)
        self.examples = preview_examples(json.loads(FIXTURE.read_text())["log"].split("\n"), load_usage())
        self.batch = {k: torch.as_tensor(np.stack([e["arrays"][k] for e in self.examples])) for k in ARRAY_KEYS}
        self.batch["plan_ok"] = torch.as_tensor(np.array([e["plan_ok"] for e in self.examples]))
        self.batch["complete"] = torch.as_tensor([e["complete"] for e in self.examples])

    def test_labels_are_consistent(self):
        from champions.model.policy import PAIRS, PLANS, SUBSETS
        self.assertEqual(len(PLANS), 90)
        for e in self.examples:
            ok = [PLANS[i] for i, v in enumerate(e["plan_ok"]) if v]
            self.assertTrue(ok)
            self.assertEqual(len({p for _, p in ok}), 1)              # the leads are always seen
            for s, p in ok:
                self.assertTrue(set(PAIRS[p]) <= set(SUBSETS[s]))
            if e["complete"]:
                self.assertEqual(len(ok), 1)

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
        self.assertEqual(net.preview_logits(self.batch).shape, (len(self.examples), 90))

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


if __name__ == "__main__":
    unittest.main()
