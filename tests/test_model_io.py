import tempfile
import unittest
from pathlib import Path

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "run with .venv/bin/python (needs torch)")
class ModelIoTest(unittest.TestCase):
    def batch(self):
        import json
        import numpy as np
        from champions.training.decisions import extract
        from champions.model.encoding import ARRAY_KEYS
        sample = Path(__file__).parent / "fixtures" / "replay_sample.json"
        examples = extract(json.loads(sample.read_text())["log"].split("\n"))
        return {k: torch.as_tensor(np.stack([e["arrays"][k] for e in examples])) for k in ARRAY_KEYS}

    def test_round_trip_keeps_the_model_identical(self):
        from champions.model.model_io import load_policy, save_policy
        from champions.model.policy import PolicyNet
        torch.manual_seed(0)
        net = PolicyNet(dim=32, layers=2, heads=2, dropout=0.0).eval()
        with tempfile.TemporaryDirectory() as tmp:
            save_policy(net, Path(tmp, "m.pt"))
            loaded = load_policy(Path(tmp, "m.pt"))
        b = self.batch()
        self.assertEqual(loaded.config(), net.config())
        self.assertTrue(torch.allclose(net(b)[0], loaded(b)[0]))

    def test_a_deeper_model_starts_out_playing_exactly_like_the_old_one(self):
        from champions.model.model_io import adapt_state
        from champions.model.policy import PolicyNet
        torch.manual_seed(1)
        old = PolicyNet(dim=32, layers=1, heads=2, dropout=0.0).eval()
        bigger = PolicyNet(dim=32, layers=3, heads=2, dropout=0.0).eval()
        adapted, report = adapt_state(bigger, old.state_dict())
        bigger.load_state_dict(adapted, strict=False)
        self.assertTrue(report["new"])
        b = self.batch()
        with torch.no_grad():
            self.assertTrue(torch.allclose(old(b)[0], bigger(b)[0], atol=1e-5))

    def test_a_missing_input_branch_and_a_smaller_embedding_are_carried_over(self):
        from champions.model.model_io import adapt_state
        from champions.model.policy import PolicyNet
        torch.manual_seed(2)
        net = PolicyNet(dim=32, layers=1, heads=2, dropout=0.0).eval()
        saved = {k: v for k, v in net.state_dict().items() if not k.startswith("partner_")}
        saved["kind_emb.weight"] = saved["kind_emb.weight"][:2].clone()      # a model from before the spare kinds existed
        adapted, report = adapt_state(net, saved)
        self.assertIn("kind_emb.weight", report["padded"])
        self.assertTrue(torch.equal(adapted["kind_emb.weight"][:2], saved["kind_emb.weight"]))
        self.assertTrue(any(n.startswith("partner_") for n in report["new"]))


if __name__ == "__main__":
    unittest.main()
