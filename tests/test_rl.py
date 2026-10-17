import copy
import random
import unittest
from argparse import Namespace
from pathlib import Path

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from champions.simulate import RUNNER

FIXTURES = Path(__file__).parent / "fixtures"
SHOWDOWN_BUILT = (RUNNER.parent.parent / "vendor" / "pokemon-showdown" / "dist" / "sim").exists()


@unittest.skipUnless(HAVE_TORCH and SHOWDOWN_BUILT, "needs torch (.venv) and the built Showdown engine")
class PpoUpdateTest(unittest.TestCase):
    def test_update_changes_the_policy_and_stays_finite(self):
        from champions.policy import PolicyNet
        from champions.train_rl import play_batch, ppo_update, to_batch, training_jobs
        torch.manual_seed(0)
        pool = [{"text": (FIXTURES / name).read_text()} for name in ("team_tr_alpha.txt", "team_cybertron.txt")]
        net = PolicyNet(dim=32, layers=1, heads=2)
        reference = copy.deepcopy(net).eval()
        jobs = training_jobs(pool, 12, {"random": 1.0}, random.Random(0), "t")
        steps, record = play_batch(net, jobs, workers=2, seed=0, shaping=0.3)
        self.assertGreater(len(steps), 20)
        self.assertEqual(sum(g for _, g in record.values()), 12)

        before = [p.clone() for p in net.parameters()]
        args = Namespace(epochs=2, minibatch=64, clip=0.2, entropy=0.01, kl=0.03, gae_lambda=0.9)
        stats = ppo_update(net, reference, torch.optim.AdamW(net.parameters(), lr=1e-3), to_batch(steps), args, False)
        self.assertTrue(all(v == v and abs(v) < 1e6 for v in stats.values()), stats)   # no NaN or blow-up
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, net.parameters())))

    def test_advantages_reduce_to_result_minus_value_without_shaping(self):
        from champions.train_rl import advantages
        reward, last = torch.tensor([0.0, 0.0, 1.0, 0.0, -1.0]), torch.tensor([False, False, True, False, True])
        value = torch.tensor([0.2, 0.5, 0.9, -0.1, 0.0])
        expected = torch.tensor([1 - 0.2, 1 - 0.5, 1 - 0.9, -1 + 0.1, -1.0])
        self.assertTrue(torch.allclose(advantages(reward, last, value, 1.0), expected))

    def test_recorded_probabilities_match_the_policy(self):
        """The log-probability stored at play time must equal what the network gives when re-scored."""
        from champions.policy import PolicyNet
        from champions.train_rl import play_batch, to_batch, training_jobs
        torch.manual_seed(1)
        pool = [{"text": (FIXTURES / name).read_text()} for name in ("team_tr_alpha.txt", "team_cybertron.txt")]
        net = PolicyNet(dim=32, layers=1, heads=2)
        steps, _ = play_batch(net, training_jobs(pool, 6, {"random": 1.0}, random.Random(1), "t"), workers=1, seed=1)
        batch = to_batch(steps)
        with torch.no_grad():
            logp, _, _, _ = net.evaluate(batch, batch["action"], batch["target"], batch["target_mask"])
        self.assertTrue(torch.allclose(logp, batch["old_logp"], atol=1e-4), (logp - batch["old_logp"]).abs().max())


if __name__ == "__main__":
    unittest.main()
