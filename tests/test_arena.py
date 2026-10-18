import json
import unittest
from pathlib import Path

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from champions.sim.simulate import RUNNER

FIXTURES = Path(__file__).parent / "fixtures"
SHOWDOWN_BUILT = (RUNNER.parent.parent / "vendor" / "pokemon-showdown" / "dist" / "sim").exists()


@unittest.skipUnless(HAVE_TORCH and SHOWDOWN_BUILT, "needs torch (.venv) and the built Showdown engine")
class ArenaTest(unittest.TestCase):
    def test_untrained_policy_finishes_games_with_legal_choices(self):
        from champions.model.agent import NeuralAgent
        from champions.sim.arena import challenger_wins, paired_jobs, run_jobs
        from champions.model.policy import PolicyNet
        torch.manual_seed(0)
        pool = [{"text": (FIXTURES / name).read_text()} for name in ("team_tr_alpha.txt", "team_cybertron.txt")]
        agent = NeuralAgent(PolicyNet(dim=32, layers=1, heads=2))
        results = run_jobs(paired_jobs(pool, 8, "neural", "random", seed=1), agent, workers=2)
        wins, decided, fallbacks = challenger_wins(results)
        self.assertEqual(len(results), 8)
        self.assertEqual(decided, 8)
        self.assertEqual(fallbacks, 0)   # every choice the policy made was accepted by the engine

    def test_a_plugged_in_team_preview_chooses_the_lineup(self):
        from champions.model.agent import NeuralAgent
        from champions.sim.arena import paired_jobs, run_jobs
        from champions.model.policy import PolicyNet
        torch.manual_seed(0)
        pool = [{"text": (FIXTURES / name).read_text()} for name in ("team_tr_alpha.txt", "team_cybertron.txt")]
        asked = []

        def preview(state, side, request):
            asked.append(len(request["side"]["pokemon"]))
            return "team 5612"          # bring Pokémon 5, 6, 1 and 2; the first two listed lead

        agent = NeuralAgent(PolicyNet(dim=32, layers=1, heads=2), preview=preview)
        results = run_jobs(paired_jobs(pool, 4, "neural", "random", seed=2), agent, workers=1)
        self.assertEqual(len(results), 4)
        self.assertEqual(set(asked), {6})        # shown all six of its own team
        self.assertEqual(sum(r["fallbacks"] for r in results), 0)

    def test_results_report_margin_idle_turns_and_timeouts_and_few_jobs_do_not_hang(self):
        from champions.model.agent import NeuralAgent
        from champions.sim.arena import challenger_margin, paired_jobs, run_jobs
        from champions.model.policy import PolicyNet
        torch.manual_seed(4)
        pool = [{"text": (FIXTURES / name).read_text()} for name in ("team_tr_alpha.txt", "team_cybertron.txt")]
        # 4 games on 6 workers: fewer jobs than workers must still finish (idle workers used to wait forever)
        results = run_jobs(paired_jobs(pool, 4, "neural", "random", seed=5), NeuralAgent(PolicyNet(dim=32, layers=1, heads=2)), workers=6)
        for r in results:
            self.assertTrue(-1.0 <= r["margin"] <= 1.0)
            self.assertIn(r["timeout"], (True, False))
            self.assertGreaterEqual(r["idleTurns"], 0)
        mean, se = challenger_margin(results)
        self.assertTrue(-1.0 <= mean <= 1.0 and se >= 0)

    def test_a_game_past_the_turn_cap_is_ended_and_flagged(self):
        from champions.model.agent import NeuralAgent
        from champions.sim.arena import run_jobs
        from champions.model.policy import PolicyNet
        text = (FIXTURES / "team_tr_alpha.txt").read_text()
        job = {"id": "cap", "a": text, "b": text, "seed": 1, "agentA": "neural", "agentB": "random", "maxTurns": 1}
        torch.manual_seed(0)
        result = run_jobs([job], NeuralAgent(PolicyNet(dim=32, layers=1, heads=2)), workers=1)[0]
        self.assertTrue(result["timeout"])
        self.assertEqual(result["winner"], "tie")

    def test_two_different_networks_can_play_each_other(self):
        from champions.model.agent import NeuralAgent
        from champions.sim.arena import run_jobs
        from champions.model.policy import PolicyNet
        text = (FIXTURES / "team_tr_alpha.txt").read_text()
        torch.manual_seed(1)
        agents = {"learner": NeuralAgent(PolicyNet(dim=32, layers=1, heads=2)), "snap3": NeuralAgent(PolicyNet(dim=32, layers=1, heads=2))}
        jobs = [{"id": f"g{i}", "a": text, "b": text, "seed": i, "agentA": "learner", "agentB": "snap3"} for i in range(3)]
        trajectories: dict = {}
        results = run_jobs(jobs, None, 2, trajectories=trajectories, agents=agents, record_names={"learner"})
        self.assertEqual(sum(r["fallbacks"] for r in results), 0)
        self.assertTrue(all(side == "p1" for _, side in trajectories))     # only the learner's seat was recorded

    def test_games_complete_with_the_learned_preview_even_when_untrained(self):
        from champions.model.agent import NeuralAgent
        from champions.sim.arena import challenger_wins, paired_jobs, run_jobs
        from champions.model.policy import PolicyNet
        from champions.model.preview import LearnedPreview
        torch.manual_seed(3)
        pool = [{"text": (FIXTURES / name).read_text()} for name in ("team_tr_alpha.txt", "team_cybertron.txt")]
        net = PolicyNet(dim=32, layers=1, heads=2)
        agent = NeuralAgent(net, preview=LearnedPreview(net, temperature=1.0, seed=1))
        results = run_jobs(paired_jobs(pool, 6, "neural", "random", seed=3), agent, workers=2)
        wins, decided, fallbacks = challenger_wins(results)
        self.assertEqual((decided, fallbacks), (6, 0))     # every lineup it picked was accepted by the engine


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_TORCH, "needs torch (.venv)")
class ForcedMoveTest(unittest.TestCase):
    def test_locked_or_recharging_pokemon_just_plays_move_1(self):
        from champions.model.agent import NeuralAgent
        from champions.model.battle_state import BattleState
        from champions.model.policy import PolicyNet
        state = BattleState()
        for line in ["|poke|p1|Rillaboom, L50, M|", "|poke|p1|Incineroar, L50, M|", "|poke|p2|Sneasler, L50, M|",
                     "|switch|p1a: R|Rillaboom, L50, M|100/100", "|switch|p2a: S|Sneasler, L50, M|100/100", "|turn|2"]:
            state.update(line)
        request = {"active": [{"moves": [{"move": "Recharge", "id": "recharge"}]}],
                   "side": {"pokemon": [{"details": "Rillaboom, L50, M", "condition": "100/100", "active": True},
                                        {"details": "Incineroar, L50, M", "condition": "100/100", "active": False}]}}
        agent = NeuralAgent(PolicyNet(dim=32, layers=1, heads=2))
        self.assertEqual(agent.decide(state, "p1", request), "move 1")


@unittest.skipUnless(HAVE_TORCH, "needs torch (.venv)")
class ForcedDoubleSwitchTest(unittest.TestCase):
    def test_two_forced_switches_with_one_pokemon_left_send_it_in_once(self):
        from champions.model.agent import NeuralAgent
        from champions.model.battle_state import BattleState
        from champions.model.policy import PolicyNet
        state = BattleState()
        for line in ["|poke|p1|Rillaboom, L50, M|", "|poke|p1|Incineroar, L50, M|", "|poke|p1|Sneasler, L50, M|",
                     "|poke|p2|Garchomp, L50, M|", "|switch|p1a: R|Rillaboom, L50, M|100/100",
                     "|switch|p1b: I|Incineroar, L50, M|100/100", "|switch|p2a: G|Garchomp, L50, M|100/100",
                     "|faint|p1a: R", "|faint|p1b: I", "|turn|3"]:
            state.update(line)
        request = {"forceSwitch": [True, True],
                   "side": {"pokemon": [{"details": "Rillaboom, L50, M", "condition": "0 fnt", "active": True},
                                        {"details": "Incineroar, L50, M", "condition": "0 fnt", "active": True},
                                        {"details": "Sneasler, L50, M", "condition": "100/100", "active": False}]}}
        for seed in range(5):
            torch.manual_seed(seed)
            choice = NeuralAgent(PolicyNet(dim=32, layers=1, heads=2)).decide(state, "p1", request)
            self.assertEqual(sorted(choice.split(", ")), ["pass", "switch 3"])
