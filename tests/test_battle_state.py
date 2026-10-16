import json
import unittest
from pathlib import Path

from champions.battle_state import BattleState, parse_hp, parse_position

FIXTURE = Path(__file__).parent / "fixtures" / "replay_sample.json"


def play(lines):
    state = BattleState()
    for line in lines:
        state.update(line)
    return state


class ParsingTest(unittest.TestCase):
    def test_hp_exact_and_percent_and_faint(self):
        self.assertAlmostEqual(parse_hp("77/164 par")[0], 77 / 164)
        self.assertEqual(parse_hp("77/164 par")[1], "par")
        self.assertEqual(parse_hp("0 fnt"), (0.0, "fnt"))
        self.assertEqual(parse_hp("100/100"), (1.0, None))

    def test_positions(self):
        self.assertEqual(parse_position("p2b: Nick Name"), ("p2", 1, "Nick Name"))
        self.assertEqual(parse_position("p1: Player"), ("p1", None, "Player"))


class SmallBattleTest(unittest.TestCase):
    def test_switch_damage_boost_weather_and_faint(self):
        state = play([
            "|poke|p1|Rillaboom, L50, M|", "|poke|p1|Incineroar, L50, M|",
            "|poke|p2|Gholdengo, L50|", "|poke|p2|Torkoal, L50, F|",
            "|switch|p1a: Rilla|Rillaboom, L50, M|100/100",
            "|switch|p2a: Gold|Gholdengo, L50|100/100",
            "|turn|1",
            "|-damage|p1a: Rilla|50/100",
            "|-boost|p2a: Gold|spa|2", "|-unboost|p2a: Gold|spa|1",
            "|-weather|SunnyDay", "|-fieldstart|move: Trick Room",
            "|-sidestart|p1: Player|move: Tailwind",
            "|move|p2a: Gold|Make It Rain|p1a: Rilla",
            "|-damage|p1a: Rilla|0 fnt", "|faint|p1a: Rilla",
        ])
        rilla, gold = state.sides["p1"].team[0], state.sides["p2"].team[0]
        self.assertTrue(rilla.fainted)
        self.assertEqual(state.sides["p1"].active[0], None)
        self.assertEqual(gold.boosts["spa"], 1)
        self.assertEqual(gold.moves, ["Make It Rain"])
        self.assertEqual((state.weather, state.trick_room), ("sun", True))
        self.assertIn("tailwind", state.sides["p1"].conditions)
        self.assertEqual(gold.turns_active, 1)

    def test_moves_called_by_other_moves_are_not_recorded(self):
        state = play(["|poke|p1|Rillaboom, L50, M|", "|switch|p1a: R|Rillaboom, L50, M|100/100",
                      "|move|p1a: R|Hyper Voice||[from]Sleep Talk"])
        self.assertEqual(state.sides["p1"].team[0].moves, [])


class FormChangeTest(unittest.TestCase):
    def test_form_names_that_differ_from_preview_still_find_the_same_pokemon(self):
        state = play(["|poke|p1|Floette-Eternal, L50, F|", "|poke|p1|Mimikyu, L50|",
                      "|switch|p1a: F|Floette, L50, F|100/100", "|switch|p1b: M|Mimikyu-Busted, L50|100/100"])
        self.assertEqual(len(state.sides["p1"].team), 2)
        self.assertEqual(state.sides["p1"].active, [0, 1])


class ReplayFixtureTest(unittest.TestCase):
    def setUp(self):
        self.state = play(json.loads(FIXTURE.read_text())["log"].split("\n"))

    def test_teams_and_winner(self):
        self.assertEqual([len(s.team) for s in self.state.sides.values()], [6, 6])
        self.assertEqual(self.state.winner, "p2")
        self.assertEqual(self.state.turn, 7)

    def test_mega_and_form_tracked(self):
        meowstic = next(m for m in self.state.sides["p2"].team if m.species == "Meowstic-F")
        self.assertTrue(meowstic.mega)
        self.assertEqual(meowstic.form, "Meowstic-F-Mega")
        self.assertTrue(self.state.sides["p1"].mega_used)

    def test_revealed_count_matches_brought(self):
        self.assertEqual(sum(m.revealed for m in self.state.sides["p1"].team), 4)


if __name__ == "__main__":
    unittest.main()


class HpColourTest(unittest.TestCase):
    def test_replay_colour_suffix(self):
        self.assertEqual(parse_hp("100/100g"), (1.0, None))
        self.assertAlmostEqual(parse_hp("45/100y par")[0], 0.45)


class HindsightTest(unittest.TestCase):
    def test_sets_are_rebuilt_from_the_whole_game(self):
        from champions.hindsight import complete_sets
        from champions.meta import load_usage
        state = play(json.loads(FIXTURE.read_text())["log"].split("\n"))
        info = complete_sets(state, "p1", load_usage())
        raichu = info[3]
        self.assertEqual(raichu["item"], "Raichunite Y")                 # shown when it Mega Evolved
        self.assertTrue({"Zap Cannon", "Protect", "Focus Blast", "Fake Out"} <= set(raichu["moves"]))
        self.assertEqual(len(info[2]["moves"]), 4)                        # Milotic never moved: filled from ladder stats
        self.assertEqual([info[i]["brought"] for i in range(6)], [False, True, True, True, True, False])
