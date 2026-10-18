import unittest
from pathlib import Path

from champions.teams.parse import check_team, parse_team, serialise_team

FIXTURES = Path(__file__).parent / "fixtures"


class TeamsTest(unittest.TestCase):
    def load(self, name):
        return (FIXTURES / name).read_text()

    def test_parse_reads_six_mons(self):
        mons = parse_team(self.load("team_tr_alpha.txt"))
        self.assertEqual(len(mons), 6)
        self.assertEqual(mons[0]["name"], "Farigiraf")
        self.assertEqual(mons[0]["item"], "Mental Herb")
        self.assertEqual(mons[0]["evs"]["HP"], 32)
        self.assertEqual(mons[0]["moves"][0], "Trick Room")

    def test_round_trip_is_stable(self):
        for name in ("team_tr_alpha.txt", "team_cybertron.txt"):
            mons = parse_team(self.load(name))
            self.assertEqual(parse_team(serialise_team(mons)), mons)

    def test_fixtures_are_legal(self):
        for name in ("team_tr_alpha.txt", "team_cybertron.txt"):
            self.assertEqual(check_team(parse_team(self.load(name))), [])

    def test_check_catches_duplicate_item_and_overspend(self):
        mons = parse_team(self.load("team_tr_alpha.txt"))
        mons[1]["item"] = mons[0]["item"]
        mons[2]["evs"]["Atk"] = 40
        problems = check_team(mons)
        self.assertTrue(any("item clause" in p for p in problems))
        self.assertTrue(any("one stat" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
