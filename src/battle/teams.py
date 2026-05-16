import random
from pathlib import Path
from poke_env.teambuilder import Teambuilder

# ---------------------------------------------------------------------------
# Team loader — reads from /teams/*.txt relative to the project root
# ---------------------------------------------------------------------------

_TEAMS_DIR = Path(__file__).parent.parent / "data" / "teams"


def load_team(name: str) -> str:
    """Load a team string from teams/<name>.txt"""
    path = _TEAMS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Team file not found: {path}")
    return path.read_text()


class RotatingTeambuilder(Teambuilder):
    """Picks a random team from a pool on every yield_team() call."""

    def __init__(self, *team_strings: str):
        self._packed = [
            self.join_team(self.parse_showdown_team(t)) for t in team_strings
        ]

    def yield_team(self) -> str:
        return random.choice(self._packed)


# ---------------------------------------------------------------------------
# Our teams
# ---------------------------------------------------------------------------

TEAM_ALPHA    = load_team("alpha")      # Sun/TR hybrid (original)
TEAM_TR_ALPHA = load_team("tr_alpha")  # Bulky TR offense (Farigiraf core)

# ---------------------------------------------------------------------------
# Opponent teams
# ---------------------------------------------------------------------------

TEAM_WOLFE     = load_team("wolfe")      # Sand + wide guard + tailwind
TEAM_CYBERTRON = load_team("cybertron")  # Mega Charizard Y sun
TEAM_ROD       = load_team("rod")        # Mega Charizard X / TR snow
TEAM_YUZU      = load_team("yuzu")       # Mega Gengar perish trap
TEAM_KURO      = load_team("kuro")       # Mega Blastoise rain
TEAM_WDMICHAEL = load_team("wdmichael")  # Mega Golurk trick room
TEAM_HYUNGWOO  = load_team("hyungwoo")   # Sun offense
TEAM_JORGE     = load_team("jorge")      # Mega Charizard X / Mega Tyranitar
