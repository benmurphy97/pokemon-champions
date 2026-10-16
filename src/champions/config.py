"""Single place for regulation-specific settings. Change these when a new regulation starts."""

REGULATION = "M-C"  # live 2026-09-09 to 2026-12-02

# Pikalytics-style ID. Confirm the exact Showdown string in Phase 1.
FORMAT_ID = "gen9championsvgc2026regmc"

TEAM_SIZE = 6
BRING = 4  # doubles: bring 4 of 6, lead 2
LEVEL = 50

# Stat Points replace EVs/IVs in Champions. Showdown still writes them on the "EVs:" line.
STAT_POINT_TOTAL = 66
STAT_POINT_MAX_PER_STAT = 32
STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
