"""Single PBT agent: owns a team string, a model checkpoint, and fitness history."""

import json
import os
import shutil
from dataclasses import dataclass, field


@dataclass
class Agent:
    agent_id: int
    team: str                        # current Showdown team string
    checkpoint_dir: str              # e.g. checkpoints/league/agent_0
    fitness: float = 0.0
    fitness_history: list[float] = field(default_factory=list)
    generation: int = 0              # how many training rounds completed
    mon_tenure: dict = field(default_factory=dict)  # pokemon_name -> generations on team

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "model")

    def update_tenure(self):
        """Increment tenure for Pokémon still on the team; new arrivals start at 1."""
        from genetic_algorithm.chromosome import parse_team
        current_names = {m["name"] for m in parse_team(self.team)}
        self.mon_tenure = {name: self.mon_tenure.get(name, 0) + 1 for name in current_names}

    def save_team(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        with open(os.path.join(self.checkpoint_dir, "team.txt"), "w") as f:
            f.write(self.team)
        with open(os.path.join(self.checkpoint_dir, "tenure.json"), "w") as f:
            json.dump(self.mon_tenure, f)

    def load_team(self) -> bool:
        path = os.path.join(self.checkpoint_dir, "team.txt")
        if not os.path.exists(path):
            return False
        with open(path) as f:
            self.team = f.read()
        tenure_path = os.path.join(self.checkpoint_dir, "tenure.json")
        if os.path.exists(tenure_path):
            with open(tenure_path) as f:
                self.mon_tenure = json.load(f)
        return True

    def copy_from(self, other: "Agent"):
        """Copy model weights only from another agent (exploit step). Team is preserved."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        src = other.checkpoint_path + ".zip"
        dst = self.checkpoint_path + ".zip"
        if os.path.exists(src):
            shutil.copy2(src, dst)
        self.generation = other.generation
