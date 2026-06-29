from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from horizon_sim.world.tile import Tile

Position = tuple[int, int]


@dataclass
class World:
    width: int
    height: int
    grid: list[list[Tile]]
    agent_positions: dict[int, Position] = field(default_factory=dict)
    resource_regrowth: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)

    @classmethod
    def random(
        cls,
        width: int,
        height: int,
        terrain_distribution: dict[str, float],
        resource_regrowth: dict[str, dict[str, dict[str, float]]] | None = None,
        seed: int | None = None,
    ) -> "World":
        rng = random.Random(seed)
        terrains = list(terrain_distribution)
        weights = [terrain_distribution[t] for t in terrains]
        grid: list[list[Tile]] = []
        for x in range(width):
            col = []
            for _y in range(height):
                terrain = rng.choices(terrains, weights=weights, k=1)[0]
                resources = {}
                if terrain == "forest":
                    resources = {"food": rng.randint(0, 3), "wood": rng.randint(1, 3)}
                elif terrain == "plains":
                    resources = {"food": rng.randint(0, 2)}
                elif terrain == "mountain":
                    resources = {"stone": rng.randint(0, 2)}
                col.append(Tile(terrain=terrain, resources=resources))
            grid.append(col)
        return cls(width, height, grid, resource_regrowth=resource_regrowth or {}, rng=rng)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_observations(self, pos: Position, radius: int) -> list[dict]:
        observations = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = pos[0] + dx, pos[1] + dy
                if self.in_bounds(x, y):
                    tile = self.grid[x][y]
                    observations.append(
                        {
                            "position": (x, y),
                            "terrain": tile.terrain,
                            "resources": tile.resources.copy(),
                            "structure": tile.structure,
                            "owner": tile.owner_id,
                        }
                    )
        return observations

    def regenerate_resources(self) -> None:
        for x in range(self.width):
            for y in range(self.height):
                tile = self.grid[x][y]
                for resource, terrain_rules in self.resource_regrowth.items():
                    rule = terrain_rules.get(tile.terrain)
                    if not rule:
                        continue
                    current = tile.resources.get(resource, 0)
                    if current < int(rule.get("max", current)) and self.rng.random() < float(rule.get("chance", 0.0)):
                        tile.resources[resource] = current + 1

    def move_agent(self, agent_id: int, dx: int, dy: int) -> bool:
        x, y = self.agent_positions[agent_id]
        nx, ny = x + max(-1, min(1, dx)), y + max(-1, min(1, dy))
        if self.in_bounds(nx, ny) and self.grid[nx][ny].terrain != "water":
            self.agent_positions[agent_id] = (nx, ny)
            return True
        return False

    def harvest(self, agent_id: int, resource: str, amount: int) -> int:
        x, y = self.agent_positions[agent_id]
        available = self.grid[x][y].resources.get(resource, 0)
        taken = min(max(0, amount), available)
        self.grid[x][y].resources[resource] = available - taken
        return taken
