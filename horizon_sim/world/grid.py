from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

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
    # Node-based scarcity model — populated by World.random when n_nodes_per_resource > 0.
    # Distinct geographic clusters per resource force agents to travel and trade
    # rather than self-supplying from diffuse per-tile yields.
    node_positions: dict[str, list[Position]] = field(default_factory=dict)
    node_yield: int = 0  # capacity cap per node; 0 = diffuse (legacy) mode

    @classmethod
    def random(
        cls,
        width: int,
        height: int,
        terrain_distribution: dict[str, float],
        resource_regrowth: dict[str, dict[str, dict[str, float]]] | None = None,
        seed: int | None = None,
        n_nodes_per_resource: int = 3,
        node_yield: int = 7,
    ) -> "World":
        rng = random.Random(seed)
        terrains = list(terrain_distribution)
        weights = [terrain_distribution[t] for t in terrains]
        grid: list[list[Tile]] = []
        for x in range(width):
            col = []
            for _y in range(height):
                terrain = rng.choices(terrains, weights=weights, k=1)[0]
                # Start every tile resource-free; nodes are placed below.
                col.append(Tile(terrain=terrain, resources={}))
            grid.append(col)

        world = cls(
            width, height, grid,
            resource_regrowth=resource_regrowth or {},
            rng=rng,
            node_yield=node_yield,
        )

        if n_nodes_per_resource > 0 and node_yield > 0:
            world.node_positions = _place_resource_nodes(
                rng, grid, width, height,
                resource_types=["food", "wood", "stone"],
                n_nodes=n_nodes_per_resource,
                node_yield=node_yield,
            )

        return world

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
        if self.node_positions:
            # Node mode: only regrow at designated node tiles up to node_yield.
            # Non-node tiles stay empty so scarcity is preserved.
            for resource, positions in self.node_positions.items():
                for (x, y) in positions:
                    tile = self.grid[x][y]
                    current = tile.resources.get(resource, 0)
                    if current < self.node_yield and self.rng.random() < 0.15:
                        tile.resources[resource] = current + 1
            return
        # Diffuse mode: terrain-based regrowth (kept for direct World construction in tests)
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


def _place_resource_nodes(
    rng: random.Random,
    grid: list[list[Tile]],
    width: int,
    height: int,
    resource_types: list[str],
    n_nodes: int,
    node_yield: int,
) -> dict[str, list[Position]]:
    """Place n_nodes clustered nodes per resource type in distinct geographic regions.

    Distinct regions mean agents must travel to different parts of the map for
    different resource types, creating inter-agent dependence and making nodes
    worth cornering (both trade-failure and monopoly-failure are addressed).
    """
    # Choose well-separated cluster centers (one per resource type)
    min_sep = max(5, int(min(width, height) * 0.3))
    centers: list[Position] = []
    for _ in range(len(resource_types)):
        chosen: Position | None = None
        for _attempt in range(500):
            x = rng.randrange(width)
            y = rng.randrange(height)
            if grid[x][y].terrain == "water":
                continue
            if all(math.sqrt((x - cx) ** 2 + (y - cy) ** 2) >= min_sep for cx, cy in centers):
                chosen = (x, y)
                break
        if chosen is None:
            # Fallback: first non-water, non-center tile
            for x in range(width):
                for y in range(height):
                    if grid[x][y].terrain != "water" and (x, y) not in centers:
                        chosen = (x, y)
                        break
                if chosen:
                    break
        if chosen:
            centers.append(chosen)

    cluster_radius = max(2, min(width, height) // 8)
    node_positions: dict[str, list[Position]] = {}

    for res_type, center in zip(resource_types, centers):
        nodes: list[Position] = []
        attempts = 0
        while len(nodes) < n_nodes and attempts < 400:
            attempts += 1
            ox = rng.randint(-cluster_radius, cluster_radius)
            oy = rng.randint(-cluster_radius, cluster_radius)
            nx, ny = center[0] + ox, center[1] + oy
            if (
                0 <= nx < width
                and 0 <= ny < height
                and grid[nx][ny].terrain != "water"
                and (nx, ny) not in nodes
            ):
                nodes.append((nx, ny))
                grid[nx][ny].resources[res_type] = node_yield
        node_positions[res_type] = nodes

    return node_positions
