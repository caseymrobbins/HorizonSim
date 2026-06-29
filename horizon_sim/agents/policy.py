from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    kind: str
    params: dict


class UtilityMaxPolicy:
    """Small baseline policy: harvest preferred local resources, otherwise wander."""

    def choose_action(self, agent, world) -> Action:
        x, y = world.agent_positions[agent.id]
        tile = world.grid[x][y]
        best_resource = None
        best_score = 0.0
        for resource, amount in tile.resources.items():
            score = agent.preferences.get(resource, 0.0) * amount
            if amount > 0 and score > best_score:
                best_resource = resource
                best_score = score
        if best_resource is not None:
            return Action("HARVEST", {"resource": best_resource, "amount": 1})
        return Action("MOVE", {"dx": world.rng.choice([-1, 0, 1]), "dy": world.rng.choice([-1, 0, 1])})
