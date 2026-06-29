from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MapCell:
    terrain: str = "unknown"
    resources: dict[str, float] = field(default_factory=dict)
    owner: Optional[int] = None
    last_observation: int = -1
    confidence: float = 0.0


class SpatialMap:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.cells = [[MapCell() for _ in range(height)] for _ in range(width)]

    def decay(self, rate: float = 0.02) -> None:
        for column in self.cells:
            for cell in column:
                cell.confidence = max(0.0, cell.confidence * (1.0 - rate))

    def apply_observations(self, observations: list[dict], turn: int) -> None:
        for obs in observations:
            x, y = obs["position"]
            self.cells[x][y] = MapCell(
                terrain=obs["terrain"],
                resources={k: float(v) for k, v in obs["resources"].items()},
                owner=obs["owner"],
                last_observation=turn,
                confidence=1.0,
            )
