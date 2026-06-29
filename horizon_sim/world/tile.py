from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Tile:
    """Ground-truth state for a single world tile."""

    terrain: str
    resources: dict[str, int] = field(default_factory=dict)
    structure: Optional[str] = None
    owner_id: Optional[int] = None
