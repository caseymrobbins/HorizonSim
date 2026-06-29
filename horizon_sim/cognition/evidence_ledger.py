from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    id: int
    source: int
    type: str
    proposition_id: str
    claim: str
    confidence: float
    status: str
    turn_received: int
