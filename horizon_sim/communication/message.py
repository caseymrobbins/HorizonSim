from __future__ import annotations

from dataclasses import dataclass, field

MSG_TYPES = frozenset({"TELL", "ASK", "OFFER", "ACCEPT", "REJECT", "INTRODUCE"})


@dataclass(frozen=True)
class Message:
    sender: int
    receiver: int
    msg_type: str  # One of TELL, ASK, OFFER, ACCEPT, REJECT, INTRODUCE
    content: dict = field(default_factory=dict)
    proposition_id: str | None = None
    introduced_agent: int | None = None
    confidence: float = 0.7
    timestamp: int = 0
