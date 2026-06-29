from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    sender: int
    receiver: int
    msg_type: str
    content: dict = field(default_factory=dict)
