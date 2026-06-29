from __future__ import annotations

import math
from dataclasses import dataclass, field

from horizon_sim.cognition.evidence_ledger import Evidence


@dataclass
class Proposition:
    strength: float = 0.0
    supporting_evidence: list[int] = field(default_factory=list)
    contradicting_evidence: list[int] = field(default_factory=list)
    last_updated: int = -1


def update_strength(prop: Proposition, evidence: list[Evidence], trust_lookup, turn: int) -> None:
    weighted_sum = 0.0
    total_weight = 0.0
    for ev_id in prop.supporting_evidence + prop.contradicting_evidence:
        ev = evidence[ev_id]
        if ev.status == "pending":
            continue
        direction = 1.0 if ev.status == "verified" else -1.0
        source_trust = 1.0 if ev.source == -1 else trust_lookup(ev.source)
        weight = source_trust * ev.confidence
        weighted_sum += direction * weight
        total_weight += weight
    prop.strength = math.tanh(weighted_sum / total_weight) if total_weight else 0.0
    prop.last_updated = turn
