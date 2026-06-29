from __future__ import annotations

import math
from dataclasses import dataclass, field

from horizon_sim.cognition.evidence_ledger import Evidence


@dataclass
class Proposition:
    id: str
    claim: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    strength: float = 0.0
    supporting_evidence: list[int] = field(default_factory=list)
    contradicting_evidence: list[int] = field(default_factory=list)
    last_updated: int = -1


def parse_claim(claim: str) -> tuple[str, str, str]:
    parts = claim.split()
    if len(parts) >= 3:
        return parts[0], parts[1], " ".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return claim, "", ""


def make_proposition(prop_id: str, claim: str) -> Proposition:
    subject, predicate, obj = parse_claim(claim)
    return Proposition(prop_id, claim, subject, predicate, obj)


def update_strength(prop: Proposition, evidence: list[Evidence], credibility_lookup, turn: int) -> None:
    weighted_sum = 0.0
    total_weight = 0.0
    for ev_id in prop.supporting_evidence + prop.contradicting_evidence:
        ev = evidence[ev_id]
        if ev.status == "pending":
            continue
        direction = 1.0 if ev.status == "verified" else -1.0
        source_credibility = credibility_lookup(ev.source, prop.id)
        weight = source_credibility * ev.confidence
        weighted_sum += direction * weight
        total_weight += weight
    prop.strength = math.tanh(weighted_sum / total_weight) if total_weight else 0.0
    prop.last_updated = turn
