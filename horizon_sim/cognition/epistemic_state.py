from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from horizon_sim.cognition.evidence_ledger import Evidence


@dataclass
class EpistemicState:
    """Proposition-dependent source credibility derived from resolved evidence."""

    default_credibility: float = 0.5
    source_proposition_credibility: dict[int, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))

    def credibility(self, source_id: int, proposition_id: str, proposition_graph: dict[str, object]) -> float:
        if source_id == -1:
            return 1.0
        source_scores = self.source_proposition_credibility.get(source_id, {})
        if proposition_id in source_scores:
            return source_scores[proposition_id]

        weighted_sum = 0.0
        total_weight = 0.0
        target = proposition_graph.get(proposition_id)
        for known_id, score in source_scores.items():
            weight = proposition_similarity(target, proposition_graph.get(known_id))
            if weight <= 0.0:
                continue
            weighted_sum += weight * score
            total_weight += weight
        return weighted_sum / total_weight if total_weight else self.default_credibility

    def update_from_evidence(self, evidence_ledger: list[Evidence], proposition_graph: dict[str, object]) -> None:
        resolved: dict[int, dict[str, list[Evidence]]] = defaultdict(lambda: defaultdict(list))
        for ev in evidence_ledger:
            if ev.source == -1 or ev.status == "pending":
                continue
            resolved[ev.source][ev.proposition_id].append(ev)

        self.source_proposition_credibility = defaultdict(dict)
        for source_id, by_proposition in resolved.items():
            for proposition_id, records in by_proposition.items():
                verified_weight = sum(ev.confidence for ev in records if ev.status == "verified")
                total_weight = sum(ev.confidence for ev in records)
                self.source_proposition_credibility[source_id][proposition_id] = (
                    verified_weight / total_weight if total_weight else self.default_credibility
                )


def proposition_similarity(left: object | None, right: object | None) -> float:
    """Approximate local proposition similarity without hard-coded domains."""
    if left is None or right is None:
        return 0.0
    if getattr(left, "id", None) == getattr(right, "id", None):
        return 1.0

    weight = 0.0
    if getattr(left, "predicate", None) and getattr(left, "predicate", None) == getattr(right, "predicate", None):
        weight += 0.6
    if getattr(left, "subject", None) and getattr(left, "subject", None) == getattr(right, "subject", None):
        weight += 0.3
    if getattr(left, "object", None) and getattr(left, "object", None) == getattr(right, "object", None):
        weight += 0.2
    return min(weight, 1.0)
