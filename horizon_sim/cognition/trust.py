from __future__ import annotations

from horizon_sim.cognition.evidence_ledger import Evidence


def compute_trust(evidence_ledger: list[Evidence], target_id: int) -> float:
    resolved = [e for e in evidence_ledger if e.source == target_id and e.status != "pending"]
    if not resolved:
        return 0.5
    verified = sum(1 for e in resolved if e.status == "verified")
    return verified / len(resolved)
