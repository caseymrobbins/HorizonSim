from __future__ import annotations

from dataclasses import dataclass, field, replace

from horizon_sim.cognition.belief_graph import Proposition, update_strength
from horizon_sim.cognition.evidence_ledger import Evidence
from horizon_sim.cognition.spatial_map import SpatialMap
from horizon_sim.cognition.trust import compute_trust
from horizon_sim.communication.message import Message


@dataclass
class Agent:
    id: int
    position: tuple[int, int]
    preferences: dict[str, float]
    inventory: dict[str, int] = field(default_factory=dict)
    policy: object | None = None
    known_agents: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.spatial_map: SpatialMap | None = None
        self.belief_graph: dict[str, Proposition] = {}
        self.evidence_ledger: list[Evidence] = []

    def attach_map(self, width: int, height: int) -> None:
        self.spatial_map = SpatialMap(width, height)

    def add_evidence(self, source: int, evidence_type: str, claim: str, confidence: float, turn: int, status: str = "pending") -> int:
        ev_id = len(self.evidence_ledger)
        self.evidence_ledger.append(Evidence(ev_id, source, evidence_type, claim, max(0.0, min(1.0, confidence)), status, turn))
        prop = self.belief_graph.setdefault(claim, Proposition())
        prop.supporting_evidence.append(ev_id)
        return ev_id

    def ingest_observations(self, observations: list[dict], turn: int) -> None:
        if self.spatial_map is not None:
            self.spatial_map.decay()
            self.spatial_map.apply_observations(observations, turn)
        for obs in observations:
            pos = obs["position"]
            self.add_evidence(-1, "observation", f"Cell{pos} terrain={obs['terrain']}", 0.99, turn, "verified")
            for res, amount in obs["resources"].items():
                status = "verified" if amount >= 1 else "contradicted"
                self.add_evidence(-1, "observation", f"Cell{pos} contains {res} >= 1", 0.99, turn, status)

    def ingest_messages(self, messages: list[Message], turn: int) -> None:
        for message in messages:
            self.known_agents.add(message.sender)
            if message.msg_type == "TELL" and "claim" in message.content:
                confidence = float(message.content.get("confidence", self.get_trust(message.sender)))
                self.add_evidence(message.sender, "communication", message.content["claim"], confidence, turn)

    def resolve_evidence_against_observations(self, observations: list[dict]) -> None:
        observable_claims = set()
        false_resource_claims = set()
        for obs in observations:
            pos = obs["position"]
            observable_claims.add(f"Cell{pos} terrain={obs['terrain']}")
            for res, amount in obs["resources"].items():
                claim = f"Cell{pos} contains {res} >= 1"
                (observable_claims if amount >= 1 else false_resource_claims).add(claim)
        for idx, ev in enumerate(self.evidence_ledger):
            if ev.status != "pending":
                continue
            if ev.claim in observable_claims:
                self.evidence_ledger[idx] = replace(ev, status="verified")
            elif ev.claim in false_resource_claims:
                self.evidence_ledger[idx] = replace(ev, status="contradicted")

    def update_beliefs(self, turn: int = 0) -> None:
        for prop in self.belief_graph.values():
            update_strength(prop, self.evidence_ledger, self.get_trust, turn)

    def get_trust(self, target_id: int) -> float:
        return compute_trust(self.evidence_ledger, target_id)
