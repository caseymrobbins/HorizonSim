from __future__ import annotations

from dataclasses import dataclass, field, replace

from horizon_sim.cognition.belief_graph import Proposition, make_proposition, update_strength
from horizon_sim.cognition.epistemic_state import EpistemicState
from horizon_sim.cognition.evidence_ledger import Evidence
from horizon_sim.cognition.spatial_map import SpatialMap
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
        self.epistemic_state = EpistemicState()
        self._claim_to_proposition_id: dict[str, str] = {}

    def attach_map(self, width: int, height: int) -> None:
        self.spatial_map = SpatialMap(width, height)

    def _proposition_for_claim(self, claim: str) -> Proposition:
        prop_id = self._claim_to_proposition_id.get(claim)
        if prop_id is None:
            prop_id = f"P{len(self._claim_to_proposition_id) + 1:03d}"
            self._claim_to_proposition_id[claim] = prop_id
            self.belief_graph[prop_id] = make_proposition(prop_id, claim)
        return self.belief_graph[prop_id]

    def add_evidence(self, source: int, evidence_type: str, claim: str, confidence: float, turn: int, status: str = "pending") -> int:
        ev_id = len(self.evidence_ledger)
        prop = self._proposition_for_claim(claim)
        self.evidence_ledger.append(Evidence(ev_id, source, evidence_type, prop.id, claim, max(0.0, min(1.0, confidence)), status, turn))
        prop.supporting_evidence.append(ev_id)
        if status != "pending":
            self.epistemic_state.update_from_evidence(self.evidence_ledger, self.belief_graph)
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
                claim = message.content["claim"]
                prop = self._proposition_for_claim(claim)
                confidence = float(message.content.get("confidence", self.get_credibility(message.sender, prop.id)))
                self.add_evidence(message.sender, "communication", claim, confidence, turn)

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
        self.epistemic_state.update_from_evidence(self.evidence_ledger, self.belief_graph)

    def update_beliefs(self, turn: int = 0) -> None:
        for prop in self.belief_graph.values():
            update_strength(prop, self.evidence_ledger, self.get_credibility, turn)

    def get_credibility(self, source_id: int, proposition_id: str) -> float:
        return self.epistemic_state.credibility(source_id, proposition_id, self.belief_graph)
