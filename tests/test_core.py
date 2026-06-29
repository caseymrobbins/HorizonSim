from horizon_sim.agents.agent import Agent
from horizon_sim.communication.message import Message
from horizon_sim.cognition.trust import compute_trust
from horizon_sim.main import build_default_simulation


def test_trust_defaults_to_neutral_and_tracks_resolved_claims():
    agent = Agent(0, (0, 0), {})
    assert compute_trust(agent.evidence_ledger, 1) == 0.5
    agent.add_evidence(1, "communication", "Cell(0, 0) contains food >= 1", 0.8, 0, "verified")
    agent.add_evidence(1, "communication", "Cell(0, 1) contains food >= 1", 0.8, 0, "contradicted")
    assert compute_trust(agent.evidence_ledger, 1) == 0.5


def test_tell_message_creates_pending_evidence_and_belief():
    agent = Agent(0, (0, 0), {})
    agent.ingest_messages([Message(1, 0, "TELL", {"claim": "Cell(1, 1) terrain=forest", "confidence": 0.7})], 3)
    assert agent.evidence_ledger[0].source == 1
    assert agent.evidence_ledger[0].status == "pending"
    assert "Cell(1, 1) terrain=forest" in agent.belief_graph


def test_simulation_steps_and_harvests_or_moves():
    sim = build_default_simulation(seed=11)
    sim.step()
    assert sim.turn == 1
    assert set(sim.world.agent_positions) == {0, 1}
    assert all(agent.spatial_map is not None for agent in sim.agents)
