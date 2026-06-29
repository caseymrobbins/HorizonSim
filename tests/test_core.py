from horizon_sim.agents.agent import Agent
from horizon_sim.communication.message import Message
from horizon_sim.main import build_default_simulation


def test_credibility_defaults_to_neutral_and_tracks_resolved_claims_by_proposition():
    agent = Agent(0, (0, 0), {})
    first = agent.add_evidence(1, "communication", "Cell(0, 0) contains food >= 1", 0.8, 0, "verified")
    second = agent.add_evidence(1, "communication", "Cell(0, 1) contains food >= 1", 0.8, 0, "contradicted")
    first_prop = agent.evidence_ledger[first].proposition_id
    second_prop = agent.evidence_ledger[second].proposition_id

    assert agent.get_credibility(2, first_prop) == 0.5
    assert agent.get_credibility(1, first_prop) == 1.0
    assert agent.get_credibility(1, second_prop) == 0.0


def test_credibility_generalizes_to_similar_propositions_not_globally():
    agent = Agent(0, (0, 0), {})
    agent.add_evidence(1, "communication", "Cell(0, 0) contains food >= 1", 0.8, 0, "verified")
    similar_prop = agent._proposition_for_claim("Cell(0, 1) contains food >= 1")
    unrelated_prop = agent._proposition_for_claim("Carol is dishonest")

    assert agent.get_credibility(1, similar_prop.id) == 1.0
    assert agent.get_credibility(1, unrelated_prop.id) == 0.5


def test_tell_message_creates_pending_evidence_and_belief():
    agent = Agent(0, (0, 0), {})
    agent.ingest_messages([Message(1, 0, "TELL", {"claim": "Cell(1, 1) terrain=forest", "confidence": 0.7})], 3)
    assert agent.evidence_ledger[0].source == 1
    assert agent.evidence_ledger[0].status == "pending"
    assert agent.evidence_ledger[0].proposition_id in agent.belief_graph


def test_simulation_steps_and_harvests_or_moves():
    sim = build_default_simulation(seed=11)
    sim.step()
    assert sim.turn == 1
    assert set(sim.world.agent_positions) == {0, 1}
    assert all(agent.spatial_map is not None for agent in sim.agents)
