"""
Success-criteria checks for the ownership / rent / theft / enforcement layer.

Each test is self-contained: it constructs the minimal state required,
calls the relevant method directly, and asserts the outcome.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from horizon_sim.agents.agent import Agent
from horizon_sim.agents.policy import Action
from horizon_sim.communication.message import Message
from horizon_sim.simulation.loop import OwnershipConfig, Simulation
from horizon_sim.world.grid import World
from horizon_sim.world.tile import Tile


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_world(seed: int = 1) -> World:
    """5×5 flat world, no water, predictable terrain."""
    grid = [[Tile(terrain="plains", resources={"food": 3}) for _ in range(5)] for _ in range(5)]
    world = World(5, 5, grid, resource_regrowth={})
    world.rng.seed(seed)
    return world


def _make_sim(*agents: Agent, ownership: OwnershipConfig | None = None) -> Simulation:
    world = _make_world()
    kw = {"ownership": ownership} if ownership else {}
    sim = Simulation(world, list(agents), **kw)
    return sim


def _place(sim: Simulation, agent: Agent, x: int, y: int) -> None:
    sim.world.agent_positions[agent.id] = (x, y)
    agent.position = (x, y)


# ── 1. Ownership claim ─────────────────────────────────────────────────────────

def test_claim_sets_owner():
    """Agent on an unowned tile takes CLAIM action → tile.owner_id = agent.id."""
    a0 = Agent(0, (2, 2), {})
    sim = _make_sim(a0)
    _place(sim, a0, 2, 2)
    sim.execute(a0, Action("CLAIM", {}))
    assert sim.world.grid[2][2].owner_id == a0.id


def test_claim_disabled_leaves_tile_unowned():
    """CLAIM_ENABLED=False: CLAIM action has no effect."""
    a0 = Agent(0, (2, 2), {})
    sim = _make_sim(a0, ownership=OwnershipConfig(claim_enabled=False))
    _place(sim, a0, 2, 2)
    sim.execute(a0, Action("CLAIM", {}))
    assert sim.world.grid[2][2].owner_id is None


def test_claim_does_not_overwrite_existing_owner():
    """A tile already owned by agent 0 cannot be claimed by agent 1."""
    a0 = Agent(0, (2, 2), {})
    a1 = Agent(1, (2, 2), {})
    sim = _make_sim(a0, a1)
    sim.world.grid[2][2].owner_id = a0.id
    _place(sim, a1, 2, 2)
    sim.execute(a1, Action("CLAIM", {}))
    assert sim.world.grid[2][2].owner_id == a0.id  # Unchanged


# ── 2. Free extraction for owner ──────────────────────────────────────────────

def test_owner_extracts_free():
    """Owner harvesting from their own tile pays no rent."""
    a0 = Agent(0, (2, 2), {})
    sim = _make_sim(a0)
    sim.world.grid[2][2].owner_id = a0.id
    _place(sim, a0, 2, 2)
    before = a0.inventory.get("wealth", 0)
    ownership_events: list[dict] = []
    sim.execute(a0, Action("HARVEST", {"resource": "food", "amount": 1}), ownership_events)
    # No ownership_events for the owner harvesting their own tile
    assert len(ownership_events) == 0
    sim._resolve_ownership_consequences(ownership_events)
    assert a0.inventory.get("wealth", 0) == before  # Wealth unchanged


# ── 3. Rent paid by non-owner with sufficient wealth ─────────────────────────

def test_rent_paid_by_non_owner():
    """Non-owner with enough wealth pays RENT_DEFAULT per unit to the owner."""
    a0 = Agent(0, (0, 0), {})  # Owner (absent)
    a1 = Agent(1, (2, 2), {})  # Harvester
    sim = _make_sim(a0, a1, ownership=OwnershipConfig(rent_default=3))
    sim.world.grid[2][2].owner_id = a0.id
    _place(sim, a0, 0, 0)  # Owner is not co-located
    _place(sim, a1, 2, 2)
    a1.inventory["wealth"] = 10

    ownership_events: list[dict] = []
    sim.execute(a1, Action("HARVEST", {"resource": "food", "amount": 1}), ownership_events)
    sim._resolve_ownership_consequences(ownership_events)

    assert a1.inventory.get("wealth", 0) == 7   # Paid 3 rent
    assert a0.inventory.get("wealth", 0) == 13  # Received 3 rent


# ── 4. Theft detected when owner co-located ───────────────────────────────────

def test_theft_detected_owner_present():
    """Non-owner with no wealth steals from owned tile; owner is present → ENFORCEMENT."""
    cfg = OwnershipConfig(fine_flat=10, fine_mult=5, rent_default=3, unpaid_fine="debt")
    a0 = Agent(0, (2, 2), {})   # Owner, co-located
    a1 = Agent(1, (2, 2), {})   # Thief
    sim = _make_sim(a0, a1, ownership=cfg)
    sim.world.grid[2][2].owner_id = a0.id
    _place(sim, a0, 2, 2)
    _place(sim, a1, 2, 2)
    a1.inventory["wealth"] = 0   # Cannot pay rent → theft

    ownership_events: list[dict] = []
    sim.execute(a1, Action("HARVEST", {"resource": "food", "amount": 1}), ownership_events)
    # Thief briefly has the food; enforcement seizes it back
    sim._resolve_ownership_consequences(ownership_events)

    # Stolen goods seized
    assert a1.inventory.get("food", 0) == 0
    # Fine exceeds wealth; debt created
    # fine = max(10, 5*1) = 10; thief has 0 wealth → debt = 10
    assert a1.debt == 10
    # ENFORCEMENT event logged
    enforcement = [e for e in sim.event_ledger if e.event_type == "ENFORCEMENT"]
    assert len(enforcement) == 1
    assert enforcement[0].details["fine"] == 10


# ── 5. Theft anonymous when owner absent ──────────────────────────────────────

def test_theft_anonymous_when_owner_absent():
    """Non-owner steals; owner is not present → THEFT logged but no ENFORCEMENT."""
    a0 = Agent(0, (0, 0), {})  # Owner, elsewhere
    a1 = Agent(1, (2, 2), {})  # Thief
    sim = _make_sim(a0, a1)
    sim.world.grid[2][2].owner_id = a0.id
    _place(sim, a0, 0, 0)      # Not co-located
    _place(sim, a1, 2, 2)
    a1.inventory["wealth"] = 0

    ownership_events: list[dict] = []
    sim.execute(a1, Action("HARVEST", {"resource": "food", "amount": 1}), ownership_events)
    sim._resolve_ownership_consequences(ownership_events)

    # THEFT event present but NOT attributed to the thief in enforcement
    theft_events = [e for e in sim.event_ledger if e.event_type == "THEFT"]
    assert len(theft_events) == 1
    assert theft_events[0].details["detected"] is False
    # No ENFORCEMENT event
    assert not any(e.event_type == "ENFORCEMENT" for e in sim.event_ledger)
    # Thief keeps the food (no detection, no seizure)
    assert a1.inventory.get("food", 0) == 1


# ── 6. Contradiction path lowers source trust (Bug #1 verify) ─────────────────

def test_contradicted_evidence_routes_to_contradicting_list():
    """Evidence added with status='contradicted' lands in prop.contradicting_evidence."""
    agent = Agent(0, (0, 0), {})
    ev_id = agent.add_evidence(1, "communication", "Cell(0,0) contains food >= 1", 0.9, 0, "contradicted")
    prop_id = agent.evidence_ledger[ev_id].proposition_id
    prop = agent.belief_graph[prop_id]
    assert ev_id in prop.contradicting_evidence
    assert ev_id not in prop.supporting_evidence


def test_contradicted_testimony_lowers_source_credibility():
    """When a communicated claim is later contradicted, the source's trust drops."""
    agent = Agent(0, (0, 0), {})
    # Source 1 claims food is present
    ev_id = agent.add_evidence(1, "communication", "Cell(0,0) contains food >= 1", 0.9, 0)
    prop_id = agent.evidence_ledger[ev_id].proposition_id
    # Before resolution: pending → default credibility 0.5
    assert agent.get_credibility(1, prop_id) == 0.5

    # Observation contradicts the claim: mark as contradicted and move to correct list
    ev = agent.evidence_ledger[ev_id]
    agent.evidence_ledger[ev_id] = replace(ev, status="contradicted")
    prop = agent.belief_graph[prop_id]
    if ev_id in prop.supporting_evidence:
        prop.supporting_evidence.remove(ev_id)
        prop.contradicting_evidence.append(ev_id)
    agent.epistemic_state.update_from_evidence(agent.evidence_ledger, agent.belief_graph)

    # Credibility should drop to 0 (contradicted, no verified evidence)
    assert agent.get_credibility(1, prop_id) == 0.0


# ── 7. Introducer blame on caught thief (Bug #2 verify) ───────────────────────

def test_introducer_blamed_when_thief_caught():
    """Introducer's credibility drops in victim's view when the introduced agent steals."""
    cfg = OwnershipConfig(fine_flat=5, fine_mult=5, rent_default=3)
    a0 = Agent(0, (2, 2), {})   # Victim / owner, co-located
    a1 = Agent(1, (2, 2), {})   # Thief
    a2 = Agent(2, (0, 0), {})   # Introducer (introduced a1 to a0)
    sim = _make_sim(a0, a1, a2, ownership=cfg)
    sim.world.grid[2][2].owner_id = a0.id
    _place(sim, a0, 2, 2)
    _place(sim, a1, 2, 2)
    _place(sim, a2, 0, 0)

    a1.inventory["wealth"] = 0  # Thief is broke → theft

    # Record that a2 introduced a1 to a0
    a0.introducers[a1.id] = a2.id

    # Check a0 has no opinion on a2 yet
    dummy_prop = a0._proposition_for_claim(f"Agent_{a2.id} vouches honestly")
    assert a0.get_credibility(a2.id, dummy_prop.id) == 0.5  # Default

    ownership_events: list[dict] = []
    sim.execute(a1, Action("HARVEST", {"resource": "food", "amount": 1}), ownership_events)
    sim._resolve_ownership_consequences(ownership_events)

    # After enforcement: a0's credibility for a2 on the vouching claim drops
    vouching_prop_id = a0._claim_to_proposition_id.get(f"Agent_{a2.id} vouches honestly")
    assert vouching_prop_id is not None
    cred = a0.get_credibility(a2.id, vouching_prop_id)
    assert cred < 0.5, f"Expected introducer credibility < 0.5, got {cred}"


# ── 8. Debt when fine exceeds wealth ──────────────────────────────────────────

def test_debt_created_when_fine_exceeds_wealth():
    """Thief can only partially pay fine; remainder becomes debt."""
    # rent_default=10 so wealth=5 cannot cover rent → theft path
    cfg = OwnershipConfig(fine_flat=20, fine_mult=10, rent_default=10, unpaid_fine="debt")
    a0 = Agent(0, (2, 2), {})
    a1 = Agent(1, (2, 2), {})
    sim = _make_sim(a0, a1, ownership=cfg)
    sim.world.grid[2][2].owner_id = a0.id
    _place(sim, a0, 2, 2)
    _place(sim, a1, 2, 2)
    a1.inventory["wealth"] = 5  # 5 < rent=10 → theft; too little to cover fine=20

    ownership_events: list[dict] = []
    sim.execute(a1, Action("HARVEST", {"resource": "food", "amount": 1}), ownership_events)
    sim._resolve_ownership_consequences(ownership_events)

    # fine = max(20, 10*1) = 20; paid = min(5, 20) = 5; debt = 15
    assert a1.inventory.get("wealth", 0) == 0
    assert a1.debt == 15


# ── 9. Seize-inventory mode ────────────────────────────────────────────────────

def test_seize_inventory_when_unpaid_fine_mode():
    """When unpaid_fine='seize_inventory', non-wealth goods are taken to cover shortfall."""
    cfg = OwnershipConfig(fine_flat=20, fine_mult=10, rent_default=3, unpaid_fine="seize_inventory")
    a0 = Agent(0, (2, 2), {})
    a1 = Agent(1, (2, 2), {})
    sim = _make_sim(a0, a1, ownership=cfg)
    sim.world.grid[2][2].owner_id = a0.id
    _place(sim, a0, 2, 2)
    _place(sim, a1, 2, 2)
    a1.inventory["wealth"] = 0   # Zero cash
    a1.inventory["wood"] = 10    # Has inventory that can be seized

    ownership_events: list[dict] = []
    sim.execute(a1, Action("HARVEST", {"resource": "food", "amount": 1}), ownership_events)
    sim._resolve_ownership_consequences(ownership_events)

    # Debt should be 0; wood was seized to cover
    assert a1.debt == 0
    assert a1.inventory.get("wood", 0) < 10  # Some wood taken
