from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from horizon_sim.agents.agent import Agent
from horizon_sim.agents.policy import Action, UtilityMaxPolicy
from horizon_sim.communication.message import Message
from horizon_sim.world.grid import World


@dataclass
class OwnershipConfig:
    """Exposed tuning knobs for the ownership / rent / enforcement layer."""
    fine_flat: int = 20       # Minimum fine regardless of stolen value
    fine_mult: int = 10       # Fine = max(fine_flat, fine_mult * stolen_value)
    rent_default: int = 2     # Per-unit lease price posted by owners (v1: uniform)
    claim_enabled: bool = True
    detection: str = "present"  # "present" = owner within detection_radius (Euclidean)
    fine_payee: str = "owner"   # "owner" | "void"
    unpaid_fine: str = "debt"   # "debt" | "seize_inventory"
    detection_radius: int = 2   # Euclidean tiles; owner within this distance detects theft


@dataclass(frozen=True)
class SimulationEvent:
    turn: int
    agent_id: int | None
    event_type: str
    details: dict


@dataclass
class Simulation:
    world: World
    agents: list[Agent]
    observation_radius: int = 2
    ownership: OwnershipConfig = field(default_factory=OwnershipConfig)
    turn: int = 0
    event_ledger: list[SimulationEvent] = field(default_factory=list)
    metrics_history: list[dict] = field(default_factory=list)
    production_totals: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    trade_count: int = 0
    trade_volume: int = 0
    # Cumulative information economy totals
    total_messages_sent: int = 0
    total_introductions_sent: int = 0
    total_evidence_created: int = 0
    total_evidence_verified: int = 0
    total_evidence_contradicted: int = 0
    total_belief_updates: int = 0
    # Cumulative ownership / enforcement totals
    total_claims: int = 0
    total_rent_paid: int = 0
    total_theft_attempts: int = 0
    total_theft_detected: int = 0
    total_fines_levied: int = 0
    total_debt_created: int = 0
    # Cumulative deception totals (Phase 3)
    total_lies_emitted: int = 0
    total_lies_detected: int = 0

    def __post_init__(self) -> None:
        self._agent_by_id: dict[int, Agent] = {a.id: a for a in self.agents}
        # (receiver_id, sender_id, claim_text) awaiting contradiction verification
        self._pending_lie_checks: set[tuple[int, int, str]] = set()
        for agent in self.agents:
            self.world.agent_positions[agent.id] = agent.position
            agent.attach_map(self.world.width, self.world.height)
            agent.inventory.setdefault("wealth", 10)
            if agent.policy is None:
                agent.policy = UtilityMaxPolicy()

    def step(self) -> None:
        # Phase 0: World regeneration
        self.world.regenerate_resources()
        self.event_ledger.append(SimulationEvent(self.turn, None, "WORLD_REGENERATE", {}))

        order = self.agents[:]
        self.world.rng.shuffle(order)

        # Snapshot evidence counts before the turn for delta computation
        pre_ev = {a.id: _evidence_snapshot(a) for a in self.agents}

        # Phase 1-2: Observe & Generate Observation Evidence
        observations: dict[int, list[dict]] = {}
        for agent in order:
            agent.position = self.world.agent_positions[agent.id]
            obs = self.world.get_observations(agent.position, self.observation_radius)
            observations[agent.id] = obs
            agent.ingest_observations(obs, self.turn)

        # Phase 3: Update Beliefs (pre-communication, based on observations)
        for agent in order:
            agent.update_beliefs(self.turn)

        # Phase 4: Generate One Communication per agent (mandatory)
        comm_inbox: dict[int, list[Message]] = defaultdict(list)
        turn_messages = 0
        turn_introductions = 0
        turn_lies = 0
        for agent in order:
            message = agent.policy.generate_communication(agent, self.world, self.turn)
            if message is not None:
                # Intercept lie markers before delivery: log ground-truth LIE event,
                # queue for detection check, then strip the marker so receivers see
                # only a normal TELL (trust graph must catch it, not the wire format).
                if "_lie_believed" in message.content:
                    believed_strength = message.content["_lie_believed"]
                    claim_text = message.content["claim"]
                    message.content.pop("_lie_believed")
                    self.event_ledger.append(SimulationEvent(
                        self.turn, agent.id, "LIE",
                        {"receiver": message.receiver, "claimed": claim_text,
                         "believed": believed_strength},
                    ))
                    self._pending_lie_checks.add((message.receiver, agent.id, claim_text))
                    turn_lies += 1
                    self.total_lies_emitted += 1
                comm_inbox[message.receiver].append(message)
                turn_messages += 1
                if message.msg_type == "INTRODUCE":
                    turn_introductions += 1
                self._emit_communication_event(agent.id, message)

        # Phase 5: Communications delivered (collected in comm_inbox above)

        # Phase 6: Convert Messages into Evidence
        for agent in order:
            agent.ingest_messages(comm_inbox[agent.id], self.turn)

        # Phase 7: Resolve Verified / Contradicted Evidence
        for agent in order:
            agent.resolve_evidence_against_observations(observations[agent.id])

        # Phase 7.5: Check whether any pending lies have been caught.
        # A lie is detected when the receiver has contradicted the claimed proposition
        # and the sender's credibility for it has dropped below 0.5.
        turn_lies_detected = self._check_lie_detections()

        # Phase 8: Update Epistemic Model
        for agent in order:
            agent.update_beliefs(self.turn)

        # Phase 9-10: Plan & Execute One Physical Action
        ownership_events: list[dict] = []
        for agent in order:
            action = agent.policy.choose_action(agent, self.world)
            self.execute(agent, action, ownership_events)

        # Phase 10.5: Resolve ownership consequences (rent / theft / enforcement)
        turn_claims, turn_rent, turn_theft, turn_detected, turn_fines, turn_debt = (
            self._resolve_ownership_consequences(ownership_events)
        )

        # Phase 11: Update World
        self.settle_local_trades()

        # Compute per-turn information economy deltas
        post_ev = {a.id: _evidence_snapshot(a) for a in self.agents}
        turn_ev_created = sum(post_ev[a.id]["total"] - pre_ev[a.id]["total"] for a in self.agents)
        turn_ev_verified = sum(max(0, post_ev[a.id]["verified"] - pre_ev[a.id]["verified"]) for a in self.agents)
        turn_ev_contradicted = sum(max(0, post_ev[a.id]["contradicted"] - pre_ev[a.id]["contradicted"]) for a in self.agents)
        turn_belief_updates = sum(
            sum(1 for prop in a.belief_graph.values() if prop.last_updated == self.turn)
            for a in self.agents
        )
        turn_credibility_updates = sum(
            sum(len(scores) for scores in a.epistemic_state.source_proposition_credibility.values())
            for a in self.agents
        )

        self.total_messages_sent += turn_messages
        self.total_introductions_sent += turn_introductions
        self.total_evidence_created += turn_ev_created
        self.total_evidence_verified += turn_ev_verified
        self.total_evidence_contradicted += turn_ev_contradicted
        self.total_belief_updates += turn_belief_updates
        self.total_claims += turn_claims
        self.total_rent_paid += turn_rent
        self.total_theft_attempts += turn_theft
        self.total_theft_detected += turn_detected
        self.total_fines_levied += turn_fines
        self.total_debt_created += turn_debt
        self.total_lies_detected += turn_lies_detected

        self.metrics_history.append(self.snapshot_metrics(
            turn_messages=turn_messages,
            turn_introductions=turn_introductions,
            turn_ev_created=turn_ev_created,
            turn_ev_verified=turn_ev_verified,
            turn_ev_contradicted=turn_ev_contradicted,
            turn_belief_updates=turn_belief_updates,
            turn_credibility_updates=turn_credibility_updates,
            turn_claims=turn_claims,
            turn_rent=turn_rent,
            turn_theft=turn_theft,
            turn_detected=turn_detected,
            turn_fines=turn_fines,
            turn_debt=turn_debt,
            turn_lies=turn_lies,
            turn_lies_detected=turn_lies_detected,
        ))
        self.turn += 1

    def _emit_communication_event(self, agent_id: int, message: Message) -> None:
        details: dict = {
            "receiver": message.receiver,
            "msg_type": message.msg_type,
            "confidence": message.confidence,
        }
        if message.msg_type == "INTRODUCE" and message.introduced_agent is not None:
            details["introduced_agent"] = message.introduced_agent
        elif "claim" in message.content:
            details["proposition"] = message.content["claim"]
        self.event_ledger.append(SimulationEvent(self.turn, agent_id, "COMMUNICATE", details))

    def execute(self, agent: Agent, action: Action, ownership_events: list | None = None) -> None:
        if action.kind == "CLAIM":
            x, y = self.world.agent_positions[agent.id]
            tile = self.world.grid[x][y]
            if tile.owner_id is None and self.ownership.claim_enabled:
                tile.owner_id = agent.id
                self.event_ledger.append(
                    SimulationEvent(self.turn, agent.id, "CLAIM", {"position": (x, y)})
                )
        elif action.kind == "MOVE":
            old_position = self.world.agent_positions[agent.id]
            moved = self.world.move_agent(agent.id, int(action.params.get("dx", 0)), int(action.params.get("dy", 0)))
            agent.position = self.world.agent_positions[agent.id]
            self.event_ledger.append(
                SimulationEvent(
                    self.turn, agent.id, "MOVE",
                    {"from": old_position, "to": agent.position, "success": moved},
                )
            )
        elif action.kind == "HARVEST":
            x, y = self.world.agent_positions[agent.id]
            tile = self.world.grid[x][y]
            tile_owner = tile.owner_id
            resource = str(action.params["resource"])
            taken = self.world.harvest(agent.id, resource, int(action.params.get("amount", 1)))
            agent.inventory[resource] = agent.inventory.get(resource, 0) + taken
            self.production_totals[resource] += taken
            self.event_ledger.append(
                SimulationEvent(
                    self.turn, agent.id, "PRODUCTION",
                    {"resource": resource, "amount": taken, "position": (x, y)},
                )
            )
            # Queue for ownership resolution if someone else owns this tile
            if tile_owner is not None and tile_owner != agent.id and taken > 0 and ownership_events is not None:
                ownership_events.append({
                    "harvester_id": agent.id,
                    "position": (x, y),
                    "resource": resource,
                    "amount": taken,
                    "owner_id": tile_owner,
                })

    def _resolve_ownership_consequences(self, events: list[dict]) -> tuple[int, int, int, int, int, int]:
        """Process rent and theft for harvests on owned tiles.

        Returns (claims, rent_paid, theft_attempts, theft_detected, fines, debt).
        `claims` is always 0 here — CLAIM events are counted in execute().
        """
        rent_paid = 0
        theft_attempts = 0
        theft_detected = 0
        fines_levied = 0
        debt_created = 0

        # Count CLAIM events emitted this turn
        claims = sum(
            1 for e in self.event_ledger
            if e.turn == self.turn and e.event_type == "CLAIM"
        )

        for ev in events:
            harvester = self._agent_by_id.get(ev["harvester_id"])
            owner = self._agent_by_id.get(ev["owner_id"])
            if harvester is None or owner is None:
                continue

            position = tuple(ev["position"])
            resource = ev["resource"]
            amount = ev["amount"]
            rent_due = self.ownership.rent_default * amount

            # Debt-holders cannot take the rent path; debt is repaid first (Phase 2).
            # This gives fines teeth: insolvent thieves accrue debt that blocks future rent.
            debtor = harvester.debt > 0
            can_pay_rent = not debtor and harvester.inventory.get("wealth", 0) >= rent_due

            if can_pay_rent:
                # ── RENT PATH ──────────────────────────────────────────────
                harvester.inventory["wealth"] -= rent_due
                if self.ownership.fine_payee == "owner":
                    owner.inventory["wealth"] = owner.inventory.get("wealth", 0) + rent_due
                rent_paid += rent_due
                self.event_ledger.append(SimulationEvent(
                    self.turn, harvester.id, "RENT_PAID",
                    {"position": position, "resource": resource, "amount": amount,
                     "rent": rent_due, "owner": owner.id},
                ))
            else:
                # ── THEFT PATH (includes debt-blocked harvesters) ──────────
                if debtor:
                    # Repay outstanding debt from available wealth before the fine
                    repaid = min(harvester.inventory.get("wealth", 0), harvester.debt)
                    if repaid > 0:
                        harvester.inventory["wealth"] = harvester.inventory.get("wealth", 0) - repaid
                        harvester.debt -= repaid
                theft_attempts += 1
                owner_pos = self.world.agent_positions.get(owner.id)
                # Detection: owner within detection_radius (Euclidean) of the stolen tile
                if owner_pos is None:
                    detected = False
                else:
                    _dx = owner_pos[0] - position[0]
                    _dy = owner_pos[1] - position[1]
                    detected = math.sqrt(_dx * _dx + _dy * _dy) <= self.ownership.detection_radius

                self.event_ledger.append(SimulationEvent(
                    self.turn, harvester.id, "THEFT",
                    {"position": position, "resource": resource, "amount": amount,
                     "owner": owner.id, "detected": detected},
                ))

                if not detected:
                    continue  # Anonymous — log only, owner does not know who

                # ── DETECTED ───────────────────────────────────────────────
                theft_detected += 1

                # Seize stolen goods back to owner
                seized = min(harvester.inventory.get(resource, 0), amount)
                harvester.inventory[resource] = harvester.inventory.get(resource, 0) - seized
                owner.inventory[resource] = owner.inventory.get(resource, 0) + seized

                # Fine = max(fine_flat, fine_mult × stolen value)
                stolen_value = amount  # 1 wealth per resource unit
                fine = max(self.ownership.fine_flat, self.ownership.fine_mult * stolen_value)
                fines_levied += fine

                available = harvester.inventory.get("wealth", 0)
                paid = min(available, fine)
                harvester.inventory["wealth"] = available - paid
                shortfall = fine - paid

                if self.ownership.fine_payee == "owner":
                    owner.inventory["wealth"] = owner.inventory.get("wealth", 0) + paid

                if shortfall > 0:
                    if self.ownership.unpaid_fine == "debt":
                        harvester.debt += shortfall
                        debt_created += shortfall
                    elif self.ownership.unpaid_fine == "seize_inventory":
                        remaining = shortfall
                        for r in list(harvester.inventory):
                            if r == "wealth" or remaining <= 0:
                                continue
                            qty = harvester.inventory.get(r, 0)
                            take = min(qty, remaining)
                            harvester.inventory[r] -= take
                            if self.ownership.fine_payee == "owner":
                                owner.inventory[r] = owner.inventory.get(r, 0) + take
                            remaining -= take

                # ── TRUST PENALTY (Bug #2 fix: direct + introducer) ───────
                # Inject a contradicted evidence record attributed to the thief
                # so the victim's credibility model lowers the thief's trust.
                owner.add_evidence(
                    source=harvester.id,
                    evidence_type="enforcement",
                    claim=f"Agent_{harvester.id} honest",
                    confidence=1.0,
                    turn=self.turn,
                    status="contradicted",
                )
                owner.update_beliefs(self.turn)

                # Introducer blame: lower trust in whoever vouched for the thief
                introducer_id = owner.introducers.get(harvester.id)
                if introducer_id is not None:
                    owner.add_evidence(
                        source=introducer_id,
                        evidence_type="enforcement",
                        claim=f"Agent_{introducer_id} vouches honestly",
                        confidence=0.9,
                        turn=self.turn,
                        status="contradicted",
                    )
                    owner.update_beliefs(self.turn)

                self.event_ledger.append(SimulationEvent(
                    self.turn, harvester.id, "ENFORCEMENT",
                    {"position": position, "owner": owner.id, "fine": fine,
                     "fine_paid": paid, "debt": shortfall},
                ))

        return claims, rent_paid, theft_attempts, theft_detected, fines_levied, debt_created

    def settle_local_trades(self) -> None:
        by_position: dict[tuple[int, int], list[Agent]] = defaultdict(list)
        for agent in self.agents:
            by_position[self.world.agent_positions[agent.id]].append(agent)
        for position, agents in by_position.items():
            if len(agents) < 2:
                continue
            buyers = sorted(agents, key=lambda a: a.preferences.get("wealth", 0.0), reverse=True)
            sellers = sorted(agents, key=lambda a: sum(v for k, v in a.inventory.items() if k != "wealth"), reverse=True)
            for buyer in buyers:
                for seller in sellers:
                    if buyer.id == seller.id or buyer.inventory.get("wealth", 0) <= 0:
                        continue
                    resource = self._tradable_resource(seller)
                    if resource is None:
                        continue
                    seller.inventory[resource] -= 1
                    seller.inventory["wealth"] = seller.inventory.get("wealth", 0) + 1
                    buyer.inventory["wealth"] -= 1
                    buyer.inventory[resource] = buyer.inventory.get(resource, 0) + 1
                    self.trade_count += 1
                    self.trade_volume += 1
                    self.event_ledger.append(
                        SimulationEvent(
                            self.turn, None, "TRADE",
                            {"buyer": buyer.id, "seller": seller.id, "resource": resource, "amount": 1, "price": 1, "position": position},
                        )
                    )
                    break

    @staticmethod
    def _tradable_resource(agent: Agent) -> str | None:
        resources = [(name, amount) for name, amount in agent.inventory.items() if name != "wealth" and amount > 1]
        if not resources:
            return None
        return max(resources, key=lambda item: item[1])[0]

    def _check_lie_detections(self) -> int:
        """Check pending lies for detection: credibility < 0.5 means the receiver
        has seen contradicting evidence and no longer trusts the sender's claim."""
        newly_detected = 0
        done: set[tuple[int, int, str]] = set()
        for (receiver_id, sender_id, claim) in self._pending_lie_checks:
            receiver = self._agent_by_id.get(receiver_id)
            if receiver is None:
                done.add((receiver_id, sender_id, claim))
                continue
            prop_id = receiver._claim_to_proposition_id.get(claim)
            if prop_id is None:
                continue  # Evidence not yet ingested; check again next turn
            cred = receiver.get_credibility(sender_id, prop_id)
            if cred < 0.5:
                newly_detected += 1
                done.add((receiver_id, sender_id, claim))
        self._pending_lie_checks -= done
        return newly_detected

    def _compute_concentration_metrics(self) -> dict:
        result: dict = {}
        # Ownership HHI per resource: fraction of nodes controlled by each agent, squared and summed.
        # Unowned nodes each count as their own unique "owner" (dispersed baseline = 1/n).
        # Monopoly by one agent → 1.0; fully dispersed (no claims) → 1/n_nodes.
        if self.world.node_positions:
            for resource, positions in self.world.node_positions.items():
                if not positions:
                    continue
                owner_counts: dict = defaultdict(int)
                for i, (x, y) in enumerate(positions):
                    oid = self.world.grid[x][y].owner_id
                    key = oid if oid is not None else f"_unc_{i}"
                    owner_counts[key] += 1
                n = len(positions)
                result[f"ownership_hhi_{resource}"] = round(
                    sum((cnt / n) ** 2 for cnt in owner_counts.values()), 4
                )
        # Wealth Gini coefficient (0=perfect equality, 1=one agent holds all)
        n = len(self.agents)
        gini = 0.0
        if n > 1:
            wealth = sorted(a.inventory.get("wealth", 0) for a in self.agents)
            S = sum(wealth)
            if S > 0:
                cumulative = sum((i + 1) * w for i, w in enumerate(wealth))
                gini = round((2 * cumulative) / (n * S) - (n + 1) / n, 4)
        result["wealth_gini"] = gini
        return result

    def _compute_network_metrics(self) -> dict:
        n = len(self.agents)
        if n < 2:
            return {
                "mean_address_book_size": 0.0,
                "average_network_degree": 0.0,
                "communication_graph_diameter": 0,
                "information_diffusion_rate": 0.0,
            }

        adj = {a.id: set(a.address_book) for a in self.agents}
        agent_ids = [a.id for a in self.agents]

        total_out = sum(len(adj[aid]) for aid in agent_ids)
        avg_degree = total_out / n

        # BFS from each node to compute diameter and reachability
        max_dist = 0
        total_reachable = 0
        for start_id in agent_ids:
            dist: dict[int, int] = {start_id: 0}
            queue = [start_id]
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                for nbr in adj.get(node, set()):
                    if nbr not in dist:
                        dist[nbr] = dist[node] + 1
                        if dist[nbr] > max_dist:
                            max_dist = dist[nbr]
                        queue.append(nbr)
            total_reachable += len(dist) - 1

        diffusion_rate = total_reachable / (n * (n - 1))

        return {
            "mean_address_book_size": avg_degree,
            "average_network_degree": avg_degree,
            "communication_graph_diameter": max_dist,
            "information_diffusion_rate": diffusion_rate,
        }

    def snapshot_metrics(self, **kw: int) -> dict:
        total_wealth = sum(a.inventory.get("wealth", 0) for a in self.agents)
        resource_totals: dict[str, int] = defaultdict(int)
        for agent in self.agents:
            for resource, amount in agent.inventory.items():
                if resource != "wealth":
                    resource_totals[resource] += amount

        network = self._compute_network_metrics()
        concentration = self._compute_concentration_metrics()

        return {
            "turn": self.turn,
            # Physical economy
            "trade_count": self.trade_count,
            "trade_volume": self.trade_volume,
            "production": dict(self.production_totals),
            "total_wealth": total_wealth,
            "mean_wealth": total_wealth / len(self.agents) if self.agents else 0.0,
            "resources_held": dict(resource_totals),
            # Information economy — per-turn
            "messages_sent": kw.get("turn_messages", 0),
            "messages_received": kw.get("turn_messages", 0),
            "introductions_sent": kw.get("turn_introductions", 0),
            "evidence_created": kw.get("turn_ev_created", 0),
            "evidence_verified": kw.get("turn_ev_verified", 0),
            "evidence_contradicted": kw.get("turn_ev_contradicted", 0),
            "belief_updates": kw.get("turn_belief_updates", 0),
            "credibility_updates": kw.get("turn_credibility_updates", 0),
            # Information economy — network
            "mean_address_book_size": network["mean_address_book_size"],
            "average_network_degree": network["average_network_degree"],
            "communication_graph_diameter": network["communication_graph_diameter"],
            "information_diffusion_rate": network["information_diffusion_rate"],
            # Information economy — cumulative
            "total_messages_sent": self.total_messages_sent,
            "total_introductions_sent": self.total_introductions_sent,
            "total_evidence_created": self.total_evidence_created,
            "total_evidence_verified": self.total_evidence_verified,
            "total_evidence_contradicted": self.total_evidence_contradicted,
            "total_belief_updates": self.total_belief_updates,
            # Ownership economy — per-turn
            "claims": kw.get("turn_claims", 0),
            "rent_paid": kw.get("turn_rent", 0),
            "theft_attempts": kw.get("turn_theft", 0),
            "theft_detected": kw.get("turn_detected", 0),
            "fines_levied": kw.get("turn_fines", 0),
            "debt_created": kw.get("turn_debt", 0),
            # Ownership economy — cumulative
            "total_claims": self.total_claims,
            "total_rent_paid": self.total_rent_paid,
            "total_theft_attempts": self.total_theft_attempts,
            "total_theft_detected": self.total_theft_detected,
            "total_fines_levied": self.total_fines_levied,
            "total_debt_created": self.total_debt_created,
            # Deception layer (Phase 3) — per-turn and cumulative
            "lies_emitted": kw.get("turn_lies", 0),
            "lies_detected": kw.get("turn_lies_detected", 0),
            "total_lies_emitted": self.total_lies_emitted,
            "total_lies_detected": self.total_lies_detected,
            # Concentration / inequality (Phase 1)
            **concentration,
        }


def _evidence_snapshot(agent: Agent) -> dict:
    total = len(agent.evidence_ledger)
    verified = sum(1 for ev in agent.evidence_ledger if ev.status == "verified")
    contradicted = sum(1 for ev in agent.evidence_ledger if ev.status == "contradicted")
    return {"total": total, "verified": verified, "contradicted": contradicted}
