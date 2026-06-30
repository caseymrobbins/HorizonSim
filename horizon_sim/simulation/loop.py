from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from horizon_sim.agents.agent import Agent
from horizon_sim.agents.policy import Action, UtilityMaxPolicy
from horizon_sim.communication.message import Message
from horizon_sim.world.grid import World


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

    def __post_init__(self) -> None:
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
        for agent in order:
            message = agent.policy.generate_communication(agent, self.world, self.turn)
            if message is not None:
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

        # Phase 8: Update Epistemic Model
        for agent in order:
            agent.update_beliefs(self.turn)

        # Phase 9-10: Plan & Execute One Physical Action
        for agent in order:
            action = agent.policy.choose_action(agent, self.world)
            self.execute(agent, action)

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

        self.metrics_history.append(self.snapshot_metrics(
            turn_messages=turn_messages,
            turn_introductions=turn_introductions,
            turn_ev_created=turn_ev_created,
            turn_ev_verified=turn_ev_verified,
            turn_ev_contradicted=turn_ev_contradicted,
            turn_belief_updates=turn_belief_updates,
            turn_credibility_updates=turn_credibility_updates,
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

    def execute(self, agent: Agent, action: Action) -> None:
        if action.kind == "MOVE":
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
            resource = str(action.params["resource"])
            taken = self.world.harvest(agent.id, resource, int(action.params.get("amount", 1)))
            agent.inventory[resource] = agent.inventory.get(resource, 0) + taken
            self.production_totals[resource] += taken
            self.event_ledger.append(
                SimulationEvent(
                    self.turn, agent.id, "PRODUCTION",
                    {"resource": resource, "amount": taken, "position": agent.position},
                )
            )

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
        }


def _evidence_snapshot(agent: Agent) -> dict:
    total = len(agent.evidence_ledger)
    verified = sum(1 for ev in agent.evidence_ledger if ev.status == "verified")
    contradicted = sum(1 for ev in agent.evidence_ledger if ev.status == "contradicted")
    return {"total": total, "verified": verified, "contradicted": contradicted}
