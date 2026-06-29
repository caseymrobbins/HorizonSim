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
    pending_messages: list[Message] = field(default_factory=list)
    turn: int = 0
    event_ledger: list[SimulationEvent] = field(default_factory=list)
    metrics_history: list[dict] = field(default_factory=list)
    production_totals: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    trade_count: int = 0
    trade_volume: int = 0

    def __post_init__(self) -> None:
        for agent in self.agents:
            self.world.agent_positions[agent.id] = agent.position
            agent.attach_map(self.world.width, self.world.height)
            agent.inventory.setdefault("wealth", 10)
            if agent.policy is None:
                agent.policy = UtilityMaxPolicy()

    def step(self) -> None:
        self.world.regenerate_resources()
        self.event_ledger.append(SimulationEvent(self.turn, None, "WORLD_REGENERATE", {}))
        inbox = defaultdict(list)
        for message in self.pending_messages:
            inbox[message.receiver].append(message)
        self.pending_messages = []

        order = self.agents[:]
        self.world.rng.shuffle(order)
        for agent in order:
            agent.position = self.world.agent_positions[agent.id]
            observations = self.world.get_observations(agent.position, self.observation_radius)
            agent.ingest_observations(observations, self.turn)
            agent.ingest_messages(inbox[agent.id], self.turn)
            agent.resolve_evidence_against_observations(observations)
            agent.update_beliefs(self.turn)
            action = agent.policy.choose_action(agent, self.world)
            self.execute(agent, action)
        self.settle_local_trades()
        self.metrics_history.append(self.snapshot_metrics())
        self.turn += 1

    def execute(self, agent: Agent, action: Action) -> None:
        if action.kind == "MOVE":
            old_position = self.world.agent_positions[agent.id]
            moved = self.world.move_agent(agent.id, int(action.params.get("dx", 0)), int(action.params.get("dy", 0)))
            agent.position = self.world.agent_positions[agent.id]
            self.event_ledger.append(
                SimulationEvent(
                    self.turn,
                    agent.id,
                    "MOVE",
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
                    self.turn,
                    agent.id,
                    "PRODUCTION",
                    {"resource": resource, "amount": taken, "position": agent.position},
                )
            )
        elif action.kind == "COMMUNICATE":
            message = action.params["message"]
            self.pending_messages.append(message)
            self.event_ledger.append(
                SimulationEvent(
                    self.turn,
                    agent.id,
                    "COMMUNICATE",
                    {"receiver": message.receiver, "msg_type": message.msg_type, "content": message.content},
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
                            self.turn,
                            None,
                            "TRADE",
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

    def snapshot_metrics(self) -> dict:
        total_wealth = sum(agent.inventory.get("wealth", 0) for agent in self.agents)
        resource_totals: dict[str, int] = defaultdict(int)
        for agent in self.agents:
            for resource, amount in agent.inventory.items():
                if resource != "wealth":
                    resource_totals[resource] += amount
        return {
            "turn": self.turn,
            "trade_count": self.trade_count,
            "trade_volume": self.trade_volume,
            "production": dict(self.production_totals),
            "total_wealth": total_wealth,
            "mean_wealth": total_wealth / len(self.agents) if self.agents else 0.0,
            "resources_held": dict(resource_totals),
        }
