from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from horizon_sim.agents.agent import Agent
from horizon_sim.agents.policy import Action, UtilityMaxPolicy
from horizon_sim.communication.message import Message
from horizon_sim.world.grid import World


@dataclass
class Simulation:
    world: World
    agents: list[Agent]
    observation_radius: int = 2
    pending_messages: list[Message] = field(default_factory=list)
    turn: int = 0

    def __post_init__(self) -> None:
        for agent in self.agents:
            self.world.agent_positions[agent.id] = agent.position
            agent.attach_map(self.world.width, self.world.height)
            if agent.policy is None:
                agent.policy = UtilityMaxPolicy()

    def step(self) -> None:
        self.world.regenerate_resources()
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
        self.turn += 1

    def execute(self, agent: Agent, action: Action) -> None:
        if action.kind == "MOVE":
            self.world.move_agent(agent.id, int(action.params.get("dx", 0)), int(action.params.get("dy", 0)))
            agent.position = self.world.agent_positions[agent.id]
        elif action.kind == "HARVEST":
            resource = str(action.params["resource"])
            taken = self.world.harvest(agent.id, resource, int(action.params.get("amount", 1)))
            agent.inventory[resource] = agent.inventory.get(resource, 0) + taken
        elif action.kind == "COMMUNICATE":
            self.pending_messages.append(action.params["message"])
