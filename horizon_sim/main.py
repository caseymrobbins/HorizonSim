from __future__ import annotations

from horizon_sim.agents.agent import Agent
from horizon_sim.agents.preferences import normalize_preferences
from horizon_sim.simulation.loop import Simulation
from horizon_sim.world.grid import World


def build_default_simulation(seed: int | None = 7) -> Simulation:
    terrain_distribution = {"plains": 0.5, "forest": 0.3, "mountain": 0.15, "water": 0.05}
    regrowth = {
        "food": {"forest": {"chance": 0.1, "max": 5}, "plains": {"chance": 0.05, "max": 2}},
        "wood": {"forest": {"chance": 0.15, "max": 4}},
    }
    world = World.random(50, 50, terrain_distribution, regrowth, seed=seed)
    agents = [
        Agent(0, (5, 5), normalize_preferences({"food": 0.9, "wealth": 0.4, "knowledge": 0.3})),
        Agent(1, (40, 40), normalize_preferences({"food": 0.3, "wealth": 0.8, "knowledge": 0.6})),
    ]
    return Simulation(world, agents)


def main(turns: int = 10) -> None:
    sim = build_default_simulation()
    for _ in range(turns):
        sim.step()
    for agent in sim.agents:
        print(f"agent={agent.id} pos={agent.position} inventory={agent.inventory}")


if __name__ == "__main__":
    main()
