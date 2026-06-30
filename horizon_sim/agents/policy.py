from __future__ import annotations

from dataclasses import dataclass

from horizon_sim.communication.message import Message


@dataclass(frozen=True)
class Action:
    kind: str
    params: dict


class UtilityMaxPolicy:
    """Baseline policy: harvest preferred local resources or wander. Communicates each turn."""

    def choose_action(self, agent, world) -> Action:
        x, y = world.agent_positions[agent.id]
        tile = world.grid[x][y]
        best_resource = None
        best_score = 0.0
        for resource, amount in tile.resources.items():
            score = agent.preferences.get(resource, 0.0) * amount
            if amount > 0 and score > best_score:
                best_resource = resource
                best_score = score
        if best_resource is not None:
            return Action("HARVEST", {"resource": best_resource, "amount": 1})
        return Action("MOVE", {"dx": world.rng.choice([-1, 0, 1]), "dy": world.rng.choice([-1, 0, 1])})

    def generate_communication(self, agent, world, turn: int) -> Message | None:
        if not agent.address_book:
            return None

        contacts = list(agent.address_book)
        receiver_id = world.rng.choice(contacts)
        other_contacts = [c for c in contacts if c != receiver_id]

        if not other_contacts:
            # Self-introduction: tell the receiver who we are so they can contact us back.
            # This bootstraps the ring into a bidirectional graph on the first turn.
            return Message(
                sender=agent.id,
                receiver=receiver_id,
                msg_type="INTRODUCE",
                introduced_agent=agent.id,
                confidence=0.99,
                timestamp=turn,
            )

        if world.rng.random() < 0.5:
            # INTRODUCE a third agent to broaden the receiver's address book
            introduced_id = world.rng.choice(other_contacts)
            return Message(
                sender=agent.id,
                receiver=receiver_id,
                msg_type="INTRODUCE",
                introduced_agent=introduced_id,
                confidence=0.99,
                timestamp=turn,
            )

        # TELL: share strongest belief about the world
        strong_beliefs = sorted(
            [(pid, prop) for pid, prop in agent.belief_graph.items() if abs(prop.strength) > 0.3],
            key=lambda x: abs(x[1].strength),
            reverse=True,
        )
        if strong_beliefs:
            _, prop = strong_beliefs[0]
            return Message(
                sender=agent.id,
                receiver=receiver_id,
                msg_type="TELL",
                content={"claim": prop.claim},
                proposition_id=prop.id,
                confidence=min(0.99, max(0.1, abs(prop.strength))),
                timestamp=turn,
            )

        # Fallback: ASK about resources at agent's current position
        x, y = world.agent_positions[agent.id]
        return Message(
            sender=agent.id,
            receiver=receiver_id,
            msg_type="ASK",
            content={"claim": f"Cell({x},{y}) contains food >= 1"},
            confidence=0.5,
            timestamp=turn,
        )
