from __future__ import annotations

import re
from dataclasses import dataclass

from horizon_sim.communication.message import Message

# Matches observation-format claims: "Cell(x, y) contains <resource> >= 1"
_RESOURCE_CLAIM_RE = re.compile(r"Cell\((\d+),\s*(\d+)\) contains \w+ >= 1")


def _parse_resource_claim_pos(claim: str) -> tuple[int, int] | None:
    """Extract (x, y) from a resource-presence claim, or None if not that form."""
    m = _RESOURCE_CLAIM_RE.match(claim)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _sign(n: int) -> int:
    return 1 if n > 0 else (-1 if n < 0 else 0)


@dataclass(frozen=True)
class Action:
    kind: str
    params: dict


class UtilityMaxPolicy:
    """Baseline policy: harvest preferred local resources or move toward believed resource
    cells. Communicates each turn; may emit lies when lie_propensity > 0."""

    def __init__(self, lie_propensity: float = 0.0) -> None:
        # lie_propensity=0.0 → always honest (default); raise to study deception.
        self.lie_propensity = lie_propensity

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
            # Claim unowned resource tiles with 30% probability instead of harvesting.
            # This lets ownership emerge without halting production entirely.
            if tile.owner_id is None and world.rng.random() < 0.3:
                return Action("CLAIM", {})
            return Action("HARVEST", {"resource": best_resource, "amount": 1})

        # No local resource: navigate toward the strongest positively-believed resource cell.
        # This gives testimony causal power — a false belief misdirects movement,
        # which is the precondition for lies to have payoff consequences.
        best_pos: tuple[int, int] | None = None
        best_strength = 0.3  # minimum belief strength to follow
        for prop in agent.belief_graph.values():
            if prop.strength <= best_strength:
                continue
            pos = _parse_resource_claim_pos(prop.claim)
            if pos is not None and pos != (x, y) and world.in_bounds(*pos):
                best_pos = pos
                best_strength = prop.strength

        if best_pos is not None:
            dx = _sign(best_pos[0] - x)
            dy = _sign(best_pos[1] - y)
            return Action("MOVE", {"dx": dx, "dy": dy})

        return Action("MOVE", {"dx": world.rng.choice([-1, 0, 1]), "dy": world.rng.choice([-1, 0, 1])})

    def generate_communication(self, agent, world, turn: int) -> Message | None:
        if not agent.address_book:
            return None

        contacts = list(agent.address_book)
        receiver_id = world.rng.choice(contacts)
        other_contacts = [c for c in contacts if c != receiver_id]

        if not other_contacts:
            # Self-introduction: bootstraps the ring into a bidirectional graph.
            return Message(
                sender=agent.id,
                receiver=receiver_id,
                msg_type="INTRODUCE",
                introduced_agent=agent.id,
                confidence=0.99,
                timestamp=turn,
            )

        # Lie path: payoff-triggered deception when lie_propensity > 0.
        # Fires before honest paths so propensity is a true marginal rate.
        if self.lie_propensity > 0 and world.rng.random() < self.lie_propensity:
            lie_msg = self._try_lie(agent, world, turn, receiver_id)
            if lie_msg is not None:
                return lie_msg

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

    def _try_lie(self, agent, world, turn: int, receiver_id: int) -> Message | None:
        """Attempt a payoff-aware lie: claim a known-depleted cell is resource-rich.

        Incentive proxy: agent has positive wealth (something to protect / gain).
        Scope boundary: no theory-of-mind targeting — flat propensity only.
        The _lie_believed key is stripped by the simulation before delivery;
        it exists solely for ground-truth audit logging — receivers never see it.
        """
        # Require at least some stake (exploration investment) to lie strategically
        if agent.inventory.get("wealth", 0) <= 0:
            return None

        # Find the most negatively-believed resource cell (we know it's depleted)
        # and claim it's rich, to misdirect a rival's movement.
        best_prop = None
        for prop in agent.belief_graph.values():
            if prop.strength > -0.05:
                continue  # Require at least weak confidence the cell is depleted
            pos = _parse_resource_claim_pos(prop.claim)
            if pos is not None:
                if best_prop is None or prop.strength < best_prop.strength:
                    best_prop = prop  # Pick the most confidently-believed-empty cell

        if best_prop is None:
            return None

        # Lie: send the depleted-cell claim with plausible positive confidence.
        # The content dict is mutable even on frozen Message; _lie_believed is stripped
        # before delivery by the simulation's Phase 4 handler.
        return Message(
            sender=agent.id,
            receiver=receiver_id,
            msg_type="TELL",
            content={"claim": best_prop.claim, "_lie_believed": best_prop.strength},
            confidence=min(0.85, max(0.3, abs(best_prop.strength))),
            timestamp=turn,
        )
