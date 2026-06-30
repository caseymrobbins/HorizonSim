from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from horizon_sim.agents.agent import Agent
from horizon_sim.agents.preferences import normalize_preferences
from horizon_sim.simulation.loop import Simulation
from horizon_sim.world.grid import World


def build_default_simulation(
    seed: int | None = 7,
    num_agents: int = 2,
    world_width: int = 50,
    world_height: int = 50,
) -> Simulation:
    terrain_distribution = {"plains": 0.5, "forest": 0.3, "mountain": 0.15, "water": 0.05}
    regrowth = {
        "food": {"forest": {"chance": 0.1, "max": 5}, "plains": {"chance": 0.05, "max": 2}},
        "wood": {"forest": {"chance": 0.15, "max": 4}},
    }
    world = World.random(world_width, world_height, terrain_distribution, regrowth, seed=seed)
    profiles = [
        {"food": 0.9, "wealth": 0.4, "knowledge": 0.3, "wood": 0.2, "stone": 0.1},
        {"food": 0.3, "wealth": 0.8, "knowledge": 0.6, "wood": 0.5, "stone": 0.2},
        {"food": 0.4, "wealth": 0.5, "knowledge": 0.4, "wood": 0.8, "stone": 0.3},
        {"food": 0.5, "wealth": 0.6, "knowledge": 0.2, "wood": 0.2, "stone": 0.9},
    ]
    agents = []
    for agent_id in range(num_agents):
        position = _random_land_position(world)
        preferences = normalize_preferences(profiles[agent_id % len(profiles)])
        agents.append(Agent(agent_id, position, preferences))

    # Initialize address books as a directed ring: agent i → agent (i+1) % n
    # This creates a strongly connected graph where every agent has exactly one
    # outgoing contact and one incoming contact.
    for i, agent in enumerate(agents):
        next_agent = agents[(i + 1) % len(agents)]
        agent.address_book.add(next_agent.id)

    return Simulation(world, agents)


def _random_land_position(world: World) -> tuple[int, int]:
    for _ in range(max(100, world.width * world.height)):
        x = world.rng.randrange(world.width)
        y = world.rng.randrange(world.height)
        if world.grid[x][y].terrain != "water":
            return x, y
    return 0, 0


def select_accelerator(preferred: str) -> dict[str, str]:
    """Report the fastest available accelerator without making the CPU-bound sim depend on it."""
    if preferred in {"cpu", "none"}:
        return {"requested": preferred, "selected": "cpu", "note": "CPU execution selected."}
    try:
        import torch

        if preferred in {"auto", "a100", "cuda"} and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return {"requested": preferred, "selected": "cuda", "device": name, "note": "CUDA available; A100 will be used if this host exposes one."}
    except ImportError:
        pass
    try:
        import jax

        tpus = [device for device in jax.devices() if device.platform == "tpu"]
        if preferred in {"auto", "tpu"} and tpus:
            return {"requested": preferred, "selected": "tpu", "device": str(tpus[0]), "note": "TPU available through JAX."}
    except ImportError:
        pass
    return {
        "requested": preferred,
        "selected": "cpu",
        "note": "No supported CUDA/A100 or TPU runtime was detected. The current discrete-event simulation is CPU-bound unless vectorized backends are added.",
    }


def format_progress(step: int, total_steps: int, started_at: float, now: float | None = None) -> str:
    now = time.monotonic() if now is None else now
    elapsed = max(0.0, now - started_at)
    percent = (step / total_steps * 100.0) if total_steps else 100.0
    steps_per_second = (step / elapsed) if elapsed > 0 and step else 0.0
    remaining_steps = max(0, total_steps - step)
    eta_seconds = (remaining_steps / steps_per_second) if steps_per_second else 0.0
    return (
        f"progress step={step}/{total_steps} ({percent:.1f}%) "
        f"elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s rate={steps_per_second:.2f} steps/s"
    )


def run_simulation(args: argparse.Namespace) -> Simulation:
    sim = build_default_simulation(args.seed, args.agents, args.world_width, args.world_height)
    started_at = time.monotonic()
    progress_interval = max(0, int(args.progress_interval))
    for step in range(1, args.steps + 1):
        sim.step()
        should_report = progress_interval and (step == 1 or step == args.steps or step % progress_interval == 0)
        if should_report:
            print(format_progress(step, args.steps, started_at), flush=True)
    return sim


_INFO_ECONOMY_FIELDS = [
    "messages_sent", "messages_received", "introductions_sent",
    "evidence_created", "evidence_verified", "evidence_contradicted",
    "belief_updates", "credibility_updates",
    "mean_address_book_size", "average_network_degree",
    "communication_graph_diameter", "information_diffusion_rate",
    "total_messages_sent", "total_introductions_sent",
    "total_evidence_created", "total_evidence_verified",
    "total_evidence_contradicted", "total_belief_updates",
]

_OWNERSHIP_FIELDS = [
    "claims", "rent_paid", "theft_attempts", "theft_detected", "fines_levied", "debt_created",
    "total_claims", "total_rent_paid", "total_theft_attempts", "total_theft_detected",
    "total_fines_levied", "total_debt_created",
]


def save_outputs(sim: Simulation, output_dir: Path, accelerator: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "accelerator.json").write_text(json.dumps(accelerator, indent=2, sort_keys=True))
    (output_dir / "events.jsonl").write_text("\n".join(json.dumps(asdict(event), sort_keys=True) for event in sim.event_ledger) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(sim.metrics_history, indent=2, sort_keys=True))
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        base_fields = ["turn", "trade_count", "trade_volume", "total_wealth", "mean_wealth", "production", "resources_held"]
        fieldnames = base_fields + _INFO_ECONOMY_FIELDS + _OWNERSHIP_FIELDS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sim.metrics_history:
            writer.writerow({
                **row,
                "production": json.dumps(row["production"], sort_keys=True),
                "resources_held": json.dumps(row["resources_held"], sort_keys=True),
            })
    agents_dir = output_dir / "agents"
    agents_dir.mkdir(exist_ok=True)
    for agent in sim.agents:
        payload = {
            "id": agent.id,
            "position": agent.position,
            "preferences": agent.preferences,
            "inventory": agent.inventory,
            "address_book": sorted(agent.address_book),
            "debt": agent.debt,
            "introducers": {str(k): v for k, v in agent.introducers.items()},
            "belief_graph": {prop_id: asdict(prop) for prop_id, prop in agent.belief_graph.items()},
            "evidence_ledger": [asdict(ev) for ev in agent.evidence_ledger],
        }
        (agents_dir / f"agent_{agent.id}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HorizonSim and export metrics, events, and per-agent graphs.")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents to simulate.")
    parser.add_argument("--world-width", type=int, default=50, help="World width in cells.")
    parser.add_argument("--world-height", type=int, default=50, help="World height in cells.")
    parser.add_argument("--world-size", type=int, help="Shortcut that sets both world width and height.")
    parser.add_argument("--steps", type=int, default=10, help="Number of simulation steps to iterate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible runs.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latest"), help="Directory where run outputs are saved.")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "none", "cuda", "a100", "tpu"], default="auto", help="Preferred hardware target to detect/report.")
    parser.add_argument("--progress-interval", type=int, default=10, help="Print rough progress and ETA every N steps; set to 0 to disable.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.world_size is not None:
        args.world_width = args.world_size
        args.world_height = args.world_size
    accelerator = select_accelerator(args.accelerator)
    sim = run_simulation(args)
    save_outputs(sim, args.output_dir, accelerator)
    print(f"completed turns={sim.turn} agents={len(sim.agents)} world={sim.world.width}x{sim.world.height}")
    print(f"outputs={args.output_dir}")
    print(f"accelerator={accelerator['selected']} note={accelerator['note']}")


if __name__ == "__main__":
    main()
