# HorizonSim v1

HorizonSim is a minimal, emergence-first social simulation in Python. It models a 2D world, independently optimizing agents, evidence-based belief updates, proposition-dependent epistemic credibility, simple communication primitives, production, local trade, and run exports.

## Run

```bash
python -m horizon_sim.main
```

Useful run controls:

```bash
python -m horizon_sim.main \
  --agents 100 \
  --world-size 200 \
  --steps 1000 \
  --output-dir runs/experiment-001 \
  --accelerator auto
```

You can also set rectangular worlds with `--world-width` and `--world-height`. The `--accelerator` flag accepts `auto`, `cpu`, `cuda`, `a100`, and `tpu`. HorizonSim currently runs as a CPU-bound discrete-event simulation, but the command records whether CUDA/A100 or TPU runtimes are visible so accelerator-backed implementations can be selected consistently later.

## Outputs

Each run writes these files under `--output-dir`:

- `metrics.json` and `metrics.csv`: per-step trade count, trade volume, production totals, resource holdings, total wealth, and mean wealth.
- `events.jsonl`: chronological event ledger for world regeneration, movement, production, communication, and trade events.
- `agents/agent_<id>.json`: each agent's final position, inventory, preferences, evidence ledger, and belief graph so graph visualizers can inspect agent cognition.
- `accelerator.json`: requested and selected hardware target details.

## Test

```bash
python -m pytest
```
