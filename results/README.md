# Experiment Results

Organised per Theory-to-Experiment Map experiment family.

## Families

- `pointer/` — Pointer chasing temporal execution (§9): proper-unseen (thesis main) and seen-list sanity check (appendix)
- `mnist/` — MNIST static classification with overlap decoding and margin analysis (§3)
- `dfa/` — DFA sequential state evaluation (§8, to be implemented)

## Conventions

- Each result directory contains `raw_results.csv`, `summary.csv`, `config_snapshot.yaml`, and a `plots/` directory
- Plot generation is automatic via `run_experiment_suite.py`
- Historical `config_snapshot.yaml` files inside result directories are kept unmodified
- MNIST sequence probes under `mnist/probes/` are exploratory only, not thesis benchmarks
- Pointer results separate the seen-list mechanism from proper-unseen generalization; do not merge their conclusions

## Reproduction

```bash
PYTHONPATH=pyac/src .venv/bin/python run_experiment_suite.py --config experiments/<config>.yaml
```

See `results/pointer/README.md` and `results/mnist/README.md` for per-experiment documentation.
