# Quantum Resource and Quantum Simulation

This repository contains numerical data and reproduction code for the project
"Taming Trotter errors with quantum resources".

The original exploratory scripts, notebooks, and generated figures are
preserved in `legacy/` for provenance. The cleaned reproduction workflow is in
`reproduce/` and writes new outputs to `reproduce/outputs/` by default.

## Repository Layout

- `legacy/`: original scripts, notebooks, and figures used during the project.
- `reproduce/`: cleaned scripts and reusable helpers for new runs.
- `DATA_PROVENANCE.md`: mapping from existing data directories to the original
  generation workflows.
- `requirements.txt`: minimal Python environment for the cleaned scripts.
- `vardata/`: existing variance and entanglement data.
- `data/`: existing magic and kurtosis data.
- `newdata/`: existing three-distribution data for the joint effect of magic
  and entanglement.
- `mag_time_data/`: existing resource-growth data.
- `LongtimeSim/`: existing long-time simulation data.
- `OtherHamiltonians/`: existing data for Hamiltonians beyond the main text.

## Quick Start

Install the minimal environment:

```bash
pip install -r requirements.txt
```

Run a small seeded smoke test:

```bash
python reproduce/scripts/run_variance.py --case typical --n 4 --depth 2 --samples 3 --seed 1234
python reproduce/scripts/run_variance.py --case atypical --n 4 --depth 2 --samples 3 --seed 5678
```

Run the full cleaned variance workflow:

```bash
python reproduce/scripts/run_variance.py --case typical --seed 1234
python reproduce/scripts/run_variance.py --case atypical --seed 5678
```

Or open the notebook example:

```text
reproduce/notebooks/run_variance_then_plot.ipynb
```

The notebook calls `run_variance.py`, loads the generated data, computes
bootstrap confidence intervals, and plots the entanglement/variance figure.

The cleaned scripts do not overwrite the existing data files. See
`reproduce/README.md` for details.
