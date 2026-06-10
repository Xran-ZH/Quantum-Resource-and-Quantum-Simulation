# Quantum Resource and Quantum Simulation

This repository contains numerical data and reproduction code for the project
"Taming Trotter errors with quantum resources".

The original exploratory scripts, notebooks, and generated figures are
preserved in `legacy/` for provenance. The cleaned reproduction workflow is in
`reproduce/` and writes new outputs to `reproduce/outputs/` by default.

## Repository Layout

- `legacy/`: original scripts, notebooks, figures, and data used during the
  project.
- `legacy/data_original/`: archived original data directories from the
  exploratory workflow.
- `reproduce/`: cleaned scripts and reusable helpers for new runs.
- `DATA_PROVENANCE.md`: mapping from existing data directories to the original
  generation workflows.
- `requirements.txt`: minimal Python environment for the cleaned scripts.
- `reproduce/outputs/`: cleaned workflow outputs, including the committed
  notebook example data set.

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

Run the cleaned magic/kurtosis workflow:

```bash
python reproduce/scripts/run_magic_kurtosis.py --seed 1234 --state-seed 4321 --delta-t 0.1 --steps 1
```

For longer accumulated-time Trotter errors, increase `--steps`. For example,
`--delta-t 0.1 --steps 100` computes errors at total time `T=10`:

```bash
python reproduce/scripts/run_magic_kurtosis.py --seed 1234 --state-seed 4321 --delta-t 0.1 --steps 100
```

Run the cleaned resource-growth workflow:

```bash
python reproduce/scripts/run_resource_growth.py --case both
```

Or open the notebook example:

```text
reproduce/notebooks/run_variance_then_plot.ipynb
```

The notebook calls `run_variance.py`, loads the generated data, computes
bootstrap confidence intervals, and plots the entanglement/variance figure.

The helper `sample_trotter_errors` in `reproduce/src/variance_experiment.py`
can also be used for the three-distribution analysis in the original
repository: choose the three representative states, sample local Clifford
frames, compute the Trotter error for each sampled state, and then plot the
resulting error distributions. Local Clifford sampling is used as an efficient
proxy for local Haar sampling because the single-qubit Clifford group forms a
unitary 3-design; therefore products of single-qubit Cliffords reproduce local
Haar averages for the corresponding low-order moments. For higher-order
statistics, it should be regarded as a structured and reproducible local-Haar
proxy.

The cleaned scripts do not overwrite the existing data files. See
`reproduce/README.md` for details.
