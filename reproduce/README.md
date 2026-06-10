# Clean Reproduction Workflow

This directory contains the cleaned reproduction workflow. It is separate from
the historical files in `legacy/` and writes new results to `reproduce/outputs/`
by default.

## Variance and Entanglement Data

Run a small smoke test first:

```bash
python reproduce/scripts/run_variance.py --case typical --n 4 --depth 2 --samples 3 --seed 1234
python reproduce/scripts/run_variance.py --case atypical --n 4 --depth 2 --samples 3 --seed 5678
```

Run the full settings used by the legacy variance scripts:

```bash
python reproduce/scripts/run_variance.py --case typical --seed 1234
python reproduce/scripts/run_variance.py --case atypical --seed 5678
```

The cleaned script generates all indices from `0` through `depth` by default.
For historical compatibility with the legacy scripts, use `--start-index 2`.

Random Clifford sampling is seeded through the `--seed` argument. Re-running
with the same software environment and seed should reproduce the same sampled
Clifford sequence.

## Notebook Example

The notebook `notebooks/run_variance_then_plot.ipynb` demonstrates the complete
workflow:

1. call `scripts/run_variance.py` for typical and atypical data;
2. load the generated `.npy` files;
3. compute bootstrap confidence intervals using `src/bootstrap.py`;
4. plot entanglement entropy together with the variance of the rescaled error.

The notebook writes outputs to `reproduce/outputs/run_variance_then_plot/`,
which is ignored by Git.
