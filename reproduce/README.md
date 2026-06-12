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

Random Clifford sampling is seeded through the `--seed` argument. The global
Clifford samples in the magic/kurtosis workflow use the Bravyi-Maslov tableau
sampler through Qiskit's compiled tableau backend, with 64-bit seeds drawn from
a NumPy random generator so the sampled sequence is reproducible from the seed.

The helper `src/variance_experiment.py::sample_trotter_errors` can also be
reused for the three-distribution analysis from the original workflow. Given
representative states such as low-magic/low-entanglement,
high-magic/low-entanglement, and high-magic/high-entanglement states, it samples
local Clifford frames and computes one Trotter-error sample per frame. Local
Clifford sampling is an efficient proxy for local Haar sampling because the
single-qubit Clifford group is a unitary 3-design, so products of single-qubit
Cliffords reproduce local Haar averages for the corresponding low-order
moments. Higher-order distributional statistics should be interpreted as
coming from this structured local-Haar proxy.

## Notebook Example

The notebook `notebooks/run_variance_then_plot.ipynb` demonstrates the complete
workflow:

1. call `scripts/run_variance.py` for typical and atypical data;
2. load the generated `.npy` files;
3. compute bootstrap confidence intervals using `src/bootstrap.py`;
4. plot entanglement entropy together with the variance of the rescaled error.

The notebook output in `reproduce/outputs/run_variance_then_plot/` is included
as a small example data set. Other generated output directories remain ignored
by Git.

## Magic and Kurtosis Data

Run a small smoke test:

```bash
python reproduce/scripts/run_magic_kurtosis.py --n 3 --samples 5 --bootstrap-samples 10 --seed 1234 --state-seed 4321 --delta-t 0.1 --steps 1
```

Run the full cleaned settings:

```bash
python reproduce/scripts/run_magic_kurtosis.py --seed 1234 --state-seed 4321 --delta-t 0.1 --steps 1
```

Run a longer accumulated-time version by increasing `--steps`. For example,
the following command uses step size `delta_t=0.1`, `100` Trotter steps, and
therefore total evolution time `T=10`:

```bash
python reproduce/scripts/run_magic_kurtosis.py --seed 1234 --state-seed 4321 --delta-t 0.1 --steps 100
```

Append additional random-Clifford error samples to an existing output directory
by using a new sampling seed:

```bash
python reproduce/scripts/run_magic_kurtosis.py --append --samples 5000 --seed 5678 --state-seed 4321 --delta-t 0.1 --steps 1
```

When `--append` is set, existing states and magic values are reused, new errors
are concatenated to the existing error files, and the kurtosis/statistics/plots
are recomputed from the combined errors. The append compatibility check ignores
`samples`, `seed`, and bootstrap settings, but requires the physical/model
parameters, state seed, and Clifford sampler to match.

This workflow generates `n+1` states by applying a random global Clifford, then
`i` T gates for `i=0,...,n`, and then another random global Clifford. It then
computes each state's magic, samples global-random-Clifford Trotter errors for
the accumulated evolution time `T = delta_t * steps`,
computes the kurtosis of the rescaled squared errors, and saves a
kurtosis-versus-magic plot together with an error-distribution plot.

The default output directory is
`reproduce/outputs/magic_kurtosis_n10_state_seed4321_dt0.1_order1_steps1_bravyi-maslov-qiskit-tableau-u64-seed/`.

## Magic and Entanglement Growth

Run a small smoke test:

```bash
python reproduce/scripts/run_resource_growth.py --case both --n 4 --steps 3 --time-max 0.3
```

Run the full cleaned settings:

```bash
python reproduce/scripts/run_resource_growth.py --case both
```

This workflow evolves the all-zero initial state under the typical and/or
atypical Hamiltonian, computes the subsystem entanglement entropy and magic at
each time point, and saves both per-case and combined plots.
