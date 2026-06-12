# Data Provenance

This file records how the existing data files in the public repository relate
to the original exploratory scripts and notebooks. The original files are kept
for provenance in `legacy/`.

## Archived Original Data Directories

- `legacy/data_original/vardata/`: variance and entanglement data used by the
  variance figures. The original typical workflow is preserved in
  `legacy/scripts/VarianceTy.py`; the original atypical workflow is preserved in
  `legacy/scripts/VarianceAty.py`.
- `legacy/data_original/data/`: magic-dependent states and error distributions
  used for the distribution and kurtosis figures. The archived repository
  contains these selected states and error distributions, but not the complete
  original state-selection script.
- `legacy/data_original/newdata/`: three error distributions for
  low-magic/low-entanglement, high-magic/low-entanglement, and
  high-magic/high-entanglement states.
- `legacy/data_original/mag_time_data/`: magic and entanglement growth data.
  The historical magic recomputation script is preserved as
  `legacy/scripts/mag_time_tystate.py`.
- `legacy/data_original/LongtimeSim/`: long-time simulation data used by
  `legacy/notebooks/LongTimeSimulation.ipynb`.
- `legacy/data_original/OtherHamiltonians/`: data used by
  `legacy/notebooks/OtherHamiltonians.ipynb`.

## Cleaned Reproduction Outputs

Newly generated results from the cleaned scripts are written to
`reproduce/outputs/` by default. Generated output directories are ignored by
Git unless explicitly added.

The cleaned variance script `reproduce/scripts/run_variance.py` fixes random
seeds for random Clifford sampling and generates `ent_0.npy`, `ent_1.npy`,
`antient_0.npy`, and `antient_1.npy` by default. In the legacy scripts, the
indexed error-sample loop starts from `j > 1` because the `j=0` and `j=1` files
had already been generated during earlier test runs, and the production loop
was then continued from `j=2`. The resulting archived data are complete; the
loop bound in the legacy cleanup version is a provenance/cleanup artifact rather
than a separate data-generation protocol.

The cleaned magic/kurtosis script `reproduce/scripts/run_magic_kurtosis.py`
uses an explicit seeded state family: a random global Clifford, followed by
`i` T gates for `i=0,...,n`, followed by another random global Clifford. This
provides a reproducible state-generation protocol for the magic/kurtosis
analysis. Global Clifford samples are generated with Qiskit's compiled
Bravyi-Maslov tableau backend using 64-bit seeds drawn from the explicit
command-line random seed.
