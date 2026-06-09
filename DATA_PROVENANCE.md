# Data Provenance

This file records how the existing data files in the public repository relate
to the original exploratory scripts and notebooks. The original files are kept
for provenance in `legacy/`.

## Existing Data Directories

- `vardata/`: variance and entanglement data used by the variance figures.
  The original typical workflow is preserved in `legacy/scripts/VarianceTy.py`;
  the original atypical workflow is preserved in `legacy/scripts/VarianceAty.py`.
- `data/`: magic-dependent states and error distributions used for the
  distribution and kurtosis figures.
- `newdata/`: three error distributions for low-magic/low-entanglement,
  high-magic/low-entanglement, and high-magic/high-entanglement states.
- `mag_time_data/`: magic and entanglement growth data. The historical magic
  recomputation script is preserved as `legacy/scripts/mag_time_tystate.py`.
- `LongtimeSim/`: long-time simulation data used by
  `legacy/notebooks/LongTimeSimulation.ipynb`.
- `OtherHamiltonians/`: data used by
  `legacy/notebooks/OtherHamiltonians.ipynb`.

## Cleaned Reproduction Outputs

Newly generated results from the cleaned scripts are written to
`reproduce/outputs/` by default and do not overwrite the existing data files.

The cleaned variance script `reproduce/scripts/run_variance.py` fixes random
seeds for random Clifford sampling and generates `ent_0.npy`, `ent_1.npy`,
`antient_0.npy`, and `antient_1.npy` by default. The legacy scripts generated
the indexed error samples only for `j > 1`; use `--start-index 2` to reproduce
that historical indexing convention.
