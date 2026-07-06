# Reproducibility code for "The Gravitational Baseline Principle"

This folder contains everything needed to reproduce every numerical claim, table,
and figure in the paper. All scripts use the same combinatorial graph-Laplacian
construction (L = D - A, A = MI/max(MI)) throughout, for consistency.

## Requirements

```
pip install numpy scipy matplotlib quimb --break-system-packages
```

`quimb` (DMRG) is only needed for `dmrg_mi_laplacian.py` / the N=20-32 Heisenberg
data; everything else uses exact diagonalization and needs only numpy/scipy.

## Script-to-figure/table map

| Script | Produces | Paper location |
|---|---|---|
| `gravitational_baseline_numerics.py` | Figs. 1-6, 9; Table II | Secs. V A-E, V H |
| `dmrg_mi_laplacian.py` + `step1_dmrg.py` + `step2_mi_batch.py` + `assemble.py` | `scaling_results.json` (N=20,24,28,32 Heisenberg data used by Fig. 6 / Table II) | Sec. V E |
| `stability_test.py` | Table I | Sec. V (perturbation stability) |
| `cft_universality_test.py` (via `run_test.py`), then `plot_cft_test.py` | Fig. 7, Table III (`cft_test_results.json`) | Sec. V F |
| `dimerization_test.py` (via `run_dimerization.py`), then `plot_dimerization.py` | Fig. 8, Table IV (`dimerization_results.json`) | Sec. V G |

Note: `run_test.py` and `run_dimerization.py` only compute and save the raw
numerical data (JSON); the corresponding `plot_*.py` script re-fits that data
and produces the figure. Both plot scripts recompute all fits from the JSON
rather than hardcoding numbers, so re-running the full pipeline end-to-end
(data generation, then plotting) is a genuine reproducibility check, not just
a re-draw.

## Validation

Before trusting any N>16 Heisenberg result, `dmrg_mi_laplacian.py`'s `validate()`
function checks the DMRG pipeline against exact diagonalization at N=12 (expect
agreement to ~1e-6 after correcting quimb's log-base-2 entropy convention to
match the natural-log convention used everywhere else in this codebase — see
the `_LN2` conversion factor in that file). Do not trust larger-N DMRG output
if this validation does not pass.

## Reproducing specific numbers

- **Table II** (linear vs quadratic R²): run `gravitational_baseline_numerics.py`
  for N=12,16, then the DMRG pipeline above for N=20-32; the script auto-loads
  `scaling_results.json` if present in the same directory.
- **Table III** (cross-model exponents): `python run_test.py` then
  `python plot_cft_test.py` (the latter prints the table and saves Fig. 7).
- **Table IV** (dimerization crossover): `python run_dimerization.py` then
  `python plot_dimerization.py` (prints the table and saves Fig. 8). N=8-16,
  delta=0,0.1,0.3,0.6. Runtime is a few minutes on a single core.

## Known limitations of this code

- The 2D pipelines (`build_2d_nn_laplacian`, `build_gaussian_laplacian`) assume
  spatial coordinates as input — see the paper's explicit caveats in Secs. V A-C.
  This is not a bug; it's the reason those results are presented as pipeline
  validation rather than evidence of coordinate emergence.
- `fig6_graviton_dispersion()` and the "classical 1D lattice" reference panels
  elsewhere compute a plain nearest-neighbour path-graph Laplacian, not the
  MI-derived Laplacian — this is intentional (see the in-code docstring), but
  worth knowing if you're modifying the script rather than just running it.
