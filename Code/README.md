# Reproducibility code for "The Gravitational Baseline Principle"

Two numerical tests of the open problems identified in the paper's Sec. 4.1
and Sec. 5, both using exact diagonalization of the spin-1/2 antiferromagnetic
Heisenberg chain (critical) and its dimerized version (gapped, used as a
control throughout).

```
substrate.py                  shared physics: Hamiltonians (full Hilbert
                               space and fixed-Sz-sector bit-manipulation
                               constructions), mutual information, classical
                               MDS embedding
test_a_metric_encoding.py     Table 1, Fig. 1 (Sec. 4.1): does the mutual-
                               information matrix encode a Euclidean metric?
test_b_lorentz_dispersion.py  Table 2, Fig. 2 (Sec. 5): does the substrate
                               show emergent Lorentz-consistent dispersion?
requirements.txt
```

## Quick start

```
pip install -r requirements.txt
python test_a_metric_encoding.py     # ~1 minute
python test_b_lorentz_dispersion.py  # ~8 minutes (dominated by N=24, 26)
```

Both scripts print their results table to stdout and save the corresponding
figure to `../figures/`.

## What each test does and why

**`test_a_metric_encoding.py`** computes the full mutual-information matrix
$I(i,j)$ for the ground state of the critical and gapped chains at $N=16,18$
(exact diagonalization, full $2^N$ Hilbert space -- needed because the test
requires the complete ground-state vector to compute entanglement entropy
across arbitrary bipartitions). It then tests whether the candidate distance
$d(i,j) = -\log(I(i,j)/I_{\max})$ behaves like a genuine metric: real-space
scaling against the *chord distance* on the periodic ring (not naive hop
count -- see the script's docstring for why this matters), triangle-inequality
violation rate, and classical (Torgerson) MDS embedding quality into 2D.

**`test_b_lorentz_dispersion.py`** computes the finite-size energy gap
$E_{\rm gap}(N)$ between the ground state and the lowest excitation in the
$S^z=1$ sector, for $N=12$ through $26$, using a fixed-Sz-sector
bit-manipulation basis (much faster than full-Hilbert-space diagonalization,
and validated against it to machine precision at smaller $N$ -- see
`substrate.py`'s docstring). Linear scaling of the gap with $1/N$ is the
established finite-size signature of emergent Lorentz-invariant (linear)
dispersion at a 1D quantum critical point.

## Validation

The fixed-Sz-sector Hamiltonian construction (`build_H_sector_pbc`,
`build_H_sector_dimerized_pbc`) is a second, independent implementation from
the full-Hilbert-space construction (`build_heisenberg_H_pbc`,
`build_dimerized_H_pbc`). Cross-checking their ground-state energies against
each other at, e.g., $N=10$ agrees to $\sim10^{-15}$ (machine precision) --
run this yourself with:

```python
from substrate import build_heisenberg_H_pbc, ground_state, build_H_sector_pbc, lowest_eigenvalue_sector
E_full, _ = ground_state(build_heisenberg_H_pbc(10, delta=1.0))
E_sector = lowest_eigenvalue_sector(build_H_sector_pbc(10, Nup=5, delta=1.0))[0]
print(abs(E_full - E_sector))  # ~1e-15
```

## Known limitations of this code

- `test_a_metric_encoding.py` is capped at $N=16,18$ because it needs the
  full $2^N$-dimensional ground-state vector to compute mutual information
  across arbitrary bipartitions; this is not extendable to the larger-$N$
  sector-restricted method used in `test_b` without a DMRG-based mutual
  information pipeline (out of scope here).
- Both tests use a single model (the 1D Heisenberg chain / its dimerized
  deformation) as a concrete stand-in for the paper's abstract substrate.
  Sec. 9 of the paper lists extending these tests to other universality
  classes and to genuinely higher-dimensional substrates as open work.
- Neither script simulates the substrate's own proposed evolution equation
  (paper Eqs. 6-7); both test properties of a static many-body ground state
  used as a concrete example substrate.
