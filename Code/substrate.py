"""
substrate.py
============

Shared physics primitives for the two numerical tests reported in Secs. 4.1
and 5 of the paper:

  Test A (test_a_metric_encoding.py): does some function of the
  mutual-information matrix I(i,j) satisfy the metric/kernel conditions
  required by the graph-Laplacian convergence theorems (Belkin-Niyogi 2008;
  Singer 2006) invoked in Sec. 4.1?

  Test B (test_b_lorentz_dispersion.py): does the substrate's low-energy
  excitation spectrum show linear (Lorentz-type) dispersion, the established
  signature of emergent conformal/Lorentz symmetry at a 1D quantum critical
  point (des Cloizeaux-Pearson 1962), as required by Sec. 5?

Both Hamiltonians below use periodic boundary conditions (a ring), so
translation invariance holds exactly and the standard CFT finite-size
scaling formulas apply cleanly:
  - critical isotropic Heisenberg chain (gapless, c=1 CFT)
  - dimerized Heisenberg chain (gapped), used throughout as the control /
    contrast case -- exponentially, not algebraically, decaying correlations.

Two independent implementations of exact diagonalization are used and
cross-validated against each other (see Code/README.md "Validation"):
  - a full 2**N Hilbert space construction via sparse Kronecker products
    (site_op / build_heisenberg_H_pbc / build_dimerized_H_pbc), needed to
    compute the full ground-state vector for the mutual-information test;
  - a fixed-Sz-sector, bit-manipulation basis construction
    (build_H_sector_pbc / build_H_sector_dimerized_pbc), which reaches much
    larger N (dispersion test only needs eigenvalues, not the full state).
"""
import numpy as np
from scipy.sparse import kron, eye, csr_matrix, coo_matrix
from scipy.sparse.linalg import eigsh

I2 = eye(2, format="csr")
_SX = [[0, 0.5], [0.5, 0]]
_SY = [[0, -0.5], [0.5, 0]]
_SZ = [[0.5, 0], [0, -0.5]]


# ============================================================
# Full 2**N Hilbert space construction (needed for the ground-state vector
# used by the mutual-information test; feasible up to N ~ 18-20)
# ============================================================

def site_op(op_matrix, site, N):
    result = eye(1, format="csr")
    for j in range(N):
        result = kron(result, csr_matrix(np.array(op_matrix, dtype=float)) if j == site else I2, format="csr")
    return result


def _bond_ops(i, j, N):
    return (
        site_op(_SX, i, N), site_op(_SX, j, N),
        site_op(_SY, i, N), site_op(_SY, j, N),
        site_op(_SZ, i, N), site_op(_SZ, j, N),
    )


def build_heisenberg_H_pbc(N, delta=1.0, J=1.0):
    """Isotropic (delta=1) Heisenberg chain, periodic boundary conditions."""
    H = csr_matrix((2**N, 2**N), dtype=float)
    for i in range(N):
        j = (i + 1) % N
        Sx, Sx1, Sy, Sy1, Sz, Sz1 = _bond_ops(i, j, N)
        H += J * (Sx @ Sx1 - Sy @ Sy1 + delta * (Sz @ Sz1))
    return H


def build_dimerized_H_pbc(N, delta):
    """Dimerized chain, PBC, alternating bond strength J_i = 1 + delta*(-1)^i."""
    H = csr_matrix((2**N, 2**N), dtype=float)
    for i in range(N):
        j = (i + 1) % N
        Ji = 1.0 + delta * ((-1) ** i)
        Sx, Sx1, Sy, Sy1, Sz, Sz1 = _bond_ops(i, j, N)
        H += Ji * (Sx @ Sx1 - Sy @ Sy1 + Sz @ Sz1)
    return H


def ground_state(H):
    evals, evecs = eigsh(H, k=1, which="SA")
    return evals[0], evecs[:, 0]


def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-15]
    return -np.sum(evals * np.log(evals))


def partial_trace(psi, keep, N):
    psi = psi.reshape([2] * N)
    trace_over = [i for i in range(N) if i not in keep]
    psi = np.transpose(psi, keep + trace_over)
    psi = psi.reshape(2 ** len(keep), 2 ** len(trace_over))
    return psi @ psi.conj().T


def MI_matrix(psi, N):
    """All-pairs mutual information matrix for a pure state psi."""
    I_mat = np.zeros((N, N))
    S = [von_neumann_entropy(partial_trace(psi, [i], N)) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            Sij = von_neumann_entropy(partial_trace(psi, [i, j], N))
            I_mat[i, j] = I_mat[j, i] = S[i] + S[j] - Sij
    return I_mat


# ============================================================
# Fixed-Sz-sector exact diagonalization (bit-manipulation basis), reaching
# much larger N for the dispersion/gap test where only eigenvalues are
# needed. Vectorized over the basis for speed.
# ============================================================

def enumerate_basis(N, Nup):
    """All N-bit integers with exactly Nup bits set, sorted ascending."""
    states = np.arange(2 ** N, dtype=np.int64)
    popcount = np.zeros_like(states)
    tmp = states.copy()
    for _ in range(N):
        popcount += tmp & 1
        tmp >>= 1
    return states[popcount == Nup]


def _build_H_sector_generic(N, Nup, bond_strengths):
    """Vectorized XXZ-type sector Hamiltonian builder, PBC. bond_strengths is
    a length-N list of (Jxy_i, Jz_i) for the bond (i, i+1 mod N)."""
    basis = enumerate_basis(N, Nup)
    dim = len(basis)

    diag = np.zeros(dim, dtype=np.float64)
    rows, cols, data = [], [], []

    for i in range(N):
        j = (i + 1) % N
        Jxy, Jz = bond_strengths[i]
        bi = (basis >> i) & 1
        bj = (basis >> j) & 1
        szi = bi - 0.5
        szj = bj - 0.5
        diag += Jz * szi * szj

        mask = bi != bj
        if np.any(mask):
            src = basis[mask]
            new_states = src ^ (np.int64(1) << i) ^ (np.int64(1) << j)
            new_idx = np.searchsorted(basis, new_states)
            src_idx = np.nonzero(mask)[0]
            rows.append(new_idx)
            cols.append(src_idx)
            data.append(np.full(len(src_idx), 0.5 * Jxy))

    rows.append(np.arange(dim))
    cols.append(np.arange(dim))
    data.append(diag)

    H = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                    shape=(dim, dim)).tocsr()
    return H


def build_H_sector_pbc(N, Nup, delta=1.0, J=1.0):
    """Heisenberg (XXZ with parameter delta) chain, PBC, restricted to the
    fixed-Sz sector with Nup up-spins."""
    bond_strengths = [(J, J * delta) for _ in range(N)]
    return _build_H_sector_generic(N, Nup, bond_strengths)


def build_H_sector_dimerized_pbc(N, Nup, delta):
    """Dimerized chain, PBC, restricted to fixed-Sz sector."""
    bond_strengths = [(1.0 + delta * ((-1) ** i), 1.0 + delta * ((-1) ** i)) for i in range(N)]
    return _build_H_sector_generic(N, Nup, bond_strengths)


def lowest_eigenvalue_sector(H_sector, k=1):
    if H_sector.shape[0] <= 200:
        evals = np.linalg.eigvalsh(H_sector.toarray())
        return np.sort(evals)[:k]
    evals = eigsh(H_sector, k=k, which="SA", return_eigenvectors=False)
    return np.sort(evals)


# ============================================================
# Classical (Torgerson) MDS -- no external dependency (sklearn not required)
# ============================================================

def classical_mds(D, dims=2):
    """Embed points into `dims`-D Euclidean space from a distance matrix D,
    via double-centering + eigendecomposition. Returns (coords, embedded
    pairwise distances, all eigenvalues of the centered Gram matrix)."""
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    B = (B + B.T) / 2
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    pos_evals = np.clip(evals[:dims], 0, None)
    X = evecs[:, :dims] * np.sqrt(pos_evals)
    diff = X[:, None, :] - X[None, :, :]
    D_embed = np.sqrt(np.sum(diff ** 2, axis=-1))
    return X, D_embed, evals


FIGURES_DIR_NAME = "figures"


def savefig(name, dpi=300):
    import os
    import matplotlib.pyplot as plt
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, FIGURES_DIR_NAME)
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, name), dpi=dpi, bbox_inches="tight")
    plt.close()
