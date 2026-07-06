"""
Extend the Heisenberg-chain MI-Laplacian to larger N via DMRG (quimb),
validated against exact diagonalization, following the same discipline
as the universal-emergent-geometry repo.

Builds the COMBINATORIAL Laplacian L = D - A (not normalized), since
that's what the original Gravitational Baseline / emergent-geometry-heisenberg
papers used, and computes the full all-pairs mutual-information matrix
(not just entanglement entropy of a bipartition).
"""
import numpy as np
import time
import json

import quimb as qu
import quimb.tensor as qtn

# ============================================================
# Exact diagonalization reference (small N) - reused from before
# ============================================================
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh

I2 = eye(2, format="csr")

def site_op(op_matrix, site, N):
    result = eye(1, format="csr")
    for j in range(N):
        result = kron(result, csr_matrix(np.array(op_matrix, dtype=float)) if j == site else I2, format="csr")
    return result

def build_sparse_H(N, J=1.0):
    H = csr_matrix((2**N, 2**N), dtype=float)
    for i in range(N - 1):
        Sx = site_op([[0, 0.5], [0.5, 0]], i, N)
        Sx1 = site_op([[0, 0.5], [0.5, 0]], i + 1, N)
        Sy = site_op([[0, -0.5], [0.5, 0]], i, N)
        Sy1 = site_op([[0, -0.5], [0.5, 0]], i + 1, N)
        Sz = site_op([[0.5, 0], [0, -0.5]], i, N)
        Sz1 = site_op([[0.5, 0], [0, -0.5]], i + 1, N)
        H += J * (Sx @ Sx1 - Sy @ Sy1 + Sz @ Sz1)
    return H

def ground_state_ed(N):
    H = build_sparse_H(N)
    evals, evecs = eigsh(H, k=1, which="SA")
    return evecs[:, 0]

def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-15]
    return -np.sum(evals * np.log(evals))

def partial_trace(psi, keep, N):
    psi = psi.reshape([2] * N)
    trace_over = [i for i in range(N) if i not in keep]
    psi = np.transpose(psi, keep + trace_over)
    psi = psi.reshape(2**len(keep), 2**len(trace_over))
    return psi @ psi.conj().T

def MI_matrix_ed(psi, N):
    I_mat = np.zeros((N, N))
    S = [von_neumann_entropy(partial_trace(psi, [i], N)) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            Sij = von_neumann_entropy(partial_trace(psi, [i, j], N))
            I_mat[i, j] = I_mat[j, i] = S[i] + S[j] - Sij
    return I_mat

def laplacian_from_MI(I_mat):
    A = I_mat / np.max(I_mat)
    np.fill_diagonal(A, 0)
    D = np.diag(A.sum(axis=1))
    return D - A

# ============================================================
# DMRG (quimb) - Heisenberg chain, isotropic (delta=1)
# ============================================================

def build_heisenberg_mpo(N):
    H = qtn.SpinHam1D(S=0.5, cyclic=False)
    H += 1.0, 'X', 'X'
    H += 1.0, 'Y', 'Y'
    H += 1.0, 'Z', 'Z'
    return H.build_mpo(N)

def dmrg_ground_state(N, bond_dims=(20, 40, 80, 120)):
    H_mpo = build_heisenberg_mpo(N)
    dmrg = qtn.DMRG2(H_mpo, bond_dims=list(bond_dims), cutoffs=1e-10)
    dmrg.solve(tol=1e-10, verbosity=0)
    return dmrg.state

_LN2 = np.log(2.0)  # quimb's qu.entropy() uses log2 (bits); convert to nats
                     # to match the ED code's natural-log convention.

def MI_matrix_dmrg(psi, N):
    I_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            rho_ij = psi.partial_trace_to_mpo([i, j]).to_dense()
            rho_i = qu.partial_trace(rho_ij, [2, 2], keep=[0])
            rho_j = qu.partial_trace(rho_ij, [2, 2], keep=[1])
            mi = (qu.entropy(rho_i) + qu.entropy(rho_j) - qu.entropy(rho_ij)) * _LN2
            I_mat[i, j] = I_mat[j, i] = mi
    return I_mat

# ============================================================
# Validation: DMRG vs exact diagonalization at N=12
# ============================================================

def validate(N=12):
    print(f"[validate] Comparing DMRG vs exact diagonalization at N={N}...")
    t0 = time.time()
    psi_ed = ground_state_ed(N)
    I_ed = MI_matrix_ed(psi_ed, N)
    L_ed = laplacian_from_MI(I_ed)
    evals_ed = eigvalsh(L_ed)
    print(f"  ED done in {time.time()-t0:.1f}s")

    t0 = time.time()
    psi_dmrg = dmrg_ground_state(N)
    I_dmrg = MI_matrix_dmrg(psi_dmrg, N)
    L_dmrg = laplacian_from_MI(I_dmrg)
    evals_dmrg = eigvalsh(L_dmrg)
    print(f"  DMRG done in {time.time()-t0:.1f}s")

    max_I_diff = np.max(np.abs(I_ed - I_dmrg))
    max_eval_diff = np.max(np.abs(evals_ed[:6] - evals_dmrg[:6]))
    print(f"  Max |I_ed - I_dmrg| = {max_I_diff:.2e}")
    print(f"  Max |eval_ed - eval_dmrg| (first 6) = {max_eval_diff:.2e}")
    print(f"  ED   low evals: {np.round(evals_ed[1:6],6)}")
    print(f"  DMRG low evals: {np.round(evals_dmrg[1:6],6)}")
    ok = max_I_diff < 1e-5 and max_eval_diff < 1e-5
    print(f"  VALIDATION: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("DMRG pipeline does not match exact diagonalization -- do not trust larger-N results.")
    return ok

if __name__ == "__main__":
    validate(N=12)
