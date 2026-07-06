"""
Test: does the MI-Laplacian's critical-structure signature degrade or vanish
away from criticality?

We dimerize the Heisenberg chain, H = sum_i J_i (Sx Sx + Sy Sy + Sz Sz)_{i,i+1},
with alternating bond strength J_i = 1 + delta*(-1)^i. At delta=0 this is the
critical isotropic Heisenberg chain (c=1 CFT). For delta != 0, the ground
state is gapped, with a finite correlation length xi ~ 1/delta (dimerization
opens a gap in this model). If the paper's "quantum-critical diagnostic"
interpretation is right, both the finite-size scaling of lambda_1(N) and the
real-space MI decay should qualitatively change once the system is gapped:
power-law behavior should give way to saturation/exponential decay once N
(or r) exceeds the correlation length.
"""
import numpy as np
import time
import json
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh
from scipy.optimize import curve_fit

I2 = eye(2, format="csr")

def site_op(op_matrix, site, N):
    result = eye(1, format="csr")
    for j in range(N):
        result = kron(result, csr_matrix(np.array(op_matrix, dtype=float)) if j == site else I2, format="csr")
    return result

def build_dimerized_H(N, delta):
    H = csr_matrix((2**N, 2**N), dtype=float)
    for i in range(N - 1):
        Ji = 1.0 + delta * ((-1) ** i)
        Sx = site_op([[0, 0.5], [0.5, 0]], i, N); Sx1 = site_op([[0, 0.5], [0.5, 0]], i + 1, N)
        Sy = site_op([[0, -0.5], [0.5, 0]], i, N); Sy1 = site_op([[0, -0.5], [0.5, 0]], i + 1, N)
        Sz = site_op([[0.5, 0], [0, -0.5]], i, N); Sz1 = site_op([[0.5, 0], [0, -0.5]], i + 1, N)
        H += Ji * (Sx @ Sx1 - Sy @ Sy1 + Sz @ Sz1)
    return H

def ground_state(H):
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

def MI_matrix(psi, N):
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

print("Module loaded.")
