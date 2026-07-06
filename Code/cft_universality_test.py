"""
Test: does the MI-Laplacian's finite-size scaling exponent b (lambda_1 ~ N^-b)
track known critical/CFT structure (varies by model and by Luttinger parameter K),
or is it a universal geometric constant independent of the microscopic model?

If b varies systematically across models/parameters in a way that tracks known
critical exponents, that supports the "CFT/quantum-critical diagnostic" reading
of Sec. V E in the paper. If b is roughly constant regardless of model, that
would instead support some kind of universal geometric interpretation.

Models implemented (same combinatorial Laplacian pipeline as the paper: exact
diagonalization -> MI matrix -> A = I/max(I) -> L = D-A -> eigvalsh):

  1. XXZ chain at various Delta in (-1, 1] (includes isotropic Heisenberg at Delta=1).
     H = sum_i [Sx Sx + Sy Sy + Delta Sz Sz]
     Standard Luttinger parameter: K = pi / (2*(pi - arccos(Delta)))  [Delta in (-1,1)]
     (Giamarchi, "Quantum Physics in One Dimension"; stated here as the standard
     reference value for internal cross-checking, not re-derived.)

  2. Transverse-field Ising model (TFIM) at its critical point h=1.
     H = -sum_i Sx_i Sx_{i+1} - h * sum_i Sz_i   (critical at h=1 in this normalization)
     Central charge c=1/2 (vs c=1 for the XXZ line) -- a genuinely different
     universality class, included as a contrast case.
"""
import numpy as np
import time
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh

I2 = eye(2, format="csr")

def site_op(op_matrix, site, N):
    result = eye(1, format="csr")
    for j in range(N):
        result = kron(result, csr_matrix(np.array(op_matrix, dtype=float)) if j == site else I2, format="csr")
    return result

def build_xxz_H(N, delta):
    H = csr_matrix((2**N, 2**N), dtype=float)
    for i in range(N - 1):
        Sx = site_op([[0, 0.5], [0.5, 0]], i, N); Sx1 = site_op([[0, 0.5], [0.5, 0]], i + 1, N)
        Sy = site_op([[0, -0.5], [0.5, 0]], i, N); Sy1 = site_op([[0, -0.5], [0.5, 0]], i + 1, N)
        Sz = site_op([[0.5, 0], [0, -0.5]], i, N); Sz1 = site_op([[0.5, 0], [0, -0.5]], i + 1, N)
        H += (Sx @ Sx1 - Sy @ Sy1 + delta * (Sz @ Sz1))
    return H

def build_tfim_H(N, h=1.0):
    # H = -sum Sx_i Sx_{i+1} - h sum Sz_i, critical at h=1 (standard normalization
    # with S=1/2 Pauli-like operators rescaled; overall energy scale is irrelevant
    # to the ground-state entanglement structure we extract).
    H = csr_matrix((2**N, 2**N), dtype=float)
    for i in range(N - 1):
        Sx = site_op([[0, 0.5], [0.5, 0]], i, N); Sx1 = site_op([[0, 0.5], [0.5, 0]], i + 1, N)
        H += -4.0 * (Sx @ Sx1)  # factor to match conventional sigma_x sigma_x = 4 Sx Sx1
    for i in range(N):
        Sz = site_op([[0.5, 0], [0, -0.5]], i, N)
        H += -2.0 * h * Sz
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
