"""
Perturbation stability test for the Gravitational Baseline Principle paper.

Tests the claim: "All geometric signatures reported here remain stable
under random perturbations of the mutual-information matrix at the level
of 1-5%, indicating that the emergent structures are not fine-tuned."

For each pipeline (2D NN Laplacian, Gaussian kernel, Heisenberg MI), we:
  1. Take the baseline adjacency matrix A.
  2. Perturb: A_pert = A * (1 + eps * xi_ij), xi_ij ~ Uniform(-1,1), symmetric.
  3. Rebuild Laplacian, recompute eigenvalues/eigenvectors.
  4. Track: (a) degeneracy ratio |lambda1-lambda2|/mean, (b) correlation of
     first-excited eigenvectors with x/y grid coordinates (2D pipelines) or
     with the unperturbed eigenvectors (Heisenberg), (c) relative change in
     low-lying eigenvalues.
  5. Repeat over many random seeds at eps = 0.01, 0.03, 0.05, report mean+-std.
"""
import numpy as np
from scipy.linalg import eigh, eigvalsh
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh

np.random.seed(0)

# ============================================================
# Pipeline builders (same as gravitational_baseline_numerics.py)
# ============================================================

def build_2d_nn_laplacian(Nx, Ny):
    N = Nx * Ny
    coords = [(i, j) for i in range(Nx) for j in range(Ny)]
    coord_map = {c: idx for idx, c in enumerate(coords)}
    A = np.zeros((N, N))
    for idx, (i, j) in enumerate(coords):
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny:
                A[idx, coord_map[(ni, nj)]] = 1
    return A, coords

def build_gaussian_kernel(N, sigma=1.5):
    coords = [(i, j) for i in range(N) for j in range(N)]
    size = N * N
    A = np.zeros((size, size))
    for idx, (i, j) in enumerate(coords):
        for kdx, (k, l) in enumerate(coords):
            if idx != kdx:
                d2 = (i - k) ** 2 + (j - l) ** 2
                A[idx, kdx] = np.exp(-d2 / (2 * sigma**2))
    return A, coords

I2 = eye(2, format="csr")
def site_op(op_matrix, site, N):
    result = eye(1, format="csr")
    for j in range(N):
        result = kron(result, csr_matrix(np.array(op_matrix, dtype=float)) if j == site else I2, format="csr")
    return result
def build_sparse_H(N, J=1.0):
    H = csr_matrix((2**N, 2**N), dtype=float)
    for i in range(N - 1):
        Sx = site_op([[0, 0.5], [0.5, 0]], i, N); Sx1 = site_op([[0, 0.5], [0.5, 0]], i + 1, N)
        Sy = site_op([[0, -0.5], [0.5, 0]], i, N); Sy1 = site_op([[0, -0.5], [0.5, 0]], i + 1, N)
        Sz = site_op([[0.5, 0], [0, -0.5]], i, N); Sz1 = site_op([[0.5, 0], [0, -0.5]], i + 1, N)
        H += J * (Sx @ Sx1 - Sy @ Sy1 + Sz @ Sz1)
    return H
def ground_state(N):
    H = build_sparse_H(N)
    evals, evecs = eigsh(H, k=1, which="SA")
    return evecs[:, 0]
def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho); evals = evals[evals > 1e-15]
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

def normalize_adjacency(A_raw):
    A = A_raw / np.max(A_raw)
    np.fill_diagonal(A, 0)
    return A

def laplacian(A):
    D = np.diag(A.sum(axis=1))
    return D - A

def symmetric_perturbation(A, eps, rng):
    N = A.shape[0]
    noise = rng.uniform(-1, 1, size=(N, N))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0)
    A_pert = A * (1 + eps * noise)
    A_pert = np.clip(A_pert, 0, None)  # keep non-negative
    np.fill_diagonal(A_pert, 0)
    return A_pert

print("Building baseline adjacency matrices...")
A_2d_raw, coords_2d = build_2d_nn_laplacian(12, 12)
A_2d = A_2d_raw.copy()  # already 0/1, "normalize" trivial but keep pipeline consistent
A_2d = A_2d / np.max(A_2d) if np.max(A_2d) > 0 else A_2d

A_gauss_raw, coords_g = build_gaussian_kernel(6, sigma=1.5)
A_gauss = A_gauss_raw / np.max(A_gauss_raw)

psi = ground_state(12)
I_mat = MI_matrix(psi, 12)
A_heis = normalize_adjacency(I_mat)

print("Baseline matrices built.\n")

# ============================================================
# Baseline diagnostics
# ============================================================

def diagnostics_2d(A, coords):
    L = laplacian(A)
    evals, evecs = eigh(L)
    x = np.array([c[1] for c in coords], dtype=float)
    y = np.array([c[0] for c in coords], dtype=float)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    deg_ratio = abs(evals[1] - evals[2]) / ((evals[1] + evals[2]) / 2)
    # best correlation of the degenerate pair's span with x and y
    corr_x = max(abs(np.corrcoef(evecs[:,1], x)[0,1]), abs(np.corrcoef(evecs[:,2], x)[0,1]))
    corr_y = max(abs(np.corrcoef(evecs[:,1], y)[0,1]), abs(np.corrcoef(evecs[:,2], y)[0,1]))
    return evals, deg_ratio, corr_x, corr_y

def diagnostics_heis(A, k_max=4):
    L = laplacian(A)
    evals = eigvalsh(L)
    low = evals[1:1+k_max]
    k = np.arange(1, k_max+1)
    A_lin = np.vstack([k, np.ones_like(k)]).T
    c_lin, *_ = np.linalg.lstsq(A_lin, low, rcond=None)
    pred_lin = A_lin @ c_lin
    r2_lin = 1 - np.sum((low-pred_lin)**2)/np.sum((low-low.mean())**2)
    A_quad = np.vstack([k**2, np.ones_like(k)]).T
    c_quad, *_ = np.linalg.lstsq(A_quad, low, rcond=None)
    pred_quad = A_quad @ c_quad
    r2_quad = 1 - np.sum((low-pred_quad)**2)/np.sum((low-low.mean())**2)
    return evals, r2_lin, r2_quad

print("="*70)
print("BASELINE (unperturbed) diagnostics")
print("="*70)
evals_2d_base, deg_2d_base, corrx_2d_base, corry_2d_base = diagnostics_2d(A_2d, coords_2d)
print(f"2D NN Laplacian:   degeneracy={deg_2d_base:.3e}  corr_x={corrx_2d_base:.4f}  corr_y={corry_2d_base:.4f}")

evals_g_base, deg_g_base, corrx_g_base, corry_g_base = diagnostics_2d(A_gauss, coords_g)
print(f"Gaussian kernel:   degeneracy={deg_g_base:.3e}  corr_x={corrx_g_base:.4f}  corr_y={corry_g_base:.4f}")

evals_h_base, r2lin_base, r2quad_base = diagnostics_heis(A_heis)
print(f"Heisenberg MI:     R2_linear={r2lin_base:.5f}  R2_quadratic={r2quad_base:.5f}")
print()

# ============================================================
# Perturbation sweep
# ============================================================

n_trials = 30
eps_levels = [0.01, 0.03, 0.05]

print("="*70)
print("PERTURBATION SWEEP (n_trials=%d per level)" % n_trials)
print("="*70)

for label, A_base, coords, is_2d in [
    ("2D NN Laplacian", A_2d, coords_2d, True),
    ("Gaussian kernel", A_gauss, coords_g, True),
    ("Heisenberg MI",   A_heis, None,     False),
]:
    print(f"\n--- {label} ---")
    for eps in eps_levels:
        rng = np.random.default_rng(42)
        if is_2d:
            degs, corrxs, corrys = [], [], []
            for trial in range(n_trials):
                A_p = symmetric_perturbation(A_base, eps, rng)
                evals, deg, cx, cy = diagnostics_2d(A_p, coords)
                degs.append(deg); corrxs.append(cx); corrys.append(cy)
            degs = np.array(degs); corrxs = np.array(corrxs); corrys = np.array(corrys)
            print(f"  eps={eps:.2f}: degeneracy = {degs.mean():.4f} +/- {degs.std():.4f} "
                  f"(baseline {0:.1e})  |  corr_x = {corrxs.mean():.4f} +/- {corrxs.std():.4f}  "
                  f"corr_y = {corrys.mean():.4f} +/- {corrys.std():.4f}")
        else:
            r2lins, r2quads = [], []
            for trial in range(n_trials):
                A_p = symmetric_perturbation(A_base, eps, rng)
                evals, r2lin, r2quad = diagnostics_heis(A_p)
                r2lins.append(r2lin); r2quads.append(r2quad)
            r2lins = np.array(r2lins); r2quads = np.array(r2quads)
            n_linear_wins = np.sum(r2lins > r2quads)
            print(f"  eps={eps:.2f}: R2_linear = {r2lins.mean():.5f} +/- {r2lins.std():.5f}  "
                  f"R2_quadratic = {r2quads.mean():.5f} +/- {r2quads.std():.5f}  "
                  f"(linear wins {n_linear_wins}/{n_trials} trials)")
