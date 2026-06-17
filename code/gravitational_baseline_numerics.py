"""
gravitational_baseline_numerics.py
====================================

Unified numerical engine for the Gravitational Baseline Principle paper.

Three independent pipelines:
  Pipeline A: Heisenberg chain mutual-information Laplacian (physical substrate)
  Pipeline B: 2D nearest-neighbour Laplacian (clean geometric signatures)
  Pipeline C: Gaussian kernel on 6x6 substrate (coordinate chart emergence)

Generates seven figures into ./figures/:
  1. heisenberg_spectrum.png     -- 1D finite-size scaling N=20,40,80
  2. gaussian_eigenmodes.png     -- Emergent coordinate fields 6x6
  3. 2d_degeneracy.png           -- Exact degeneracy of first excited pair
  4. 2d_eigenmodes.png           -- 2D coordinate chart eigenmodes
  5. spectral_dimension.png      -- d_s(t) flow for all three pipelines
  6. graviton_dispersion.png     -- Massless dispersion ω = k
  7. pipeline_comparison.png     -- Side-by-side geometric signatures

Author: Paul Jarvis
Requires: numpy, scipy, matplotlib
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, eigvalsh

# ============================================================
# Safe figure saving helper
# ============================================================

def savefig(path):
    """Save figure to path, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Sparse Heisenberg Hamiltonian and mutual information
# ============================================================

I2 = eye(2, format="csr")


def site_op(op_matrix, site, N):
    """Return operator acting with op_matrix on given site, identity elsewhere."""
    result = eye(1, format="csr")
    for j in range(N):
        result = kron(
            result,
            csr_matrix(np.array(op_matrix, dtype=float)) if j == site else I2,
            format="csr",
        )
    return result


def build_sparse_H(N, J=1.0):
    """Build sparse Heisenberg Hamiltonian for N spins."""
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


def ground_state(N):
    """Compute ground state of Heisenberg chain of length N."""
    H = build_sparse_H(N)
    evals, evecs = eigsh(H, k=1, which="SA")
    return evecs[:, 0]


def von_neumann_entropy(rho):
    """Von Neumann entropy of density matrix rho."""
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-15]
    return -np.sum(evals * np.log(evals))


def partial_trace(psi, keep, N):
    """Partial trace over all sites except those in keep."""
    psi = psi.reshape([2] * N)
    trace_over = [i for i in range(N) if i not in keep]
    psi = np.transpose(psi, keep + trace_over)
    psi = psi.reshape(2**len(keep), 2**len(trace_over))
    return psi @ psi.conj().T


def MI_matrix(psi, N):
    """Mutual-information matrix for all pairs of sites."""
    I_mat = np.zeros((N, N))
    S = [von_neumann_entropy(partial_trace(psi, [i], N)) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            Sij = von_neumann_entropy(partial_trace(psi, [i, j], N))
            I_mat[i, j] = I_mat[j, i] = S[i] + S[j] - Sij
    return I_mat


def laplacian_from_MI(I_mat):
    """Graph Laplacian from mutual-information adjacency matrix."""
    A = I_mat / np.max(I_mat)
    np.fill_diagonal(A, 0)
    D = np.diag(A.sum(axis=1))
    return D - A


# ============================================================
# 2D nearest-neighbour Laplacian
# ============================================================

def build_2d_nn_laplacian(Nx, Ny):
    """Build 2D nearest-neighbour Laplacian on Nx x Ny grid."""
    N = Nx * Ny
    coords = [(i, j) for i in range(Nx) for j in range(Ny)]
    coord_map = {c: idx for idx, c in enumerate(coords)}
    A = np.zeros((N, N))
    for idx, (i, j) in enumerate(coords):
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny:
                A[idx, coord_map[(ni, nj)]] = 1
    D = np.diag(A.sum(axis=1))
    return D - A, coords


# ============================================================
# Gaussian kernel Laplacian
# ============================================================

def build_gaussian_laplacian(N, sigma=1.5):
    """Build Gaussian kernel Laplacian on N x N grid."""
    coords = [(i, j) for i in range(N) for j in range(N)]
    size = N * N
    A = np.zeros((size, size))
    for idx, (i, j) in enumerate(coords):
        for kdx, (k, l) in enumerate(coords):
            if idx != kdx:
                d2 = (i - k) ** 2 + (j - l) ** 2
                A[idx, kdx] = np.exp(-d2 / (2 * sigma**2))
    D = np.diag(A.sum(axis=1))
    return D - A, coords


# ============================================================
# Spectral dimension
# ============================================================

def spectral_dimension(evals, t_range, threshold=1e-6):
    """Compute spectral dimension d_s(t) from Laplacian eigenvalues."""
    evals_nz = evals[evals > threshold]
    ds = []
    for t in t_range:
        weights = np.exp(-t * evals_nz)
        K = np.sum(weights)
        if K < 1e-300:
            ds.append(0.0)
            continue
        dlogK = -np.dot(evals_nz, weights) / K
        ds.append(-2 * t * dlogK)
    return np.array(ds)


# ============================================================
# Figure 1: 1D finite-size scaling
# ============================================================

def fig1_heisenberg_scaling():
    print("[Fig 1] Heisenberg 1D finite-size scaling...")

    results = {}
    for N in [20, 40, 80]:
        t0 = time.time()
        A = np.zeros((N, N))
        for i in range(N - 1):
            A[i, i + 1] = A[i + 1, i] = 1
        L = np.diag(A.sum(axis=1)) - A
        evals = eigvalsh(L)

        k_vals = np.arange(1, 6)
        low = evals[1:6]
        a_vals = low * (N / np.pi) ** 2 / k_vals**2
        a_N = np.mean(a_vals)
        results[N] = (evals, a_N, k_vals, low)
        print(f"  N={N}: a(N)={a_N:.6f} ({time.time() - t0:.2f}s)")

    heis_results = {}
    for N in [12, 16]:
        t0 = time.time()
        psi = ground_state(N)
        I_mat = MI_matrix(psi, N)
        L = laplacian_from_MI(I_mat)
        evals = eigvalsh(L)
        k_vals = np.arange(1, 5)
        low = evals[1:5]
        coeffs = np.polyfit(k_vals**2, low, 1)
        heis_results[N] = (evals, coeffs[0])
        print(f"  Heisenberg N={N}: a(N)={coeffs[0]:.6f} ({time.time() - t0:.1f}s)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors_nn = ["tab:blue", "tab:orange", "tab:green"]
    k_fine = np.linspace(0, 6, 200)
    for (N, (evals, a_N, k_vals, low)), color in zip(results.items(), colors_nn):
        axes[0].plot(k_vals**2, low * (N / np.pi) ** 2, "o", color=color, ms=7,
                     label=f"N={N}", zorder=5)
        axes[0].plot(k_fine**2, a_N * k_fine**2, "-", color=color,
                     alpha=0.6, lw=1.5)

    axes[0].set_xlabel(r"$k^2$")
    axes[0].set_ylabel(r"$\lambda_k (N/\pi)^2$")
    axes[0].set_title("1D NN Laplacian: continuum scaling")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for i, (N, (_, a_N, _, _)) in enumerate(results.items()):
        axes[0].annotate(f"$a({N})={a_N:.4f}$",
                         xy=(0.98, 0.15 + 0.12 * i),
                         xycoords="axes fraction", ha="right")

    k_h = np.arange(1, 5)
    for (N, (evals, a_N)), color in zip(heis_results.items(),
                                        ["tab:purple", "tab:red"]):
        axes[1].plot(k_h**2, evals[1:5], "o-", color=color, ms=7, lw=2,
                     label=f"Heisenberg MI N={N} (a={a_N:.3f})")

    axes[1].set_xlabel(r"$k^2$")
    axes[1].set_ylabel(r"$\lambda_k$")
    axes[1].set_title("Heisenberg MI Laplacian scaling")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("figures/heisenberg_spectrum.png")


# ============================================================
# Figure 2: Gaussian kernel eigenmodes (6x6)
# ============================================================

def fig2_gaussian_eigenmodes():
    print("[Fig 2] Gaussian kernel eigenmodes 6x6...")
    N = 6
    L, coords = build_gaussian_laplacian(N, sigma=1.5)
    evals, evecs = eigh(L)

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3)

    titles = [f"Mode 0, $\\lambda$={evals[0]:.3f}",
              f"Mode 1, $\\lambda$={evals[1]:.3f}",
              f"Mode 2, $\\lambda$={evals[2]:.3f}",
              f"Mode 3, $\\lambda$={evals[3]:.3f}",
              f"Mode 4, $\\lambda$={evals[4]:.3f}",
              f"Mode 5, $\\lambda$={evals[5]:.3f}"]

    for idx in range(6):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        mode = evecs[:, idx].reshape(N, N)
        im = ax.imshow(mode, cmap="RdBu_r", origin="lower",
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(titles[idx])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Gaussian kernel eigenmodes (6x6)")
    plt.tight_layout()
    savefig("figures/gaussian_eigenmodes.png")


# ============================================================
# Figure 3: 2D degeneracy and coordinate correlation
# ============================================================

def fig3_2d_degeneracy():
    print("[Fig 3] 2D degeneracy test...")
    Nx, Ny = 12, 12
    L, coords = build_2d_nn_laplacian(Nx, Ny)
    evals, evecs = eigh(L)

    x_coords = np.array([c[1] for c in coords], dtype=float)
    y_coords = np.array([c[0] for c in coords], dtype=float)
    x_coords = (x_coords - x_coords.mean()) / x_coords.std()
    y_coords = (y_coords - y_coords.mean()) / y_coords.std()

    v1 = evecs[:, 1]
    v2 = evecs[:, 2]

    corr_v1_x = np.corrcoef(v1, x_coords)[0, 1]
    corr_v1_y = np.corrcoef(v1, y_coords)[0, 1]
    corr_v2_x = np.corrcoef(v2, x_coords)[0, 1]
    corr_v2_y = np.corrcoef(v2, y_coords)[0, 1]

    x_rot = -y_coords
    corr_v1_xrot = np.corrcoef(v1, x_rot)[0, 1]

    deg_ratio = abs(evals[1] - evals[2]) / ((evals[1] + evals[2]) / 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    k_show = np.arange(10)
    axes[0].bar(k_show, evals[1:11])
    axes[0].set_xlabel("Mode index")
    axes[0].set_ylabel(r"$\lambda_k$")
    axes[0].set_title(f"2D NN Laplacian spectrum\nDegeneracy={deg_ratio:.1e}")
    axes[0].grid(True, alpha=0.3)

    axes[1].axis("off")
    table_data = [
        ["", "corr with $x$", "corr with $y$"],
        ["$v^{(1)}$", f"{corr_v1_x:.4f}", f"{corr_v1_y:.4f}"],
        ["$v^{(2)}$", f"{corr_v2_x:.4f}", f"{corr_v2_y:.4f}"],
        ["", "", ""],
        ["Rotation test:", "", ""],
        ["$v^{(1)}$ vs $x_{rot}$", f"{corr_v1_xrot:.4f}", ""],
    ]
    table = axes[1].table(cellText=table_data[1:],
                          colLabels=table_data[0],
                          loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    plt.tight_layout()
    savefig("figures/2d_degeneracy.png")


# ============================================================
# Figure 4: 2D eigenmodes visualised
# ============================================================

def fig4_2d_eigenmodes():
    print("[Fig 4] 2D eigenmodes...")
    Nx, Ny = 12, 12
    L, coords = build_2d_nn_laplacian(Nx, Ny)
    evals, evecs = eigh(L)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, mode_idx in enumerate([0, 1, 2, 3, 4, 5]):
        mode = evecs[:, mode_idx].reshape(Nx, Ny)
        im = axes[idx].imshow(mode, cmap="RdBu_r", origin="lower")
        plt.colorbar(im, ax=axes[idx], shrink=0.8)
        axes[idx].set_title(f"Mode {mode_idx}, $\\lambda={evals[mode_idx]:.4f}$")
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    plt.tight_layout()
    savefig("figures/2d_eigenmodes.png")


# ============================================================
# Figure 5: Spectral dimension flow — all pipelines
# ============================================================

def fig5_spectral_dimension():
    print("[Fig 5] Spectral dimension flow...")

    t_range = np.logspace(-2, 2, 200)

    L_2d, _ = build_2d_nn_laplacian(12, 12)
    evals_2d = eigvalsh(L_2d)
    ds_2d = spectral_dimension(evals_2d, t_range)

    L_g, _ = build_gaussian_laplacian(6, sigma=1.5)
    evals_g = eigvalsh(L_g)
    ds_g = spectral_dimension(evals_g, t_range, threshold=1e-4)

    psi = ground_state(12)
    I_mat = MI_matrix(psi, 12)
    L_h = laplacian_from_MI(I_mat)
    evals_h = eigvalsh(L_h)
    ds_h = spectral_dimension(evals_h, t_range)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].semilogx(t_range, ds_2d, "-", lw=2.5, color="tab:blue",
                     label="2D NN Laplacian (12x12)")
    axes[0].semilogx(t_range, np.clip(ds_g, 0, 4), "-", lw=2.5,
                     color="tab:orange", label="Gaussian kernel (6x6)")
    axes[0].semilogx(t_range, np.clip(ds_h, 0, 3), "--", lw=2.0,
                     color="tab:green", label="Heisenberg MI (N=12)")
    axes[0].axhline(y=2, color="gray", linestyle=":", alpha=0.7,
                    label="$d_s=2$")
    axes[0].axhline(y=1, color="gray", linestyle="-.", alpha=0.5,
                    label="$d_s=1$")
    axes[0].set_xlabel(r"Heat-kernel time $t$")
    axes[0].set_ylabel(r"Spectral dimension $d_s(t)$")
    axes[0].set_title("Spectral dimension flow: all pipelines")
    axes[0].set_ylim(-0.2, 3.5)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(t_range, ds_2d, "-", lw=2.5, color="tab:blue")
    axes[1].axhline(y=2, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_xlabel(r"Heat-kernel time $t$")
    axes[1].set_ylabel(r"$d_s(t)$")
    axes[1].set_title("Spectral dimension: 2D NN Laplacian (12x12)")
    axes[1].set_ylim(-0.2, 3.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("figures/spectral_dimension.png")


# ============================================================
# Figure 6: Massless graviton dispersion
# ============================================================

def fig6_graviton_dispersion():
    print("[Fig 6] Graviton dispersion...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for N, color, ls in [(20, "tab:blue", "-"),
                         (40, "tab:orange", "--"),
                         (80, "tab:green", ":")]:
        A = np.zeros((N, N))
        for i in range(N - 1):
            A[i, i + 1] = A[i + 1, i] = 1
        L = np.diag(A.sum(axis=1)) - A
        evals = eigvalsh(L)

        k_vals = np.arange(1, min(21, N))
        omega_norm = np.sqrt(evals[1:len(k_vals) + 1]) * N / np.pi

        axes[0].plot(k_vals, omega_norm, ls, color=color, lw=2,
                     label=f"N={N}")

    k_ref = np.linspace(0, 20, 200)
    axes[0].plot(k_ref, k_ref, "k--", alpha=0.5, lw=1.5,
                 label="$\\omega = k$")
    axes[0].set_xlabel("Mode number $k$")
    axes[0].set_ylabel(r"$\omega_k = \sqrt{\lambda_k} \cdot N/\pi$")
    axes[0].set_title("Normalised dispersion relation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for N, color, ls in [(20, "tab:blue", "-"),
                         (40, "tab:orange", "--"),
                         (80, "tab:green", ":")]:
        A = np.zeros((N, N))
        for i in range(N - 1):
            A[i, i + 1] = A[i + 1, i] = 1
        L = np.diag(A.sum(axis=1)) - A
        evals = eigvalsh(L)
        k_vals = np.arange(1, min(11, N))
        omega_norm = np.sqrt(evals[1:len(k_vals) + 1]) * N / np.pi
        dev = np.abs(omega_norm - k_vals) / k_vals
        axes[1].semilogy(k_vals, dev, ls, color=color, lw=2, label=f"N={N}")

    axes[1].set_xlabel("Mode number $k$")
    axes[1].set_ylabel(r"$|\omega_k - k|/k$")
    axes[1].set_title("Fractional deviation from massless dispersion")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    savefig("figures/graviton_dispersion.png")


# ============================================================
# Figure 7: Pipeline comparison summary
# ============================================================

def fig7_pipeline_comparison():
    print("[Fig 7] Pipeline comparison summary...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    L_g, coords_g = build_gaussian_laplacian(6, sigma=1.5)
    evals_g, evecs_g = eigh(L_g)

    axes[0, 0].imshow(evecs_g[:, 0].reshape(6, 6), cmap="RdBu_r",
                      origin="lower")
    axes[0, 0].set_title("Gaussian 6x6\nMode 0 (zero mode)")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    axes[0, 1].imshow(evecs_g[:, 1].reshape(6, 6), cmap="RdBu_r",
                      origin="lower")
    deg_g = abs(evals_g[1] - evals_g[2]) / ((evals_g[1] + evals_g[2]) / 2)
    axes[0, 1].set_title(f"Gaussian 6x6\nMode 1 (coord chart)\n"
                         f"deg={deg_g:.1e}")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    axes[0, 2].imshow(evecs_g[:, 2].reshape(6, 6), cmap="RdBu_r",
                      origin="lower")
    axes[0, 2].set_title("Gaussian 6x6\nMode 2 (orthogonal chart)")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])

    L_2d, _ = build_2d_nn_laplacian(12, 12)
    evals_2d, evecs_2d = eigh(L_2d)

    axes[1, 0].imshow(evecs_2d[:, 0].reshape(12, 12), cmap="RdBu_r",
                      origin="lower")
    axes[1, 0].set_title("NN 12x12\nMode 0 (zero mode)")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    axes[1, 1].imshow(evecs_2d[:, 1].reshape(12, 12), cmap="RdBu_r",
                      origin="lower")
    deg_2d = abs(evals_2d[1] - evals_2d[2]) / ((evals_2d[1] + evals_2d[2]) / 2)
    axes[1, 1].set_title(f"NN 12x12\nMode 1 (coord chart)\n"
                         f"deg={deg_2d:.1e}")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    axes[1, 2].imshow(evecs_2d[:, 2].reshape(12, 12), cmap="RdBu_r",
                      origin="lower")
    axes[1, 2].set_title("NN 12x12\nMode 2 (orthogonal chart)")
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    fig.suptitle("Pipeline comparison: independent routes to emergent 2D geometry")
    plt.tight_layout()
    savefig("figures/pipeline_comparison.png")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    t_total = time.time()
    print("=" * 60)
    print("Gravitational Baseline Principle - Numerical Evidence")
    print("=" * 60)

    fig1_heisenberg_scaling()
    fig2_gaussian_eigenmodes()
    fig3_2d_degeneracy()
    fig4_2d_eigenmodes()
    fig5_spectral_dimension()
    fig6_graviton_dispersion()
    fig7_pipeline_comparison()

    print("\n" + "=" * 60)
    print(f"All figures saved to ./figures/. Total time: {time.time() - t_total:.1f}s")
    print("=" * 60)
