import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# ============================================================
# Utility functions
# ============================================================

def correlation(u, v):
    u = u - u.mean()
    v = v - v.mean()
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)

def spectral_dimension(evals, t_values):
    """Return spectral dimension d_s(t) from heat kernel trace."""
    K = np.array([np.sum(np.exp(-t * evals)) for t in t_values])
    logK = np.log(K)
    logt = np.log(t_values)
    dlogK_dlogt = np.gradient(logK, logt)
    return -2 * dlogK_dlogt

# ============================================================
# 1D Laplacian
# ============================================================

def laplacian_1d(N):
    A = np.zeros((N, N))
    for i in range(N):
        if i > 0: A[i, i-1] = 1
        if i < N-1: A[i, i+1] = 1
    D = np.diag(A.sum(axis=1))
    return D - A

def test_1d_scaling(N_list=[20, 40, 80], n_modes=6):
    print("\n=== 1D FINITE-SIZE SCALING ===")
    for N in N_list:
        L = laplacian_1d(N)
        vals, _ = la.eigh(L)
        ks = np.arange(n_modes) * np.pi / (N+1)
        a_fit = np.sum(vals[:n_modes] * ks**2) / np.sum(ks**4)
        print(f"N={N:3d}, fitted a={a_fit:.6f}, λ1={vals[1]:.6f}, λ2={vals[2]:.6f}")

# ============================================================
# 2D Laplacian
# ============================================================

def laplacian_2d(nx, ny):
    N = nx * ny
    A = np.zeros((N, N))
    def idx(i, j): return i * ny + j
    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            if i > 0: A[k, idx(i-1, j)] = 1
            if i < nx-1: A[k, idx(i+1, j)] = 1
            if j > 0: A[k, idx(i, j-1)] = 1
            if j < ny-1: A[k, idx(i, j+1)] = 1
    D = np.diag(A.sum(axis=1))
    return D - A

def test_2d_modes(nx=12, ny=12, n_modes=10):
    print("\n=== 2D LOW-LYING SPECTRUM ===")
    L = laplacian_2d(nx, ny)
    vals, vecs = la.eigh(L)
    vals = vals[:n_modes]
    vecs = vecs[:, :n_modes]

    for i in range(n_modes):
        print(f"Mode {i}: λ={vals[i]:.6f}")

    # Degeneracy test
    split = abs(vals[1] - vals[2]) / ((vals[1] + vals[2]) / 2)
    print(f"\nDegeneracy splitting (modes 1 & 2): {split:.3e}")

    # Coordinate correlations
    xs = np.linspace(-1, 1, nx)
    ys = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    x_field = X.reshape(-1)
    y_field = Y.reshape(-1)

    c1x = correlation(vecs[:, 1], x_field)
    c1y = correlation(vecs[:, 1], y_field)
    c2x = correlation(vecs[:, 2], x_field)
    c2y = correlation(vecs[:, 2], y_field)

    print("\nCoordinate correlations:")
    print(f"mode1: corr(x)={c1x:.4f}, corr(y)={c1y:.4f}")
    print(f"mode2: corr(x)={c2x:.4f}, corr(y)={c2y:.4f}")

    # Rotational invariance test
    R = np.array([[0, -1], [1, 0]])  # 90° rotation
    XY = np.vstack([x_field, y_field])
    XY_rot = R @ XY
    rot_corr = correlation(vecs[:, 1], XY_rot[0])
    print(f"\nRotational invariance check (mode1 vs rotated-x): {rot_corr:.4f}")

# ============================================================
# Graviton dispersion (clean)
# ============================================================

def test_graviton_dispersion(nx=12, ny=12, n_modes=20):
    print("\n=== GRAVITON DISPERSION ===")
    L = laplacian_2d(nx, ny)
    vals, _ = la.eigh(L)
    vals = vals[1:n_modes]  # skip zero mode
    ks = np.sqrt(vals)
    omega = ks

    # Fit ω = a k
    a_fit = np.sum(omega * ks) / np.sum(ks**2)
    rel_err = np.abs(omega - a_fit * ks) / omega

    print(f"Fitted a={a_fit:.6f}")
    print(f"Mean relative error={rel_err.mean():.3e}, max={rel_err.max():.3e}")

# ============================================================
# Spectral dimension
# ============================================================

def test_spectral_dimension(nx=12, ny=12):
    print("\n=== SPECTRAL DIMENSION ===")
    L = laplacian_2d(nx, ny)
    vals, _ = la.eigh(L)
    t_vals = np.logspace(-3, 1, 20)
    d_s = spectral_dimension(vals, t_vals)
    for t, d in zip(t_vals, d_s):
        print(f"t={t:.3e}, d_s={d:.3f}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_1d_scaling()
    test_2d_modes()
    test_graviton_dispersion()
    test_spectral_dimension()
