import numpy as np
import matplotlib.pyplot as plt

def build_2d_kernel(N=6, a=1.0, sigma=1.0):
    """
    Build a mutual-information kernel A_ij on an N×N lattice.
    """
    coords = []
    for i in range(N):
        for j in range(N):
            coords.append(np.array([i*a, j*a]))
    coords = np.array(coords)

    num = N*N
    A = np.zeros((num, num))

    for i in range(num):
        for j in range(num):
            r2 = np.sum((coords[i] - coords[j])**2)
            A[i, j] = np.exp(-r2 / (2*sigma**2))

    # Zero diagonal (no self-connection)
    np.fill_diagonal(A, 0.0)
    return A, coords


def graph_laplacian(A):
    """
    L_ij = degree(i) δ_ij – A_ij
    """
    D = np.diag(np.sum(A, axis=1))
    return D - A


# ---- Run the 2D spectral emergence calculation ----
N = 6
A, coords = build_2d_kernel(N=N, sigma=1.0)
L = graph_laplacian(A)

# Compute exact eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eigh(L)
eigvals_sorted = np.sort(eigvals)

# ---- Plot the 6 lowest low-lying modes ----
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
axes = axes.flatten()

for idx in range(6):
    # Reshape the 36-element eigenvector back into the 6x6 spatial lattice matrix
    mode = eigvecs[:, idx].reshape(N, N)
    
    # Render using the RdBu divergent colormap for clear nodal planes
    im = axes[idx].imshow(mode, cmap='RdBu', origin='lower')
    
    # Raw string combination to safely escape the LaTeX lambda command
    axes[idx].set_title(f"Mode {idx}, " + r"$\lambda$=" + f"{eigvals_sorted[idx]:.3f}", fontsize=10)
    axes[idx].axis('off')

plt.suptitle("Emergent 2D Coordinate Fields from Substrate Entanglement Kernel", 
             fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()

# Save automatically as a crisp, high-resolution PNG asset for your LaTeX document
plt.savefig("eigenmodes.png", dpi=300, bbox_inches='tight')
print("Successfully generated and saved: eigenmodes.png")

plt.show()
