"""
Generates cft_universality_test.png (Fig. 7) and prints Table III values,
computed directly from cft_test_results.json (produced by run_test.py).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

with open("cft_test_results.json") as f:
    results = json.load(f)

def powerlaw(x, a, b):
    return a * x**(-b)

model_order = ["XXZ_delta_1.0_Heisenberg", "XXZ_delta_0.5",
               "XXZ_delta_0.0_freefermion", "TFIM_critical"]
labels = [r'XXZ $\Delta=1.0$' + '\n(Heisenberg)', r'XXZ $\Delta=0.5$',
          r'XXZ $\Delta=0.0$' + '\n(free fermion)', 'TFIM\n(critical)']
K_vals = [0.5, 0.75, 1.0, None]

b_vals, eta_vals = [], []
for name in model_order:
    data = results[name]
    Ns = np.array(data["Ns"], dtype=float)
    lam1 = np.array(data["lambda1"])
    popt_b, _ = curve_fit(powerlaw, Ns, lam1, p0=[1, 0.5], maxfev=10000)
    b_vals.append(popt_b[1])

    mi_row0 = np.array(data["mi_row0_maxN"])
    N = int(data["Ns"][-1])
    r = np.arange(1, N // 2 + 1)
    mi_vals = mi_row0[1:N // 2 + 1]
    mask = mi_vals > 1e-8
    popt_eta, _ = curve_fit(powerlaw, r[mask], mi_vals[mask], p0=[1, 1], maxfev=10000)
    eta_vals.append(popt_eta[1])

b_vals = np.array(b_vals)
eta_vals = np.array(eta_vals)

print("Model                          b        eta_MI")
for name, b, eta in zip(model_order, b_vals, eta_vals):
    print(f"{name:<30} {b:.4f}   {eta:.4f}")
corr = np.corrcoef(b_vals, eta_vals)[0, 1]
print(f"\nCorrelation(b, eta_MI) = {corr:.4f}")

slope, intercept = np.polyfit(K_vals[:3], b_vals[:3], 1)
pred = slope * np.array(K_vals[:3]) + intercept
r2 = 1 - np.sum((b_vals[:3] - pred) ** 2) / np.sum((b_vals[:3] - b_vals[:3].mean()) ** 2)
print(f"Within XXZ line: b = {slope:.3f}*K + {intercept:.3f}, R2={r2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:red']
markers = ['o', 'o', 'o', 's']
for i in range(4):
    axes[0].scatter(eta_vals[i], b_vals[i], s=120, color=colors[i], marker=markers[i], zorder=5)
    axes[0].annotate(labels[i], (eta_vals[i], b_vals[i]), textcoords='offset points', xytext=(8, 8), fontsize=9)
axes[0].set_xlabel(r'Direct MI-decay exponent $\eta_{MI}$ (measured from $I(r)\propto r^{-\eta}$)')
axes[0].set_ylabel(r'Laplacian finite-size exponent $b$ ($\lambda_1\propto N^{-b}$)')
axes[0].set_title(f'Two independent diagnostics agree\n(correlation r={corr:.2f})')
axes[0].grid(alpha=0.3)
axes[0].axhline(0, color='gray', lw=0.7)

axes[1].plot(K_vals[:3], b_vals[:3], 'o-', color='tab:blue', ms=10, lw=2,
             label='XXZ line (same c=1 universality class)')
for i in range(3):
    axes[1].annotate(f'$\\Delta={[1.0,0.5,0.0][i]}$', (K_vals[i], b_vals[i]),
                      textcoords='offset points', xytext=(8, -12), fontsize=9)
axes[1].set_xlabel(r'Luttinger parameter $K$ (standard theory value)')
axes[1].set_ylabel(r'Laplacian finite-size exponent $b$')
axes[1].set_title(f'Within one universality class:\nb varies continuously with K (R²={r2:.3f})')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/cft_universality_test.png", dpi=200)
print("\nsaved figures/cft_universality_test.png")
