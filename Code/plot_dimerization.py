"""
Generates dimerization_crossover_test.png (Fig. 8) and prints Table IV values,
computed directly from dimerization_results.json (produced by run_dimerization.py).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

with open("dimerization_results.json") as f:
    results = json.load(f)

def powerlaw(x, a, b):
    return a * x**(-b)

def expdecay(r, a, xi):
    return a * np.exp(-r / xi)

deltas = [0.0, 0.1, 0.3, 0.6]
b_vals, xi_vals = [], []

print("delta    b        xi (exp fit, if applicable)")
for delta in deltas:
    key = f"delta_{delta}"
    data = results[key]
    Ns = np.array(data["Ns"], dtype=float)
    lam1 = np.array(data["lambda1"])
    popt_b, _ = curve_fit(powerlaw, Ns, lam1, p0=[1, 1], maxfev=10000)
    b_vals.append(popt_b[1])

    mi_row0 = np.array(data["mi_row0_maxN"])
    N = int(data["Ns"][-1])
    r = np.arange(1, N // 2 + 1)
    mi_vals = mi_row0[1:N // 2 + 1]
    mask = mi_vals > 1e-10
    if delta == 0.0:
        xi_vals.append(None)
        print(f"{delta:<8} {popt_b[1]:.4f}   -- (power-law, no finite xi)")
    else:
        popt_e, _ = curve_fit(expdecay, r[mask], mi_vals[mask], p0=[1, 1], maxfev=10000)
        xi_vals.append(popt_e[1])
        print(f"{delta:<8} {popt_b[1]:.4f}   {popt_e[1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(deltas, b_vals, 'o-', color='tab:purple', ms=10, lw=2)
axes[0].axhline(2.0, color='gray', ls=':', label='continuum-Laplacian value (b=2)')
axes[0].axhline(b_vals[0], color='tab:blue', ls='--', alpha=0.6,
                label=f'critical-point value (b={b_vals[0]:.3f}, this work)')
axes[0].set_xlabel(r'Dimerization strength $\delta$ (0 = critical)')
axes[0].set_ylabel(r'Finite-size exponent $b$')
axes[0].set_title('Exponent crosses over from critical\nto continuum-geometric as the gap opens')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i, delta in enumerate(deltas):
    key = f"delta_{delta}"
    mi_row0 = np.array(results[key]["mi_row0_maxN"])
    N = int(results[key]["Ns"][-1])
    r = np.arange(1, N // 2 + 1)
    mi_vals = mi_row0[1:N // 2 + 1]
    mask = mi_vals > 1e-8
    axes[1].semilogy(r[mask], mi_vals[mask], 'o-', color=colors[i], label=f'$\\delta$={delta}')

axes[1].set_xlabel('Site separation $r$')
axes[1].set_ylabel('Mutual information $I(r)$ (log scale)')
axes[1].set_title('Real-space MI decay: power-law (critical)\nsteepens toward a sharp cutoff (gapped)')
axes[1].legend()
axes[1].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig("figures/dimerization_crossover_test.png", dpi=200)
print("\nsaved figures/dimerization_crossover_test.png")
