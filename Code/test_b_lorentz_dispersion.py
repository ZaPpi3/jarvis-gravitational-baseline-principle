"""
test_b_lorentz_dispersion.py
=============================

Tests the first of the two prerequisites Sec. 5 of the paper identifies for
the Fierz-Pauli graviton argument to apply: that the substrate's low-energy
excitations acquire an emergent Lorentz-covariant (linear) dispersion.

Standard CFT/Bethe-ansatz result for the critical isotropic Heisenberg
chain: the Sz=1 sector's lowest excitation gap above the Sz=0 ground state
scales as

    E_gap(N) ~ (pi*v/N) * (1 + c/ln N + ...),   v = pi*J/2 exactly,

i.e. LINEARLY in 1/N -- the finite-size signature of a gapless spectrum with
linear dispersion omega(k) ~ v|k| near k=0 (des Cloizeaux & Pearson, J. Phys.
Chem. Solids 23, 133 (1962)). This is the standard, well-established
signature of emergent Lorentz/conformal invariance at a 1D quantum critical
point (see e.g. Cardy, "Scaling and Renormalization in Statistical Physics").

Control case: the dimerized (gapped) chain should NOT extrapolate to zero;
E_gap(N) should approach a finite constant (the spin gap), cleanly
distinguishing "gapless with linear dispersion" from "gapped".

Usage: python test_b_lorentz_dispersion.py
Produces figures/lorentz_dispersion_test.png and prints Table V.
Runtime: a few minutes (dominated by N=24, 26).
"""
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from substrate import build_H_sector_pbc, build_H_sector_dimerized_pbc, lowest_eigenvalue_sector, savefig

Ns = [12, 14, 16, 18, 20, 22, 24, 26]


def powerlaw(N, A, p):
    return A * N ** (-p)


def powerlaw_fixed_p2(N, A):
    return A * N ** (-2)


def const_plus_powerlaw(N, A, p, C):
    return C + A * N ** (-p)


def log_corrected(N, A, c):
    return (A / N) * (1 + c / np.log(N))


def main():
    results = {"critical": {}, "gapped_delta0.3": {}}

    print("=== Critical isotropic Heisenberg chain (delta=1.0) ===")
    for N in Ns:
        t0 = time.time()
        H0 = build_H_sector_pbc(N, Nup=N // 2, delta=1.0)
        E0 = lowest_eigenvalue_sector(H0, k=1)[0]
        H1 = build_H_sector_pbc(N, Nup=N // 2 + 1, delta=1.0)
        E1 = lowest_eigenvalue_sector(H1, k=1)[0]
        gap = E1 - E0
        results["critical"][N] = {"E0": float(E0), "E1": float(E1), "gap": float(gap)}
        print(f"  N={N:3d}: gap={gap:.6f}  ({time.time()-t0:.1f}s)")

    print("\n=== Gapped dimerized chain (delta=0.3), control case ===")
    for N in Ns:
        t0 = time.time()
        H0 = build_H_sector_dimerized_pbc(N, Nup=N // 2, delta=0.3)
        E0 = lowest_eigenvalue_sector(H0, k=1)[0]
        H1 = build_H_sector_dimerized_pbc(N, Nup=N // 2 + 1, delta=0.3)
        E1 = lowest_eigenvalue_sector(H1, k=1)[0]
        gap = E1 - E0
        results["gapped_delta0.3"][N] = {"E0": float(E0), "E1": float(E1), "gap": float(gap)}
        print(f"  N={N:3d}: gap={gap:.6f}  ({time.time()-t0:.1f}s)")

    with open("test_b_results.json", "w") as f:
        json.dump(results, f, indent=2)

    Ns_arr = np.array(Ns, dtype=float)
    gaps_crit = np.array([results["critical"][N]["gap"] for N in Ns])
    gaps_gap = np.array([results["gapped_delta0.3"][N]["gap"] for N in Ns])

    # Free power-law fit
    popt, pcov = curve_fit(powerlaw, Ns_arr, gaps_crit, p0=[1.5, 1.0], maxfev=20000)
    A, p = popt
    perr = np.sqrt(np.diag(pcov))
    r2 = 1 - np.sum((gaps_crit - powerlaw(Ns_arr, *popt)) ** 2) / np.sum((gaps_crit - gaps_crit.mean()) ** 2)

    # Log-corrected fit (standard finite-size form at the SU(2) point)
    popt2, pcov2 = curve_fit(log_corrected, Ns_arr, gaps_crit, p0=[np.pi**2 / 2, -1.0], maxfev=20000)
    r2_2 = 1 - np.sum((gaps_crit - log_corrected(Ns_arr, *popt2)) ** 2) / np.sum((gaps_crit - gaps_crit.mean()) ** 2)

    # Fixed p=2 alternative (non-relativistic / Schrodinger scaling)
    popt3, _ = curve_fit(powerlaw_fixed_p2, Ns_arr, gaps_crit, p0=[1.0], maxfev=20000)
    r2_3 = 1 - np.sum((gaps_crit - powerlaw_fixed_p2(Ns_arr, *popt3)) ** 2) / np.sum((gaps_crit - gaps_crit.mean()) ** 2)

    # Gapped control: gap -> finite constant
    popt4, pcov4 = curve_fit(const_plus_powerlaw, Ns_arr, gaps_gap, p0=[1.0, 1.0, 0.3], maxfev=20000)
    r2_4 = 1 - np.sum((gaps_gap - const_plus_powerlaw(Ns_arr, *popt4)) ** 2) / np.sum((gaps_gap - gaps_gap.mean()) ** 2)

    print("\n" + "=" * 70)
    print("Table V: finite-size gap scaling, critical vs gapped chain")
    print("=" * 70)
    print(f"Critical:  E_gap ~ A/N^p :  A={A:.4f}+/-{perr[0]:.4f}  p={p:.4f}+/-{perr[1]:.4f}  R2={r2:.6f}")
    print(f"           log-corrected:  A={popt2[0]:.4f}+/-{np.sqrt(pcov2[0,0]):.4f}  "
          f"c={popt2[1]:.4f}+/-{np.sqrt(pcov2[1,1]):.4f}  R2={r2_2:.6f}")
    print(f"           fixed p=2 (non-relativistic alternative): R2={r2_3:.6f}")
    print(f"           exact Bethe-ansatz prediction: p=1, A=pi^2/2={np.pi**2/2:.4f}")
    print(f"Gapped:    E_gap = C + A/N^p :  C(N->inf gap)={popt4[2]:.4f}+/-{np.sqrt(pcov4[2,2]):.4f}  R2={r2_4:.6f}")

    # ============================================================
    # Figure
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    N_fine = np.linspace(min(Ns), max(Ns), 200)
    axes[0].loglog(Ns_arr, gaps_crit, "ko", ms=9, zorder=5, label="ED data (critical)")
    axes[0].loglog(N_fine, powerlaw(N_fine, *popt), "-", lw=2, color="tab:blue",
                    label=f"free fit: $p$={p:.3f} ($R^2$={r2:.5f})")
    axes[0].loglog(N_fine, powerlaw_fixed_p2(N_fine, *popt3), "--", lw=2, color="tab:red",
                    label=f"fixed $p$=2 ($R^2$={r2_3:.3f})")
    axes[0].set_xlabel("$N$")
    axes[0].set_ylabel(r"Finite-size gap $E_{\rm gap}(N)$")
    axes[0].set_title("Critical chain: linear ($p{\\approx}1$) beats\nquadratic dispersion, decisively")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, which="both")

    axes[1].plot(Ns_arr, gaps_crit, "o-", color="tab:blue", ms=8, label="critical (gapless, $c$=1 CFT)")
    axes[1].plot(Ns_arr, gaps_gap, "s-", color="tab:orange", ms=8, label="gapped ($\\delta$=0.3), control")
    axes[1].axhline(popt4[2], color="tab:orange", ls=":", alpha=0.7,
                     label=f"gapped $N\\to\\infty$ limit = {popt4[2]:.3f}")
    axes[1].set_xlabel("$N$")
    axes[1].set_ylabel(r"Finite-size gap $E_{\rm gap}(N)$")
    axes[1].set_title("Critical gap $\\to 0$; gapped control\nsaturates at a finite spin gap")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("lorentz_dispersion_test.png", dpi=200)
    print("\nsaved figures/lorentz_dispersion_test.png")


if __name__ == "__main__":
    main()
