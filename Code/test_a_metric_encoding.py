"""
test_a_metric_encoding.py
===========================

Tests the central open problem Sec. 4.1 of the paper identifies: does some
function of the mutual-information matrix I(i,j) -- candidate distance
d(i,j) = -log(I(i,j)/I_max) -- satisfy the metric/kernel conditions required
by the graph-Laplacian convergence theorems (Belkin-Niyogi 2008; Singer
2006) for the discrete-to-continuum limit (Eq. 5) to hold?

Three independent, pre-registered checks, run on both the critical
(gapless) and gapped (dimerized) Heisenberg chain as a contrast pair:

  1. Real-space scaling of d(r) against the CFT-correct notion of distance
     on a periodic ring -- the *chord distance* (N/pi) sin(pi r / N), not
     the naive hop-count r (using naive r is the less charitable, and
     geometrically incorrect, version of this test; using chord distance is
     what "geodesic distance on the manifold" actually means for this
     topology, and is what the convergence theorems require). Linear-in-
     chord (Euclidean-kernel-consistent) vs logarithmic-in-chord
     (holographic/CFT-consistent; equivalent to I(r) being a power law in
     chord distance, i.e. Calabrese & Cardy, J. Stat. Mech. 0406, P06002
     (2004)) are compared.
  2. Triangle-inequality violation rate on the full N x N distance matrix.
  3. Classical (Torgerson) MDS embedding into 2D Euclidean space, with
     embedding-quality R^2 between embedded and target distances.

Usage: python test_a_metric_encoding.py
Produces figures/mi_metric_encoding_test.png and prints Table VI.
Runtime: well under a minute (N=16,18 only; needs the full 2**N ground
state, unlike test_b which uses the larger-N sector method).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from substrate import (
    build_heisenberg_H_pbc, build_dimerized_H_pbc, ground_state, MI_matrix,
    classical_mds, savefig,
)


def linear(x, a, b):
    return a * x + b


def logarithmic(x, a, b):
    return a * np.log(x) + b


def analyze(I_mat, N):
    A = I_mat / np.max(I_mat)
    np.fill_diagonal(A, 1.0)
    d = -np.log(A)
    np.fill_diagonal(d, 0.0)

    coords = np.arange(N)
    ring_dist = np.minimum(np.abs(coords - 0), N - np.abs(coords - 0))
    r_vals = np.arange(1, N // 2 + 1)
    I_of_r = np.array([I_mat[0, np.where(ring_dist == r)[0][0]] for r in r_vals])
    chord = (N / np.pi) * np.sin(np.pi * r_vals / N)

    d_of_r = -np.log(I_of_r / np.max(I_mat))

    popt_lin, _ = curve_fit(linear, chord, d_of_r)
    r2_lin = 1 - np.sum((d_of_r - linear(chord, *popt_lin)) ** 2) / np.sum((d_of_r - d_of_r.mean()) ** 2)
    popt_log, _ = curve_fit(logarithmic, chord, d_of_r)
    r2_log = 1 - np.sum((d_of_r - logarithmic(chord, *popt_log)) ** 2) / np.sum((d_of_r - d_of_r.mean()) ** 2)

    n_viol, n_total, max_viol = 0, 0, 0.0
    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            for k in range(N):
                if k == i or k == j:
                    continue
                n_total += 1
                lhs, rhs = d[i, k], d[i, j] + d[j, k]
                if lhs > rhs + 1e-9:
                    n_viol += 1
                    max_viol = max(max_viol, lhs - rhs)

    X2, D_embed2, evals = classical_mds(d, dims=2)
    mask = ~np.eye(N, dtype=bool)
    target, embedded = d[mask], D_embed2[mask]
    r2_embed = 1 - np.sum((target - embedded) ** 2) / np.sum((target - target.mean()) ** 2)

    return {
        "r_vals": r_vals.tolist(), "chord": chord.tolist(), "I_of_r": I_of_r.tolist(),
        "d_of_r": d_of_r.tolist(),
        "popt_lin": popt_lin.tolist(), "r2_lin": r2_lin,
        "popt_log": popt_log.tolist(), "r2_log": r2_log,
        "triangle_violation_rate": n_viol / n_total, "max_violation": max_viol,
        "mds_r2": r2_embed, "X2": X2.tolist(),
    }


def main():
    all_results = {}
    for N in [16, 18]:
        print(f"\nN={N}: computing ground states and MI matrices (PBC)...")
        H_crit = build_heisenberg_H_pbc(N, delta=1.0)
        _, psi_crit = ground_state(H_crit)
        I_crit = MI_matrix(psi_crit, N)

        H_gap = build_dimerized_H_pbc(N, delta=0.3)
        _, psi_gap = ground_state(H_gap)
        I_gap = MI_matrix(psi_gap, N)

        res_crit = analyze(I_crit, N)
        res_gap = analyze(I_gap, N)
        all_results[N] = {"critical": res_crit, "gapped": res_gap}

        print(f"  critical: linear R2={res_crit['r2_lin']:.4f}  log R2={res_crit['r2_log']:.4f}  "
              f"triangle-viol={100*res_crit['triangle_violation_rate']:.1f}%  MDS-2D R2={res_crit['mds_r2']:.4f}")
        print(f"  gapped:   linear R2={res_gap['r2_lin']:.4f}  log R2={res_gap['r2_log']:.4f}  "
              f"triangle-viol={100*res_gap['triangle_violation_rate']:.1f}%  MDS-2D R2={res_gap['mds_r2']:.4f}")

    with open("test_a_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("Table VI: mutual-information metric-encoding test")
    print("=" * 70)
    print(f"{'N':>4s} {'':10s} {'lin R2':>10s} {'log R2':>10s} {'tri.viol%':>10s} {'MDS-2D R2':>10s}")
    for N, results in all_results.items():
        for label in ("critical", "gapped"):
            res = results[label]
            print(f"{N:4d} {label:10s} {res['r2_lin']:10.4f} {res['r2_log']:10.4f} "
                  f"{100*res['triangle_violation_rate']:9.2f}% {res['mds_r2']:10.4f}")

    # ============================================================
    # Figure (use N=18, the larger/more resolved system, as the primary
    # illustration; the table above documents N=16 for robustness)
    # ============================================================
    res_crit = all_results[18]["critical"]
    res_gap = all_results[18]["gapped"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, res, label, color in [(axes[0], res_crit, "Critical (gapless, $c$=1 CFT)", "tab:blue"),
                                    (axes[1], res_gap, "Gapped ($\\delta$=0.3), control", "tab:orange")]:
        chord = np.array(res["chord"])
        d_of_r = np.array(res["d_of_r"])
        chord_fine = np.linspace(chord.min(), chord.max(), 200)
        ax.plot(chord, d_of_r, "o", ms=9, color=color, zorder=5, label="ED data")
        ax.plot(chord_fine, linear(chord_fine, *res["popt_lin"]), "--", lw=2, color="gray",
                label=f"linear ($R^2$={res['r2_lin']:.4f})")
        ax.plot(chord_fine, logarithmic(chord_fine, *res["popt_log"]), "-", lw=2, color=color,
                label=f"log ($R^2$={res['r2_log']:.4f})")
        ax.set_xlabel("Chord distance $(N/\\pi)\\sin(\\pi r/N)$")
        ax.set_ylabel(r"Candidate distance $d=-\log(I/I_{\max})$")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("mi_metric_encoding_test.png", dpi=200)
    print("\nsaved figures/mi_metric_encoding_test.png")


if __name__ == "__main__":
    main()
