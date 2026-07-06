import numpy as np
import time
import json
from scipy.linalg import eigvalsh
from dimerization_test import build_dimerized_H, ground_state, MI_matrix, laplacian_from_MI

Ns = [8, 10, 12, 14, 16]
deltas = [0.0, 0.1, 0.3, 0.6]

results = {}

for delta in deltas:
    key = f"delta_{delta}"
    print(f"\n=== delta={delta} ===")
    lam1 = []
    mi_row0_maxN = None
    for N in Ns:
        t0 = time.time()
        H = build_dimerized_H(N, delta)
        psi = ground_state(H)
        I_mat = MI_matrix(psi, N)
        L = laplacian_from_MI(I_mat)
        evals = eigvalsh(L)
        lam1.append(evals[1])
        print(f"  N={N:2d}  lambda_1={evals[1]:.6f}  ({time.time()-t0:.1f}s)")
        if N == Ns[-1]:
            mi_row0_maxN = I_mat[0, :].tolist()
    results[key] = {"delta": delta, "Ns": Ns, "lambda1": lam1, "mi_row0_maxN": mi_row0_maxN}

with open("dimerization_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved dimerization_results.json")
