import numpy as np
import time
import json
from scipy.linalg import eigvalsh
from cft_universality_test import (build_xxz_H, build_tfim_H, ground_state,
                                     MI_matrix, laplacian_from_MI)

Ns = [8, 10, 12, 14, 16]

models = {
    "XXZ_delta_1.0_Heisenberg": lambda N: build_xxz_H(N, 1.0),
    "XXZ_delta_0.5":            lambda N: build_xxz_H(N, 0.5),
    "XXZ_delta_0.0_freefermion":lambda N: build_xxz_H(N, 0.0),
    "TFIM_critical":            lambda N: build_tfim_H(N, 1.0),
}

results = {}

for name, H_builder in models.items():
    print(f"\n=== {name} ===")
    lam1 = []
    mi_row0_at_maxN = None
    for N in Ns:
        t0 = time.time()
        H = H_builder(N)
        psi = ground_state(H)
        I_mat = MI_matrix(psi, N)
        L = laplacian_from_MI(I_mat)
        evals = eigvalsh(L)
        lam1.append(evals[1])
        print(f"  N={N:2d}  lambda_1={evals[1]:.6f}  ({time.time()-t0:.1f}s)")
        if N == Ns[-1]:
            mi_row0_at_maxN = I_mat[0, :].tolist()
    results[name] = {"Ns": Ns, "lambda1": lam1, "mi_row0_maxN": mi_row0_at_maxN}

with open("cft_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved cft_test_results.json")
