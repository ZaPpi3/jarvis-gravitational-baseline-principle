import sys, json, numpy as np
from scipy.linalg import eigvalsh

N = int(sys.argv[1])
with open(f"mi_partial_N{N}.json") as f:
    mi_dict = json.load(f)

I_mat = np.zeros((N, N))
for k, v in mi_dict.items():
    i, j = map(int, k.split("_"))
    I_mat[i, j] = I_mat[j, i] = v

A = I_mat / np.max(I_mat)
np.fill_diagonal(A, 0)
D = np.diag(A.sum(axis=1))
L = D - A
evals = eigvalsh(L)
low = evals[1:8].tolist()
print(f"N={N} low evals (k=1..7): {np.round(low,6)}")

results = {}
import os
if os.path.exists("scaling_results.json"):
    with open("scaling_results.json") as f:
        results = json.load(f)
results[str(N)] = {"low_evals": low}
with open("scaling_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved to scaling_results.json")
