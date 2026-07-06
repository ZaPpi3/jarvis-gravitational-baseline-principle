import sys, pickle, time, os, json
import numpy as np
import quimb as qu

N = int(sys.argv[1])
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 40

psi_fname = f"psi_N{N}.pkl"
mi_fname = f"mi_partial_N{N}.json"

with open(psi_fname, "rb") as f:
    psi = pickle.load(f)

pairs_all = [(i, j) for i in range(N) for j in range(i + 1, N)]

mi_dict = {}
if os.path.exists(mi_fname):
    with open(mi_fname) as f:
        mi_dict = json.load(f)

done_pairs = set(tuple(map(int, k.split("_"))) for k in mi_dict.keys())
remaining = [p for p in pairs_all if p not in done_pairs]

_LN2 = np.log(2.0)
t0 = time.time()
n_done_this_run = 0
for (i, j) in remaining:
    rho_ij = psi.partial_trace_to_mpo([i, j]).to_dense()
    rho_i = qu.partial_trace(rho_ij, [2, 2], keep=[0])
    rho_j = qu.partial_trace(rho_ij, [2, 2], keep=[1])
    mi = (qu.entropy(rho_i) + qu.entropy(rho_j) - qu.entropy(rho_ij)) * _LN2
    mi_dict[f"{i}_{j}"] = mi
    n_done_this_run += 1
    if n_done_this_run >= batch_size:
        break

with open(mi_fname, "w") as f:
    json.dump(mi_dict, f)

total_pairs = len(pairs_all)
print(f"N={N}: {len(mi_dict)}/{total_pairs} pairs done "
      f"({n_done_this_run} this run, {time.time()-t0:.1f}s)")
if len(mi_dict) == total_pairs:
    print("ALL PAIRS COMPLETE")
