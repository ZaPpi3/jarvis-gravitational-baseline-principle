import sys, pickle, time, os
from dmrg_mi_laplacian import dmrg_ground_state

N = int(sys.argv[1])
fname = f"psi_N{N}.pkl"
if os.path.exists(fname):
    print(f"N={N}: psi already saved.")
    sys.exit(0)

t0 = time.time()
bd = (30, 60, 100) if N <= 20 else (40, 80, 150, 200)
psi = dmrg_ground_state(N, bond_dims=bd)
with open(fname, "wb") as f:
    pickle.dump(psi, f)
print(f"N={N}: DMRG done and saved in {time.time()-t0:.1f}s, max_bond={psi.max_bond()}")
