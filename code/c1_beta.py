"""c1: fc 의 탐색예산 의존성 — GIST β∈{512,1024,2048,4096}, SIFT β∈{512,2048} / hub·random / 3 seeds"""
import numpy as np, time
from c_common import *
t0 = time.time(); out = {}
PLAN = [("gist", [512, 1024, 2048, 4096]), ("sift", [512, 2048])]
for ds, betas in PLAN:
    for seed in [0, 1, 2]:
        base, q = load(ds, seed)
        nbrs, indeg = build_hnsw(base)
        rng = np.random.default_rng(500 + seed)
        od = orders(base, indeg, seed)
        for mode in ["random", "hub"]:
            for beta in betas:
                rec = []
                for f in F_GRID:
                    alive = np.ones(len(base), bool)
                    alive[od[mode][:int(f * len(base))]] = False
                    rec.append(recall_at(base, alive, nbrs, q, rng, beam=beta))
                fc = fc_from_curve(F_GRID, rec)
                if f == F_GRID[0] and rec[0] < 0.98:
                    print(f"  [selfcheck WARN] {ds} s{seed} {mode} b{beta} rec0={rec[0]:.3f}")
                out[f"{ds}_{mode}_b{beta}_s{seed}"] = np.array(rec)
                print(f"  [{ds}] s{seed} {mode:6s} beta={beta:4d} fc={fc} rec0={rec[0]:.3f} ({time.time()-t0:.0f}s)", flush=True)
np.savez("../results/c1_beta.npz", F=F_GRID, **out)
print("===== c1 요약: fc(beta) =====")
for ds, betas in PLAN:
    for mode in ["random", "hub"]:
        row = []
        for beta in betas:
            fcs = [fc_from_curve(F_GRID, out[f"{ds}_{mode}_b{beta}_s{s}"]) for s in [0,1,2]]
            fcs = [x for x in fcs if x is not None]
            row.append(f"b{beta}:{np.mean(fcs):.3f}±{np.std(fcs):.3f}" if fcs else f"b{beta}:>0.95")
        print(f"  {ds} {mode}: " + "  ".join(row))
print(f"[done] {time.time()-t0:.0f}s")
