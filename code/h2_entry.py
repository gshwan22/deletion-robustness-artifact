import numpy as np, c_common
from c_common import load, build_hnsw, orders, recall_at
print("===== h2: 단일 고정 진입점 편향 (R=1 vs R=16) =====", flush=True)
for ds in ["sift", "gist"]:
    base, q = load(ds, 0); n = len(base)
    nbrs0, indeg = build_hnsw(base); od = orders(base, indeg, 0)
    for nm in ["hub", "random"]:
        for R in [1, 16]:
            c_common.R_ENTRY = R
            recs = []
            for f in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                k = int(f * n); alive = np.ones(n, bool); nbrs = nbrs0.copy()
                d = od[nm][:k]; alive[d] = False; nbrs[d] = -1
                recs.append(recall_at(base, alive, nbrs, q, np.random.default_rng(11)))
            print(f"  [{ds} {nm:6s} R={R:2d}] f=0/.2/.3/.4/.5/.6/.7: " + " ".join(f"{x:.3f}" for x in recs), flush=True)
c_common.R_ENTRY = 16
