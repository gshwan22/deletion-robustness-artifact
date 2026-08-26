"""c2: 대안 예측자 비교 — h, k99비, top1% 질량, Gini, CV vs 기존 Δfc (HNSW 4코퍼스)"""
import numpy as np, time
from c_common import *
t0 = time.time()
KNOWN_DFC = {"sift": 0.136, "glove": 0.118, "gist": 0.539, "textbge": 0.123}
DATA["glove"] = "../data/glove-100-angular.hdf5"
rows = {}
for ds in ["sift", "glove", "gist", "textbge"]:
    stats = []
    for seed in [0, 1, 2]:
        base, _ = load(ds, seed)
        if ds == "glove":
            base = base / np.linalg.norm(base, axis=1, keepdims=True)
        nbrs, indeg = build_hnsw(base)
        k = np.sort(indeg)[::-1]; tot = k.sum(); n = len(k)
        h = k[0] / k.mean()
        k99 = np.percentile(indeg, 99) / k.mean()
        top1 = k[:n//100].sum() / tot
        gini = (2*np.sum((np.arange(1,n+1))*np.sort(indeg)) / (n*tot)) - (n+1)/n
        cv = indeg.std() / indeg.mean()
        stats.append((h, k99, top1, gini, cv))
        print(f"  [{ds}] s{seed} h={h:.1f} k99r={k99:.2f} top1%={top1:.3f} gini={gini:.3f} cv={cv:.2f} ({time.time()-t0:.0f}s)", flush=True)
    rows[ds] = np.mean(stats, axis=0)
np.savez("../results/c2_tail.npz", **{d: v for d, v in rows.items()})
print("===== c2 요약: 예측자 후보 vs Δfc =====")
names = ["h", "k99_ratio", "top1%mass", "gini", "cv"]
dss = list(rows)
dfc = np.array([KNOWN_DFC[d] for d in dss])
for j, nm in enumerate(names):
    vals = np.array([rows[d][j] for d in dss])
    rho = np.corrcoef(np.argsort(np.argsort(vals)), np.argsort(np.argsort(dfc)))[0,1]
    print(f"  {nm:10s} " + " ".join(f"{d}:{rows[d][j]:.3f}" for d in dss) + f"  | rank-corr(Δfc)={rho:.2f}")
print(f"[done] {time.time()-t0:.0f}s")
