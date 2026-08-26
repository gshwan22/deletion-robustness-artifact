"""c7: 저-f 정밀 그리드 (0~0.20, step 0.02) — hub/random × repair on/off × SIFT/GIST, 3 seeds
EDBT 저-f 표('5/10/20% 삭제에서 무엇을 잃는가') 재료"""
import numpy as np, time
import c_common as C
from c_common import *
C_F = np.round(np.arange(0.0, 0.21, 0.02), 2)

def build_rev(nbrs, n):
    rev = [[] for _ in range(n)]
    src, dst = np.where(nbrs >= 0)
    for u, j in zip(src, dst):
        rev[nbrs[u, j]].append(int(u))
    return rev

def repair_batch(base, nbrs, rev, alive, newly, cap):
    rewritten = 0
    for v in newly:
        outs_v = [int(w) for w in nbrs[v] if w >= 0 and alive[w]]
        for u in rev[v]:
            if not alive[u]: continue
            row = nbrs[u]
            if v not in row: continue
            cand = [int(w) for w in row if w >= 0 and alive[w] and w != v]
            cs = set(cand)
            for w in outs_v:
                if w != u and w not in cs:
                    cand.append(w); cs.add(w)
            if cand:
                d = ((base[cand] - base[u]) ** 2).sum(1)
                keep = [cand[i] for i in np.argsort(d)[:cap]]
            else:
                keep = []
            row[:] = -1; row[:len(keep)] = keep
            for w in keep: rev[w].append(int(u))
            rewritten += 1
    return rewritten

t0 = time.time(); out = {}
for ds in ["sift", "gist"]:
    for seed in [0, 1, 2]:
        base, q = load(ds, seed)
        nbrs0, indeg = build_hnsw(base)
        n, cap = len(base), nbrs0.shape[1]
        od = orders(base, indeg, seed)
        for mode in ["random", "hub"]:
            for rep in [False, True]:
                rng = np.random.default_rng(1500 + seed)
                nbrs = nbrs0.copy()
                rev = build_rev(nbrs, n) if rep else None
                alive = np.ones(n, bool); prev = 0; rec = []
                for f in C_F:
                    ndel = int(f * n)
                    newly = od[mode][prev:ndel]; newly = newly[alive[newly]]
                    if len(newly):
                        alive[newly] = False
                        if rep: repair_batch(base, nbrs, rev, alive, newly, cap)
                    prev = ndel
                    rec.append(recall_at(base, alive, nbrs, q, rng))
                out[f"{ds}_{mode}_{'rep' if rep else 'norep'}_s{seed}"] = np.array(rec)
                print(f"  [{ds}] s{seed} {mode:6s} repair={int(rep)} rec@0.05/0.10/0.20="
                      f"{rec[2]:.3f}/{rec[5]:.3f}/{rec[10]:.3f} ({time.time()-t0:.0f}s)", flush=True)
np.savez("../results/c7_lowf.npz", F=C_F, **out)
print("===== c7 요약: recall @ f=0.05/0.10/0.20 (3s 평균) =====")
for ds in ["sift", "gist"]:
    for mode in ["random", "hub"]:
        for rep in ["norep", "rep"]:
            m = np.mean([out[f"{ds}_{mode}_{rep}_s{s}"] for s in [0,1,2]], axis=0)
            print(f"  {ds} {mode:6s} {rep:5s}: {m[2]:.3f} / {m[5]:.3f} / {m[10]:.3f}")
print(f"[done] {time.time()-t0:.0f}s")
