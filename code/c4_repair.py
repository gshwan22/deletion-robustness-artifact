"""c4: 로컬 엣지 수리(FreshDiskANN식 delete-consolidation) 하에서 허브 취약성이 잔존하는가
수리 규칙: v 삭제 시, v의 생존 in-이웃 u마다 후보 = (u의 생존 out-이웃) ∪ (v의 생존 out-이웃),
거리 상위 cap(=원 out-degree 폭)으로 절단 재연결. 스윕을 따라 그래프가 누적 진화."""
import numpy as np, time
from c_common import *
t0 = time.time(); out = {}

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
            if v not in row: continue          # 이미 다른 수리로 끊김
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
            row[:] = -1
            row[:len(keep)] = keep
            for w in keep:
                rev[w].append(int(u))
            rewritten += 1
    return rewritten

for ds in ["sift", "gist"]:
    for seed in [0, 1, 2]:
        base, q = load(ds, seed)
        nbrs0, indeg = build_hnsw(base)
        n, cap = len(base), nbrs0.shape[1]
        rng = np.random.default_rng(900 + seed)
        od = orders(base, indeg, seed)
        for mode in ["random", "hub"]:
            for rep in [False, True]:
                nbrs = nbrs0.copy()
                rev = build_rev(nbrs, n) if rep else None
                alive = np.ones(n, bool)
                rec, rew_total, prev = [], 0, 0
                for f in F_GRID:
                    ndel = int(f * n)
                    newly = od[mode][prev:ndel]
                    alive[newly] = False
                    if rep and len(newly):
                        rew_total += repair_batch(base, nbrs, rev, alive, newly, cap)
                    prev = ndel
                    rec.append(recall_at(base, alive, nbrs, q, rng))
                fc = fc_from_curve(F_GRID, rec)
                tag = f"{ds}_{mode}_{'rep' if rep else 'norep'}_s{seed}"
                out[tag] = np.array(rec); out[tag + "_rew"] = np.array([rew_total])
                if rec[0] < 0.98:
                    print(f"  [selfcheck WARN] {tag} rec0={rec[0]:.3f}")
                print(f"  [{ds}] s{seed} {mode:6s} repair={int(rep)} fc={fc} rec@0.5={rec[10]:.3f} rewrites={rew_total} ({time.time()-t0:.0f}s)", flush=True)
np.savez("../results/c4_repair.npz", F=F_GRID, **out)
print("===== c4 요약: fc (no-repair -> repair) =====")
for ds in ["sift", "gist"]:
    for mode in ["random", "hub"]:
        line = []
        for rep in ["norep", "rep"]:
            fcs = [fc_from_curve(F_GRID, out[f"{ds}_{mode}_{rep}_s{s}"]) for s in [0,1,2]]
            fcs = [x for x in fcs if x is not None]
            line.append(f"{rep}:{np.mean(fcs):.3f}±{np.std(fcs):.3f}" if fcs else f"{rep}:>0.95")
        print(f"  {ds} {mode}: " + " -> ".join(line))
print("  판정: hub의 rep-norep 격차가 random 대비 얼마나 좁혀지는가 = 수리의 허브 취약성 완화 정도")
print(f"[done] {time.time()-t0:.0f}s")
