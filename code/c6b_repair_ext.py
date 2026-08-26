"""c6: 수리 결과의 일반화 — (a) textbge 지식 코퍼스 수리 on/off (b) GIST adaptive×repair 군비경쟁
(c) c4의 gist hub/random 을 5-seed 로 승격 (seed 3,4 추가분)"""
import numpy as np, time
from c_common import *

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

def run_sweep(base, q, nbrs0, order_fn, rep, seed, adaptive=False):
    n, cap = len(base), nbrs0.shape[1]
    rng = np.random.default_rng(1300 + seed)
    nbrs = nbrs0.copy()
    rev = build_rev(nbrs, n) if rep else None
    alive = np.ones(n, bool)
    rec, prev = [], 0
    for f in F_GRID:
        ndel = int(f * n)
        newly = np.array([], dtype=np.int64)
        if adaptive:
            # 현재 그래프(수리 반영)의 in-degree 상위에서 삭제
            need = ndel - prev
            if need > 0:
                indeg_now = np.bincount(nbrs[alive][nbrs[alive] >= 0], minlength=n)
                indeg_now[~alive] = -1
                newly = np.argsort(-indeg_now, kind="stable")[:need]
        else:
            newly = order_fn[prev:ndel]
            newly = newly[alive[newly]]
        if len(newly):
            alive[newly] = False
            if rep:
                repair_batch(base, nbrs, rev, alive, newly, cap)
        prev = ndel
        rec.append(recall_at(base, alive, nbrs, q, rng))
    return rec

t0 = time.time(); out = {}
# (a) textbge — 완료분 스킵
for seed in []:
    base, q = load("textbge", seed)
    nbrs0, indeg = build_hnsw(base)
    od = orders(base, indeg, seed)
    for mode in ["random", "hub"]:
        for rep in [False, True]:
            rec = run_sweep(base, q, nbrs0, od[mode], rep, seed)
            fc = fc_from_curve(F_GRID, rec)
            out[f"textbge_{mode}_{'rep' if rep else 'norep'}_s{seed}"] = np.array(rec)
            print(f"  [textbge] s{seed} {mode:6s} repair={int(rep)} fc={fc} ({time.time()-t0:.0f}s)", flush=True)
# (b) GIST adaptive × repair
for seed in [0, 1, 2]:
    base, q = load("gist", seed)
    nbrs0, indeg = build_hnsw(base)
    for rep in [False, True]:
        rec = run_sweep(base, q, nbrs0, None, rep, seed, adaptive=True)
        fc = fc_from_curve(F_GRID, rec)
        out[f"gist_adaptive_{'rep' if rep else 'norep'}_s{seed}"] = np.array(rec)
        print(f"  [gist] s{seed} adaptive repair={int(rep)} fc={fc} ({time.time()-t0:.0f}s)", flush=True)
# (c) c4 gist 5-seed 승격분 (seed 3,4)
for seed in [3, 4]:
    base, q = load("gist", seed)
    nbrs0, indeg = build_hnsw(base)
    od = orders(base, indeg, seed)
    for mode in ["random", "hub"]:
        for rep in [False, True]:
            rec = run_sweep(base, q, nbrs0, od[mode], rep, seed)
            fc = fc_from_curve(F_GRID, rec)
            out[f"gist_{mode}_{'rep' if rep else 'norep'}_s{seed}"] = np.array(rec)
            print(f"  [gist] s{seed} {mode:6s} repair={int(rep)} fc={fc} ({time.time()-t0:.0f}s)", flush=True)
np.savez("../results/c6b_repair_ext.npz", F=F_GRID, **out)
print("===== c6 요약 =====")
for key_ds, modes in [("textbge", ["random", "hub"]), ("gist", ["adaptive"])]:
    for mode in modes:
        line = []
        for rep in ["norep", "rep"]:
            fcs = [fc_from_curve(F_GRID, out[f"{key_ds}_{mode}_{rep}_s{s}"]) for s in [0,1,2]]
            fcs2 = [x for x in fcs if x is not None]
            line.append(f"{rep}:{np.mean(fcs2):.3f}" if fcs2 else f"{rep}:>0.95")
        print(f"  {key_ds} {mode}: " + " -> ".join(line))
print("  판정: textbge(저-h)는 수리로 치유되는지 / adaptive 는 수리를 뚫는지")
print(f"[done] {time.time()-t0:.0f}s")
