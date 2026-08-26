"""d6: 수리의 h-지배가 MRNG 계열에서도 성립하는가 — MRNG-GIST(h~40) hub/random × repair on/off, 2 seeds"""
import numpy as np, time, faiss
from c_common import load, F_GRID, recall_at, fc_from_curve, orders
from c_repair import build_rev, repair_batch
t0=time.time(); out={}

def build_mrng(base, K=64, cap=32):
    n,d = base.shape
    flat = faiss.IndexFlatL2(d); flat.add(base)
    _, I = flat.search(base, K+1)          # self 포함
    nbrs = np.full((n,cap), -1, np.int32)
    for i in range(n):
        cand = [c for c in I[i] if c != i][:K]
        cvec = base[cand]
        dic = ((cvec - base[i])**2).sum(1)
        order = np.argsort(dic)
        kept = []
        for oi in order:
            c = cand[oi]; dc = dic[oi]
            ok = True
            for k in kept:
                if ((base[c]-base[k])**2).sum() < dc:   # occlusion
                    ok = False; break
            if ok:
                kept.append(c)
                if len(kept) >= cap: break
        nbrs[i,:len(kept)] = kept
        if (i+1) % 20000 == 0:
            print(f"    mrng build {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    indeg = np.bincount(nbrs[nbrs>=0], minlength=n).astype(np.int64)
    return nbrs, indeg

for seed in [0,1]:
    base,q = load("gist",seed)
    nbrs0, indeg = build_mrng(base)
    h = indeg.max()/indeg.mean()
    print(f"  [mrng-gist] s{seed} h={h:.1f}", flush=True)
    n,cap = len(base), nbrs0.shape[1]
    od = orders(base, indeg, seed)
    rng0 = np.random.default_rng(1900+seed)
    for mode in ["random","hub"]:
        for rep in [False,True]:
            rng = np.random.default_rng(1900+seed)
            nbrs = nbrs0.copy(); rev = build_rev(nbrs,n) if rep else None
            alive = np.ones(n,bool); prev=0; rec=[]
            for f in F_GRID:
                ndel=int(f*n); newly=od[mode][prev:ndel]; newly=newly[alive[newly]]
                if len(newly):
                    alive[newly]=False
                    if rep: repair_batch(base,nbrs,rev,alive,newly,cap)
                prev=ndel
                rec.append(recall_at(base,alive,nbrs,q,rng))
            fc = fc_from_curve(F_GRID, rec)
            out[f"mrnggist_{mode}_{'rep' if rep else 'norep'}_s{seed}"]=np.array(rec)
            print(f"  [mrng-gist] s{seed} {mode:6s} repair={int(rep)} fc={fc} ({time.time()-t0:.0f}s)",flush=True)
np.savez("../results/d6_mrng_repair.npz", F=F_GRID, **out)
print("===== d6 요약 =====")
for mode in ["random","hub"]:
    line=[]
    for rep in ["norep","rep"]:
        fcs=[fc_from_curve(F_GRID,out[f"mrnggist_{mode}_{rep}_s{s}"]) for s in [0,1]]
        fcs2=[x for x in fcs if x is not None]
        line.append(f"{rep}:{np.mean(fcs2):.3f}" if fcs2 else f"{rep}:>0.95")
    print(f"  mrng-gist {mode}: "+" -> ".join(line))
print("  판정: 극단-h(~40)에서 수리 불충분이 재현되면 h-지배의 계열 일반화")
print(f"[done] {time.time()-t0:.0f}s")
