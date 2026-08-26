"""d1: c5 동일비용 비교 완성 — hubmass θ∈{0.45,0.60}+repair, GIST hub/random, 3 seeds"""
import numpy as np, time
from c_common import *
from c_repair import build_rev, repair_batch
t0=time.time(); out={}; BATCH,CUM=0.025,0.60
for seed in [0,1,2]:
    base,q = load("gist",seed); nbrs0,indeg0 = build_hnsw(base)
    n,cap = len(base), nbrs0.shape[1]; od = orders(base,indeg0,seed)
    for mode in ["hub","random"]:
        for theta in [0.45, 0.60]:
            rng=np.random.default_rng(1700+seed)
            nbrs=nbrs0.copy(); rev=build_rev(nbrs,n); alive=np.ones(n,bool)
            indeg_ref=indeg0.copy(); mass_ref=indeg_ref.sum()
            dm=0; rebuilds=0; recs=[]; ptr=0
            for st in range(int(CUM/BATCH)):
                newly=od[mode][ptr:ptr+int(BATCH*n)]; newly=newly[alive[newly]]
                alive[newly]=False
                repair_batch(base,nbrs,rev,alive,newly,cap)
                dm+=indeg_ref[newly].sum(); ptr+=int(BATCH*n)
                if dm/mass_ref>theta:
                    ids=np.where(alive)[0]; sub=np.ascontiguousarray(base[ids])
                    nbrs_s,indeg_s=build_hnsw(sub)
                    nbrs=np.full_like(nbrs0,-1)
                    conv=np.where(nbrs_s>=0, ids[np.clip(nbrs_s,0,None)], -1)
                    nbrs[ids,:nbrs_s.shape[1]]=conv
                    rev=build_rev(nbrs,n)
                    indeg_ref=np.zeros(n,np.int64); indeg_ref[ids]=indeg_s
                    mass_ref=indeg_ref.sum(); dm=0; rebuilds+=1
                recs.append(recall_at(base,alive,nbrs,q,rng))
            emax=1-min(recs)
            out[f"gist_{mode}_t{int(theta*100)}_s{seed}"]=np.array([emax,rebuilds])
            print(f"  s{seed} {mode:6s} theta={theta} emax={emax:.3f} rebuilds={rebuilds} ({time.time()-t0:.0f}s)",flush=True)
np.savez("../results/d1_theta.npz", **out)
print("===== d1 요약 (3s 평균) =====")
for mode in ["hub","random"]:
    for t in [45,60]:
        m=np.mean([out[f"gist_{mode}_t{t}_s{s}"] for s in [0,1,2]],axis=0)
        print(f"  gist {mode} theta=0.{t}: emax={m[0]:.3f} rebuilds={m[1]:.1f}")
print("  기준: fixed20+rep = 0.038/r2 — 동일 r에서 emax 비교")
print(f"[done] {time.time()-t0:.0f}s")
