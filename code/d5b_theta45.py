"""d5b: 혼합 스트림에 hubmass45 정책 1점 추가 (fixed20의 r=2와 동일 비용 비교용), GIST, 3 seeds"""
import numpy as np, time, faiss, h5py
from c_common import DATA, SAMPLE_Q, recall_at
t0=time.time(); out={}
N0, POOL_N, STEPS, DEL_R, INS_R = 80_000, 20_000, 24, 0.02, 0.01
def extract(index, ncur):
    h = index.hnsw
    neighbors = faiss.vector_to_array(h.neighbors)
    offsets = faiss.vector_to_array(h.offsets).astype(np.int64)
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
    n0 = int(cum[1]); idx = offsets[:-1][:, None] + np.arange(n0)[None, :]
    nbrs = neighbors[idx][:ncur]
    return np.where(nbrs<0, np.int32(-1), nbrs).astype(np.int32)
with h5py.File(DATA["gist"],"r") as f:
    tr = np.array(f["train"],dtype=np.float32); te = np.array(f["test"],dtype=np.float32)
for seed in [0,1,2]:
    rng = np.random.default_rng(2100+seed)
    sel = rng.choice(len(tr), N0+POOL_N, replace=False)
    vecs = np.ascontiguousarray(tr[sel]); q = np.ascontiguousarray(te[:SAMPLE_Q])
    THETA = 0.45
    index = faiss.IndexHNSWFlat(vecs.shape[1],16); index.hnsw.efConstruction=200
    index.add(vecs[:N0]); ncur=N0
    alive=np.ones(N0+POOL_N,bool); alive[N0:]=False; inserted=N0
    nbrs=extract(index,ncur)
    indeg_ref=np.bincount(nbrs[nbrs>=0],minlength=N0+POOL_N).astype(np.int64)
    mass_ref=indeg_ref.sum(); n_ref=N0
    dm=0; rebuilds=0; recs=[]; ins_enabled=True
    srng=np.random.default_rng(2200+seed)
    for st in range(STEPS):
        k=int(DEL_R*alive.sum())
        cand=np.argsort(-np.where(alive, indeg_ref, -1), kind="stable")[:k]
        alive[cand]=False; dm+=indeg_ref[cand].sum()
        m=int(INS_R*n_ref)
        if ins_enabled and inserted+m <= N0+POOL_N:
            index.add(vecs[inserted:inserted+m])
            alive[inserted:inserted+m]=True; inserted+=m; ncur=inserted
            nbrs=extract(index,ncur)
        if dm/mass_ref>THETA:
            ids=np.where(alive[:ncur])[0]
            index=faiss.IndexHNSWFlat(vecs.shape[1],16); index.hnsw.efConstruction=200
            index.add(np.ascontiguousarray(vecs[ids]))
            sub=extract(index,len(ids))
            nbrs=np.full((ncur,sub.shape[1]),-1,np.int32)
            conv=np.where(sub>=0, ids[np.clip(sub,0,None)], -1)
            nbrs[ids]=conv
            indeg_ref=np.zeros(N0+POOL_N,np.int64)
            d_loc=np.bincount(sub[sub>=0],minlength=len(ids)); indeg_ref[ids]=d_loc
            mass_ref=indeg_ref.sum(); n_ref=alive.sum(); dm=0; rebuilds+=1
            ins_enabled=False
        recs.append(recall_at(vecs[:max(ncur,N0)], alive[:max(ncur,N0)], nbrs, q, srng))
    emax=1-min(recs)
    out[f"gist_hm45_s{seed}"]=np.array([emax,rebuilds])
    print(f"  s{seed} hubmass45 emax={emax:.3f} rebuilds={rebuilds} ({time.time()-t0:.0f}s)",flush=True)
np.savez("../results/d5b_theta45.npz", **out)
m=np.mean([out[f"gist_hm45_s{s}"] for s in [0,1,2]],axis=0)
print(f"===== d5b: gist 혼합 hubmass45 = emax {m[0]:.3f} / r{m[1]:.1f} (비교: fixed20 0.057/r2, hm30 0.025/r3) =====")
print(f"[done] {time.time()-t0:.0f}s")
