"""e2b: 개수 스윕 — hub 스트림을 '원 차수 고정(static)'으로 (옛 3.9x 가 프로토콜 산물인지 확정)
e6 : 혼합 삽입 스트림 개수 스윕 (fixed 10/15/25 + mass 15/45 추가; 기존 d5의 never/fixed20/mass30·d5b mass45 와 합쳐 프런티어)"""
import numpy as np, time, faiss, h5py
from c_common import load, build_hnsw, recall_at, DATA, SAMPLE_Q
t0 = time.time()

def remap(sub, ids, n):
    g = np.full((n, sub.shape[1]), -1, np.int32)
    g[ids] = np.where(sub >= 0, ids[np.clip(sub, 0, None)], -1)
    return g

# ═══════════ e2b: static hub ═══════════
STEP, NSTEP = 0.025, 24
POL = ["fixed05", "fixed10", "fixed15", "fixed20", "fixed25", "mass15", "mass30"]
out = {}
def sim_static(ds, policy, seed):
    base, q = load(ds, seed); n = len(base)
    nbrs0, indeg0 = build_hnsw(base)
    nbrs = nbrs0.copy(); alive = np.ones(n, bool)
    order = np.argsort(-indeg0, kind="stable")          # 정적: 원 차수로 고정
    indeg_ref = indeg0.astype(np.int64); mass_ref = indeg_ref.sum(); n_ref = n
    dcount = dmass = rebuilds = 0; recs = []; ptr = 0
    for st in range(NSTEP):
        k = int(STEP * n)
        cand = order[ptr:ptr + k]; ptr += k
        alive[cand] = False; nbrs[cand] = -1
        dcount += k; dmass += indeg_ref[cand].sum()
        fire = (dcount / n_ref > int(policy[5:]) / 100) if policy.startswith("fixed") else (dmass / mass_ref > int(policy[4:]) / 100)
        if fire:
            ids = np.where(alive)[0]
            sub, dloc = build_hnsw(np.ascontiguousarray(base[ids]))
            nbrs = remap(sub, ids, n)
            indeg_ref = np.zeros(n, np.int64); indeg_ref[ids] = dloc
            mass_ref = indeg_ref.sum(); n_ref = alive.sum(); dcount = dmass = 0; rebuilds += 1
        recs.append(recall_at(base, alive, nbrs, q, np.random.default_rng(3000 + seed + st)))
    return 1 - min(recs), rebuilds
for ds, seeds in [("gist", [0, 1, 2, 3, 4]), ("sift", [0, 1, 2])]:
    for pol in POL:
        em, rb = [], []
        for s in seeds:
            e, r = sim_static(ds, pol, s); em.append(e); rb.append(r)
            out[f"{ds}_hubstatic_{pol}_s{s}"] = np.array([e, r])
        print(f"  [e2b] {ds} hub-static {pol:8s}: emax={np.mean(em):.3f}±{np.std(em):.3f} C_r={np.mean(rb):.1f} ({time.time()-t0:.0f}s)", flush=True)
np.savez("../results/e2b_count_static.npz", **out)
print("===== e2b 요약 (정적 hub 스트림) =====")
for ds, seeds in [("gist", [0, 1, 2, 3, 4]), ("sift", [0, 1, 2])]:
    for pol in POL:
        v = np.mean([out[f"{ds}_hubstatic_{pol}_s{s}"] for s in seeds], axis=0)
        print(f"  {ds} {pol:8s} C_r={v[1]:.1f} emax={v[0]:.3f}")

# ═══════════ e6: mixed-insertion count sweep ═══════════
N0, POOL_N, STEPS, DEL_R, INS_R = 80_000, 20_000, 24, 0.02, 0.01
def extract(index, ncur):
    h = index.hnsw
    neighbors = faiss.vector_to_array(h.neighbors)
    offsets = faiss.vector_to_array(h.offsets).astype(np.int64)
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
    n0 = int(cum[1]); idx = offsets[:-1][:, None] + np.arange(n0)[None, :]
    nb = neighbors[idx][:ncur]
    return np.where(nb < 0, np.int32(-1), nb).astype(np.int32)
with h5py.File(DATA["gist"], "r") as f:
    tr = np.array(f["train"], dtype=np.float32); te = np.array(f["test"], dtype=np.float32)
POL6 = ["fixed10", "fixed15", "fixed25", "mass15", "mass45"]
out6 = {}
for pol in POL6:
    em, rb = [], []
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(2100 + seed)
        sel = rng.choice(len(tr), N0 + POOL_N, replace=False)
        vecs = np.ascontiguousarray(tr[sel]); q = np.ascontiguousarray(te[:SAMPLE_Q])
        index = faiss.IndexHNSWFlat(vecs.shape[1], 16); index.hnsw.efConstruction = 200
        index.add(vecs[:N0]); ncur = N0
        alive = np.ones(N0 + POOL_N, bool); alive[N0:] = False; inserted = N0
        nbrs = extract(index, ncur)
        indeg_ref = np.bincount(nbrs[nbrs >= 0], minlength=N0 + POOL_N).astype(np.int64)
        mass_ref = indeg_ref.sum(); n_ref = N0
        dm = dc = 0; rebuilds = 0; recs = []; ins_enabled = True
        srng = np.random.default_rng(2200 + seed)
        for st in range(STEPS):
            k = int(DEL_R * alive.sum())
            cand = np.argsort(-np.where(alive, indeg_ref, -1), kind="stable")[:k]
            alive[cand] = False; dm += indeg_ref[cand].sum(); dc += k
            m = int(INS_R * n_ref)
            if ins_enabled and inserted + m <= N0 + POOL_N:
                index.add(vecs[inserted:inserted + m])
                alive[inserted:inserted + m] = True; inserted += m; ncur = inserted
                nbrs = extract(index, ncur)
            fire = (dc / n_ref > int(pol[5:]) / 100) if pol.startswith("fixed") else (dm / mass_ref > int(pol[4:]) / 100)
            if fire:
                ids = np.where(alive[:ncur])[0]
                index = faiss.IndexHNSWFlat(vecs.shape[1], 16); index.hnsw.efConstruction = 200
                index.add(np.ascontiguousarray(vecs[ids]))
                sub = extract(index, len(ids))
                nbrs = np.full((ncur, sub.shape[1]), -1, np.int32)
                nbrs[ids] = np.where(sub >= 0, ids[np.clip(sub, 0, None)], -1)
                indeg_ref = np.zeros(N0 + POOL_N, np.int64)
                indeg_ref[ids] = np.bincount(sub[sub >= 0], minlength=len(ids))
                mass_ref = indeg_ref.sum(); n_ref = alive.sum(); dm = dc = 0; rebuilds += 1
                ins_enabled = False
            recs.append(recall_at(vecs[:max(ncur, N0)], alive[:max(ncur, N0)], nbrs, q, srng))
        e = 1 - min(recs); em.append(e); rb.append(rebuilds)
        out6[f"gist_mixed_{pol}_s{seed}"] = np.array([e, rebuilds])
    print(f"  [e6] gist mixed {pol:8s}: emax={np.mean(em):.3f}±{np.std(em):.3f} C_r={np.mean(rb):.1f} ({time.time()-t0:.0f}s)", flush=True)
np.savez("../results/e6_mixed_count.npz", **out6)
print("===== e6 요약 (기존 d5: never 0.167/r0, fixed20 0.057/r2, mass30 0.025/r3 ; d5b: mass45 0.048/r2) =====")
for pol in POL6:
    v = np.mean([out6[f"gist_mixed_{pol}_s{s}"] for s in [0, 1, 2]], axis=0)
    print(f"  {pol:8s} C_r={v[1]:.1f} emax={v[0]:.3f}")
print(f"[done] {time.time()-t0:.0f}s")
