#!/usr/bin/env python3
"""
figdata2 : Fig1 용 5모드 recall(f) + S(f) 곡선, 3 seeds (mean/std)
-> results/figdata_curves5_v2.npz
"""
import time
import numpy as np
import h5py
import faiss
import heapq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DATA_PATH = "../data/sift-128-euclidean.hdf5"
N = 100_000; M = 16; EF_CONSTR = 200; EF = 512; K = 10
SAMPLE_Q = 300; R_ENTRIES = 16; T_REL = 50; N_CLUSTERS = 400
F_GRID = np.round(np.linspace(0.0, 0.95, 20), 3)
MODES = ["random", "hub", "clustered", "antihub", "relevance"]
SEEDS = [0, 1, 2, 3, 4]


def load():
    with h5py.File(DATA_PATH, "r") as f:
        full = np.asarray(f["train"], dtype=np.float32)
        q = np.asarray(f["test"], dtype=np.float32)
    return full, np.ascontiguousarray(q[:SAMPLE_Q])


def build(base):
    n, d = base.shape
    ix = faiss.IndexHNSWFlat(d, M); ix.hnsw.efConstruction = EF_CONSTR; ix.add(base)
    h = ix.hnsw
    nb = faiss.vector_to_array(h.neighbors)
    off = faiss.vector_to_array(h.offsets).astype(np.int64)
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
    n0 = int(cum[1])
    idx = off[:-1][:, None] + np.arange(n0)[None, :]
    g = nb[idx]; g = np.where(g < 0, np.int32(-1), g).astype(np.int32)
    indeg = np.bincount(g[g >= 0], minlength=n).astype(np.int64)
    rows = np.repeat(np.arange(n), n0); cols = g.ravel(); mk = cols >= 0
    A = coo_matrix((np.ones(mk.sum(), np.uint8), (rows[mk], cols[mk])), shape=(n, n)).tocsr()
    A = A + A.T; A.data[:] = 1
    return g, indeg, A


def sqd(x, q):
    d = x - q
    return float(d @ d)


def beam(qv, g, alive, base, ent, ef, k):
    cand, res, seen = [], [], set()
    for e in ent:
        if alive[e] and e not in seen:
            de = sqd(base[e], qv)
            heapq.heappush(cand, (de, e)); heapq.heappush(res, (-de, e)); seen.add(e)
    if not cand:
        return []
    while cand:
        d, u = heapq.heappop(cand)
        if d > -res[0][0] and len(res) >= ef:
            break
        for v in g[u]:
            if v < 0 or v in seen or not alive[v]:
                continue
            seen.add(v)
            dv = sqd(base[v], qv)
            if len(res) < ef or dv < -res[0][0]:
                heapq.heappush(cand, (dv, v)); heapq.heappush(res, (-dv, v))
                if len(res) > ef:
                    heapq.heappop(res)
    return [i for _, i in sorted((-nd, ni) for nd, ni in res)[:k]]


t0 = time.time()
full, q = load()
rec_all = {m: [] for m in MODES}
S_all = {m: [] for m in MODES}
for seed in SEEDS:
    rng0 = np.random.default_rng(1000 + seed)
    base = np.ascontiguousarray(full[rng0.choice(full.shape[0], size=N, replace=False)])
    g, indeg, A = build(base)
    ixf = faiss.IndexFlatL2(base.shape[1]); ixf.add(base)
    _, Irel = ixf.search(q, T_REL)
    relscore = np.bincount(Irel.ravel(), minlength=N).astype(np.int64)
    km = faiss.Kmeans(base.shape[1], N_CLUSTERS, niter=10, seed=seed, verbose=False)
    km.train(base); _, a = km.index.search(base, 1); a = a.ravel()
    rr = np.random.default_rng(seed)
    orders = {
        "random": rr.permutation(N),
        "hub": np.argsort(-indeg, kind="stable"),
        "antihub": np.argsort(indeg, kind="stable"),
        "relevance": np.argsort(-relscore, kind="stable"),
        "clustered": np.concatenate([np.where(a == c)[0]
                                     for c in rr.permutation(N_CLUSTERS)]).astype(np.int64),
    }
    rng = np.random.default_rng(5000 + seed)
    for m in MODES:
        od = orders[m]
        rec, Ss = [], []
        for f in F_GRID:
            alive = np.ones(N, bool); alive[od[:int(round(f * N))]] = False
            ids = np.where(alive)[0]
            sub = A[ids][:, ids]
            _, lab = connected_components(sub, directed=False)
            Ss.append(np.bincount(lab).max() / ids.size)
            ent = rng.choice(ids, size=min(R_ENTRIES, ids.size), replace=False)
            ixo = faiss.IndexFlatL2(base.shape[1]); ixo.add(np.ascontiguousarray(base[ids]))
            _, loc = ixo.search(q, K); ora = ids[loc]
            hit = sum(len(set(beam(q[i], g, alive, base, ent, EF, K)).intersection(ora[i]))
                      for i in range(q.shape[0]))
            rec.append(hit / (q.shape[0] * K))
        rec_all[m].append(rec); S_all[m].append(Ss)
        print(f"  seed{seed} [{m}] rec_last={rec[-1]:.3f} S_last={Ss[-1]:.3f}")

out = {"F": F_GRID}
for m in MODES:
    out[f"{m}_rec_mu"] = np.mean(rec_all[m], axis=0)
    out[f"{m}_rec_sd"] = np.std(rec_all[m], axis=0)
    out[f"{m}_S_mu"] = np.mean(S_all[m], axis=0)
    out[f"{m}_S_sd"] = np.std(S_all[m], axis=0)
np.savez_compressed("../results/figdata_curves5_5s.npz", **out)
print(f"[done] {time.time()-t0:.1f}s -> figdata_curves5_v2.npz")
