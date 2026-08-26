"""EDBT 보강 실험 공통 모듈 (b10 검증 패턴 기반)"""
import numpy as np, h5py, faiss, heapq, time

DATA = {"sift": "../data/sift-128-euclidean.hdf5",
        "gist": "../data/gist-960-euclidean.hdf5",
        "textbge": "../data/text-bge-100k.hdf5"}
N_BASE, SAMPLE_Q, K, R_ENTRY = 100_000, 300, 10, 16
F_GRID = np.round(np.arange(0.0, 0.96, 0.05), 2)

def load(ds, seed):
    with h5py.File(DATA[ds], "r") as f:
        tr = np.array(f["train"], dtype=np.float32)
        te = np.array(f["test"], dtype=np.float32)
    if ds == "textbge":
        base = tr[:N_BASE] if len(tr) >= N_BASE else tr
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(tr), N_BASE, replace=False))
        base = tr[idx]
    q = te[:SAMPLE_Q].copy()
    return np.ascontiguousarray(base), np.ascontiguousarray(q)

def build_hnsw(base, M=16, efc=200):
    n, d = base.shape
    index = faiss.IndexHNSWFlat(d, M); index.hnsw.efConstruction = efc
    index.add(base); h = index.hnsw
    neighbors = faiss.vector_to_array(h.neighbors)
    offsets = faiss.vector_to_array(h.offsets).astype(np.int64)
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
    n0 = int(cum[1])
    idx = offsets[:-1][:, None] + np.arange(n0)[None, :]
    nbrs = neighbors[idx]
    nbrs = np.where(nbrs < 0, np.int32(-1), nbrs).astype(np.int32)
    indeg = np.bincount(nbrs[nbrs >= 0], minlength=n).astype(np.int64)
    return np.ascontiguousarray(nbrs), indeg

def beam_search(base, alive, nbrs, q, entries, beam, k):
    d0 = ((base[entries] - q) ** 2).sum(1)
    cand = [(d0[i], int(e)) for i, e in enumerate(entries) if alive[e]]
    heapq.heapify(cand)
    best, seen = [], set(e for _, e in cand)
    frontier = list(cand)
    while frontier:
        d, u = heapq.heappop(frontier)
        if best and len(best) >= beam and d > -best[0][0]:
            break
        heapq.heappush(best, (-d, u))
        if len(best) > beam: heapq.heappop(best)
        for v in nbrs[u]:
            if v < 0 or v in seen or not alive[v]: continue
            seen.add(int(v))
            dv = ((base[v] - q) ** 2).sum()
            heapq.heappush(frontier, (dv, int(v)))
    return [u for _, u in sorted([(-nd, u) for nd, u in best])[:k]]

def oracle_topk(base, alive, queries, k):
    ids = np.where(alive)[0]
    flat = faiss.IndexFlatL2(base.shape[1]); flat.add(base[ids])
    _, I = flat.search(queries, k)
    return [set(ids[row]) for row in I]

def recall_at(base, alive, nbrs, queries, rng, beam=512, k=K):
    gold = oracle_topk(base, alive, queries, k)
    pool = rng.choice(len(base), 8 * R_ENTRY, replace=False)
    entries = [int(i) for i in pool if alive[i]][:R_ENTRY]
    hits = 0
    for qi in range(len(queries)):
        top = beam_search(base, alive, nbrs, queries[qi], entries, beam, k)
        hits += len(set(top) & gold[qi])
    return hits / (len(queries) * k)

def fc_from_curve(F, rec):
    half = rec[0] / 2
    for i in range(len(rec)):
        if rec[i] < half:
            if i == 0: return 0.0
            f0, f1, r0, r1 = F[i-1], F[i], rec[i-1], rec[i]
            return float(f0 + (r0 - half) / (r0 - r1) * (f1 - f0))
    return None  # no collapse

def orders(base, indeg, seed):
    rng = np.random.default_rng(100 + seed)
    return {"random": rng.permutation(len(base)),
            "hub": np.argsort(-indeg, kind="stable"),
            "oldest": np.arange(len(base)),
            "newest": np.arange(len(base))[::-1].copy()}
