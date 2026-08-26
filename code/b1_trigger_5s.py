#!/usr/bin/env python3
"""
ANN-Percolation — Phase 4b : trigger v2 (위험이 실재하는 레짐)
=============================================================
v1 문제: SIFT + indeg^2 확률가중 + F_MAX 0.6 으로는 never 조차 안 무너짐(min 0.983).
        위험이 없는데 정책 비교 -> 안전성 주장 불성립.
v2 수정:
  - hubtargeted 시나리오 추가: 결정적 상위 in-degree 우선 삭제
    (GDPR 일괄삭제/인기항목 회전/adversarial 의 현실 표현)
  - GIST 추가 (fc_hub=0.372 -> F_MAX 0.6 안에서 위험 실재)
  - 정책: never / fixed20 / hubmass(θ=0.15, 0.30)
  - 채점: min recall(스트림 전체) + rebuild 수 + 최저점 발생 f
기대: GIST×hubtargeted 에서 never·fixed20 침몰, hubmass 방어.
"""
import time
import numpy as np
import h5py
import faiss
import heapq

DATASETS = {
    "sift": "../data/sift-128-euclidean.hdf5",
    "gist": "../data/gist-960-euclidean.hdf5",
}
N         = 100_000
M         = 16
EF_CONSTR = 200
EF        = 512
K         = 10
SAMPLE_Q  = 200
R_ENTRIES = 16
BATCH     = 0.025
F_MAX     = 0.60
THETAS    = [0.15, 0.30]
N_CLUSTERS = 400
SEEDS     = [0, 1, 2, 3, 4]
SCENARIOS = ["random", "hubtargeted", "clustered"]


def load(name):
    with h5py.File(DATASETS[name], "r") as f:
        full = np.asarray(f["train"], dtype=np.float32)
        query = np.asarray(f["test"], dtype=np.float32)
    return full, np.ascontiguousarray(query[:SAMPLE_Q])


def subsample(full, seed):
    rng = np.random.default_rng(1000 + seed)
    return np.ascontiguousarray(full[rng.choice(full.shape[0], size=N, replace=False)])


def build_graph(vecs):
    n, d = vecs.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = EF_CONSTR
    index.add(vecs)
    h = index.hnsw
    neighbors = faiss.vector_to_array(h.neighbors).astype(np.int64)
    offsets   = faiss.vector_to_array(h.offsets).astype(np.int64)
    cum       = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
    n0 = int(cum[1])
    idx = offsets[:-1][:, None] + np.arange(n0)[None, :]
    nbrs0 = np.where(neighbors[idx] < 0, -1, neighbors[idx]).astype(np.int32)
    indeg = np.bincount(nbrs0[nbrs0 >= 0], minlength=n).astype(np.int64)
    return nbrs0, indeg


def sqd(x, q):
    diff = x - q
    return float(diff @ diff)


def beam_multi(q, nbrs, alive, vecs, entries, ef, k):
    cand, res, seen = [], [], set()
    for e in entries:
        if alive[e] and e not in seen:
            de = sqd(vecs[e], q)
            heapq.heappush(cand, (de, e)); heapq.heappush(res, (-de, e)); seen.add(e)
    if not cand:
        return []
    while cand:
        d, u = heapq.heappop(cand)
        if d > -res[0][0] and len(res) >= ef:
            break
        for v in nbrs[u]:
            if v < 0 or v in seen or not alive[v]:
                continue
            seen.add(v)
            dv = sqd(vecs[v], q)
            if len(res) < ef or dv < -res[0][0]:
                heapq.heappush(cand, (dv, v)); heapq.heappush(res, (-dv, v))
                if len(res) > ef:
                    heapq.heappop(res)
    return [ni for _, ni in sorted((-nd, ni) for nd, ni in res)[:k]]


def measure_recall(vecs, nbrs, alive, query, rng):
    ids = np.where(alive)[0]
    idxf = faiss.IndexFlatL2(vecs.shape[1])
    idxf.add(np.ascontiguousarray(vecs[ids]))
    _, loc = idxf.search(query, K)
    ora = ids[loc]
    ent = rng.choice(ids, size=min(R_ENTRIES, ids.size), replace=False)
    hit = sum(len(set(beam_multi(query[i], nbrs, alive, vecs, ent, EF, K))
                  .intersection(ora[i])) for i in range(query.shape[0]))
    return hit / (query.shape[0] * K)


def stream_order(vecs, indeg, scenario, seed):
    rng = np.random.default_rng(seed)
    n = vecs.shape[0]
    if scenario == "random":
        return rng.permutation(n)
    if scenario == "hubtargeted":                       # 결정적 상위차수 우선
        return np.argsort(-indeg, kind="stable")
    if scenario == "clustered":
        km = faiss.Kmeans(vecs.shape[1], N_CLUSTERS, niter=10, seed=seed, verbose=False)
        km.train(vecs)
        _, a = km.index.search(vecs, 1); a = a.ravel()
        return np.concatenate([np.where(a == c)[0]
                               for c in rng.permutation(N_CLUSTERS)]).astype(np.int64)
    raise ValueError(scenario)


def run(vecs0, query, scenario, policy, theta, seed):
    rng = np.random.default_rng(9000 + seed)
    vecs = vecs0.copy()
    nbrs, indeg = build_graph(vecs)
    alive = np.ones(vecs.shape[0], bool)
    glob_order = stream_order(vecs0, indeg, scenario, seed)
    glob_deleted = np.zeros(vecs0.shape[0], bool)
    glob2loc = np.arange(vecs0.shape[0])

    n_batches = int(round(F_MAX / BATCH))
    batch_sz = int(round(BATCH * vecs0.shape[0]))
    ptr = 0
    rebuilds = 0
    recalls = []
    for b in range(n_batches):
        cnt = 0
        while cnt < batch_sz and ptr < len(glob_order):
            g = glob_order[ptr]; ptr += 1
            if glob_deleted[g]:
                continue
            glob_deleted[g] = True
            l = glob2loc[g]
            if l >= 0:
                alive[l] = False
            cnt += 1
        deleted_mass = indeg[~alive].sum() / max(indeg.sum(), 1)
        deleted_cnt = (~alive).mean()
        do_rebuild = ((policy == "fixed20" and deleted_cnt >= 0.20) or
                      (policy == "hubmass" and deleted_mass >= theta))
        if do_rebuild:
            surv_glob = np.where(~glob_deleted)[0]
            vecs = np.ascontiguousarray(vecs0[surv_glob])
            nbrs, indeg = build_graph(vecs)
            alive = np.ones(vecs.shape[0], bool)
            glob2loc = -np.ones(vecs0.shape[0], np.int64)
            glob2loc[surv_glob] = np.arange(surv_glob.size)
            rebuilds += 1
        recalls.append(measure_recall(vecs, nbrs, alive, query, rng))
    return np.array(recalls), rebuilds


def main():
    t0 = time.time()
    rows = []
    for ds in DATASETS:
        full, query = load(ds)
        print(f"[{ds}] N={N} q={query.shape}")
        policies = [("never", None), ("fixed20", None)] + [("hubmass", th) for th in THETAS]
        for sc in SCENARIOS:
            for po, th in policies:
                mins, rbs, fmins = [], [], []
                for seed in SEEDS:
                    vecs0 = subsample(full, seed)
                    rec, nrb = run(vecs0, query, sc, po, th, seed)
                    mins.append(rec.min()); rbs.append(nrb)
                    fmins.append(BATCH * (np.argmin(rec) + 1))
                tag = po if th is None else f"{po}{th}"
                rows.append((ds, sc, tag, np.mean(mins), np.std(mins),
                             np.mean(rbs), np.mean(fmins)))
                print(f"  [{ds}][{sc:11s}][{tag:10s}] min={np.mean(mins):.3f}±{np.std(mins):.3f} "
                      f"rebuilds={np.mean(rbs):.1f} (min@f={np.mean(fmins):.2f})")

    print("\n===== v2 요약: ds x scenario x policy -> min recall (rebuilds) =====")
    tags = ["never", "fixed20"] + [f"hubmass{th}" for th in THETAS]
    print(f"  {'ds':>5} {'scenario':>12} | " + " | ".join(f"{t:>15}" for t in tags))
    for ds in DATASETS:
        for sc in SCENARIOS:
            cells = []
            for t in tags:
                r = [x for x in rows if x[0] == ds and x[1] == sc and x[2] == t][0]
                cells.append(f"{r[3]:.3f} r={r[5]:.0f}".rjust(15))
            print(f"  {ds:>5} {sc:>12} | " + " | ".join(cells))
    print("\n  성립 조건: gist×hubtargeted 에서 never·fixed20 의 min << hubmass 의 min")
    np.savez("../results/trigger_5s.npz",
             rows=np.array([(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows], dtype=object))
    print(f"\n[done] {time.time()-t0:.1f}s -> ../results/trigger_v2.npz")


if __name__ == "__main__":
    main()
