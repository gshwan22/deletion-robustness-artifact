#!/usr/bin/env python3
"""
ANN-Percolation — 검증 4순위(최종) : #8 데이터셋 일반화 + #7 query 표본
=====================================================================
예측(사전 등록): hubness 가 심한 데이터셋일수록 attack-vs-failure gap 이 크다.
  SIFT(128d) < glove(100d angular, hubness 악명) ~ GIST(960d 고차원)

측정: 각 데이터셋에서
  - 실물 HNSW level-0, hubness(max/mean + 상위1% in-edge 점유율)
  - 4모드(random/hub/antihub/relevance) f_c  [3 seeds]
  - gap = f_c(random) - f_c(hub)
  - #7: 쿼리 표본 2벌(head300 / rand300) 로 f_c 재현
판정:
  - 모든 데이터셋에서 hub << random, antihub 무해, relevance ~ random -> 인과 일반화
  - hubness 순위와 gap 순위 일치 -> 메커니즘(hubness 지배) 지지
"""
import time
import numpy as np
import h5py
import faiss
import heapq
import matplotlib.pyplot as plt

DATASETS = {
    "sift":  ("../data/sift-128-euclidean.hdf5",  "l2"),
    "glove": ("../data/glove-100-angular.hdf5",   "angular"),
    "gist":  ("../data/gist-960-euclidean.hdf5",  "l2"),
}
N         = 100_000
M         = 16
EF_CONSTR = 200
EF        = 512
K         = 10
T_REL     = 50
SAMPLE_Q  = 300
R_ENTRIES = 16
F_GRID    = np.round(np.linspace(0.0, 0.95, 20), 3)
MODES     = ["random", "hub", "antihub", "relevance"]
SEEDS     = [0, 1, 2]
REL       = 0.5


def load(name):
    path, metric = DATASETS[name]
    with h5py.File(path, "r") as f:
        full = np.asarray(f["train"], dtype=np.float32)
        query = np.asarray(f["test"], dtype=np.float32)
    if metric == "angular":                      # 정규화 -> L2 로 통일
        full = full / (np.linalg.norm(full, axis=1, keepdims=True) + 1e-12)
        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    return full, query


def subsample(full, seed):
    rng = np.random.default_rng(1000 + seed)
    n = min(N, full.shape[0])
    return np.ascontiguousarray(full[rng.choice(full.shape[0], size=n, replace=False)])


def build(base):
    n, d = base.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = EF_CONSTR
    index.add(base)
    h = index.hnsw
    neighbors = faiss.vector_to_array(h.neighbors).astype(np.int64)
    offsets   = faiss.vector_to_array(h.offsets).astype(np.int64)
    cum       = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
    n0 = int(cum[1])
    idx = offsets[:-1][:, None] + np.arange(n0)[None, :]
    nbrs0 = np.where(neighbors[idx] < 0, -1, neighbors[idx]).astype(np.int32)
    indeg = np.bincount(nbrs0[nbrs0 >= 0], minlength=n).astype(np.int64)
    return nbrs0, indeg


def hubness_stats(indeg):
    mm = indeg.max() / indeg.mean()
    top1 = np.sort(indeg)[::-1][: max(1, len(indeg)//100)].sum() / indeg.sum()
    return mm, top1     # max/mean, 상위1% in-edge 점유율


def exact_topk(base, query, k):
    idx = faiss.IndexFlatL2(base.shape[1])
    idx.add(base)
    _, I = idx.search(query, k)
    return I


def removal_order(base, indeg, relscore, mode, seed):
    rng = np.random.default_rng(seed)
    n = base.shape[0]
    if mode == "random":
        return rng.permutation(n)
    if mode == "hub":
        return np.argsort(-indeg, kind="stable")
    if mode == "antihub":
        return np.argsort(indeg, kind="stable")
    if mode == "relevance":
        return np.argsort(-relscore, kind="stable")
    raise ValueError(mode)


def sqd(x, q):
    diff = x - q
    return float(diff @ diff)


def beam_multi(q, nbrs, alive, base, entries, ef, k):
    cand, res, seen = [], [], set()
    for e in entries:
        if alive[e] and e not in seen:
            de = sqd(base[e], q)
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
            dv = sqd(base[v], q)
            if len(res) < ef or dv < -res[0][0]:
                heapq.heappush(cand, (dv, v)); heapq.heappush(res, (-dv, v))
                if len(res) > ef:
                    heapq.heappop(res)
    return [ni for _, ni in sorted((-nd, ni) for nd, ni in res)[:k]]


def oracle_survivor(base, alive_ids, query, k):
    idx = faiss.IndexFlatL2(base.shape[1])
    idx.add(np.ascontiguousarray(base[alive_ids]))
    _, loc = idx.search(query, k)
    return alive_ids[loc]


def fcross(recall):
    thr = recall[0] * REL
    for i in range(1, len(F_GRID)):
        if recall[i] < thr <= recall[i-1]:
            f0, f1, r0, r1 = F_GRID[i-1], F_GRID[i], recall[i-1], recall[i]
            return f0 + (r0 - thr) / (r0 - r1 + 1e-12) * (f1 - f0)
    return None


def recall_curve(base, query, nbrs, indeg, order, rng):
    n = base.shape[0]
    out = []
    for f in F_GRID:
        alive = np.ones(n, bool)
        alive[order[:int(round(f * n))]] = False
        ids = np.where(alive)[0]
        if ids.size < K + 1:
            out.append(0.0); continue
        ent = rng.choice(ids, size=min(R_ENTRIES, ids.size), replace=False)
        ora = oracle_survivor(base, ids, query, K)
        hit = sum(len(set(beam_multi(query[i], nbrs, alive, base, ent, EF, K))
                      .intersection(ora[i])) for i in range(query.shape[0]))
        out.append(hit / (query.shape[0] * K))
    return np.array(out)


def run_dataset(name):
    full, queries = load(name)
    qA = np.ascontiguousarray(queries[:SAMPLE_Q])                       # head300
    rngq = np.random.default_rng(777)
    qB = np.ascontiguousarray(queries[rngq.choice(queries.shape[0],
                                                  size=min(SAMPLE_Q, queries.shape[0]),
                                                  replace=False)])      # rand300 (#7)
    fcs = {m: [] for m in MODES}
    fcs_qB_hub, fcs_qB_rand = [], []
    hub_mm, hub_t1 = [], []
    for seed in SEEDS:
        base = subsample(full, seed)
        nbrs, indeg = build(base)
        mm, t1 = hubness_stats(indeg)
        hub_mm.append(mm); hub_t1.append(t1)
        relscore = np.bincount(exact_topk(base, qA, T_REL).ravel(),
                               minlength=base.shape[0]).astype(np.int64)
        rng = np.random.default_rng(5000 + seed)
        for m in MODES:
            order = removal_order(base, indeg, relscore, m, seed)
            fc = fcross(recall_curve(base, qA, nbrs, indeg, order, rng))
            fcs[m].append(fc)
        # #7: qB 로 hub/random 만 재측정
        for m, sink in [("hub", fcs_qB_hub), ("random", fcs_qB_rand)]:
            order = removal_order(base, indeg, relscore, m, seed)
            sink.append(fcross(recall_curve(base, qB, nbrs, indeg, order, rng)))
        print(f"  [{name}] seed{seed} " +
              " ".join(f"{m}:{fcs[m][-1] if fcs[m][-1] is None else round(fcs[m][-1],3)}"
                       for m in MODES) +
              f"  hubness={mm:.1f}/top1%={t1:.2f}")
    def agg(v):
        u = [x for x in v if x is not None]
        return (np.mean(u), np.std(u)) if u else (None, None)
    out = {m: agg(fcs[m]) for m in MODES}
    out["hub_qB"] = agg(fcs_qB_hub); out["rand_qB"] = agg(fcs_qB_rand)
    out["hubness"] = (np.mean(hub_mm), np.mean(hub_t1))
    return out


def main():
    t0 = time.time()
    results = {}
    for name in DATASETS:
        print(f"[{name}] running ...")
        results[name] = run_dataset(name)

    print("\n===== #8 데이터셋 일반화 =====")
    print(f"  {'ds':>6} {'hub(mm)':>8} {'top1%':>6} {'fc_hub':>7} {'fc_rand':>8} {'gap':>6} "
          f"{'antihub':>8} {'relev':>6}")
    fmt = lambda x: " >rng" if x is None else f"{x:.3f}"
    for name, r in results.items():
        gh, gr = r["hub"][0], r["random"][0]
        gap = None if (gh is None or gr is None) else gr - gh
        print(f"  {name:>6} {r['hubness'][0]:>8.1f} {r['hubness'][1]:>6.2f} "
              f"{fmt(gh):>7} {fmt(gr):>8} {fmt(gap):>6} "
              f"{fmt(r['antihub'][0]):>8} {fmt(r['relevance'][0]):>6}")
    print("  판정: 모든 ds 에서 hub<<rand & antihub 무해 & relev~rand -> 인과 일반화")
    print("        hubness 순위와 gap 순위 일치 -> 메커니즘(hubness 지배) 지지")

    print("\n===== #7 query 표본 (head300 vs rand300) =====")
    for name, r in results.items():
        print(f"  [{name}] hub: qA={fmt(r['hub'][0])} qB={fmt(r['hub_qB'][0])}   "
              f"random: qA={fmt(r['random'][0])} qB={fmt(r['rand_qB'][0])}")
    print("  판정: qA~qB 면 표본 편향 없음")

    np.savez("../results/dataset_generalization.npz",
             **{f"{name}_{m}_mu": np.array(results[name][m][0] if results[name][m][0] is not None else np.nan)
                for name in results for m in MODES},
             **{f"{name}_hubness": np.array(results[name]["hubness"])
                for name in results})
    print(f"\n[done] {time.time()-t0:.1f}s -> ../results/dataset_generalization.npz")


if __name__ == "__main__":
    main()
