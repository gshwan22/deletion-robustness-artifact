#!/usr/bin/env python3
"""
ANN-Percolation — W2 마감 : attack 스펙트럼 (리뷰어 "왜 degree만?" 선제 차단)
==========================================================================
기존: random / hub(정적 초기 in-degree). 추가:
  adaptive  : 배치-적응형 degree attack — 0.5% 지울 때마다 in-degree 재계산
              (고전 결과: recomputed > initial. 얼마나 더 센가?)
  traversal : 라우팅 사용빈도 중심성 — 파일럿 쿼리(평가셋과 분리) beam search 에서
              각 노드 방문 횟수 집계 -> 많이 경유되는 노드부터 삭제.
              betweenness 의 검색-현실 버전. hub 과 근사하면 '차수=싼 프록시' 정당화,
              더 세면 'in-degree 는 취약성의 하한'.
  kcore     : coreness 내림차순 삭제 (무향 level-0 그래프 peeling)
측정: sift/gist 100k, 3 seeds, f_c(recall<0.5*baseline). random/hub 도 같은 런에서 재측정.
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
SAMPLE_Q  = 300          # 평가 쿼리 = test[:300]
PILOT_Q   = 300          # traversal 파일럿 = test[300:600] (분리)
R_ENTRIES = 16
F_GRID    = np.round(np.linspace(0.0, 0.95, 20), 3)
MODES     = ["random", "hub", "adaptive", "traversal", "kcore"]
SEEDS     = [0, 1, 2, 3, 4]
REL       = 0.5
ADAPT_BATCH = 0.005      # 적응형 재계산 배치 (0.5%)


def load(name):
    with h5py.File(DATASETS[name], "r") as f:
        full = np.asarray(f["train"], dtype=np.float32)
        query = np.asarray(f["test"], dtype=np.float32)
    return full, np.ascontiguousarray(query[:SAMPLE_Q]), \
           np.ascontiguousarray(query[SAMPLE_Q:SAMPLE_Q + PILOT_Q])


def subsample(full, seed):
    rng = np.random.default_rng(1000 + seed)
    return np.ascontiguousarray(full[rng.choice(full.shape[0], size=N, replace=False)])


def build(base):
    n, d = base.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = EF_CONSTR
    index.add(base)
    h = index.hnsw
    neighbors = faiss.vector_to_array(h.neighbors)
    offsets   = faiss.vector_to_array(h.offsets).astype(np.int64)
    cum       = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
    n0 = int(cum[1])
    idx = offsets[:-1][:, None] + np.arange(n0)[None, :]
    nbrs0 = neighbors[idx]
    nbrs0 = np.where(nbrs0 < 0, np.int32(-1), nbrs0).astype(np.int32)
    indeg = np.bincount(nbrs0[nbrs0 >= 0], minlength=n).astype(np.int64)
    return nbrs0, indeg


def sqd(x, q):
    diff = x - q
    return float(diff @ diff)


def beam_multi(q, nbrs, alive, base, entries, ef, k, visit_counter=None):
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
    if visit_counter is not None:
        for u in seen:
            visit_counter[u] += 1
    return [ni for _, ni in sorted((-nd, ni) for nd, ni in res)[:k]]


def traversal_order(base, query_pilot, nbrs, indeg, seed):
    """intact 그래프에서 파일럿 쿼리 beam search -> 노드 방문빈도 내림차순."""
    n = base.shape[0]
    alive = np.ones(n, bool)
    rng = np.random.default_rng(4000 + seed)
    ent = rng.choice(n, size=R_ENTRIES, replace=False)
    visits = np.zeros(n, np.int64)
    for i in range(query_pilot.shape[0]):
        beam_multi(query_pilot[i], nbrs, alive, base, ent, EF, K, visit_counter=visits)
    # 동률은 indeg 로 타이브레이크
    return np.lexsort((-indeg, -visits))


def adaptive_order(nbrs, indeg0):
    """배치-적응형 in-degree attack: B 개 지우고 out-엣지 기여 차감 후 재선정."""
    n = indeg0.shape[0]
    indeg = indeg0.astype(np.int64).copy()
    alive = np.ones(n, bool)
    B = max(1, int(round(ADAPT_BATCH * n)))
    order = np.empty(n, np.int64)
    ptr = 0
    while ptr < n:
        alive_ids = np.where(alive)[0]
        if alive_ids.size == 0:
            break
        take = alive_ids[np.argsort(-indeg[alive_ids], kind="stable")[:B]]
        for u in take:
            alive[u] = False
            for v in nbrs[u]:
                if v >= 0:
                    indeg[v] -= 1
        order[ptr:ptr + take.size] = take
        ptr += take.size
    return order[:ptr]


def kcore_order(nbrs):
    """무향 peeling -> coreness 내림차순."""
    n = nbrs.shape[0]
    # 무향 인접 (level-0 대칭화)
    adj = [[] for _ in range(n)]
    for u in range(n):
        for v in nbrs[u]:
            if v >= 0:
                adj[u].append(v); adj[v].append(u)
    deg = np.array([len(a) for a in adj], np.int64)
    core = np.zeros(n, np.int64)
    removed = np.zeros(n, bool)
    hp = [(int(deg[u]), u) for u in range(n)]
    heapq.heapify(hp)
    k = 0
    while hp:
        d, u = heapq.heappop(hp)
        if removed[u] or d != deg[u]:
            continue
        k = max(k, d)
        core[u] = k
        removed[u] = True
        for v in adj[u]:
            if not removed[v]:
                deg[v] -= 1
                heapq.heappush(hp, (int(deg[v]), v))
    return np.argsort(-core, kind="stable")


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
        idxf = faiss.IndexFlatL2(base.shape[1])
        idxf.add(np.ascontiguousarray(base[ids]))
        _, loc = idxf.search(query, K)
        ora = ids[loc]
        hit = sum(len(set(beam_multi(query[i], nbrs, alive, base, ent, EF, K))
                      .intersection(ora[i])) for i in range(query.shape[0]))
        out.append(hit / (query.shape[0] * K))
    return np.array(out)


def main():
    t0 = time.time()
    results = {}
    for name in DATASETS:
        full, qeval, qpilot = load(name)
        fcs = {m: [] for m in MODES}
        for seed in SEEDS:
            base = subsample(full, seed)
            nbrs, indeg = build(base)
            rng = np.random.default_rng(5000 + seed)
            orders = {
                "random": np.random.default_rng(seed).permutation(N),
                "hub": np.argsort(-indeg, kind="stable"),
            }
            t1 = time.time()
            orders["adaptive"] = adaptive_order(nbrs, indeg)
            t2 = time.time()
            orders["traversal"] = traversal_order(base, qpilot, nbrs, indeg, seed)
            t3 = time.time()
            orders["kcore"] = kcore_order(nbrs)
            t4 = time.time()
            print(f"  [{name}] seed{seed} orders: adapt {t2-t1:.0f}s / trav {t3-t2:.0f}s / kcore {t4-t3:.0f}s")
            for m in MODES:
                fc = fcross(recall_curve(base, qeval, nbrs, indeg, orders[m], rng))
                fcs[m].append(fc)
                print(f"    [{name}] seed{seed} [{m:9s}] f_c={fc}")
        def agg(v):
            u = [x for x in v if x is not None]
            return (np.mean(u), np.std(u)) if u else (None, None)
        results[name] = {m: agg(fcs[m]) for m in MODES}

    print("\n===== attack 스펙트럼 (f_c, mean±std, 3 seeds) =====")
    fmt = lambda t: " >rng" if t[0] is None else f"{t[0]:.3f}±{t[1]:.3f}"
    print(f"  {'ds':>5} | " + " | ".join(f"{m:>13}" for m in MODES))
    for name, r in results.items():
        print(f"  {name:>5} | " + " | ".join(f"{fmt(r[m]):>13}" for m in MODES))
    print("\n  해석 가이드:")
    print("  - adaptive < hub      : 적응형이 더 셈(고전 결과 재현) -> 얼마나?")
    print("  - traversal ~ hub     : in-degree = 라우팅 중심성의 싼 프록시 (트리거 정당화)")
    print("  - traversal << hub    : in-degree 는 취약성의 하한 (더 강한 공격 존재 경고)")
    np.savez("../results/attack_spectrum_5s.npz",
             **{f"{name}_{m}": np.array([results[name][m][0] or np.nan,
                                          results[name][m][1] or np.nan])
                for name in results for m in MODES})
    print(f"\n[done] {time.time()-t0:.1f}s -> ../results/attack_spectrum.npz")


if __name__ == "__main__":
    main()
