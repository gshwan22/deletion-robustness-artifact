#!/usr/bin/env python3
"""
ANN-Percolation — 검증 2순위 : oracle 공정성(#3) + hub-relevance 교란(#5)
=====================================================================
질문: hub 삭제의 recall 하락이
  (구조적) 라우팅 허브 제거로 길찾기 붕괴 = navigability  [강한 주장]
  (교란)  hub이 정답 밀집영역이라 정답 주변을 지운 것       [trivial, 재프레임 필요]
중 무엇인가.

측정 A: hubness vs relevance 상관 — gold(쿼리 top-K) 노드의 in-degree 분포 vs 전체.
측정 B: 4모드 대조 (동일 f, 생존자-oracle)
        random / hub(고 indeg) / antihub(저 indeg) / relevance(정답영역 삭제)
        -> hub >> relevance & hub >> antihub 이면 구조적(navigability).
측정 C: recall 두 정의 비교 (#3 oracle 공정성)
        survivor  = 생존자 중 exact top-K
        fixed     = 원래 gold(f=0 top-K) 중 '생존한 것'을 얼마나 찾나
        두 정의가 같은 결론이면 oracle 정의가 결과를 만든 게 아님.
"""
import time
import numpy as np
import h5py
import faiss
import heapq
import matplotlib.pyplot as plt

DATA_PATH = "../data/sift-128-euclidean.hdf5"  # b3: sift
N         = 100_000
M         = 16
EF_CONSTR = 200
EF        = 512
K         = 10
T_REL     = 50          # relevance 정의용 top-T
SAMPLE_Q  = 300
R_ENTRIES = 16
F_GRID    = np.round(np.linspace(0.0, 0.95, 20), 3)
MODES     = ["random", "hub", "antihub", "relevance"]
SEEDS     = [0, 1, 2, 3, 4]
REL       = 0.5


def load_full():
    with h5py.File(DATA_PATH, "r") as f:
        full = np.asarray(f["train"], dtype=np.float32)
        query = np.asarray(f["test"], dtype=np.float32)
    return full, np.ascontiguousarray(query[:SAMPLE_Q])


def subsample(full, seed):
    rng = np.random.default_rng(1000 + seed)
    return np.ascontiguousarray(full[rng.choice(full.shape[0], size=N, replace=False)])


def build_real_hnsw(base):
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


def exact_topk(base, query, k):
    idx = faiss.IndexFlatL2(base.shape[1])
    idx.add(base)
    _, I = idx.search(query, k)
    return I


def relevance_score(base, query, T):
    """각 노드가 쿼리 top-T 에 등장한 횟수 = 정답영역 근접도."""
    I = exact_topk(base, query, T)
    return np.bincount(I.ravel(), minlength=base.shape[0]).astype(np.int64)


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


def run_seed(full, query, seed):
    base = subsample(full, seed)
    nbrs, indeg = build_real_hnsw(base)
    relscore = relevance_score(base, query, T_REL)
    gold = exact_topk(base, query, K)          # (Q,K) 원래 정답 (fixed-gold 용)
    n = base.shape[0]

    # 측정 A: hubness vs relevance
    goldnodes = np.unique(gold.ravel())
    A_stat = (indeg[goldnodes].mean(), indeg.mean(),
              np.corrcoef(indeg, relscore)[0, 1])

    rng = np.random.default_rng(5000 + seed)
    res = {}
    for mode in MODES:
        order = removal_order(base, indeg, relscore, mode, seed)
        rec_surv, rec_fix = [], []
        for f in F_GRID:
            alive = np.ones(n, bool)
            alive[order[:int(round(f * n))]] = False
            ids = np.where(alive)[0]
            if ids.size < K + 1:
                rec_surv.append(0.0); rec_fix.append(0.0); continue
            ent = rng.choice(ids, size=min(R_ENTRIES, ids.size), replace=False)
            ora = oracle_survivor(base, ids, query, K)
            hs = hf = 0; fdenom = 0
            for i in range(query.shape[0]):
                got = set(beam_multi(query[i], nbrs, alive, base, ent, EF, K))
                hs += len(got.intersection(ora[i]))
                surv_gold = [g for g in gold[i] if alive[g]]   # 생존한 원래정답
                if surv_gold:
                    hf += len(got.intersection(surv_gold))
                    fdenom += len(surv_gold)
            rec_surv.append(hs / (query.shape[0] * K))
            rec_fix.append(hf / fdenom if fdenom else 0.0)
        res[mode] = (np.array(rec_surv), np.array(rec_fix))
        print(f"  seed{seed} [{mode:9s}] "
              f"fc_surv={fcross(res[mode][0])}  fc_fixed={fcross(res[mode][1])}")
    return A_stat, res


def main():
    t0 = time.time()
    full, query = load_full()
    print(f"[data] q={query.shape} seeds={SEEDS}")
    stats, allres = [], []
    for s in SEEDS:
        A, res = run_seed(full, query, s)
        stats.append(A); allres.append(res)

    # 측정 A 요약
    gm = np.mean([a[0] for a in stats]); om = np.mean([a[1] for a in stats])
    cc = np.mean([a[2] for a in stats])
    print("\n===== 측정 A: hubness vs relevance =====")
    print(f"  gold 노드 평균 in-degree={gm:.1f}  vs  전체 평균={om:.1f}  (비 {gm/om:.2f}x)")
    print(f"  corr(indeg, relevance)={cc:.3f}")
    print("  -> 비가 1에 가깝고 corr 낮으면: hub != 정답영역 (교란 없음, 구조적 주장 안전)")

    # f_c 집계
    def agg(mode, which):
        v = [fcross(allres[i][mode][which]) for i in range(len(SEEDS))]
        v = [x for x in v if x is not None]
        return (np.mean(v), np.std(v)) if v else (None, None)

    print("\n===== 측정 B+C: 모드별 f_c (survivor / fixed) =====")
    fc = {}
    for m in MODES:
        s_mu, s_sd = agg(m, 0); f_mu, f_sd = agg(m, 1)
        fc[m] = (s_mu, f_mu)
        sp = "None" if s_mu is None else f"{s_mu:.3f}±{s_sd:.3f}"
        fp = "None" if f_mu is None else f"{f_mu:.3f}±{f_sd:.3f}"
        print(f"  [{m:9s}] survivor f_c={sp}   fixed f_c={fp}")

    print("\n===== 판정 =====")
    h, r, a = fc["hub"][0], fc["relevance"][0], fc["antihub"][0]
    if h and r and a:
        print(f"  hub={h:.3f}  relevance={r:.3f}  antihub={a:.3f}  random={fc['random'][0]}")
        if h < r - 0.03 and h < a - 0.05:
            print("  ✅ 구조적: hub 이 relevance·antihub 보다 먼저 붕괴 = navigability(강한 주장)")
        elif abs(h - r) <= 0.03:
            print("  ⚠️ hub ≈ relevance: 정답영역 매개 가능성. hubness-relevance 상관 확인 필요")
        else:
            print("  (혼합) — 숫자 보고 해석")
    # #3: 두 recall 정의가 같은 순서?
    print("  oracle 공정성(#3): survivor vs fixed f_c 가 모드 순서 동일하면 oracle 정의 무관")

    # 플롯
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    c = {"random": "#2E6FBA", "hub": "#D1453B", "antihub": "#8E44AD", "relevance": "#27AE60"}
    for m in MODES:
        surv = np.mean([allres[i][m][0] for i in range(len(SEEDS))], axis=0)
        fix  = np.mean([allres[i][m][1] for i in range(len(SEEDS))], axis=0)
        ax[0].plot(F_GRID, surv, "-o", color=c[m], ms=3, label=m)
        ax[1].plot(F_GRID, fix,  "-o", color=c[m], ms=3, label=m)
    ax[0].set(title="survivor-oracle recall", xlabel="f", ylabel=f"recall@{K}")
    ax[1].set(title="fixed-gold recall (#3 control)", xlabel="f", ylabel="found/surviving-gold")
    for aa in ax:
        aa.grid(alpha=0.3); aa.legend()
    plt.tight_layout()
    plt.savefig("../figures/b3_confound_sift_5s.png", dpi=150)
    print(f"\n[done] {time.time()-t0:.1f}s -> ../figures/confound_check.png")


if __name__ == "__main__":
    main()
