"""batch_h: KEIR 잔여 지적 3건 종결
h3 (먼저, 1분): 순서별 구조 편향 비율 mu(f)/f  — 삭제 질량 분율 / 삭제 개수 분율
h2 (5분)    : 단일 고정 진입점 편향 정량화 — R=1 고정 진입 vs R=16 다중 진입, SIFT·GIST hub/random
h1 (~1h)   : recall 프로빙 기준선 — 프로브 질의 P개로 recall 재서 임계 아래면 재구축 (정답 계산 비용 포함)
             vs 개수 / 질량 프런티어 (무수리, GIST hub 재조준 + random, 3 seeds)"""
import numpy as np, time, inspect
import c_common
from c_common import load, build_hnsw, orders, recall_at, F_GRID
t0 = time.time()
LOG = lambda s: print(s, flush=True)

# ═══════════ h3: 구조 편향 비율 ═══════════
LOG("===== h3: 구조 편향 비율 mu(f)/f (시드 0~2 평균) =====")
for ds in ["sift", "gist"]:
    acc = {}
    for seed in [0, 1, 2]:
        base, _ = load(ds, seed); n = len(base)
        _, indeg = build_hnsw(base); tot = indeg.sum()
        od = orders(base, indeg, seed)
        for nm in ["random", "hub", "oldest", "newest"]:
            if nm not in od: continue
            order = od[nm]
            row = []
            for f in [0.05, 0.10, 0.20, 0.30, 0.50]:
                k = int(f * n); row.append(indeg[order[:k]].sum() / tot / f)
            acc.setdefault(nm, []).append(row)
    for nm, rows in acc.items():
        m = np.mean(rows, axis=0)
        LOG(f"  [{ds}] {nm:7s}  f=.05:{m[0]:.2f}  .10:{m[1]:.2f}  .20:{m[2]:.2f}  .30:{m[3]:.2f}  .50:{m[4]:.2f}")
LOG("  해석: 1.0 = 질량 중립, >1 = 허브 편향, <1 = 저차수 편향")

# ═══════════ h2: 진입점 편향 ═══════════
LOG("===== h2: 단일 고정 진입점 편향 (R=1 vs R=16) =====")
sig = inspect.signature(recall_at)
LOG(f"  recall_at 시그니처: {sig}")
has_R = "R" in sig.parameters or "n_entry" in sig.parameters
Rkw = "R" if "R" in sig.parameters else ("n_entry" if "n_entry" in sig.parameters else None)
if Rkw is None:
    LOG("  recall_at 에 진입점 수 인자가 없음 → c_common 내부 상수 이름 탐색")
    cands = [k for k in dir(c_common) if k.upper() in ("R", "N_ENTRY", "NENTRY", "ENTRY", "N_ENTRIES")]
    LOG(f"  후보 상수: {cands}")
for ds in ["sift", "gist"]:
    base, q = load(ds, 0); n = len(base)
    nbrs0, indeg = build_hnsw(base); od = orders(base, indeg, 0)
    for nm in ["hub", "random"]:
        for R in [1, 16]:
            recs = []
            for f in [0.0, 0.3, 0.5, 0.7]:
                k = int(f * n); alive = np.ones(n, bool); nbrs = nbrs0.copy()
                dele = od[nm][:k]; alive[dele] = False; nbrs[dele] = -1
                rng = np.random.default_rng(11)
                try:
                    if Rkw: r = recall_at(base, alive, nbrs, q, rng, **{Rkw: R})
                    else:
                        cname = [k for k in dir(c_common) if k.upper() in ("R", "N_ENTRY", "NENTRY", "ENTRY", "N_ENTRIES")]
                        if not cname: raise RuntimeError("진입점 인자/상수 없음")
                        old = getattr(c_common, cname[0]); setattr(c_common, cname[0], R)
                        r = recall_at(base, alive, nbrs, q, rng); setattr(c_common, cname[0], old)
                except Exception as e:
                    LOG(f"  [h2] 실패: {e}"); r = float("nan")
                recs.append(r)
            LOG(f"  [{ds} {nm:6s} R={R:2d}] recall@f=0/.3/.5/.7 = " + "/".join(f"{x:.3f}" for x in recs))
LOG("  해석: R=1 이 R=16 대비 낮게 + 비단조로 나오면 단일 진입점 인공물의 크기")

# ═══════════ h1: recall 프로빙 기준선 ═══════════
LOG("===== h1: recall 프로빙 정책 vs 개수/질량 =====")
STEP, NSTEP = 0.025, 24
def remap(sub, ids, n):
    g = np.full((n, sub.shape[1]), -1, np.int32)
    g[ids] = np.where(sub >= 0, ids[np.clip(sub, 0, None)], -1); return g
def probe_recall(base, alive, nbrs, probes, rng):
    # 프로브 질의 = 평가 질의와 분리된 소규모 셋. recall_at 는 생존자 정확 top-10 을 내부에서 계산
    return recall_at(base, alive, nbrs, probes, rng)
def sim(ds, stream, policy, seed):
    base, q = load(ds, seed); n = len(base)
    nbrs0, indeg0 = build_hnsw(base); nbrs = nbrs0.copy(); alive = np.ones(n, bool)
    rng0 = np.random.default_rng(5000 + seed)
    P = int(policy.split("_")[1]) if policy.startswith("probe") else 0
    thr = float(policy.split("_")[2]) if policy.startswith("probe") else 0.0
    pidx = rng0.choice(n, P, replace=False) if P else None     # 프로브는 기반 벡터에서 뽑은 별도 셋
    probes = (base[pidx] + 0.01 * rng0.standard_normal((P, base.shape[1])).astype(np.float32)) if P else None
    eval_q = q
    indeg_ref = indeg0.astype(np.int64); mass_ref = indeg_ref.sum(); n_ref = n
    rand_order = rng0.permutation(n)
    dc = dm = rebuilds = probe_calls = 0; recs = []
    for st in range(NSTEP):
        k = int(STEP * n)
        if stream == "hub": cand = np.argsort(-np.where(alive, indeg_ref, -1), kind="stable")[:k]
        else: cand = rand_order[alive[rand_order]][:k]
        alive[cand] = False; nbrs[cand] = -1; dc += k; dm += indeg_ref[cand].sum()
        fire = False
        if policy.startswith("fixed"): fire = dc / n_ref > int(policy[5:]) / 100
        elif policy.startswith("mass"): fire = dm / mass_ref > int(policy[4:]) / 100
        elif policy.startswith("probe"):
            probe_calls += 1
            pr = probe_recall(base, alive, nbrs, probes, np.random.default_rng(6000 + st))
            fire = pr < thr
        if fire:
            ids = np.where(alive)[0]
            sub, dloc = build_hnsw(np.ascontiguousarray(base[ids]))
            nbrs = remap(sub, ids, n)
            indeg_ref = np.zeros(n, np.int64); indeg_ref[ids] = dloc
            mass_ref = indeg_ref.sum(); n_ref = alive.sum(); dc = dm = 0; rebuilds += 1
        recs.append(recall_at(base, alive, nbrs, eval_q, np.random.default_rng(2000 + seed + st)))
    return 1 - min(recs), rebuilds, probe_calls * P
POLS = ["probe_20_0.95", "probe_20_0.90", "probe_100_0.95", "probe_100_0.90", "probe_100_0.97"]
out = {}
for ds, stream, seeds in [("gist", "hub", [0, 1, 2]), ("gist", "random", [0, 1, 2])]:
    for pol in POLS:
        em, rb, pq = [], [], []
        for s in seeds:
            e, r, p = sim(ds, stream, pol, s); em.append(e); rb.append(r); pq.append(p)
            out[f"{ds}_{stream}_{pol}_s{s}"] = np.array([e, r, p])
        LOG(f"  [h1] {ds} {stream:6s} {pol:15s}: emax={np.mean(em):.3f}±{np.std(em):.3f}  C_r={np.mean(rb):.1f}  probe_queries={np.mean(pq):.0f}  ({time.time()-t0:.0f}s)")
    np.savez("../results/h1_probe.npz", **out)
LOG("===== h1 요약 (비교: e2 재조준 hub — fixed10 0.033/7, fixed20 0.097/3, mass30 0.024/6, mass15 0.012/12 ; random — fixed20 0.012/3, mass30 0.013/2) =====")
for ds, stream in [("gist", "hub"), ("gist", "random")]:
    for pol in POLS:
        v = np.mean([out[f"{ds}_{stream}_{pol}_s{s}"] for s in [0, 1, 2]], axis=0)
        LOG(f"  {stream:6s} {pol:15s} C_r={v[1]:.1f} emax={v[0]:.3f} probe_q={v[2]:.0f}")
LOG(f"[done] {time.time()-t0:.0f}s")
