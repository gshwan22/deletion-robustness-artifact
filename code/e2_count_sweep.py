"""e2: 개수 임계 스윕 — "질량 트리거의 이득은 그냥 일찍 재구축한 효과 아니냐"를 닫는 실험.
fixed 5/10/15/20/25% 와 mass θ=0.15/0.30 을 같은 무수리 스트림에서 비교 → (C_r, ε_max) 프런티어.
스트림: gist hub(5s), gist random(3s), sift hub(3s). 배치 2.5%, 누적 60%."""
import numpy as np, time
from c_common import load, build_hnsw, orders, recall_at
t0 = time.time()
STEP, NSTEP = 0.025, 24
POLICIES = ["never", "fixed05", "fixed10", "fixed15", "fixed20", "fixed25", "mass15", "mass30"]
STREAMS = [("gist", "hub", [0, 1, 2, 3, 4]), ("gist", "random", [0, 1, 2]), ("sift", "hub", [0, 1, 2])]
out = {}

def remap(nbrs_local, ids, n):
    g = np.full((n, nbrs_local.shape[1]), -1, np.int32)
    g[ids] = np.where(nbrs_local >= 0, ids[np.clip(nbrs_local, 0, None)], -1)
    return g

def simulate(ds, stream, policy, seed):
    base, q = load(ds, seed)
    n = len(base)
    nbrs0, indeg0 = build_hnsw(base)
    nbrs = nbrs0.copy()
    alive = np.ones(n, bool)
    indeg_ref = indeg0.astype(np.int64); mass_ref = indeg_ref.sum(); n_ref = n
    rng = np.random.default_rng(1000 + seed)
    rand_order = rng.permutation(n)
    dcount, dmass, rebuilds = 0, 0, 0
    recs = []
    for st in range(NSTEP):
        k = int(STEP * n)
        if stream == "hub":
            cand = np.argsort(-np.where(alive, indeg_ref, -1), kind="stable")[:k]
        else:
            pool = rand_order[alive[rand_order]]
            cand = pool[:k]
        alive[cand] = False
        nbrs[cand] = -1
        dcount += k; dmass += indeg_ref[cand].sum()
        fire = False
        if policy.startswith("fixed"):
            fire = dcount / n_ref > int(policy[5:]) / 100
        elif policy.startswith("mass"):
            fire = dmass / mass_ref > int(policy[4:]) / 100
        if fire:
            ids = np.where(alive)[0]
            sub, dloc = build_hnsw(np.ascontiguousarray(base[ids]))
            nbrs = remap(sub, ids, n)
            indeg_ref = np.zeros(n, np.int64); indeg_ref[ids] = dloc
            mass_ref = indeg_ref.sum(); n_ref = alive.sum()
            dcount, dmass, rebuilds = 0, 0, rebuilds + 1
        recs.append(recall_at(base, alive, nbrs, q, np.random.default_rng(2000 + seed + st)))
    return 1 - min(recs), rebuilds

for ds, stream, seeds in STREAMS:
    for pol in POLICIES:
        em, rb = [], []
        for s in seeds:
            e, r = simulate(ds, stream, pol, s)
            em.append(e); rb.append(r)
            out[f"{ds}_{stream}_{pol}_s{s}"] = np.array([e, r])
        print(f"  {ds} {stream:6s} {pol:8s}: emax={np.mean(em):.3f}±{np.std(em):.3f}  C_r={np.mean(rb):.1f}  ({time.time()-t0:.0f}s)", flush=True)
np.savez("../results/e2_count_sweep.npz", **out)
print("===== e2 요약: (C_r, emax) 프런티어 — 동일 C_r에서 mass < fixed 이면 조기재구축 효과가 아님 =====")
for ds, stream, seeds in STREAMS:
    print(f"  [{ds} {stream}]")
    for pol in POLICIES:
        v = np.mean([out[f"{ds}_{stream}_{pol}_s{s}"] for s in seeds], axis=0)
        print(f"     {pol:8s}  C_r={v[1]:.1f}  emax={v[0]:.3f}")
print(f"[done] {time.time()-t0:.0f}s")
