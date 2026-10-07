"""e7: 표 4 단일시드 행 보강 — adaptive·k-core 를 SIFT/GIST 5시드로 (±std 확보). traversal 은 제외.
e5: 수리 활성 스트림에서의 개수 스윕 (fixed 10/15/25 + mass 15/45) — 자체 구현 국소 재연결 수리(§3.6 규칙)
    ※ e5 의 수리 구현은 c_repair 와 독립 구현이므로, 기존 표 13(c5/d1) 과 수치가 다를 수 있음 → 비교는 e5 내부에서만."""
import numpy as np, time
from c_common import load, build_hnsw, recall_at, F_GRID
t0 = time.time()
LOG = lambda s: print(s, flush=True)
def f_at(F, rec, thr):
    rec = np.asarray(rec, float); below = np.where(rec < thr)[0]
    if len(below) == 0: return None
    i = below[0]
    if i == 0: return float(F[0])
    f0, f1, r0, r1 = F[i-1], F[i], rec[i-1], rec[i]
    return float(f0 + (r0 - thr) / (r0 - r1) * (f1 - f0)) if r0 != r1 else float(f1)

# ═══════════ e7 ═══════════
out7 = {}
def sweep_adaptive(base, q, nbrs0, seed, batch=0.005):
    n = len(base); nbrs = nbrs0.copy(); alive = np.ones(n, bool); deleted = 0; recs = []
    targets = [int(f * n) for f in F_GRID]; rng = np.random.default_rng(100 + seed)
    for tgt in targets:
        while deleted < tgt:
            need = min(int(batch * n), tgt - deleted)
            if need <= 0: break
            indeg = np.bincount(nbrs[alive][nbrs[alive] >= 0], minlength=n); indeg[~alive] = -1
            newly = np.argsort(-indeg, kind="stable")[:need]
            alive[newly] = False; nbrs[newly] = -1; deleted += need
        recs.append(recall_at(base, alive, nbrs, q, rng))
    return np.array(recs)
def sweep_kcore(base, q, nbrs0, seed):
    import networkx as nx
    n = len(base)
    G = nx.Graph()
    src = np.repeat(np.arange(n), nbrs0.shape[1]); dst = nbrs0.ravel(); m = dst >= 0
    G.add_edges_from(zip(src[m].tolist(), dst[m].tolist()))
    core = nx.core_number(G); cvals = np.array([core.get(i, 0) for i in range(n)])
    order = np.argsort(-cvals, kind="stable")
    nbrs = nbrs0.copy(); alive = np.ones(n, bool); recs = []; rng = np.random.default_rng(200 + seed); ptr = 0
    for f in F_GRID:
        tgt = int(f * n)
        cand = order[ptr:tgt]; ptr = tgt
        alive[cand] = False; nbrs[cand] = -1
        recs.append(recall_at(base, alive, nbrs, q, rng))
    return np.array(recs)
LOG("===== e7: adaptive / k-core 5 seeds =====")
for ds in ["sift", "gist"]:
    fa, fk = [], []
    for seed in range(5):
        base, q = load(ds, seed); nbrs0, _ = build_hnsw(base)
        ra = sweep_adaptive(base, q, nbrs0, seed); out7[f"{ds}_adaptive_s{seed}"] = ra; fa.append(f_at(F_GRID, ra, 0.5))
        try:
            rk = sweep_kcore(base, q, nbrs0, seed); out7[f"{ds}_kcore_s{seed}"] = rk; fk.append(f_at(F_GRID, rk, 0.5))
        except Exception as e:
            LOG(f"  k-core 실패 ({e})"); fk.append(None)
        LOG(f"  [{ds}] seed {seed}: adaptive f_c={fa[-1]}  kcore f_c={fk[-1]}  ({time.time()-t0:.0f}s)")
    fa2 = [x for x in fa if x is not None]; fk2 = [x for x in fk if x is not None]
    LOG(f"  [{ds}] adaptive f_c = {np.mean(fa2):.3f}±{np.std(fa2):.3f}  | kcore f_c = {np.mean(fk2):.3f}±{np.std(fk2):.3f}" if fk2 else f"  [{ds}] adaptive f_c = {np.mean(fa2):.3f}±{np.std(fa2):.3f}")
np.savez("../results/e7_spectrum_seeds.npz", **out7)

# ═══════════ e5: 수리 활성 개수 스윕 ═══════════
LOG("===== e5: 수리 활성 스트림 개수 스윕 (자체 국소수리) =====")
STEP, NSTEP = 0.025, 24
POL5 = ["never_rep", "fixed10", "fixed15", "fixed20", "fixed25", "mass15", "mass30", "mass45"]
def build_rev(nbrs, n):
    rev = [set() for _ in range(n)]
    for u in range(n):
        for w in nbrs[u]:
            if w >= 0: rev[w].add(u)
    return rev
def repair_delete(v, nbrs, rev, base, alive, cap):
    outs_v = [int(w) for w in nbrs[v] if w >= 0 and alive[w]]
    for u in list(rev[v]):
        if not alive[u]: continue
        row = nbrs[u]
        cand = {int(w) for w in row if w >= 0 and alive[w] and w != v} | set(outs_v); cand.discard(u)
        keep = []
        if cand:
            cl = np.fromiter(cand, dtype=np.int64)
            dd = np.linalg.norm(base[cl] - base[u], axis=1)
            keep = cl[np.argsort(dd)[:cap]].tolist()
        for w in row:
            if w >= 0: rev[w].discard(u)
        newrow = np.full(nbrs.shape[1], -1, np.int32); newrow[:len(keep)] = keep
        nbrs[u] = newrow
        for w in keep: rev[w].add(u)
    rev[v] = set(); nbrs[v] = -1
def remap(sub, ids, n):
    g = np.full((n, sub.shape[1]), -1, np.int32)
    g[ids] = np.where(sub >= 0, ids[np.clip(sub, 0, None)], -1)
    return g
def sim_rep(policy, seed):
    base, q = load("gist", seed); n = len(base)
    nbrs0, indeg0 = build_hnsw(base); nbrs = nbrs0.copy(); cap = nbrs.shape[1]
    rev = build_rev(nbrs, n); alive = np.ones(n, bool)
    indeg_ref = indeg0.astype(np.int64); mass_ref = indeg_ref.sum(); n_ref = n
    dc = dm = rebuilds = 0; recs = []
    for st in range(NSTEP):
        k = int(STEP * n)
        cand = np.argsort(-np.where(alive, indeg_ref, -1), kind="stable")[:k]
        for v in cand:
            alive[v] = False; repair_delete(int(v), nbrs, rev, base, alive, cap)
        dc += k; dm += indeg_ref[cand].sum()
        fire = False
        if policy.startswith("fixed"): fire = dc / n_ref > int(policy[5:]) / 100
        elif policy.startswith("mass"): fire = dm / mass_ref > int(policy[4:]) / 100
        if fire:
            ids = np.where(alive)[0]
            sub, dloc = build_hnsw(np.ascontiguousarray(base[ids]))
            nbrs = remap(sub, ids, n); rev = build_rev(nbrs, n)
            indeg_ref = np.zeros(n, np.int64); indeg_ref[ids] = dloc
            mass_ref = indeg_ref.sum(); n_ref = alive.sum(); dc = dm = 0; rebuilds += 1
        recs.append(recall_at(base, alive, nbrs, q, np.random.default_rng(4000 + seed + st)))
    return 1 - min(recs), rebuilds
out5 = {}
for pol in POL5:
    em, rb = [], []
    for seed in [0, 1, 2]:
        e, r = sim_rep(pol, seed); em.append(e); rb.append(r)
        out5[f"gist_hub_rep_{pol}_s{seed}"] = np.array([e, r])
        LOG(f"  [e5] {pol:9s} seed {seed}: emax={e:.3f} C_r={r} ({time.time()-t0:.0f}s)")
    LOG(f"  [e5] {pol:9s}: emax={np.mean(em):.3f}±{np.std(em):.3f} C_r={np.mean(rb):.1f}")
    np.savez("../results/e5_count_repair.npz", **out5)   # 정책마다 중간 저장
LOG("===== e5 요약 (수리 활성, 자체 구현) =====")
for pol in POL5:
    v = np.mean([out5[f"gist_hub_rep_{pol}_s{s}"] for s in [0, 1, 2]], axis=0)
    LOG(f"  {pol:9s} C_r={v[1]:.1f} emax={v[0]:.3f}")
LOG(f"[done] {time.time()-t0:.0f}s")
