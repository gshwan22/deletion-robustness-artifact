"""c5: 수리 활성 상태의 유지보수 정책 비교 — repair-only vs fixed20+repair vs hubmass0.3+repair
스트림: 배치 2.5%, 누적 60%, GIST+SIFT, hub/random, 3 seeds. 재구축 = 생존 벡터로 HNSW 재빌드."""
import numpy as np, time
from c_common import *

def build_rev(nbrs, n):
    rev = [[] for _ in range(n)]
    src, dst = np.where(nbrs >= 0)
    for u, j in zip(src, dst):
        rev[nbrs[u, j]].append(int(u))
    return rev

def repair_batch(base, nbrs, rev, alive, newly, cap):
    rewritten = 0
    for v in newly:
        outs_v = [int(w) for w in nbrs[v] if w >= 0 and alive[w]]
        for u in rev[v]:
            if not alive[u]: continue
            row = nbrs[u]
            if v not in row: continue
            cand = [int(w) for w in row if w >= 0 and alive[w] and w != v]
            cs = set(cand)
            for w in outs_v:
                if w != u and w not in cs:
                    cand.append(w); cs.add(w)
            if cand:
                d = ((base[cand] - base[u]) ** 2).sum(1)
                keep = [cand[i] for i in np.argsort(d)[:cap]]
            else:
                keep = []
            row[:] = -1; row[:len(keep)] = keep
            for w in keep: rev[w].append(int(u))
            rewritten += 1
    return rewritten

t0 = time.time(); out = {}
BATCH, CUM = 0.025, 0.60
for ds in ["gist", "sift"]:
    for seed in [0, 1, 2]:
        base, q = load(ds, seed)
        nbrs0, indeg0 = build_hnsw(base)
        n, cap = len(base), nbrs0.shape[1]
        od_all = orders(base, indeg0, seed)
        for mode in ["hub", "random"]:
            order = od_all[mode]
            for policy in ["repair_only", "fixed20_rep", "hubmass30_rep"]:
                rng = np.random.default_rng(1100 + seed)
                nbrs = nbrs0.copy(); rev = build_rev(nbrs, n)
                alive = np.ones(n, bool)
                indeg_ref = indeg0.copy(); mass_ref = indeg_ref.sum()
                del_mass = 0; del_cnt_since = 0; rebuilds = 0; rew = 0
                recs = []
                steps = int(CUM / BATCH)
                ptr = 0
                for st in range(steps):
                    newly = order[ptr:ptr + int(BATCH * n)]
                    # order에는 이미 죽은 노드가 없음(전역 순서) — alive만 죽임
                    newly = newly[alive[newly]]
                    alive[newly] = False
                    rew += repair_batch(base, nbrs, rev, alive, newly, cap)
                    del_mass += indeg_ref[newly].sum(); del_cnt_since += len(newly)
                    ptr += int(BATCH * n)
                    trigger = False
                    if policy == "fixed20_rep" and del_cnt_since >= 0.20 * alive.sum() + 0:
                        trigger = del_cnt_since >= 0.20 * n * 0.999
                    if policy == "hubmass30_rep" and del_mass / mass_ref > 0.30:
                        trigger = True
                    if trigger:
                        ids = np.where(alive)[0]
                        sub = np.ascontiguousarray(base[ids])
                        nbrs_s, indeg_s = build_hnsw(sub)
                        # 전역 좌표로 환원
                        nbrs = np.full_like(nbrs0, -1)
                        conv = np.where(nbrs_s >= 0, ids[np.clip(nbrs_s, 0, None)], -1)
                        nbrs[ids, :nbrs_s.shape[1]] = conv
                        rev = build_rev(nbrs, n)
                        indeg_ref = np.zeros(n, np.int64); indeg_ref[ids] = indeg_s
                        mass_ref = indeg_ref.sum(); del_mass = 0; del_cnt_since = 0
                        rebuilds += 1
                    recs.append(recall_at(base, alive, nbrs, q, rng))
                emax = 1 - min(recs)
                tag = f"{ds}_{mode}_{policy}_s{seed}"
                out[tag] = np.array(recs); out[tag+"_meta"] = np.array([emax, rebuilds, rew])
                print(f"  [{ds}] s{seed} {mode:6s} {policy:14s} emax={emax:.3f} rebuilds={rebuilds} rewrites={rew} ({time.time()-t0:.0f}s)", flush=True)
np.savez("../results/c5_trigger_repair.npz", **out)
print("===== c5 요약: emax / rebuilds (3 seeds 평균) =====")
for ds in ["gist", "sift"]:
    for mode in ["hub", "random"]:
        line = []
        for policy in ["repair_only", "fixed20_rep", "hubmass30_rep"]:
            ms = np.mean([out[f"{ds}_{mode}_{policy}_s{s}_meta"] for s in [0,1,2]], axis=0)
            line.append(f"{policy}: {ms[0]:.3f}/r{ms[1]:.1f}")
        print(f"  {ds} {mode}: " + "  ".join(line))
print("  판정: 고-h(hub)에서 repair_only 의 emax 가 크게 남으면 -> 수리 시대에도 질량 트리거 필요")
print(f"[done] {time.time()-t0:.0f}s")
