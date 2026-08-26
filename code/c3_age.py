"""c3: 삽입 순서와 라우팅 질량 — corr(나이, k_in), oldest/newest-first 삭제의 fc·질량 궤적"""
import numpy as np, time
from c_common import *
t0 = time.time(); out = {}
for ds in ["sift", "gist", "textbge"]:
    for seed in [0, 1, 2]:
        base, q = load(ds, seed)
        nbrs, indeg = build_hnsw(base)
        n = len(base)
        corr = np.corrcoef(np.arange(n), indeg)[0, 1]
        # 질량 궤적: oldest-first 삭제 시 K_in(D)/K_in(V)
        cum_old = np.cumsum(indeg) / indeg.sum()          # index 순 누적
        mass_at = {f: cum_old[int(f*n)-1] for f in [0.1, 0.2, 0.3]}
        rng = np.random.default_rng(700 + seed)
        od = orders(base, indeg, seed)
        fcs = {}
        for mode in ["oldest", "newest", "random"]:
            rec = []
            for f in F_GRID:
                alive = np.ones(n, bool); alive[od[mode][:int(f*n)]] = False
                rec.append(recall_at(base, alive, nbrs, q, rng))
            fcs[mode] = fc_from_curve(F_GRID, rec)
            out[f"{ds}_{mode}_s{seed}"] = np.array(rec)
        print(f"  [{ds}] s{seed} corr(age,kin)={corr:+.3f} mass@10/20/30%={mass_at[0.1]:.3f}/{mass_at[0.2]:.3f}/{mass_at[0.3]:.3f} "
              f"fc old={fcs['oldest']} new={fcs['newest']} rand={fcs['random']} ({time.time()-t0:.0f}s)", flush=True)
        out[f"{ds}_corr_s{seed}"] = np.array([corr])
np.savez("../results/c3_age.npz", F=F_GRID, **out)
print("===== c3 요약 =====")
for ds in ["sift", "gist", "textbge"]:
    cs = [out[f"{ds}_corr_s{s}"][0] for s in [0,1,2]]
    print(f"  {ds}: corr(age,kin)={np.mean(cs):+.3f}±{np.std(cs):.3f}")
print("  판정: corr<0 이면 오래된 노드가 고질량 -> 만료(oldest-first) 삭제 = 자연 허브편중의 증거")
print(f"[done] {time.time()-t0:.0f}s")
