"""e4: adaptive 순서의 재계산 배치 크기 민감도 (0.25% / 0.5% / 1.0%), SIFT seed 0 — §3.4 강건성 항목"""
import numpy as np, time
from c_common import load, build_hnsw, F_GRID, recall_at
t0 = time.time()

def f_at(F, rec, thr):
    rec = np.asarray(rec, float); below = np.where(rec < thr)[0]
    if len(below) == 0: return None
    i = below[0]
    if i == 0: return float(F[0])
    f0, f1, r0, r1 = F[i-1], F[i], rec[i-1], rec[i]
    return float(f0 + (r0 - thr) / (r0 - r1) * (f1 - f0)) if r0 != r1 else float(f1)

base, q = load("sift", 0); n = len(base)
nbrs0, _ = build_hnsw(base)
out = {}
print("===== e4: adaptive 배치 크기 민감도 (SIFT, seed 0) =====", flush=True)
for batch in [0.0025, 0.005, 0.01]:
    nbrs = nbrs0.copy(); alive = np.ones(n, bool); deleted = 0; recs = []
    rng = np.random.default_rng(7)
    for f in F_GRID:
        tgt = int(f * n)
        while deleted < tgt:
            need = min(int(batch * n), tgt - deleted)
            if need <= 0: break
            indeg = np.bincount(nbrs[alive][nbrs[alive] >= 0], minlength=n); indeg[~alive] = -1
            newly = np.argsort(-indeg, kind="stable")[:need]
            alive[newly] = False; nbrs[newly] = -1; deleted += need
        recs.append(recall_at(base, alive, nbrs, q, rng))
    recs = np.array(recs); out[f"sift_adaptive_b{batch}"] = recs
    fc, f90 = f_at(F_GRID, recs, 0.5), f_at(F_GRID, recs, 0.9)
    print(f"  batch={batch*100:.2f}%: f_c={fc:.3f}  f_0.9={f90:.3f}  ({time.time()-t0:.0f}s)", flush=True)
np.savez("../results/e4_adaptive_batch.npz", **out)
print("  판정: 세 f_c 차이가 ±0.01 이내면 0.5% 배치 선택은 결과에 무관")
print(f"[done] {time.time()-t0:.0f}s")
