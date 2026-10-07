"""e3: 엄격 임계 f_0.9 (recall이 0.9 아래로 처음 떨어지는 f) — 기존 곡선에서 추출 (신규 탐색 없음)
e4: adaptive 순서의 배치 크기 민감도 (0.25% / 0.5% / 1.0%), SIFT seed 0"""
import numpy as np, time, glob
from c_common import load, build_hnsw, F_GRID, recall_at
t0 = time.time()

# ───────── e3: 엄격 임계 ─────────
def f_at(F, rec, thr):
    rec = np.asarray(rec, float)
    below = np.where(rec < thr)[0]
    if len(below) == 0: return None
    i = below[0]
    if i == 0: return float(F[0])
    f0, f1, r0, r1 = F[i-1], F[i], rec[i-1], rec[i]
    return float(f0 + (r0 - thr) / (r0 - r1) * (f1 - f0)) if r0 != r1 else float(f1)

print("===== e3: f_0.9 (recall 0.9 임계) vs f_c (0.5 임계) =====")
found = 0
for path in sorted(glob.glob("../results/*.npz")):
    try: d = np.load(path)
    except Exception: continue
    F = d["F"] if "F" in d.files else F_GRID
    rows = {}
    for k in d.files:
        v = d[k]
        if v.ndim == 1 and len(v) == len(F) and np.all((v >= 0) & (v <= 1.0001)) and any(t in k for t in ["random", "hub", "relevance", "antihub", "adaptive", "oldest", "newest"]):
            base_key = k.split("_s")[0] if "_s" in k else k
            rows.setdefault(base_key, []).append(v)
    if not rows: continue
    print(f"  [{path.split('/')[-1]}]")
    for bk, lst in sorted(rows.items()):
        m = np.mean(lst, axis=0)
        f90, f50 = f_at(F, m, 0.9), f_at(F, m, 0.5)
        fmt = lambda x: f"{x:.3f}" if x is not None else ">max"
        print(f"     {bk:34s} f_0.9={fmt(f90)}  f_c={fmt(f50)}  (n={len(lst)})")
        found += 1
print(f"  총 {found} 곡선 처리")

# ───────── e4: adaptive 배치 민감도 ─────────
print("===== e4: adaptive 배치 크기 민감도 (SIFT, seed 0) =====")
base, q = load("sift", 0)
n = len(base)
nbrs0, indeg0 = build_hnsw(base)
out = {}
for batch in [0.0025, 0.005, 0.01]:
    nbrs = nbrs0.copy(); alive = np.ones(n, bool)
    deleted = 0; recs = []
    gi = 0
    targets = [int(f * n) for f in F_GRID]
    rng = np.random.default_rng(7)
    while gi < len(targets):
        # 다음 그리드 점까지 micro-batch 로 삭제 (매 batch 마다 in-degree 재계산)
        while deleted < targets[gi]:
            need = min(int(batch * n), targets[gi] - deleted)
            if need <= 0: break
            indeg = np.bincount(nbrs[alive][nbrs[alive] >= 0], minlength=n); indeg[~alive] = -1
            newly = np.argsort(-indeg, kind="stable")[:need]
            alive[newly] = False; nbrs[newly] = -1
            deleted += need
        recs.append(recall_at(base, alive, nbrs, q, rng))
        gi += 1
    recs = np.array(recs)
    fc = f_at(F_GRID, recs, 0.5); f90 = f_at(F_GRID, recs, 0.9)
    out[f"sift_adaptive_b{batch}"] = recs
    print(f"  batch={batch*100:.2f}%: f_c={fc if fc is None else round(fc,3)}  f_0.9={f90 if f90 is None else round(f90,3)}  ({time.time()-t0:.0f}s)", flush=True)
np.savez("../results/e4_adaptive_batch.npz", **out)
print("  판정: 세 배치에서 f_c 차이가 ±0.01 이내면 0.5% 선택은 결과에 무관 (§3.4 강건성 항목 추가)")
print(f"[done] {time.time()-t0:.0f}s")
