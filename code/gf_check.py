#!/usr/bin/env python3
"""
ANN-Percolation — GF(생성함수) 항행성 이론 검증 계산
====================================================
이론 구조:
  전진차수  k_fwd(u; q) = u 의 out-이웃 중 쿼리 q 에 더 가까운 이웃 수
  G_fwd(x) = E[ x^{k_fwd} | k_fwd >= 1 ]      (전진 가능 노드 조건부, 실측 분포로)
  p~(f)    = 전진 엣지 중 머리(head)가 삭제된 비율 (모드별 실측)
  스텝 생존 = 1 - G_fwd(p~),  경로 L 스텝 -> recall_pred(f) = [1 - G_fwd(p~(f))]^L

프로토콜 (자유 파라미터 규율):
  L 은 random 곡선에만 캘리브레이션(최소자승 스캔) -> 같은 L 로 hub 곡선을 '예측'
  즉 hub 예측은 zero free parameter.

검증 항목:
  V1: P(k_fwd) 분포 (전진 이웃의 허브 편향 확인: 전진 이웃 평균 k_in vs 전체)
  V2: p~(f) vs f — hub 에서 p~ >> f (질량 편향), random 에서 p~ ~= f
  V3: 예측 recall(f) vs 실측 (figdata_curves5_v2) — 곡선 형태·f_c 재현 여부
출력: 콘솔 표 + results/gf_check.npz + figures/gf_check.png (작업용)
"""
import time
import numpy as np
import h5py
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "../data/sift-128-euclidean.hdf5"
N = 100_000; M = 16; EF_CONSTR = 200
N_Q = 50                     # 이론 통계용 쿼리 수
F_GRID = np.round(np.linspace(0.0, 0.95, 20), 3)
SEED = 0
L_SCAN = np.arange(1, 81)

def load():
    with h5py.File(DATA_PATH, "r") as f:
        full = np.asarray(f["train"], dtype=np.float32)
        q = np.asarray(f["test"], dtype=np.float32)
    rng = np.random.default_rng(1000 + SEED)
    base = np.ascontiguousarray(full[rng.choice(full.shape[0], size=N, replace=False)])
    return base, np.ascontiguousarray(q[:N_Q])

def build(base):
    n, d = base.shape
    ix = faiss.IndexHNSWFlat(d, M); ix.hnsw.efConstruction = EF_CONSTR; ix.add(base)
    h = ix.hnsw
    nb = faiss.vector_to_array(h.neighbors)
    off = faiss.vector_to_array(h.offsets).astype(np.int64)
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
    n0 = int(cum[1])
    idx = off[:-1][:, None] + np.arange(n0)[None, :]
    g = nb[idx]; g = np.where(g < 0, np.int32(-1), g).astype(np.int32)
    indeg = np.bincount(g[g >= 0], minlength=n).astype(np.int64)
    return g, indeg

t0 = time.time()
base, Q = load()
g, indeg = build(base)
n, n_slots = g.shape
valid = g >= 0
print(f"[build] n={n} slots={n_slots} <k_in>={indeg.mean():.1f}")

# 삭제 순서 (모드별)
rng = np.random.default_rng(SEED)
order = {"hub": np.argsort(-indeg, kind="stable"), "random": rng.permutation(n)}

# ---------- 쿼리별 전진 마스크 집계 ----------
# fwd[u, j] = (g[u,j] 가 u 보다 쿼리에 가깝다)
kfwd_pool = []            # V1: (q,u) 풀의 k_fwd 값들
fwd_head_indeg = []       # 전진 이웃의 k_in (허브 편향 확인)
# p~ 계산용 누적: 모드별·f별 (전진엣지 총수, head 삭제 전진엣지 수)
p_tilde = {m: np.zeros((len(F_GRID), 2)) for m in order}
removed_mask = {m: {i: None for i in range(len(F_GRID))} for m in order}
for m in order:
    for i, f in enumerate(F_GRID):
        rm = np.zeros(n, bool); rm[order[m][: int(round(f * n))]] = True
        removed_mask[m][i] = rm

for qi in range(N_Q):
    Dq = ((base - Q[qi]) ** 2).sum(1)               # (n,)
    Dn = np.where(valid, Dq[np.clip(g, 0, None)], np.inf)
    fwd = valid & (Dn < Dq[:, None])                # (n, slots)
    kf = fwd.sum(1)
    kfwd_pool.append(kf)
    heads = g[fwd]
    fwd_head_indeg.append(indeg[heads])
    for m in order:
        for i in range(len(F_GRID)):
            rm = removed_mask[m][i]
            tail_alive = ~rm
            fe = fwd & tail_alive[:, None]          # 생존 tail 의 전진 엣지
            tot = fe.sum()
            dead = (fe & rm[np.clip(g, 0, None)]).sum()
            p_tilde[m][i, 0] += tot
            p_tilde[m][i, 1] += dead
    if (qi + 1) % 10 == 0:
        print(f"  query {qi+1}/{N_Q} ({time.time()-t0:.0f}s)")

kfwd_pool = np.concatenate(kfwd_pool)
fwd_head_indeg = np.concatenate(fwd_head_indeg)

# ---------- V1: 분포·허브 편향 ----------
frac0 = (kfwd_pool == 0).mean()
kf_pos = kfwd_pool[kfwd_pool >= 1]
print(f"\n[V1] P(k_fwd=0)={frac0:.3f}  E[k_fwd|>=1]={kf_pos.mean():.2f}")
print(f"     전진 이웃 평균 k_in = {fwd_head_indeg.mean():.1f}  vs 전체 평균 {indeg.mean():.1f} "
      f"(비 {fwd_head_indeg.mean()/indeg.mean():.2f}x)  <- 허브 편향")

# ---------- V2: p~(f) ----------
pt = {m: p_tilde[m][:, 1] / np.maximum(p_tilde[m][:, 0], 1) for m in order}
print("\n[V2] p~(f) (전진엣지 head 삭제율):")
print("   f    p~_hub   p~_rand")
for i in [2, 5, 8, 11, 14, 16]:
    print(f"  {F_GRID[i]:.2f}  {pt['hub'][i]:.3f}   {pt['random'][i]:.3f}")

# ---------- V3: GF 예측 ----------
def G_fwd(x):
    return np.mean(np.power(float(x), kf_pos))

d5 = np.load("../results/figdata_curves5_v2.npz")
meas = {"hub": d5["hub_rec_mu"], "random": d5["random_rec_mu"]}

def pred_curve(m, L):
    surv = np.array([1.0 - G_fwd(pt[m][i]) for i in range(len(F_GRID))])
    return surv ** L

# L 캘리브레이션: random 만
errs = [np.mean((pred_curve("random", L) - meas["random"]) ** 2) for L in L_SCAN]
L_star = int(L_SCAN[int(np.argmin(errs))])
print(f"\n[V3] L* (random 캘리브레이션) = {L_star}")

def fcross(rec):
    thr = rec[0] * 0.5
    for i in range(1, len(F_GRID)):
        if rec[i] < thr <= rec[i-1]:
            f0, f1, r0, r1 = F_GRID[i-1], F_GRID[i], rec[i-1], rec[i]
            return f0 + (r0 - thr) / (r0 - r1 + 1e-12) * (f1 - f0)
    return None

print("   mode     f_c(측정)   f_c(예측)")
out = {"F": F_GRID, "L_star": np.array(L_star), "kfwd": kfwd_pool[:200000]}
for m in ["random", "hub"]:
    pc = pred_curve(m, L_star)
    out[f"{m}_pred"] = pc; out[f"{m}_meas"] = meas[m]; out[f"{m}_ptilde"] = pt[m]
    fm, fp = fcross(meas[m]), fcross(pc)
    print(f"   {m:7s}  {fm if fm is None else round(fm,3)}      {fp if fp is None else round(fp,3)}")
np.savez("../results/gf_check.npz", **out)

# ---------- 작업용 그림 ----------
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
ks = np.sort(kf_pos); ax[0].plot(ks, 1 - np.arange(1, len(ks)+1)/len(ks), lw=1.5)
ax[0].set(xlabel="k_fwd (>=1)", ylabel="CCDF", title="V1: forward-degree dist", yscale="log")
ax[1].plot(F_GRID, F_GRID, "--", c="#999", label="p~=f")
ax[1].plot(F_GRID, pt["random"], "-o", ms=3, c="#2E6FBA", label="random")
ax[1].plot(F_GRID, pt["hub"], "-o", ms=3, c="#D1453B", label="hub")
ax[1].set(xlabel="f", ylabel="p~(f)", title="V2: forward-edge deletion bias"); ax[1].legend()
for m, c in [("random", "#2E6FBA"), ("hub", "#D1453B")]:
    ax[2].plot(F_GRID, meas[m], "-o", ms=3, color=c, label=f"{m} measured")
    ax[2].plot(F_GRID, out[f"{m}_pred"], "--", color=c, label=f"{m} GF pred")
ax[2].set(xlabel="f", ylabel="recall", title=f"V3: prediction (L*={L_star})"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig("../figures/gf_check.png", dpi=140)
print(f"\n[done] {time.time()-t0:.1f}s -> results/gf_check.npz, figures/gf_check.png")
