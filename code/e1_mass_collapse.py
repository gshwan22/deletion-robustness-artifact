"""e1: 한 줄 주장의 직접 검증 — x축을 삭제 개수(f)에서 누적 삭제 질량으로 바꾸면 곡선들이 모이는가.
기존 npz의 recall 곡선(c3/c4/c6b) + 순서 재생성으로 질량 궤적 계산. 신규 탐색 없음(adaptive 순서 재생성만 그래프 연산).
출력: 분산 축소비 + fig8_mass_collapse.pdf + c3 oldest 수치 요약"""
import numpy as np, time, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from c_common import load, build_hnsw, orders, F_GRID, fc_from_curve
t0 = time.time()
R = "../results"
c3 = np.load(f"{R}/c3_age.npz"); c4 = np.load(f"{R}/c4_repair.npz"); c6 = np.load(f"{R}/c6b_repair_ext.npz")

def adaptive_sets(nbrs0, n, grid):
    """탐색 없이 adaptive 삭제 집합 재생성 (매 그리드 스텝 현재 in-degree 상위)"""
    nbrs = nbrs0.copy(); alive = np.ones(n, bool); prev = 0; dele = []
    for f in grid:
        need = int(f*n) - prev
        if need > 0:
            indeg = np.bincount(nbrs[alive][nbrs[alive] >= 0], minlength=n)
            indeg[~alive] = -1
            newly = np.argsort(-indeg, kind="stable")[:need]
            alive[newly] = False; dele.append(newly); prev += need
        else:
            dele.append(np.array([], dtype=np.int64))
    return dele

summary = {}
for ds in ["gist", "sift"]:
    # 순서별 (질량궤적, recall곡선) 수집 — seed 0..2 평균
    curves = {}   # name -> (mass[FG], rec[FG]) seed 평균
    acc = {}
    for seed in [0, 1, 2]:
        base, _ = load(ds, seed)
        nbrs0, indeg = build_hnsw(base)
        n = len(base); tot = indeg.sum()
        od = orders(base, indeg, seed)
        pool = {
            "random": (od["random"], c4[f"{ds}_random_norep_s{seed}"]),
            "hub":    (od["hub"],    c4[f"{ds}_hub_norep_s{seed}"]),
            "oldest": (od["oldest"], c3[f"{ds}_oldest_s{seed}"]),
            "newest": (od["newest"], c3[f"{ds}_newest_s{seed}"]),
        }
        for name, (order, rec) in pool.items():
            mass = np.array([indeg[order[:int(f*n)]].sum()/tot for f in F_GRID])
            acc.setdefault(name, []).append((mass, np.array(rec)))
        if ds == "gist":
            dele = adaptive_sets(nbrs0, n, F_GRID)
            cum = 0; mass = []
            for dset in dele:
                cum += indeg[dset].sum(); mass.append(cum/tot)
            acc.setdefault("adaptive", []).append((np.array(mass), np.array(c6[f"gist_adaptive_norep_s{seed}"])))
        print(f"  [{ds}] s{seed} orders done ({time.time()-t0:.0f}s)", flush=True)
    for name, lst in acc.items():
        m = np.mean([x[0] for x in lst], axis=0); r = np.mean([x[1] for x in lst], axis=0)
        curves[name] = (m, r)
    # 분산 비교: 동일 f 에서의 recall 표준편차 vs 동일 질량에서의 표준편차
    names = list(curves)
    recs_f = np.stack([curves[nm][1] for nm in names])
    disp_f = recs_f.std(axis=0).mean()
    mgrid = np.linspace(0.05, min(curves[nm][0].max() for nm in names), 25)
    recs_m = np.stack([np.interp(mgrid, curves[nm][0], curves[nm][1]) for nm in names])
    disp_m = recs_m.std(axis=0).mean()
    summary[ds] = (disp_f, disp_m, disp_f/disp_m if disp_m > 0 else np.inf)
    print(f"  [{ds}] 순서 {len(names)}종 | 동일-f 분산 {disp_f:.4f} vs 동일-질량 분산 {disp_m:.4f} | 축소비 {disp_f/disp_m:.2f}x", flush=True)
    # 그림
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    cols = {"random":"#7F8C8D","hub":"#C0392B","oldest":"#2980B9","newest":"#27AE60","adaptive":"#8E44AD"}
    for nm in names:
        axes[0].plot(F_GRID, curves[nm][1], color=cols[nm], lw=1.5, label=nm)
        axes[1].plot(curves[nm][0], curves[nm][1], color=cols[nm], lw=1.5)
    axes[0].set_xlabel("deleted fraction $f$"); axes[1].set_xlabel("deleted in-edge mass fraction")
    axes[0].set_ylabel("recall@10"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title(f"{ds.upper()}: by count", fontsize=9); axes[1].set_title("by mass", fontsize=9)
    fig.tight_layout(); fig.savefig(f"../figures/pdf/fig8_mass_collapse_{ds}.pdf")
np.savez(f"{R}/e1_mass_collapse.npz", **{f"{d}_dispf_dispm_ratio": np.array(v) for d, v in summary.items()})
print("===== e1 요약 =====")
for d, (a, b, r) in summary.items():
    print(f"  {d}: 동일-f 분산 {a:.4f} -> 동일-질량 분산 {b:.4f} (축소 {r:.2f}x)")
print("  판정: 축소비 >2x 면 '질량이 올바른 상태변수'의 직접 증거. ~1x 면 proxy로 강등 서술")
print("===== c3 oldest-first 수치 (§8 인용용) =====")
for ds in ["sift", "gist", "textbge"]:
    for mode in ["oldest", "newest", "random"]:
        fcs = [fc_from_curve(F_GRID, c3[f"{ds}_{mode}_s{s}"]) for s in [0,1,2]]
        fcs2 = [x for x in fcs if x is not None]
        v = f"{np.mean(fcs2):.3f}±{np.std(fcs2):.3f}" if fcs2 else ">0.95"
        print(f"  {ds} {mode}: fc={v}")
print(f"[done] {time.time()-t0:.0f}s")
