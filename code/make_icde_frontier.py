"""fig9_frontier.pdf — (a) static hub no repair (e2b), (b) adaptive hub no repair (e2), (c) adaptive hub + local repair (e5)
실행: cd ~/Desktop/Work_SH/ANN-Percolation/src && python make_icde_frontier.py"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, OUT = "../results", "../figures"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 10, "legend.fontsize": 7.5, "pdf.fonttype": 42})
COUNT_C, MASS_C = "#A5ACB8", "#4C9085"   # 마스터 팔레트: 기준선 회색 / 질량 딥틸

def mean_pts(d, prefix, pols, seeds):
    pts = []
    for p in pols:
        ks = [f"{prefix}_{p}_s{s}" for s in seeds if f"{prefix}_{p}_s{s}" in d.files]
        if not ks: continue
        v = np.mean([d[k] for k in ks], axis=0)      # [emax, C_r, ...]
        pts.append((float(v[1]), float(v[0]), p))
    return sorted(pts)

panels = [
    ("e2b_count_static.npz", "gist_hubstatic", list(range(5)),
     ["fixed25", "fixed20", "fixed15", "fixed10", "fixed05"], ["mass30", "mass15"]),
    ("e2_count_sweep.npz", "gist_hub", list(range(5)),
     ["fixed25", "fixed20", "fixed15", "fixed10", "fixed05"], ["mass30", "mass15"]),
    ("e5_count_repair.npz", "gist_hub_rep", list(range(3)),
     ["fixed25", "fixed20", "fixed15", "fixed10"], ["mass45", "mass30", "mass15"]),
]
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3), sharey=False)
for ax, (fn, pre, seeds, cpol, mpol), tag in zip(axes, panels, ["(a)", "(b)", "(c)"]):
    d = np.load(f"{R}/{fn}")
    cp = mean_pts(d, pre, cpol, seeds); mp = mean_pts(d, pre, mpol, seeds)
    ax.plot([p[0] for p in cp], [p[1] for p in cp], "s-", color=COUNT_C, ms=4.5, lw=1.3, label="Count")
    ax.plot([p[0] for p in mp], [p[1] for p in mp], "o-", color=MASS_C, ms=5, lw=1.5, label="Mass")
    ax.set_yscale("log"); ax.set_xlabel("$C_r$")
    ax.set_title(tag, loc="left", fontsize=10, fontweight="bold", pad=4)
    print(tag, "count", [(round(a, 1), round(b, 3)) for a, b, _ in cp], "| mass", [(round(a, 1), round(b, 3)) for a, b, _ in mp])
axes[0].set_ylabel(r"$\varepsilon_{\max}$")
axes[0].legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig9_frontier.pdf"); print("saved fig9_frontier.pdf")
