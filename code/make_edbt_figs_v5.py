"""EDBT 그림 v2 — 반영: 타이틀 제거, 기호만, 괄호형 범례, 인셋 분리, 팔레트·에러밴드
실행: cd ~/Desktop/Work_SH/ANN-Percolation/src && python make_edbt_figs_v2.py"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, OUT = "../results", "../paper-edbt/figures"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 10, "legend.fontsize": 7.5, "pdf.fonttype": 42})
# 마스터 팔레트 (역할 기반, 전 그림 공통)
C = {"rand": "#A5ACB8", "hub": "#FC8D62", "rel": "#E78AC3", "rep": "#4C9085",
     "ada": "#B48EAD", "old": "#7FB8DA", "new": "#E6C35C"}
SEEDS = [0, 1, 2]
def ms(d, pat):
    a = np.stack([d[pat.format(s=s)] for s in SEEDS]); return a.mean(0), a.std(0)
def tag(ax, s):  # 패널 마커: 축 밖 좌상단
    ax.set_title(s, loc="left", fontsize=10, fontweight="bold", pad=6)

# ================= fig4_decomp =================
g = np.load(f"{R}/nq_gold.npz"); F = g["F"]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
for mode, c, lab in [("random", C["rand"], "Random"), ("hub", C["hub"], "Hub"), ("relevance", C["rel"], "Relevance")]:
    m, s = ms(g, mode + "_s{s}")
    axes[0].plot(F, m[0], color=c, lw=1.6, label=lab)
    axes[0].fill_between(F, m[0]-s[0], m[0]+s[0], color=c, alpha=0.15, lw=0)
axes[0].set_xlabel("$f$"); axes[0].set_ylabel("$R_g(f)$")
axes[0].set_xlim(0, 0.95); axes[0].set_ylim(0, 1.0); axes[0].legend(frameon=False); tag(axes[0], "(a)")
for mode, c, lab in [("random", C["rand"], "Random"), ("hub", C["hub"], "Hub")]:
    hits = np.stack([g[f"{mode}_s{s}"] for s in SEEDS])
    cond = hits[:, 2] / np.clip(1 - hits[:, 1], 1e-9, None)
    m, s = cond.mean(0), cond.std(0)
    axes[1].plot(F, m, color=c, lw=1.6, label=lab)
    axes[1].fill_between(F, m-s, m+s, color=c, alpha=0.15, lw=0)
for mode, c, fc in [("hub", C["hub"], 0.796), ("random", C["rand"], 0.919)]:
    hits = np.stack([g[f"{mode}_s{s0}"] for s0 in SEEDS])
    cond = (hits[:, 2] / np.clip(1 - hits[:, 1], 1e-9, None)).mean(0)
    yfc = np.interp(fc, F, cond)
    axes[1].plot([fc], [yfc], marker="*", ms=10, color=c, mec="#555", mew=0.5, zorder=5)
    nm = "hub" if mode == "hub" else "rand"
    axes[1].annotate(f"$f_c^{{\\mathrm{{{nm}}}}}{{=}}{fc:.2f}$", (fc, yfc),
                     textcoords="offset points", xytext=(-8, 9), fontsize=7, ha="right", color=c)
axes[1].set_xlabel("$f$"); axes[1].set_ylabel("$\\tilde{U}(f)$")
axes[1].set_xlim(0, 0.95); axes[1].set_ylim(0, 1.05); axes[1].legend(frameon=False, loc="upper left"); tag(axes[1], "(b)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_decomp.pdf"); plt.close(fig); print("fig4 ok")

# ================= fig7_repair: 인셋 분리 → 2패널 =================
c4 = np.load(f"{R}/c4_repair.npz"); c6 = np.load(f"{R}/c6b_repair_ext.npz"); F = c4["F"]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
for pat, dd, c, lab, ls in [
    ("gist_hub_norep_s{s}", c4, C["hub"], "Hub (no repair)", "-"),
    ("gist_hub_rep_s{s}", c4, C["rep"], "Hub (repair)", "-"),
    ("gist_adaptive_rep_s{s}", c6, C["ada"], "Adaptive (repair)", "--"),
    ("gist_random_norep_s{s}", c4, C["rand"], "Random (no repair)", ":")]:
    m, s = ms(dd, pat)
    axes[0].plot(F, m, ls, color=c, lw=1.5, label=lab)
    axes[0].fill_between(F, m-s, m+s, color=c, alpha=0.15, lw=0)
axes[0].set_xlabel("$f$"); axes[0].set_ylabel("recall@10")
axes[0].set_xlim(0, 0.95); axes[0].set_ylim(0, 1.02)
axes[0].legend(frameon=False, loc="lower left", fontsize=7); tag(axes[0], "(a)")
hs = [4.2, 4.8, 25.2, 40.2]; rec = [1.0, 1.0, 0.45, 0.29]
lbl = ["SIFT", "NQ", "GIST", "MRNG\u00b7GIST"]
mk = ["o", "s", "^", "D"]
axes[1].semilogx(hs, rec, "-", color=C["rep"], lw=1.4, zorder=1)
for x, y, l, m in zip(hs, rec, lbl, mk):
    axes[1].semilogx([x], [y], m, color=C["rep"], ms=5.5, mec="#333", mew=0.6, zorder=3, label=l)
axes[1].legend(frameon=False, fontsize=6.8, loc="lower left", handlelength=1.0,
               borderpad=0.2, labelspacing=0.35)
import matplotlib.ticker as mtk
axes[1].set_xticks([4, 10, 20, 40])
axes[1].xaxis.set_major_formatter(mtk.ScalarFormatter())
axes[1].xaxis.set_minor_formatter(mtk.NullFormatter())
axes[1].set_xlabel("$h$"); axes[1].set_ylabel("$\\rho$")
axes[1].set_ylim(0, 1.12); axes[1].set_xlim(3.2, 60); tag(axes[1], "(b)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig7_repair.pdf"); plt.close(fig); print("fig7 ok")

# ================= fig8: 질량 재매개화 (단독 패널, GIST) =================
from c_common import load, build_hnsw, orders, F_GRID
c3 = np.load(f"{R}/c3_age.npz")
def adaptive_sets(nbrs0, n, grid):
    nbrs = nbrs0.copy(); alive = np.ones(n, bool); prev = 0; dele = []
    for f in grid:
        need = int(f*n) - prev
        if need > 0:
            indeg = np.bincount(nbrs[alive][nbrs[alive] >= 0], minlength=n); indeg[~alive] = -1
            newly = np.argsort(-indeg, kind="stable")[:need]
            alive[newly] = False; dele.append(newly); prev += need
        else:
            dele.append(np.array([], dtype=np.int64))
    return dele
LS = {"random": ":", "hub": "-", "oldest": "--", "newest": "-.", "adaptive": "-"}
acc = {}
for seed in SEEDS:
    base, _ = load("gist", seed); nbrs0, indeg = build_hnsw(base)
    n, tot = len(base), indeg.sum(); od = orders(base, indeg, seed)
    pool = {"random": (od["random"], c4[f"gist_random_norep_s{seed}"]),
            "hub": (od["hub"], c4[f"gist_hub_norep_s{seed}"]),
            "oldest": (od["oldest"], c3[f"gist_oldest_s{seed}"]),
            "newest": (od["newest"], c3[f"gist_newest_s{seed}"])}
    for nm, (order, rec) in pool.items():
        mass = np.array([indeg[order[:int(f*n)]].sum()/tot for f in F_GRID])
        acc.setdefault(nm, []).append((mass, np.array(rec)))
    cum, mtr = 0, []
    for dset in adaptive_sets(nbrs0, n, F_GRID):
        cum += indeg[dset].sum(); mtr.append(cum/tot)
    acc.setdefault("adaptive", []).append((np.array(mtr), np.array(c6[f"gist_adaptive_norep_s{seed}"])))
cols = {"random": C["rand"], "hub": C["hub"], "oldest": C["old"], "newest": C["new"], "adaptive": C["ada"]}
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for nm in ["random", "oldest", "newest", "adaptive", "hub"]:
    lst = acc[nm]
    M = np.stack([x[0] for x in lst]); Rr = np.stack([x[1] for x in lst])
    fm, rm, rs = M.mean(0), Rr.mean(0), Rr.std(0)
    ax.plot(fm, rm, LS[nm], color=cols[nm], lw=1.5, label=nm.capitalize())
    ax.fill_between(fm, rm-rs, rm+rs, color=cols[nm], alpha=0.15, lw=0)
ax.set_xlabel("$\\mu$"); ax.set_ylabel("recall@10")
ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig8_mass_gist.pdf"); plt.close(fig)
print("fig8 (single-panel) ok")

# ================= fig9_policy =================
c5 = np.load(f"{R}/c5_trigger_repair.npz"); d1 = np.load(f"{R}/d1_theta.npz")
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
pols = ["repair_only", "fixed20_rep", "hubmass30_rep"]
labs = ["Repair", "Count", "Mass"]
em = [np.mean([c5[f"gist_hub_{p}_s{s}_meta"][0] for s in SEEDS]) for p in pols]
rb = [np.mean([c5[f"gist_hub_{p}_s{s}_meta"][1] for s in SEEDS]) for p in pols]
axes[0].bar(range(3), em, color=["#E3E7EB", "#C3CBD6", "#8CB4D9"], edgecolor="#555", lw=0.5)
axes[0].set_yscale("log"); axes[0].set_ylabel("$\\varepsilon_{\\max}$")
axes[0].set_xticks(range(3)); axes[0].set_xticklabels(labs, fontsize=8.5)
for i, (e, r) in enumerate(zip(em, rb)):
    axes[0].text(i, e*1.3, f"$C_r{{=}}{r:.0f}$", ha="center", fontsize=8)
tag(axes[0], "(a)")
pts = []
for t in [60, 45]:
    m = np.mean([d1[f"gist_hub_t{t}_s{s}"] for s in SEEDS], axis=0); pts.append((m[1], m[0], f"$\\theta{{=}}0.{t}$"))
pts.append((rb[2], em[2], "$\\theta{=}0.30$")); pts.sort()
xs, ys = [p[0] for p in pts], [p[1] for p in pts]
axes[1].plot(xs, ys, "o-", color="#8CB4D9", lw=1.4, ms=5, label="Mass (repair)")
for x, y, l in pts:
    axes[1].annotate(l, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=7.5)
axes[1].plot([rb[1]], [em[1]], "s", mfc="none", mec="#333", mew=1.4, ms=9, label="Count (repair)")
axes[1].plot([rb[0]], [em[0]], "^", color=C["hub"], ms=7, label="Repair only")
axes[1].set_yscale("log"); axes[1].set_xlabel("$C_r$"); axes[1].set_ylabel("$\\varepsilon_{\\max}$")
axes[1].set_xticks([0, 1, 2, 3, 4]); axes[1].legend(frameon=False, fontsize=7); tag(axes[1], "(b)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig9_policy.pdf"); plt.close(fig); print("fig9 ok")
print("V2 DONE")
