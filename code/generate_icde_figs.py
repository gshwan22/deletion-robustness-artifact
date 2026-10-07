#!/usr/bin/env python3
"""ANN-Percolation — generate_figures v4 (3차 피드백). 필요: pip install networkx"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import os

R = "../results"; OUT = "../figures"
os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 10,
    "legend.fontsize": 7.8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})
C = {"random": "#A5ACB8", "hub": "#FC8D62", "clustered": "#E6C35C",
     "antihub": "#9DD6A9", "relevance": "#E78AC3", "tomb": "#8A8F98"}
CP = {"random": "#C6CBD3", "hub": "#FDBBA1", "clustered": "#F0DCA0",
      "antihub": "#C2E5CC", "traversal": "#B7C0CA", "kcore": "#D8B49A",
      "adaptive": "#D5B3CC"}
PASTEL = ["#8FB8DE", "#9DD6A9", "#F2CE8F", "#C9A8DC", "#8FD0C6", "#EFA9BB", "#B7C0CA", "#D8C49A"]
LAB = {"random": "Random", "hub": "Hub", "clustered": "Clustered",
       "antihub": "Anti-hub", "relevance": "Relevance"}

# ============================================================ Fig 0
def fig0():
    import networkx as nx
    sg = np.load(f"{R}/figdata_subgraph.npz")
    ed, lab, indeg = sg["edges"], sg["labels"], sg["indeg"].astype(float)
    n_full = int(lab.shape[0])
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(n_full, 1500, replace=False))
    remap = -np.ones(n_full, np.int64); remap[keep] = np.arange(keep.size)
    m0 = (remap[ed[:, 0]] >= 0) & (remap[ed[:, 1]] >= 0)
    E0 = np.stack([remap[ed[m0, 0]], remap[ed[m0, 1]]], 1)
    G = nx.Graph(); G.add_nodes_from(range(keep.size)); G.add_edges_from(map(tuple, E0))
    cc = max(nx.connected_components(G), key=len)
    cc = np.array(sorted(cc))
    remap2 = -np.ones(keep.size, np.int64); remap2[cc] = np.arange(cc.size)
    m1 = (remap2[E0[:, 0]] >= 0) & (remap2[E0[:, 1]] >= 0)
    E = np.stack([remap2[E0[m1, 0]], remap2[E0[m1, 1]]], 1)
    lab2 = lab[keep][cc]; indeg2 = indeg[keep][cc]
    n = cc.size
    G2 = nx.Graph(); G2.add_nodes_from(range(n)); G2.add_edges_from(map(tuple, E))
    pos = nx.spring_layout(G2, seed=1, k=1.1 / np.sqrt(n), iterations=120)
    xy = np.array([pos[i] for i in range(n)])

    hubset = np.zeros(n, bool)
    hubset[np.argsort(-indeg2)[: int(0.10 * n)]] = True     # 상위 10% = highway
    is_hub_edge = hubset[E[:, 0]] | hubset[E[:, 1]]
    order_hub = np.argsort(-indeg2)
    del_hub = np.zeros(n, bool); del_hub[order_hub[: int(0.30 * n)]] = True
    del_rnd = np.zeros(n, bool); del_rnd[rng.choice(n, int(0.30 * n), replace=False)] = True

    fig = plt.figure(figsize=(6.8, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.22, wspace=0.04)

    def draw(ax, deleted, tag):
        alive_e = ~(deleted[E[:, 0]] | deleted[E[:, 1]])
        for u, v in E[alive_e & ~is_hub_edge]:
            ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]],
                    lw=0.25, color="#c3cad2", alpha=0.5, zorder=1)
        for u, v in E[alive_e & is_hub_edge]:
            ax.plot([xy[u, 0], xy[v, 0]], [xy[u, 1], xy[v, 1]],
                    lw=0.7, color="#FC8D62", alpha=0.55, zorder=2)
        al = ~deleted
        s = 3 + 1.6 * np.sqrt(indeg2)
        base = al & ~hubset
        ax.scatter(xy[base, 0], xy[base, 1], s=s[base] ** 1.35,
                   c=[PASTEL[l % len(PASTEL)] for l in lab2[base]],
                   alpha=0.95, lw=0.3, edgecolor="white", zorder=3)
        hb = al & hubset
        ax.scatter(xy[hb, 0], xy[hb, 1], s=s[hb] ** 1.4,
                   c="#FC8D62", alpha=0.9, lw=0.5, edgecolor="white", zorder=4)
        ax.set(xticks=[], yticks=[])
        ax.set_title(tag, loc="left", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)

    axa = fig.add_subplot(gs[0, 0])
    draw(axa, np.zeros(n, bool), "(a)")
    axa.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#FC8D62", ms=5,
               label=r"Hub (top-10% $k_{\mathrm{in}}$)"),
        Line2D([], [], ls="-", color="#FC8D62", lw=1.2, alpha=0.7, label="Highway edge"),
        Line2D([], [], ls="-", color="#c3cad2", lw=1.2, label="Other edge")],
        frameon=False, fontsize=6.0, loc="lower left", handlelength=1.4,
        borderaxespad=0.0)
    draw(fig.add_subplot(gs[0, 1]), del_rnd, "(b)")
    draw(fig.add_subplot(gs[0, 2]), del_hub, "(c)")

    axd = fig.add_subplot(gs[1, :])
    ID = np.load(f"{R}/figdata_indeg.npz")
    series_d = [("SIFT · HNSW", "hnsw_sift", "#7FB8DA"),
                          ("GloVe · HNSW", "hnsw_glove", "#E6C35C"),
                          ("GIST · HNSW", "hnsw_gist", "#FC8D62"),
                          ("SIFT · MRNG", "mrng_sift", "#4A7FA5"),
                          ("GloVe · MRNG", "mrng_glove", "#B08F2E"),
                          ("GIST · MRNG", "mrng_gist", "#C75B33")]
    for label, key, c in series_d:
        if key not in ID.files:
            continue
        x = np.sort(ID[key].astype(float)); x = x[x > 0]
        axd.plot(x, 1 - np.arange(1, len(x) + 1) / len(x), lw=1.5, color=c, label=label)
    axd.set_xscale("log"); axd.set_yscale("log"); axd.set_ylim(1e-5, 1)
    axd.set_xlabel(r"$k_{\mathrm{in}}$"); axd.set_ylabel(r"$P(k_{\mathrm{in}})$")
    axd.set_title("(d)", loc="left", fontsize=9)
    axd.legend(frameon=False, ncol=2, loc="lower left")
    plt.savefig(f"{OUT}/fig0_anatomy.pdf"); plt.close()

# ============================================================ Fig 1 (v3 유지)
def fig1():
    d = np.load(f"{R}/figdata_curves5_5s.npz")
    F = d["F"]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for m in ["random", "hub", "clustered", "relevance", "antihub"]:
        mu, sd = d[f"{m}_rec_mu"], d[f"{m}_rec_sd"]
        ax[0].plot(F, mu, "-o", ms=2.2, lw=1.3, color=C[m], label=LAB[m])
        ax[0].fill_between(F, mu - sd, mu + sd, color=C[m], alpha=0.15, lw=0)
        muS, sdS = d[f"{m}_S_mu"], d[f"{m}_S_sd"]
        ax[1].plot(F, muS, "-o", ms=2.2, lw=1.3, color=C[m], label=LAB[m])
        ax[1].fill_between(F, muS - sdS, muS + sdS, color=C[m], alpha=0.15, lw=0)
    # tombstone baseline (hub 삭제, M=16) — compaction 전엔 손상이 안 보임
    dt = np.load(f"{R}/M_sweep_tombstone.npz")
    ax[0].plot(dt["F"], dt["M16_hub_tmb"], "--", color="#777", lw=1.2,
               label="Hub (tombstone)")
    ax[0].axvspan(0.785, 0.849, color=C["hub"], alpha=0.09)
    ax[0].set_xlabel(r"$f$"); ax[0].set_ylabel("recall@10")
    # 이론 기준선 (Molloy-Reed/Cohen, SIFT): 있으면 점선 표시
    try:
        th = np.load(f"{R}/theory_fc.npz")
        for key, cc in [("hnsw_sift_rand", C["random"]), ("hnsw_sift_hub", C["hub"])]:
            ax[1].plot(float(th[key]), 0.0, marker="*", ms=11, color=cc,
                       markeredgecolor="white", markeredgewidth=0.5,
                       clip_on=False, zorder=5)
    except FileNotFoundError:
        pass
    ax[1].set_xlabel(r"$f$"); ax[1].set_ylabel(r"$S(f)$")
    for a in ax:
        a.set_xlim(0, 1.0); a.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[0].set_title("(a)", loc="left", fontsize=9)
    ax[1].set_title("(b)", loc="left", fontsize=9)
    ax[0].legend(frameon=False, fontsize=7.0, loc="lower left")
    ax[1].legend(frameon=False, fontsize=7.0, loc="lower left")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig1_modes.pdf"); plt.close()

# ============================================================ Fig 2 (막대 파스텔)
def fig2():
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.4))
    sp = np.load(f"{R}/attack_spectrum_5s.npz")
    modes = ["adaptive", "hub", "kcore", "traversal", "random"]
    names = ["Adaptive\ndegree", "Hub\n(static)", "K-core", "Traversal", "Random"]
    x = np.arange(len(modes)); w = 0.36
    DS_STYLE = {"sift": ("#BDD5EA", "\\\\"), "gist": ("#FBC4AD", "//")}
    for off, ds in [(-w / 2, "sift"), (w / 2, "gist")]:
        col, hat = DS_STYLE[ds]
        mu = [sp[f"{ds}_{m}"][0] for m in modes]
        sd = [sp[f"{ds}_{m}"][1] for m in modes]
        ax[0].bar(x + off, mu, w, yerr=sd, capsize=2,
                  color=col, hatch=hat,
                  edgecolor="#666", linewidth=0.4, label=ds.upper())
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=7.3)
    ax[0].set_ylabel(r"$f_c$"); ax[0].set_title("(a)", loc="left", fontsize=9)
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.2, borderpad=0.15, labelspacing=0.3)
    ax[0].tick_params(axis="x", labelsize=6.0, rotation=38)
    for _t in ax[0].get_xticklabels(): _t.set_ha("right")
    fB = [0,.10,.20,.30,.40,.50,.60,.70,.72,.73,.74,.75,.76,.77,.78,.79,.80,.81,.82,.83,.84,.85,.86]
    recB = [1,1,.999,.998,.987,.960,.893,.724,.686,.662,.627,.602,.566,.534,.504,.468,.420,.378,.324,.292,.254,.227,.189]
    reachB = [1,1,1,1,1,.998,.980,.884,.860,.814,.786,.752,.704,.674,.640,.596,.552,.500,.456,.422,.376,.334,.248]
    ax[1].plot(fB, reachB, "--s", ms=2.3, lw=1.2, color="#555", label="Path exists (BFS)")
    ax[1].plot(fB, recB, "-o", ms=2.3, lw=1.3, color=C["hub"], label="Greedy finds")
    ax[1].fill_between(fB, recB, reachB, color=C["hub"], alpha=0.12)
    ax[1].set_xlabel(r"$f_{\mathrm{hub}}$"); ax[1].set_ylabel(r"$P_{\mathrm{reach}}$")
    ax[1].set_title("(b)", loc="left", fontsize=9)
    ax[1].legend(frameon=False, loc="lower left", fontsize=7.5)
    # (c) 전진-엣지 질량 편향 p~(f)
    try:
        gfc = np.load(f"{R}/gf_check.npz")
        Fg = gfc["F"]
        ax[2].plot(Fg, Fg, "-", color=C["random"], lw=1.2,
                   label=r"Random, analytic ($\tilde p{=}f$)")
        ax[2].plot(Fg, gfc["random_ptilde"], "o", ms=3.2,
                   color=C["random"], markeredgecolor="white",
                   markeredgewidth=0.4, label="Random, simulation")
        ax[2].plot(Fg, gfc["hub_ptilde"], "-o", ms=2.5, lw=1.3,
                   color=C["hub"], label="Hub, simulation")
        ax[2].set_xlabel(r"$f$"); ax[2].set_ylabel(r"$\tilde p(f)$")
        ax[2].set_xlim(0, 1.0); ax[2].set_ylim(0, 1.0)
        ax[2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax[2].set_title("(c)", loc="left", fontsize=9)
        ax[2].legend(frameon=False, fontsize=6.5, loc="lower right", handlelength=1.3, borderpad=0.15, labelspacing=0.3)
    except FileNotFoundError:
        fig.delaxes(ax[2])
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2_cause.pdf"); plt.close()

# ============================================================ Fig 3 (broken-axis 2패널 줌)
def fig3():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.8), width_ratios=[1.35, 1.0])
    fig.subplots_adjust(wspace=0.06)
    # 왼쪽: 저-h 존
    lo = [("SIFT·HNSW", "H", 4.2, 0.136, (6, -9)), ("GloVe·HNSW", "H", 6.6, 0.118, (-8, -2)),
          ("SIFT·MRNG", "M", 5.8, 0.211, (6, 3)),
          ("GloVe·MRNG", "M", 8.4, 0.198, (-7, 8)),
          ("NQ·bge-m3", "T", 4.8, 0.123, (-8, -2))]
    for label, idx, hb, gap, off in lo:
        mk = {"H": "o", "M": "^", "T": "D"}[idx]
        col = {"H": C["random"], "M": C["hub"], "T": "#4C9085"}[idx]
        a1.scatter(hb, gap, s=52, marker=mk, color=col, zorder=3, edgecolor="white", lw=0.6)
        a1.annotate(label, (hb, gap), textcoords="offset points", xytext=off,
                    fontsize=7, ha="left" if off[0] > 0 else "right")
    # SIFT 100k->1M (실선), DEEP 10M (점선 연결 + 빈 마커: 데이터셋 다름을 명시)
    a1.plot([4.2, 6.3], [0.136, 0.159], "-", color="#999", lw=1.1, zorder=2)
    a1.plot([6.3, 7.0], [0.159, 0.184], "--", color="#999", lw=1.0, zorder=2)
    a1.scatter([6.3], [0.159], s=17, color="#999", zorder=3)
    a1.scatter([7.0], [0.184], s=22, facecolor="white", edgecolor="#999", lw=1.0, zorder=3)
    a1.annotate("SIFT 1M", (6.3, 0.159), textcoords="offset points", xytext=(2, -10),
                fontsize=6.2, color="#777", ha="left")
    a1.annotate("DEEP 10M", (7.0, 0.184), textcoords="offset points", xytext=(2, 5),
                fontsize=6.2, color="#777", ha="left")
    a1.set_xlim(3.1, 9.6); a1.set_ylim(0.09, 0.24)
    a1.set_xlabel(r"$h = k_{\max}/\langle k_{\mathrm{in}}\rangle$")
    a1.set_ylabel(r"$\Delta f_c$")
    a2.legend(handles=[Line2D([], [], marker="o", ls="", color=C["random"], label="HNSW"),
                       Line2D([], [], marker="^", ls="", color=C["hub"], label="MRNG"),
                       Line2D([], [], marker="D", ls="", color="#4C9085", label="HNSW · text (bge-m3)"),
                       Line2D([], [], ls="-", color="#999", label="Scale (100k$\\to$10M)")],
              frameon=True, framealpha=1.0, edgecolor="#cccccc", loc="center left", fontsize=7.2)
    # 오른쪽: 고-h 존
    hi = [("GIST·HNSW", "H", 25.2, 0.539, (6, -9)), ("GIST·MRNG", "M", 40.2, 0.633, (-8, -9))]
    for label, idx, hb, gap, off in hi:
        mk = {"H": "o", "M": "^", "T": "D"}[idx]
        col = {"H": C["random"], "M": C["hub"], "T": "#4C9085"}[idx]
        a2.scatter(hb, gap, s=52, marker=mk, color=col, zorder=3, edgecolor="white", lw=0.6)
        ha = "left" if off[0] > 0 else "right"
        a2.annotate(label, (hb, gap), textcoords="offset points", xytext=off,
                    fontsize=7, ha=ha)
    a2.set_xlim(20, 46); a2.set_ylim(0.50, 0.67)
    a2.yaxis.tick_right()
    a2.set_xlabel(r"$h$")
    # broken-axis 표식
    for a, side in [(a1, 1), (a2, 0)]:
        d = 0.015
        kw = dict(transform=a.transAxes, color="k", clip_on=False, lw=0.8)
        a.plot((side - d, side + d), (-d, +d), **kw)
        a.plot((side - d, side + d), (1 - d, 1 + d), **kw)
        a.spines["right" if side == 1 else "left"].set_visible(False)
    plt.savefig(f"{OUT}/fig3_hubness_gap.pdf"); plt.close()

# ============================================================ Fig 4 (v3 유지)
def fig4():
    ef = np.load(f"{R}/ef_sweep.npz")
    B = ef["ef"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(B, ef["gist_random_f30"], "--o", color=C["random"], ms=2.5, lw=1.1, label=r"$f_{\mathrm{rand}}=0.3$")
    ax.plot(B, ef["gist_random_f50"], "--s", color=C["random"], ms=2.5, lw=1.1, alpha=0.75, label=r"$f_{\mathrm{rand}}=0.5$")
    ax.plot(B, ef["gist_random_f70"], "--^", color=C["random"], ms=2.5, lw=1.1, alpha=0.5, label=r"$f_{\mathrm{rand}}=0.7$")
    ax.plot(B, ef["gist_hub_f30"], "-o", color=C["hub"], ms=3, lw=1.3, label=r"$f_{\mathrm{hub}}=0.3$")
    ax.plot(B, ef["gist_hub_f50"], "-s", color=C["hub"], ms=3, lw=1.3, alpha=0.75, label=r"$f_{\mathrm{hub}}=0.5$")
    ax.plot(B, ef["gist_hub_f70"], "-^", color=C["hub"], ms=3, lw=1.3, alpha=0.5, label=r"$f_{\mathrm{hub}}=0.7$")
    ax.axhline(0.9, ls="--", c="#aaa", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel("recall@10")
    ax.legend(frameon=False, fontsize=7.0, loc="center right")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig4_no_defense.pdf"); plt.close()

# ============================================================ Fig 5 (v3 유지)
def fig5():
    d = np.load(f"{R}/M_sweep_tombstone.npz")
    F = d["F"]
    hs, ht = d["M16_hub_str"], d["M16_hub_tmb"]
    rs, rt = d["M16_random_str"], d["M16_random_tmb"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(F, ht, "--", color=C["hub"], lw=1.2, alpha=0.6, label="Hub · tombstone")
    ax.plot(F, hs, "-o", color=C["hub"], ms=2.6, lw=1.4, label="Hub · structural")
    ax.fill_between(F, hs, ht, color=C["hub"], alpha=0.10)
    ax.plot(F, rt, "--", color=C["random"], lw=1.2, alpha=0.6, label="Random · tombstone")
    ax.plot(F, rs, "-o", color=C["random"], ms=2.6, lw=1.4, label="Random · structural")
    ax.set_xlabel(r"$f$"); ax.set_ylabel("recall@10")
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=7.2, loc="lower left")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig5_tombstone.pdf"); plt.close()

# ============================================================ Fig 6 (최악손실/비용 막대)
def fig6():
    # trigger_v2 (2 seeds) 요약값
    pol = ["Never", "Fixed-count\n20%", "Hub-mass\n$\\theta$=0.30", "Hub-mass\n$\\theta$=0.15"]
    adv_min = [0.004, 0.937, 0.984, 0.991]      # 5s      # gist x hubtargeted
    adv_rb  = [0, 4, 4, 9]
    ben_min = [0.965, 0.991, 0.988, 0.992]      # 5s      # gist x random (benign)
    ben_rb  = [0, 4, 2, 5]                      # 5s
    cols = ["#E3E7EB", "#E3E7EB", "#8CB4D9", "#8CB4D9"]  # 제안(hub-mass)만 진한 회색
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    x = np.arange(4)
    loss = [1 - m for m in adv_min]
    b = ax[0].bar(x, loss, 0.6, color=cols, edgecolor="#666", lw=0.4)
    for i, (l, r) in enumerate(zip(loss, adv_rb)):
        ax[0].annotate(rf"$C_{{\mathrm{{r}}}}$={r}", (i, l), textcoords="offset points",
                       xytext=(0, 3), ha="center", fontsize=6.8)
    ax[0].set_yscale("log"); ax[0].set_ylim(5e-3, 2.0)
    ax[0].set_xticks(x); ax[0].set_xticklabels(pol, fontsize=7.0)
    ax[0].set_ylabel(r"$\varepsilon_{\max}$")
    ax[0].set_title("(a)", loc="left", fontsize=9)
    b2 = ax[1].bar(x, ben_rb, 0.6, color=cols, edgecolor="#666", lw=0.4)
    for i, (r, m) in enumerate(zip(ben_rb, ben_min)):
        ax[1].annotate(rf"$\varepsilon_{{\max}}$={1-m:.3f}", (i, r), textcoords="offset points",
                       xytext=(0, 3), ha="center", fontsize=6.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(pol, fontsize=7.0)
    ax[1].set_ylabel(r"$C_{\mathrm{r}}$")
    ax[1].set_ylim(0, 6.2)
    ax[1].set_title("(b)", loc="left", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig5_trigger.pdf"); plt.close()

# ============================================================ Fig 7 (측정 편향 vs f)
def fig7():
    d = np.load(f"{R}/figdata_curves5_v2.npz")
    F = d["F"]
    true_hub = d["hub_rec_mu"]
    v1_tomb = np.array([0.996,0.996,0.997,0.997,0.997,0.997,0.998,0.998,0.998,0.999,
                        0.999,0.999,0.999,0.999,0.999,1.0,1.0,1.0,1.0,1.0])
    bias_tomb = v1_tomb - true_hub
    fA = np.array([.72,.73,.74,.75,.76,.77,.78,.79,.80,.81,.82,.83,.84,.85,.86])
    recA = np.array([.686,.664,.628,.601,.567,.533,.504,.466,.423,0,0,.288,.253,.218,0])
    recBB = np.array([.686,.662,.627,.602,.566,.534,.504,.468,.420,.378,.324,.292,.254,.227,.189])
    bias_single = recA - recBB
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axhline(0, color="#999", lw=1)
    ax.fill_between(F, 0, bias_tomb, color=C["tomb"], alpha=0.30, lw=0)
    ax.plot(F, bias_tomb, "-o", ms=2.4, lw=1.3, color=C["tomb"], label="Tombstone API")
    ml, sl, bl = ax.stem(fA, bias_single, basefmt=" ")
    plt.setp(sl, color="#B48EAD", lw=1.3)
    plt.setp(ml, color="#B48EAD", ms=3.5)
    ax.plot([], [], "-o", color="#B48EAD", ms=3, label="Single entry")
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.55, 1.05)
    ax.set_xlabel(r"$f_{\mathrm{hub}}$")
    ax.set_ylabel(r"$\Delta$recall  (measured $-$ true)")
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig7_pitfalls.pdf"); plt.close()


if __name__ == "__main__":
    for fn in [fig0, fig1, fig2, fig3, fig4, fig6]:  # fig5 -> fig1(a) 흡수, fig7 -> Table
        fn()
        print(f"  {fn.__name__} ✓")
    print(f"[done] -> {OUT}/fig*.pdf")
