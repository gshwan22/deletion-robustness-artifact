#!/usr/bin/env bash
# =====================================================================
# ANN-Percolation — 4일 무인 배치 (2026-08-06)
# 실행: cd ~/Desktop/Work_SH/ANN-Percolation/src
#       nohup bash batch_4days.sh > ../results/batch.log 2>&1 &
# 각 단계는 독립 로그를 남기며, 실패해도 다음 단계로 진행한다.
# 예상 총 소요: 12–20시간 (GIST 5-seed 파트가 대부분)
# =====================================================================
set -u
cd "$(dirname "$0")"
TS() { date "+%F %T"; }
run() {  # run <이름> <파이썬파일>
  echo "[$(TS)] ===== START $1 ====="
  python -u "$2" > "../results/batch_$1.log" 2>&1
  rc=$?
  echo "[$(TS)] ===== END $1 (exit=$rc) ====="
}

# ---------------------------------------------------------------
# STEP 0. 스크립트 세대 준비 (기존 스크립트에서 seed/설정만 변경한 사본 생성)
# ---------------------------------------------------------------
python - << 'PYEOF'
import re, os

def derive(src_path, dst_path, subs, must=True):
    if not os.path.exists(src_path):
        print(f"[skip] {src_path} 없음 -> {dst_path} 생성 안 함")
        return
    s = open(src_path).read()
    for old, new in subs:
        if old not in s:
            msg = f"[warn] {src_path}: 패턴 미발견: {old[:60]}"
            print(msg)
            if must:
                continue
        s = s.replace(old, new)
    open(dst_path, "w").write(s)
    print(f"[ok] {dst_path}")

# (1) 트리거 5-seed 승격
derive("trigger_v2.py", "b1_trigger_5s.py", [
    ("SEEDS     = [0, 1]", "SEEDS     = [0, 1, 2, 3, 4]"),
    ('np.savez("../results/trigger_v2.npz"', 'np.savez("../results/trigger_5s.npz"'),
])

# (2) attack 스펙트럼 5-seed 승격
derive("attack_spectrum.py", "b2_spectrum_5s.py", [
    ("SEEDS     = [0, 1, 2]", "SEEDS     = [0, 1, 2, 3, 4]"),
    ('np.savez("../results/attack_spectrum.npz"', 'np.savez("../results/attack_spectrum_5s.npz"'),
])

# (3) confound(4모드 인과) 5-seed + GIST 추가
derive("confound_check.py", "b3_confound_5s.py", [
    ("SEEDS = [0, 1, 2]", "SEEDS = [0, 1, 2, 3, 4]"),
    ('DATA_PATH = "../data/sift-128-euclidean.hdf5"',
     'DATA_PATH = "../data/sift-128-euclidean.hdf5"  # b3: sift'),
    ('plt.savefig("../figures/confound_check.png", dpi=150)',
     'plt.savefig("../figures/b3_confound_sift_5s.png", dpi=150)'),
])
# GIST 판: 경로만 교체한 사본
derive("b3_confound_5s.py", "b4_confound_gist_5s.py", [
    ('DATA_PATH = "../data/sift-128-euclidean.hdf5"  # b3: sift',
     'DATA_PATH = "../data/gist-960-euclidean.hdf5"  # b4: gist'),
    ('plt.savefig("../figures/b3_confound_sift_5s.png", dpi=150)',
     'plt.savefig("../figures/b4_confound_gist_5s.png", dpi=150)'),
])

# (4) curves5 (그림 1 원곡선) 5-seed 승격
derive("figdata2.py", "b5_curves5_5s.py", [
    ("SEEDS = [0, 1, 2]", "SEEDS = [0, 1, 2, 3, 4]"),
    ('np.savez_compressed("../results/figdata_curves5_v2.npz"',
     'np.savez_compressed("../results/figdata_curves5_5s.npz"'),
])

# (5) K 강건성: 메인 4모드 sweep 을 K in {1,10,100} 으로
#     dataset_generalization 기반, sift+gist, 3 seeds, K 루프
base = open("dataset_generalization.py").read()
s = base.replace('DATASETS = {\n    "sift":  ("../data/sift-128-euclidean.hdf5",  "l2"),\n    "glove": ("../data/glove-100-angular.hdf5",   "angular"),\n    "gist":  ("../data/gist-960-euclidean.hdf5",  "l2"),\n}',
                 'DATASETS = {\n    "sift":  ("../data/sift-128-euclidean.hdf5",  "l2"),\n    "gist":  ("../data/gist-960-euclidean.hdf5",  "l2"),\n}')
s = s.replace("MODES     = [\"random\", \"hub\", \"antihub\", \"relevance\"]",
              "MODES     = [\"random\", \"hub\"]")
s = s.replace("K         = 10", "K_LIST    = [1, 10, 100]\nK         = 10")
# recall/fc 를 K 루프로 감싸는 래퍼는 단순화: run_dataset 대신 main 재정의
marker = "def main():"
i = s.index(marker)
s = s[:i] + '''def main():
    import time as _t
    t0 = _t.time()
    for name in DATASETS:
        full, queries = load(name)
        qA = np.ascontiguousarray(queries[:SAMPLE_Q])
        for seed in SEEDS:
            base = subsample(full, seed)
            nbrs, indeg = build(base)
            rng = np.random.default_rng(5000 + seed)
            for m in ["random", "hub"]:
                order = removal_order(base, indeg, None, m, seed)
                for Kv in K_LIST:
                    global K
                    K = Kv
                    fc = fcross(recall_curve(base, qA, nbrs, indeg, order, rng))
                    print(f"  [{name}] seed{seed} [{m}] K={Kv} f_c={fc}")
    print(f"[done] {_t.time()-t0:.1f}s (K-robustness)")


def _unused():
''' + s[i+len(marker):]
open("b6_k_robustness.py", "w").write(s)
print("[ok] b6_k_robustness.py")

# (6) efConstruction 민감도: M_sweep 구조 재활용, efC in {100, 400}
s2 = base.replace("EF_CONSTR = 200", "EF_CONSTR = 200  # overridden below")
marker = "def main():"
i = s2.index(marker)
s2 = s2[:i] + '''def main():
    import time as _t
    t0 = _t.time()
    import faiss as _f
    for efc in [100, 400]:
        full, queries = load("sift")
        qA = np.ascontiguousarray(queries[:SAMPLE_Q])
        for seed in [0, 1, 2]:
            base = subsample(full, seed)
            n, d = base.shape
            index = _f.IndexHNSWFlat(d, 16)
            index.hnsw.efConstruction = efc
            index.add(base)
            h = index.hnsw
            neighbors = _f.vector_to_array(h.neighbors)
            offsets = _f.vector_to_array(h.offsets).astype(np.int64)
            cum = _f.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
            n0 = int(cum[1])
            idx = offsets[:-1][:, None] + np.arange(n0)[None, :]
            nbrs = neighbors[idx]
            nbrs = np.where(nbrs < 0, np.int32(-1), nbrs).astype(np.int32)
            indeg = np.bincount(nbrs[nbrs >= 0], minlength=n).astype(np.int64)
            hub = indeg.max() / indeg.mean()
            rng = np.random.default_rng(5000 + seed)
            for m in ["random", "hub"]:
                order = removal_order(base, indeg, None, m, seed)
                fc = fcross(recall_curve(base, qA, nbrs, indeg, order, rng))
                print(f"  [efC={efc}] seed{seed} [{m}] f_c={fc} hubness={hub:.1f}")
    print(f"[done] {_t.time()-t0:.1f}s (efC sensitivity)")


def _unused():
''' + s2[i+len(marker):]
open("b7_efc_sensitivity.py", "w").write(s2)
print("[ok] b7_efc_sensitivity.py")
PYEOF

# ---------------------------------------------------------------
# STEP 1~7. 순차 실행 (가벼운 것 먼저 — 초반 실패를 빨리 발견)
# ---------------------------------------------------------------
run "b7_efc"        b7_efc_sensitivity.py
run "b6_krobust"    b6_k_robustness.py
run "b2_spectrum5s" b2_spectrum_5s.py
run "b5_curves5s"   b5_curves5_5s.py
run "b3_confound_sift5s" b3_confound_5s.py
run "b1_trigger5s"  b1_trigger_5s.py
run "b4_confound_gist5s" b4_confound_gist_5s.py

echo "[$(TS)] ===== ALL DONE ====="
echo "결과 로그: results/batch_b*.log / npz: trigger_5s, attack_spectrum_5s, figdata_curves5_5s"
