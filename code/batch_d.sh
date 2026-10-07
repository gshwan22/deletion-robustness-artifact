#!/usr/bin/env bash
# 쥐어짜기 배치: cd src && nohup bash batch_d.sh > ../results/batch_d.log 2>&1 &
set -u
cd "$(dirname "$0")"
source ../.venv/bin/activate 2>/dev/null || source ~/Desktop/Work_SH/ANN-Percolation/.venv/bin/activate
pip install hnswlib 2>&1 | tail -1
TS(){ date "+%F %T"; }
run(){ echo "[$(TS)] ===== START $1 ====="; python -u "$2" > "../results/batch_d_$1.log" 2>&1; echo "[$(TS)] ===== END $1 (exit=$?) ====="; }
run d1_theta   d1_theta.py
run d3_hnswlib d3_hnswlib.py
run d2_fig67   make_fig67.py
run d6_mrng    d6_mrng_repair.py
run d5_mixed   d5_mixed.py
echo "[$(TS)] ===== ALL DONE ====="
