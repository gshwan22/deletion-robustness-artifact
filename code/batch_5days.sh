#!/usr/bin/env bash
# EDBT 보강 5일 배치: cd src && nohup bash batch_5days.sh > ../results/batch5.log 2>&1 &
set -u
cd "$(dirname "$0")"
source ~/Desktop/Work_SH/ANN-Percolation/.venv/bin/activate 2>/dev/null || source ../.venv/bin/activate
TS(){ date "+%F %T"; }
run(){ echo "[$(TS)] ===== START $1 ====="; python -u "$2" > "../results/batch5_$1.log" 2>&1; echo "[$(TS)] ===== END $1 (exit=$?) ====="; }
run c2_tail    c2_tail.py
run c3_age     c3_age.py
run c1_beta    c1_beta.py
run c4_repair  c4_repair.py
echo "[$(TS)] ===== ALL DONE ====="
