#!/usr/bin/env bash
# 확장 배치: cd src && nohup bash batch_ext.sh > ../results/batch_ext.log 2>&1 &
set -u
cd "$(dirname "$0")"
source ../.venv/bin/activate 2>/dev/null || source ~/Desktop/Work_SH/ANN-Percolation/.venv/bin/activate
TS(){ date "+%F %T"; }
run(){ echo "[$(TS)] ===== START $1 ====="; python -u "$2" > "../results/batch_ext_$1.log" 2>&1; echo "[$(TS)] ===== END $1 (exit=$?) ====="; }
run c7_lowf   c7_lowf.py
run c6_ext    c6_repair_ext.py
run c5_policy c5_trigger_repair.py
echo "[$(TS)] ===== ALL DONE ====="
