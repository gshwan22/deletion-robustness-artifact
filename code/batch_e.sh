#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
source ../.venv/bin/activate 2>/dev/null || source ~/Desktop/Work_SH/ANN-Percolation/.venv/bin/activate
TS(){ date "+%F %T"; }
run(){ echo "[$(TS)] ===== START $1 ====="; python -u "$2" > "../results/batch_e_$1.log" 2>&1; echo "[$(TS)] ===== END $1 (exit=$?) ====="; }
run e1_mass  e1_mass_collapse.py
run d5b_t45  d5b_theta45.py
echo "[$(TS)] ===== ALL DONE ====="
