#!/usr/bin/env bash
# ICDE 보강 배치: e2(개수 스윕) → e3/e4(엄격 임계·adaptive 민감도). d7(pgvector)은 수동.
set -u
cd "$(dirname "$0")"
source ../.venv/bin/activate 2>/dev/null || source ~/Desktop/Work_SH/ANN-Percolation/.venv/bin/activate
TS(){ date "+%F %T"; }
run(){ echo "[$(TS)] ===== START $1 ====="; python -u "$2" > "../results/batch_f_$1.log" 2>&1; echo "[$(TS)] ===== END $1 (exit=$?) ====="; }
run e2_count   e2_count_sweep.py
run e3e4       e3_e4_strict_adaptive.py
echo "[$(TS)] ===== ALL DONE ====="
