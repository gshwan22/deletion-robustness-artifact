#!/usr/bin/env bash
# 5일 무인 배치 — 우선순위 순, 한 단계 실패해도 계속. 각 단계 개별 로그.
set -u
cd "$(dirname "$0")"
source ../.venv/bin/activate 2>/dev/null || source ~/Desktop/Work_SH/ANN-Percolation/.venv/bin/activate
TS(){ date "+%F %T"; }
run(){ echo "[$(TS)] ===== START $1 ====="; python -u "$2" > "../results/batch_g_$1.log" 2>&1; echo "[$(TS)] ===== END $1 (exit=$?) ====="; }
run engines   d7b_d4d_e8_engines.py     # e8 즉시 + pgvector 비용/f 스윕 + qdrant f 스윕  (~3h)
run sweeps    e2b_e6_sweeps.py          # 정적 hub 개수 스윕 + 혼합 삽입 개수 스윕       (~3h)
run long      e7_e5_long.py             # adaptive/kcore 5시드 + 수리 활성 개수 스윕      (~6-10h)
echo "[$(TS)] ===== ALL DONE ====="
