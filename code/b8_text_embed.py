#!/usr/bin/env python3
"""
b8: BEIR/NQ 위키 문단 100k -> bge-m3 임베딩 -> ann-benchmarks 호환 hdf5
=====================================================================
목적: "실제 knowledge retrieval 코퍼스에서도 4모드 패턴이 재현되는가" (KEIR 보강)
출력: ../data/text-bge-100k.hdf5  (train: 100k x 1024, test: 쿼리 임베딩, 모두 정규화 -> l2)
의존: pip install -U sentence-transformers datasets h5py
소요: GB10 GPU 기준 임베딩 ~20-40분
실행: cd src && python -u b8_text_embed.py
"""
import numpy as np
import h5py
import time

N_BASE = 100_000
POOL   = 300_000          # 앞 30만에서 무작위 10만 추출 (위키 순서 편향 완화)
SEED   = 7
OUT    = "../data/text-bge-100k.hdf5"
MODEL  = "BAAI/bge-m3"
BATCH  = 256

t0 = time.time()
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

print("[1/4] corpus streaming (BeIR/nq)...")
corpus = load_dataset("BeIR/nq", "corpus", split="corpus", streaming=True)
texts = []
for i, row in enumerate(corpus):
    if i >= POOL:
        break
    t = (row.get("title") or "") + " " + (row.get("text") or "")
    texts.append(t.strip())
    if (i + 1) % 50_000 == 0:
        print(f"  pooled {i+1}/{POOL} ({time.time()-t0:.0f}s)")
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(len(texts), size=N_BASE, replace=False))
texts = [texts[i] for i in idx]
print(f"  base texts: {len(texts)}")

print("[2/4] queries (BeIR/nq)...")
qs = load_dataset("BeIR/nq", "queries", split="queries")
queries = [q["text"] for q in qs]
print(f"  queries: {len(queries)}")

print("[3/4] embedding with bge-m3 ...")
model = SentenceTransformer(MODEL)
emb_base = model.encode(texts, batch_size=BATCH, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
emb_q = model.encode(queries, batch_size=BATCH, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
print(f"  base {emb_base.shape}, queries {emb_q.shape} ({time.time()-t0:.0f}s)")

print("[4/4] write hdf5 (ann-benchmarks 호환: train/test) ...")
with h5py.File(OUT, "w") as f:
    f.create_dataset("train", data=emb_base)
    f.create_dataset("test", data=emb_q)
print(f"[done] {time.time()-t0:.0f}s -> {OUT}")
print("정규화 완료 상태이므로 파이프라인에는 metric='l2' 로 선언하면 됨 (angular 재정규화 불필요·무해)")
