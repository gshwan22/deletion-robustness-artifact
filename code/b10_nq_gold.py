#!/usr/bin/env python3
"""
b10: NQ gold-relevance 삭제 실험 — knowledge-level 평가
=====================================================================
목적: R1("ANN fidelity != knowledge retrieval") / R3("framing") 반론 대응.
  qrels(정답 문단 주석) 기준 answer-passage hit@10 을 삭제 하에 측정하고
  실패를 (i) availability loss (정답 삭제됨) vs (ii) substrate failure
  (정답 생존인데 미도달) 로 분해.
설계:
  - corpus: BeIR/nq 스트리밍 풀에서 qrels 정답 문단 전부 강제 포함 + 무작위 채움 = 100k
  - 임베딩: bge-m3 (b8 과 동일 설정)
  - 삭제 순서: random / hub / relevance / antihub, 3 seeds
  - 지표(각 f): gold_hit@10 (질의의 정답 문단 중 하나가 top-10 에 등장),
      avail_loss(f)   = P(질의의 모든 정답 문단이 삭제됨)
      substrate_fail(f)= P(정답 생존 & top-10 미포함)
      => gold_hit = 1 - avail_loss - substrate_fail
출력: ../results/nq_gold.npz + 요약표 stdout
실행: cd src && nohup python -u b10_nq_gold.py > ../results/b10_gold.log 2>&1 &
소요: 임베딩 ~25분 + 곡선 3 seeds ~40분 (GB10)
"""
import numpy as np, h5py, time, faiss

N_BASE   = 100_000
POOL     = 300_000
SEEDS    = [0, 1, 2]
F_GRID   = np.arange(0.0, 0.96, 0.05)
K        = 10
BEAM     = 512
R_ENTRY  = 16
SAMPLE_Q = 500          # qrels 있는 질의 중 평가 표본
OUT_H5   = "../data/nq-gold-bge-100k.hdf5"
OUT_NPZ  = "../results/nq_gold.npz"

t0 = time.time()
# ---------- 1) corpus + qrels ----------
from datasets import load_dataset
print("[1/5] qrels & corpus ...")
qrels = load_dataset("BeIR/nq-qrels", split="test")          # query-id, corpus-id, score
gold_map = {}                                                # qid -> set(corpus-id)
for r in qrels:
    gold_map.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
gold_cids = set().union(*gold_map.values())
print(f"  queries with gold: {len(gold_map)}, gold passages: {len(gold_cids)}")

corpus = load_dataset("BeIR/nq", "corpus", split="corpus", streaming=True)
texts, cids = [], []
gold_texts, gold_ids = [], []
for i, row in enumerate(corpus):
    cid = str(row["_id"])
    t = ((row.get("title") or "") + " " + (row.get("text") or "")).strip()
    if cid in gold_cids:
        gold_ids.append(cid); gold_texts.append(t)
    elif i < POOL:
        cids.append(cid); texts.append(t)
    if i >= POOL and len(gold_ids) >= len(gold_cids):
        break
    if (i+1) % 500_000 == 0:
        print(f"  scanned {i+1} ({time.time()-t0:.0f}s, gold {len(gold_ids)}/{len(gold_cids)})")
print(f"  gold collected: {len(gold_ids)}, filler pool: {len(texts)}")

rng = np.random.default_rng(7)
n_fill = N_BASE - len(gold_ids)
sel = rng.choice(len(texts), size=n_fill, replace=False)
base_texts = gold_texts + [texts[j] for j in sel]
base_cids  = gold_ids   + [cids[j]  for j in sel]
cid2idx = {c: i for i, c in enumerate(base_cids)}

# ---------- 2) queries (qrels 있는 것만) ----------
qs = load_dataset("BeIR/nq", "queries", split="queries")
q_texts, q_gold = [], []
for q in qs:
    qid = str(q["_id"])
    if qid in gold_map:
        idxs = [cid2idx[c] for c in gold_map[qid] if c in cid2idx]
        if idxs:
            q_texts.append(q["text"]); q_gold.append(idxs)
print(f"[2/5] usable queries: {len(q_texts)}")
qsel = rng.choice(len(q_texts), size=min(SAMPLE_Q, len(q_texts)), replace=False)
q_texts = [q_texts[i] for i in qsel]; q_gold = [q_gold[i] for i in qsel]

# ---------- 3) embed ----------
print("[3/5] embedding (bge-m3) ...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
emb = model.encode(base_texts, batch_size=256, show_progress_bar=True,
                   normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
qemb = model.encode(q_texts, batch_size=256, normalize_embeddings=True,
                    convert_to_numpy=True).astype(np.float32)
with h5py.File(OUT_H5, "w") as f:
    f.create_dataset("train", data=emb); f.create_dataset("test", data=qemb)
print(f"  saved {OUT_H5} ({time.time()-t0:.0f}s)")

# ---------- 4) HNSW 추출 + 곡선 ----------
print("[4/5] curves ...")
def build(base):
    n, d = base.shape
    index = faiss.IndexHNSWFlat(d, 16); index.hnsw.efConstruction = 200
    index.add(base); h = index.hnsw
    neighbors = faiss.vector_to_array(h.neighbors)
    offsets = faiss.vector_to_array(h.offsets).astype(np.int64)
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
    n0 = int(cum[1])
    idx = offsets[:-1][:, None] + np.arange(n0)[None, :]
    nbrs = neighbors[idx]
    return np.where(nbrs < 0, np.int32(-1), nbrs).astype(np.int32)

def beam_search(base, alive, nbrs, q, entries, beam, k):
    import heapq
    d0 = ((base[entries] - q) ** 2).sum(1)
    cand = [(d0[i], e) for i, e in enumerate(entries) if alive[e]]
    heapq.heapify(cand)
    best, seen = [], set(e for _, e in cand)
    frontier = list(cand)
    while frontier:
        d, u = heapq.heappop(frontier)
        if best and len(best) >= beam and d > -best[0][0]:
            break
        heapq.heappush(best, (-d, u))
        if len(best) > beam: heapq.heappop(best)
        for v in nbrs[u]:
            if v < 0 or v in seen or not alive[v]: continue
            seen.add(v)
            dv = ((base[v] - q) ** 2).sum()
            heapq.heappush(frontier, (dv, v))
    top = sorted([(-nd, u) for nd, u in best])[:k]
    return [u for _, u in top]

results = {}
for seed in SEEDS:
    nbrs = build(emb)
    indeg = np.bincount(nbrs[nbrs >= 0], minlength=len(emb)).astype(np.int64)
    r = np.random.default_rng(100 + seed)
    orders = {
        "random": r.permutation(len(emb)),
        "hub": np.argsort(-indeg, kind="stable"),
        "antihub": np.argsort(indeg, kind="stable"),
    }
    # relevance: intact top-50 빈도
    flat = faiss.IndexFlatL2(emb.shape[1]); flat.add(emb)
    _, I50 = flat.search(qemb, 50)
    freq = np.bincount(I50.ravel(), minlength=len(emb))
    orders["relevance"] = np.argsort(-freq, kind="stable")

    for mode, order in orders.items():
        hit, avail, subfail = [], [], []
        for f in F_GRID:
            ndel = int(f * len(emb))
            alive = np.ones(len(emb), bool); alive[order[:ndel]] = False
            entries = [i for i in r.choice(len(emb), 4*R_ENTRY, replace=False) if alive[i]][:R_ENTRY]
            h = a = s = 0
            for qi in range(len(qemb)):
                gold_alive = [g for g in q_gold[qi] if alive[g]]
                if not gold_alive:
                    a += 1; continue
                top = beam_search(emb, alive, nbrs, qemb[qi], entries, BEAM, K)
                if set(top) & set(gold_alive): h += 1
                else: s += 1
            nq = len(qemb)
            hit.append(h/nq); avail.append(a/nq); subfail.append(s/nq)
            print(f"  [{mode}] s{seed} f={f:.2f} hit={h/nq:.3f} avail_loss={a/nq:.3f} substrate={s/nq:.3f}", flush=True)
        results[f"{mode}_s{seed}"] = np.array([hit, avail, subfail])

np.savez(OUT_NPZ, F=F_GRID, **results)
print(f"[5/5] ===== 요약 (hit@10 반감점 & f=0.5 분해) =====")
for mode in ["random", "hub", "relevance", "antihub"]:
    hs = np.mean([results[f"{mode}_s{s}"][0] for s in SEEDS], axis=0)
    av = np.mean([results[f"{mode}_s{s}"][1] for s in SEEDS], axis=0)
    sf = np.mean([results[f"{mode}_s{s}"][2] for s in SEEDS], axis=0)
    half = hs[0] / 2
    fc = next((F_GRID[i] for i in range(len(hs)) if hs[i] < half), ">0.95")
    i5 = int(0.5 / 0.05)
    print(f"  {mode:10s} fc_gold={fc}  @f=0.5: hit={hs[i5]:.3f} avail={av[i5]:.3f} substrate={sf[i5]:.3f}")
print(f"[done] {time.time()-t0:.0f}s -> {OUT_NPZ}")
