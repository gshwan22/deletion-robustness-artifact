"""d7b: pgvector — VACUUM(수리) vs REINDEX(재구축) 소요시간 + hub f∈{0.5,0.75,0.9} 회복, SIFT 1점
d4d: Qdrant — hub f∈{0.5,0.9} (컨테이너 떠 있을 때만)
e8 : 엄격 임계 f_0.9 — 모든 npz 의 1-D 곡선 키를 전수 추출(relevance/antihub/NQ 포함)"""
import numpy as np, time, glob, traceback
from c_common import load, build_hnsw, F_GRID
t0 = time.time()
LOG = lambda s: print(s, flush=True)

# ═══════════ e8 (가볍고 확실한 것 먼저) ═══════════
def f_at(F, rec, thr):
    rec = np.asarray(rec, float); below = np.where(rec < thr)[0]
    if len(below) == 0: return None
    i = below[0]
    if i == 0: return float(F[0])
    f0, f1, r0, r1 = F[i-1], F[i], rec[i-1], rec[i]
    return float(f0 + (r0 - thr) / (r0 - r1) * (f1 - f0)) if r0 != r1 else float(f1)
LOG("===== e8: 엄격 임계 전수 (키 필터 없음) =====")
for path in sorted(glob.glob("../results/*.npz")):
    try: d = np.load(path, allow_pickle=True)
    except Exception: continue
    F = d["F"] if "F" in d.files else None
    rows = {}
    for k in d.files:
        try: v = np.asarray(d[k])
        except Exception: continue
        if v.ndim != 1 or v.dtype.kind not in "fd": continue
        Fk = F if (F is not None and len(F) == len(v)) else (F_GRID if len(F_GRID) == len(v) else None)
        if Fk is None or not np.all((v >= -1e-6) & (v <= 1 + 1e-6)): continue
        bk = k.split("_s")[0] if "_s" in k and k.split("_s")[-1].isdigit() else k
        rows.setdefault(bk, []).append((Fk, v))
    if not rows: continue
    LOG(f"  [{path.split('/')[-1]}]")
    for bk, lst in sorted(rows.items()):
        Fk = lst[0][0]; m = np.mean([v for _, v in lst], axis=0)
        f90, f50 = f_at(Fk, m, 0.9), f_at(Fk, m, 0.5)
        fmt = lambda x: f"{x:.3f}" if x is not None else ">max"
        LOG(f"     {bk:36s} f_0.9={fmt(f90)}  f_c={fmt(f50)}  (n={len(lst)})")

# ═══════════ d7b: pgvector ═══════════
try:
    import psycopg2
    from psycopg2.extras import execute_values
    con = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pw", dbname="postgres")
    con.autocommit = True; cur = con.cursor()
    def exact_gold(base, q, alive):
        idx = np.where(alive)[0]; sub = base[idx]; g = []
        for qi in range(len(q)):
            dd = np.linalg.norm(sub - q[qi], axis=1); g.append(set(idx[np.argsort(dd)[:10]].tolist()))
        return g
    def recall_tbl(tbl, base, q, alive):
        gold = exact_gold(base, q, alive); cur.execute("SET hnsw.ef_search = 512;"); hits = 0
        for qi in range(len(q)):
            cur.execute(f"SELECT id FROM {tbl} ORDER BY v <-> %s::vector LIMIT 10;", (list(map(float, q[qi])),))
            hits += len({r[0] for r in cur.fetchall()} & gold[qi])
        return hits / (10 * len(q))
    def load_table(tbl, base):
        n, d = base.shape
        cur.execute(f"DROP TABLE IF EXISTS {tbl};"); cur.execute(f"CREATE TABLE {tbl} (id int PRIMARY KEY, v vector({d}));")
        B = 2000
        for i in range(0, n, B):
            rows = [(int(i + j), list(map(float, base[i + j]))) for j in range(min(B, n - i))]
            execute_values(cur, f"INSERT INTO {tbl} (id, v) VALUES %s", rows, template="(%s, %s::vector)")
        cur.execute("SET maintenance_work_mem = '2GB';")
        t = time.time(); cur.execute(f"CREATE INDEX {tbl}_hnsw ON {tbl} USING hnsw (v vector_l2_ops) WITH (m = 16, ef_construction = 200);")
        return time.time() - t
    LOG("===== d7b: pgvector 비용·f 스윕 =====")
    for ds, fs in [("gist", [0.5, 0.75, 0.9]), ("sift", [0.75])]:
        base, q = load(ds, 0); q = q[:300]; n = len(base)
        _, indeg = build_hnsw(base)
        hub = np.argsort(-indeg, kind="stable")
        t_build_full = load_table(f"pg_{ds}", base)
        LOG(f"  [{ds}] full index build {t_build_full:.0f}s, f=0 recall={recall_tbl(f'pg_{ds}', base, q, np.ones(n, bool)):.3f}")
        for f in fs:
            tbl = f"pg_{ds}_f{int(f*100)}"
            cur.execute(f"DROP TABLE IF EXISTS {tbl};"); cur.execute(f"CREATE TABLE {tbl} AS SELECT * FROM pg_{ds};")
            cur.execute(f"ALTER TABLE {tbl} ADD PRIMARY KEY (id);")
            cur.execute(f"CREATE INDEX {tbl}_hnsw ON {tbl} USING hnsw (v vector_l2_ops) WITH (m = 16, ef_construction = 200);")
            dele = hub[:int(f * n)]; alive = np.ones(n, bool); alive[dele] = False
            cur.execute(f"DELETE FROM {tbl} WHERE id = ANY(%s);", (list(map(int, dele)),))
            r_del = recall_tbl(tbl, base, q, alive)
            t = time.time(); cur.execute(f"VACUUM {tbl};"); t_vac = time.time() - t
            r_vac = recall_tbl(tbl, base, q, alive)
            # 재구축 비용: 생존자만으로 새 인덱스
            cur.execute(f"DROP TABLE IF EXISTS {tbl}_rb;"); cur.execute(f"CREATE TABLE {tbl}_rb AS SELECT * FROM {tbl};")
            t = time.time(); cur.execute(f"CREATE INDEX {tbl}_rb_hnsw ON {tbl}_rb USING hnsw (v vector_l2_ops) WITH (m = 16, ef_construction = 200);"); t_rb = time.time() - t
            LOG(f"  [{ds} hub f={f}] tombstone={r_del:.3f}  vacuum={r_vac:.3f}  | VACUUM {t_vac:.0f}s vs REINDEX(survivors) {t_rb:.0f}s  = {100*t_vac/max(t_rb,1e-9):.0f}%  ({time.time()-t0:.0f}s)")
            cur.execute(f"DROP TABLE {tbl};"); cur.execute(f"DROP TABLE {tbl}_rb;")
except Exception as e:
    LOG(f"  [d7b] 실패/건너뜀: {e}"); traceback.print_exc()

# ═══════════ d4d: Qdrant f 스윕 ═══════════
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff
    cl = QdrantClient("localhost", port=6333, timeout=120)
    LOG("===== d4d: Qdrant hub f 스윕 (GIST) =====")
    base, q = load("gist", 0); q = q[:300]; n, d = base.shape
    _, indeg = build_hnsw(base); hub = np.argsort(-indeg, kind="stable")
    def gold(alive):
        idx = np.where(alive)[0]; sub = base[idx]
        return [set(idx[np.argsort(np.linalg.norm(sub - q[i], axis=1))[:10]].tolist()) for i in range(len(q))]
    def rec(alive):
        g = gold(alive); hits = 0
        for i in range(len(q)):
            res = cl.query_points("annperc", query=q[i].tolist(), limit=10).points
            hits += len({int(r.id) for r in res} & g[i])
        return hits / (10 * len(q))
    for f in [0.5, 0.9]:
        cl.recreate_collection("annperc", vectors_config=VectorParams(size=d, distance=Distance.EUCLID))
        for i in range(0, n, 200):
            cl.upsert("annperc", points=[PointStruct(id=int(i + j), vector=base[i + j].tolist()) for j in range(min(200, n - i))])
        time.sleep(30)
        dele = hub[:int(f * n)]; alive = np.ones(n, bool); alive[dele] = False
        cl.delete("annperc", points_selector=list(map(int, dele)))
        r_t = rec(alive)
        cl.update_collection("annperc", optimizer_config=OptimizersConfigDiff(deleted_threshold=0.01, vacuum_min_vector_number=100))
        time.sleep(120)
        r_v = rec(alive)
        LOG(f"  [qdrant hub f={f}] after delete={r_t:.3f}  after vacuum={r_v:.3f}  ({time.time()-t0:.0f}s)")
except Exception as e:
    LOG(f"  [d4d] 실패/건너뜀: {e}")
LOG(f"[done] {time.time()-t0:.0f}s")
