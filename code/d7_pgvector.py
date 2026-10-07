"""d7: 엔진 2호 스팟체크 — pgvector (PostgreSQL HNSW). GIST-100k, f=0.75, hub vs random.
pgvector 삭제 의미론: DELETE = tombstone(힙 가시성), VACUUM = 인덱스에서 제거 + 링크 수리.
→ hnswlib(은폐형)·Qdrant(노출형+재작성)과 다른 세 번째 유형 기대.
사전: docker run -d --name pgv -e POSTGRES_PASSWORD=pw -p 5432:5432 pgvector/pgvector:pg16
      pip install psycopg2-binary"""
import numpy as np, time, psycopg2
from psycopg2.extras import execute_values
from c_common import load, build_hnsw
t0 = time.time()
base, q = load("gist", 0)
n, d = base.shape
q = q[:300]
_, indeg = build_hnsw(base)

con = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pw", dbname="postgres")
con.autocommit = True
cur = con.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("DROP TABLE IF EXISTS items;")
cur.execute(f"CREATE TABLE items (id int PRIMARY KEY, v vector({d}));")
B = 2000
for i in range(0, n, B):
    rows = [(int(i + j), list(map(float, base[i + j]))) for j in range(min(B, n - i))]
    execute_values(cur, "INSERT INTO items (id, v) VALUES %s", rows, template="(%s, %s::vector)")
    if (i // B) % 10 == 0: print(f"  insert {i}/{n} ({time.time()-t0:.0f}s)", flush=True)
cur.execute("SET maintenance_work_mem = '2GB';")
cur.execute("CREATE INDEX items_hnsw ON items USING hnsw (v vector_l2_ops) WITH (m = 16, ef_construction = 200);")
print(f"  index built ({time.time()-t0:.0f}s)", flush=True)

def exact_gold(alive):
    g = []
    for qi in range(len(q)):
        dd = np.linalg.norm(base[alive] - q[qi], axis=1)
        g.append(set(np.where(alive)[0][np.argsort(dd)[:10]].tolist()))
    return g

def recall(alive):
    gold = exact_gold(alive)
    cur.execute("SET hnsw.ef_search = 512;")
    hits = 0
    for qi in range(len(q)):
        cur.execute("SELECT id FROM items ORDER BY v <-> %s::vector LIMIT 10;", (list(map(float, q[qi])),))
        got = {r[0] for r in cur.fetchall()}
        hits += len(got & gold[qi])
    return hits / (10 * len(q))

print(f"  f=0.0 recall={recall(np.ones(n, bool)):.3f}", flush=True)
hub = np.argsort(-indeg, kind="stable")
rand = np.random.default_rng(42).permutation(n)
for name, order in [("hub", hub), ("random", rand)]:
    cur.execute("DROP TABLE IF EXISTS items2;")
    cur.execute("CREATE TABLE items2 AS SELECT * FROM items;")
    cur.execute("ALTER TABLE items2 ADD PRIMARY KEY (id);")
    cur.execute("CREATE INDEX items2_hnsw ON items2 USING hnsw (v vector_l2_ops) WITH (m = 16, ef_construction = 200);")
    f = 0.75
    dele = order[:int(f * n)]
    alive = np.ones(n, bool); alive[dele] = False
    cur.execute("DELETE FROM items2 WHERE id = ANY(%s);", (list(map(int, dele)),))
    # 측정은 items2 기준이므로 recall() 쿼리 테이블 치환
    def recall2(alive):
        gold = exact_gold(alive); cur.execute("SET hnsw.ef_search = 512;"); hits = 0
        for qi in range(len(q)):
            cur.execute("SELECT id FROM items2 ORDER BY v <-> %s::vector LIMIT 10;", (list(map(float, q[qi])),))
            hits += len({r[0] for r in cur.fetchall()} & gold[qi])
        return hits / (10 * len(q))
    r_del = recall2(alive)
    print(f"  [{name}] after DELETE (tombstone)  f={f}: recall={r_del:.3f}", flush=True)
    cur.execute("VACUUM items2;")
    r_vac = recall2(alive)
    print(f"  [{name}] after VACUUM (index repair) f={f}: recall={r_vac:.3f}  ({time.time()-t0:.0f}s)", flush=True)
print("===== d7 판정 =====")
print("  tombstone 단계 recall 높음 → 은폐형(hnswlib 계열) / 낮음 → 노출형(Qdrant 계열)")
print("  VACUUM 후 회복 정도 → pgvector 의 삭제 시 링크 수리 효력 (우리 §6.4 수리 결과와 대조)")
print(f"[done] {time.time()-t0:.0f}s")
