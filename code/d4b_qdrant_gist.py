"""d4: Qdrant 스팟체크 — SIFT-100k 업서트, faiss-추출 허브 순서로 30%/60% 삭제, recall 측정"""
import numpy as np, time
from c_common import load, build_hnsw
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
t0=time.time()
base,q = load("gist",0); n,d = base.shape
_,indeg = build_hnsw(base); hub=np.argsort(-indeg,kind="stable")
cl=QdrantClient("localhost",port=6333)
cl.recreate_collection("annperc", vectors_config=VectorParams(size=d,distance=Distance.EUCLID))
B=200
for i in range(0,n,B):
    cl.upsert("annperc",[PointStruct(id=int(j),vector=base[j].tolist()) for j in range(i,min(i+B,n))])
    if (i//B)%10==0: print(f"  upsert {i}/{n} ({time.time()-t0:.0f}s)",flush=True)
import faiss
def recall(alive):
    ids=np.where(alive)[0]; flat=faiss.IndexFlatL2(d); flat.add(base[ids])
    _,G=flat.search(q,10); gold=[set(ids[r]) for r in G]
    hits=0
    for qi in range(len(q)):
        res=cl.query_points("annperc",query=q[qi].tolist(),limit=10).points
        hits+=len(set(int(r.id) for r in res)&gold[qi])
    return hits/(len(q)*10)
alive=np.ones(n,bool)
print(f"  f=0.0 recall={recall(alive):.3f}")
for f in [0.75]:
    dele=hub[:int(f*n)]
    cl.delete("annperc",points_selector=[int(v) for v in dele])
    alive[:]=True; alive[dele]=False
    print(f"  [tombstone] f={f} recall={recall(alive):.3f}",flush=True)
    # 물리 반영 강제: 최적화/컴팩션 트리거 (버전 따라 API 상이 — 실패 시 수동 확인)
    try:
        cl.update_collection("annperc", optimizer_config={"deleted_threshold":0.01,"vacuum_min_vector_number":100})
        time.sleep(120)
        info = cl.get_collection("annperc")
        print(f"  [collection] points={info.points_count} indexed={info.indexed_vectors_count} segments={info.segments_count}", flush=True)
        print(f"  [post-vacuum] f={f} recall={recall(alive):.3f}",flush=True)
    except Exception as e:
        print("  vacuum trigger 실패 (버전 상이):",e)
print("[done]")
