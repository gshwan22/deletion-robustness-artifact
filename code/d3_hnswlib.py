"""d3: 실물 라이브러리(hnswlib)의 tombstone 삭제가 구조 손상을 은폐함을 확인 — 함정 표의 실엔진 데이터"""
import numpy as np, time, h5py
import hnswlib
from c_common import load, build_hnsw, F_GRID
t0=time.time(); out={}
for seed in [0,1]:
    base,q = load("sift",seed)
    n,d = base.shape
    # faiss 추출 그래프의 허브 순서 (기하 프록시 공격)
    _,indeg = build_hnsw(base)
    hub_order = np.argsort(-indeg,kind="stable")
    rnd_order = np.random.default_rng(42+seed).permutation(n)
    idx = hnswlib.Index(space="l2", dim=d)
    idx.init_index(max_elements=n, ef_construction=200, M=16)
    idx.add_items(base, np.arange(n))
    idx.set_ef(512)
    import faiss
    for mode,order in [("hub",hub_order),("random",rnd_order)]:
        # 재초기화 없이 순차 mark_deleted (누적)
        idx2 = hnswlib.Index(space="l2", dim=d)
        idx2.init_index(max_elements=n, ef_construction=200, M=16)
        idx2.add_items(base, np.arange(n)); idx2.set_ef(512)
        alive=np.ones(n,bool); prev=0; rec=[]
        for f in [0.0,0.2,0.4,0.6,0.8,0.9]:
            ndel=int(f*n)
            for v in order[prev:ndel]:
                idx2.mark_deleted(int(v))
            alive[order[prev:ndel]]=False; prev=ndel
            ids=np.where(alive)[0]
            flat=faiss.IndexFlatL2(d); flat.add(base[ids])
            _,G=flat.search(q,10); gold=[set(ids[r]) for r in G]
            L,_=idx2.knn_query(q,k=10)
            hits=sum(len(set(L[i])&gold[i]) for i in range(len(q)))
            rec.append(hits/(len(q)*10))
        out[f"{mode}_s{seed}"]=np.array(rec)
        print(f"  [hnswlib] s{seed} {mode:6s} recall@f=0/.2/.4/.6/.8/.9 = "+"/".join(f"{r:.3f}" for r in rec),flush=True)
np.savez("../results/d3_hnswlib.npz", **out)
print("===== d3 판정: tombstone(hnswlib 실물)에서 recall 이 고-f 까지 유지되면 은폐 확인 =====")
print("     (우리 구조적 삭제의 동일 지점: hub f=0.8 에서 recall ~0)")
print(f"[done] {time.time()-t0:.0f}s")
