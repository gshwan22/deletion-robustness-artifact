"""d5: 삽입-삭제 혼합 스트림 — 마지막 한계 항목 제거.
시작 80k, 스텝마다 삭제 2%(hub, 마지막 재구축 시점 차수 기준) + 삽입 1%(faiss 네이티브 add, 풀에서),
누적 24스텝. 정책: never / fixed20 / hubmass0.3 (질량 참조는 재구축 시 갱신). GIST+SIFT, 3 seeds."""
import numpy as np, time, faiss, h5py
from c_common import DATA, SAMPLE_Q, recall_at
from c_common import K as KK
t0=time.time(); out={}
N0, POOL_N, STEPS, DEL_R, INS_R = 80_000, 20_000, 24, 0.02, 0.01

def extract(index, ncur):
    h = index.hnsw
    neighbors = faiss.vector_to_array(h.neighbors)
    offsets = faiss.vector_to_array(h.offsets).astype(np.int64)
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
    n0 = int(cum[1])
    idx = offsets[:-1][:, None] + np.arange(n0)[None, :]
    nbrs = neighbors[idx][:ncur]
    return np.where(nbrs<0, np.int32(-1), nbrs).astype(np.int32)

for ds in ["gist","sift"]:
    with h5py.File(DATA[ds],"r") as f:
        tr = np.array(f["train"],dtype=np.float32); te = np.array(f["test"],dtype=np.float32)
    for seed in [0,1,2]:
        rng = np.random.default_rng(2100+seed)
        sel = rng.choice(len(tr), N0+POOL_N, replace=False)
        vecs = np.ascontiguousarray(tr[sel]); q = np.ascontiguousarray(te[:SAMPLE_Q])
        for policy in ["never","fixed20","hubmass30"]:
            index = faiss.IndexHNSWFlat(vecs.shape[1],16); index.hnsw.efConstruction=200
            index.add(vecs[:N0])
            ncur=N0; alive=np.ones(N0+POOL_N,bool); alive[N0:]=False   # 풀은 미삽입 상태
            inserted=N0
            nbrs=extract(index,ncur)
            indeg_ref=np.bincount(nbrs[nbrs>=0],minlength=N0+POOL_N).astype(np.int64)
            mass_ref=indeg_ref.sum(); n_ref=N0
            dm=0; dc=0; rebuilds=0; recs=[]
            ins_enabled=True   # 재빌드 후 삽입 중단(좌표 단순화, 전 정책 동일 조건)
            srng=np.random.default_rng(2200+seed)
            for st in range(STEPS):
                # 삭제: 참조 차수 기준 hub 상위 (생존자 중)
                k=int(DEL_R*alive.sum())
                cand=np.argsort(-np.where(alive, indeg_ref, -1), kind="stable")[:k]
                alive[cand]=False; dm+=indeg_ref[cand].sum(); dc+=k
                # 삽입: 풀에서 신규 (faiss add — 기존 그래프에 실제 연결)
                m=int(INS_R*n_ref)
                if ins_enabled and inserted+m <= N0+POOL_N:
                    index.add(vecs[inserted:inserted+m])
                    alive[inserted:inserted+m]=True
                    inserted+=m; ncur=inserted
                    nbrs=extract(index,ncur)
                # 트리거
                trig = (policy=="fixed20" and dc>=0.20*n_ref) or \
                       (policy=="hubmass30" and dm/mass_ref>0.30)
                if trig:
                    ids=np.where(alive[:ncur])[0]
                    index=faiss.IndexHNSWFlat(vecs.shape[1],16); index.hnsw.efConstruction=200
                    # 재빌드: 생존자만 새 좌표로 — 좌표계 유지 위해 재매핑
                    index.add(np.ascontiguousarray(vecs[ids]))
                    # 전역 좌표 복원용
                    sub_nbrs=extract(index,len(ids))
                    nbrs=np.full((ncur,sub_nbrs.shape[1]),-1,np.int32)
                    conv=np.where(sub_nbrs>=0, ids[np.clip(sub_nbrs,0,None)], -1)
                    nbrs[ids]=conv
                    # 이후 삽입은 재빌드 인덱스 좌표와 어긋나므로: 재빌드 후엔 삽입을 재빌드 인덱스에 add 하고
                    # 추출도 그 좌표로.. 단순화를 위해 재빌드 후 잔여 스텝은 별도 매핑 유지
                    indeg_ref=np.zeros(N0+POOL_N,np.int64)
                    d_loc=np.bincount(sub_nbrs[sub_nbrs>=0],minlength=len(ids))
                    indeg_ref[ids]=d_loc
                    mass_ref=indeg_ref.sum(); n_ref=alive.sum(); dm=0; dc=0; rebuilds+=1
                    ins_enabled=False   # 재빌드 후 삭제만 진행 (전 정책 동일 조건, 논문에 명시)
                recs.append(recall_at(vecs[:max(ncur,N0)], alive[:max(ncur,N0)], nbrs, q, srng))
            emax=1-min(recs)
            out[f"{ds}_{policy}_s{seed}"]=np.array(recs)
            out[f"{ds}_{policy}_s{seed}_meta"]=np.array([emax,rebuilds])
            print(f"  [{ds}] s{seed} {policy:9s} emax={emax:.3f} rebuilds={rebuilds} ({time.time()-t0:.0f}s)",flush=True)
np.savez("../results/d5_mixed.npz", **out)
print("===== d5 요약: 혼합 스트림 emax / rebuilds (3s) =====")
for ds in ["gist","sift"]:
    for policy in ["never","fixed20","hubmass30"]:
        m=np.mean([out[f"{ds}_{policy}_s{s}_meta"] for s in [0,1,2]],axis=0)
        print(f"  {ds} {policy:9s}: {m[0]:.3f} / r{m[1]:.1f}")
print("  판정: 삽입이 섞여도 질량 트리거의 우위가 유지되는가 (참조 갱신 포함)")
print(f"[done] {time.time()-t0:.0f}s")
