import numpy as np
from c_common import F_GRID
def f_at(F, r, thr):
    r=np.asarray(r,float); b=np.where(r<thr)[0]
    if len(b)==0: return None
    i=b[0]
    if i==0: return float(F[0])
    return float(F[i-1]+(r[i-1]-thr)/(r[i-1]-r[i])*(F[i]-F[i-1]))
def show(tag, arr, F):
    a=np.asarray(arr,float); m=a.mean(0) if a.ndim==2 else a
    if len(m)!=len(F): return
    f9,f5=f_at(F,m,0.9),f_at(F,m,0.5); fmt=lambda x:f"{x:.3f}" if x is not None else ">0.95"
    print(f"  {tag:28s} f0.9={fmt(f9)}  fc={fmt(f5)}")
for fn in ["attack_spectrum_5s.npz","c4_repair.npz","e7_spectrum_seeds.npz","attack_spectrum.npz"]:
    try: d=np.load(f"../results/{fn}", allow_pickle=True)
    except Exception as e: print(fn,"skip",e); continue
    F=d["F"] if "F" in d.files else F_GRID
    print(f"[{fn}]")
    groups={}
    for k in d.files:
        if k=="F": continue
        v=d[k]
        if v.dtype==object: continue
        base=k.rsplit("_s",1)[0] if k.rsplit("_s",1)[-1].isdigit() else k
        groups.setdefault(base,[]).append(v)
    for b,vs in sorted(groups.items()):
        a=np.array(vs); a=a.reshape(-1,a.shape[-1]) if a.ndim>=2 else a
        show(b,a,F)
