import numpy as np
def build_rev(nbrs, n):
    rev = [[] for _ in range(n)]
    src, dst = np.where(nbrs >= 0)
    for u, j in zip(src, dst):
        rev[nbrs[u, j]].append(int(u))
    return rev
def repair_batch(base, nbrs, rev, alive, newly, cap):
    rewritten = 0
    for v in newly:
        outs_v = [int(w) for w in nbrs[v] if w >= 0 and alive[w]]
        for u in rev[v]:
            if not alive[u]: continue
            row = nbrs[u]
            if v not in row: continue
            cand = [int(w) for w in row if w >= 0 and alive[w] and w != v]
            cs = set(cand)
            for w in outs_v:
                if w != u and w not in cs:
                    cand.append(w); cs.add(w)
            if cand:
                d = ((base[cand] - base[u]) ** 2).sum(1)
                keep = [cand[i] for i in np.argsort(d)[:cap]]
            else:
                keep = []
            row[:] = -1; row[:len(keep)] = keep
            for w in keep: rev[w].append(int(u))
            rewritten += 1
    return rewritten
