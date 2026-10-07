Embedded knowledge corpus (NQ / bge-m3)

100,000 Natural Questions Wikipedia passages embedded with bge-m3 (1024-d, dense, L2-normalized),
indexed with the same settings as all other corpora (faiss IndexHNSWFlat, M=16, efConstruction=200).
For the knowledge-level decomposition, every annotated answer passage of the evaluation queries is
force-included in the base, and 500 annotated queries are evaluated for any answer passage in the top 10.
Stored at data/nq_bge_m3.hdf5 with datasets `train` [n,1024] and `test` [q,1024]; the annotated variant
carries the query-to-passage judgments used by code/b10_nq_gold.py. Built by code/b8_text_embed.py.
