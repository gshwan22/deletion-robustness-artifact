Embedded knowledge corpus (NQ / bge-m3)
=======================================

The corpus used as the modern knowledge-retrieval anchor is not redistributed
here, since it derives from public sources. It is reconstructed as follows.

Source
  Natural Questions (Wikipedia passages). 100,000 passages form the base.

Embedding
  bge-m3, 1024 dimensions, dense representation, L2-normalized.

Index
  Built with the same settings as every other corpus in the paper
  (faiss IndexHNSWFlat, M = 16, efConstruction = 200).

Annotated variant (Section 4.3, Figure 4)
  For the knowledge-level decomposition, every annotated answer passage of the
  evaluation queries is force-included in the 100k base, so that availability
  loss and substrate failure can be separated without truncation artifacts.
  Evaluation uses 500 annotated queries and asks whether any answer passage of
  a query appears in the top 10 of the retrieved list.

Storage layout
  An HDF5 file at data/nq_bge_m3.hdf5 with the datasets expected by
  code/c_common.py:
      train  float32 [n, 1024]   passage embeddings
      test   float32 [q, 1024]   query embeddings
  The annotated variant additionally carries the query-to-passage judgments
  used by code/b10_nq_gold.py.

Reproduction note
  Section 4.2 reports that the ordering pattern reproduces on this corpus, and
  Section 4.3 reports the two-channel decomposition. Both use the result arrays
  provided in results/ (dataset_generalization.npz and nq_gold.npz), so the
  figures and tables can be regenerated without rebuilding the embeddings.
