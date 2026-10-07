# Artifact: What You Delete Breaks Retrieval

Reproduction package for the ICDE 2027 Experiment, Analysis, and Benchmark submission
"What You Delete Breaks Retrieval: Measuring Deletion Robustness Across the Vector-Index Maintenance Stack".

## Layout

    code/      experiment drivers, batch scripts, and figure generators
    results/   result arrays (.npz) behind every number in the paper, and run logs
    figures/   the figures as they appear in the paper (PDF)

## Environment

All experiments ran on a single NVIDIA DGX Spark node (20-core ARM CPU, 128 GB unified memory).

    python 3.12, numpy 2.5.1, faiss 1.14.3, hnswlib 0.8.0, scipy, networkx, matplotlib, h5py
    engines: Qdrant 1.19.0 (qdrant/qdrant Docker image), pgvector 0.8.6 on PostgreSQL 16
             (pgvector/pgvector:pg16 Docker image), qdrant-client, psycopg2-binary

    python -m venv .venv && source .venv/bin/activate
    pip install numpy faiss-cpu hnswlib scipy networkx matplotlib h5py qdrant-client psycopg2-binary
    docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
    docker run -d --name pgv -e POSTGRES_PASSWORD=pw -p 5432:5432 pgvector/pgvector:pg16

## Data

Benchmark corpora are public and are not redistributed. Place the HDF5 files under `data/`
as expected by `code/c_common.py`:

    sift-128-euclidean.hdf5, glove-100-angular.hdf5, gist-960-euclidean.hdf5   (ANN-Benchmarks)
    deep-image-96-angular.hdf5 (10M subset)                                     (ANN-Benchmarks)
    nq_bge_m3.hdf5   100k Natural Questions passages embedded with bge-m3 (see code/README_corpus.txt)

## Reproducing

`results/` already contains every array, so all figures regenerate without rerunning the sweeps:

    cd code
    python generate_icde_figs.py     # Figures 1, 2, 3, 6, 7
    python make_icde_figs.py         # Figures 4, 5, 8
    python make_icde_frontier.py     # Figure 9

To rerun experiments, run the scripts listed below from `code/`; the `batch_*.sh` drivers chain them.

## Conventions used in the paper

- Recall@10 against the exact top-10 of the surviving entries, multi-entry beam search
  (16 entry points, beam width 512), physical removal of deleted nodes and incident edges.
- Half-point f_c: first f where recall falls below half its undamaged value, linear interpolation.
  Reported values are means of per-seed half-points, except the strict-threshold table,
  which reads both thresholds from seed-mean curves of the repair-study runs (c4_repair).
- S(f): size of the largest connected component of the surviving graph with edge directions
  ignored, divided by the number of surviving entries.
- Normalized repair recovery rho = (f_c^rep - f_c^norep) / (f_c^rand - f_c^norep), with
  f_c^rand the random half-point without repair, censored half-points set to 0.95, rho capped at 1.
- Mass dispersion: standard deviation of recall across orderings, averaged over a common grid
  of f, or of mu after interpolating each curve onto that grid (e1_mass_collapse.py).
- Rebuild policies compare loss ceiling eps_max (maximum instantaneous recall loss over a stream)
  against rebuild count C_r. Matched-count factors interpolate the count frontier linearly.

## Paper element to script and result map

| Paper element | Script | Result file |
|---|---|---|
| Benchmark biases (tombstone, single entry point) | `d3_hnswlib.py`, `h2_entry.py` | `d3_hnswlib.npz`, `logs/` |
| Corpus ensemble, NQ ordering results | `dataset_generalization.py`, `b8_text_embed.py` | `dataset_generalization.npz` |
| Ordering spectrum (half-points) | `attack_spectrum.py`, `b2_spectrum_5s.py`, `e7_e5_long.py` (adaptive, K-core 5 seeds) | `attack_spectrum_5s.npz`, `e7_spectrum_seeds.npz` |
| Strict threshold f_0.9 | `f09_table.py` | `c4_repair.npz`, `e7_spectrum_seeds.npz`, `c3_age.npz` |
| Degradation curves and S(f) (Fig. 2) | `b5_curves5_5s.py` | `figdata_curves5_5s.npz` |
| Knowledge-level decomposition (Fig. 4) | `b10_nq_gold.py` | `nq_gold.npz` |
| Connectivity baselines | `gf_check.py` | `gf_check.npz` |
| Mass reparameterization (Fig. 5) | `e1_mass_collapse.py` | `e1_mass_collapse.npz` |
| Degree-statistic equivalence | `c2_tail.py` | `c2_tail.npz` |
| Search budget (Fig. 7) | `c1_beta.py` | `c1_beta.npz` |
| Local repair, recovery rho (Fig. 8) | `c4_repair.py`, `c6b_repair_ext.py`, `d6_mrng_repair.py` | `c4_repair.npz`, `c6b_repair_ext.npz`, `d6_mrng_repair.npz` |
| Adaptive batch sensitivity | `e4_only.py` | `e4_adaptive_batch.npz` |
| Engines: Qdrant | `d4b_qdrant_gist.py`, `d4c_qdrant_random.py` | `logs/` |
| Engines: pgvector recall and VACUUM timing | `d7_pgvector.py`, `d7b_d4d_e8_engines.py` | `logs/` |
| Rebuild rules without repair (Fig. 9a,b) | `e2_count_sweep.py`, `e2b_e6_sweeps.py` | `e2_count_sweep.npz`, `e2b_count_static.npz` |
| Recall probing comparison | `h_close_keir.py` (h1) | `h1_probe.npz` |
| Rebuild rules with repair (Fig. 9c) | `e7_e5_long.py` (e5) | `e5_count_repair.npz` |
| Interleaved insertion | `d5_mixed.py`, `d5b_theta45.py`, `e2b_e6_sweeps.py` (e6) | `d5_mixed.npz`, `d5b_theta45.npz`, `e6_mixed_count.npz` |
| Structural bias ratio mu(f)/f | `h_close_keir.py` (h3) | `logs/` |
| Age orderings | `c3_age.py` | `c3_age.npz` |
