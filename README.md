# Artifact: What You Delete Breaks Retrieval

Reproduction package for the EDBT 2027 Experiments & Analysis submission on
deletion robustness of graph-based vector indexes.

## Layout

    code/       experiment and figure-generation scripts
    scripts/    batch drivers used for the long sweeps
    results/    result arrays (.npz) backing every number in the paper
    figures/    final figures (PDF) as they appear in the paper

## Environment

    python 3.12, numpy, faiss-cpu, matplotlib, h5py
    optional: hnswlib (Section 3.5 confirmation), qdrant-client + Docker (Section 6.5)

    python -m venv .venv && source .venv/bin/activate
    pip install numpy faiss-cpu matplotlib h5py hnswlib qdrant-client

## Data

The benchmark corpora are public and are not redistributed here.
Place the HDF5 files under `data/` using the layout expected by `code/c_common.py`:

    data/sift-128-euclidean.hdf5     (ANN-Benchmarks)
    data/glove-100-angular.hdf5      (ANN-Benchmarks)
    data/gist-960-euclidean.hdf5     (ANN-Benchmarks)
    data/deep-image-96-angular.hdf5  (ANN-Benchmarks, 10M subset)
    data/nq_bge_m3.hdf5              (built from Natural Questions passages
                                      embedded with bge-m3; see code/README_corpus.txt)

## Reproducing

Each script writes a `.npz` into `results/` and prints the numbers quoted in the
paper. The provided `results/` already contains every array, so figures can be
regenerated without rerunning the sweeps:

    cd code
    python generate_figures_v24.py     # Figures 1, 2, 3, 5, 6
    python make_edbt_figs_v5.py        # Figures 4, 7, 8, 9

To rerun the experiments from scratch, use the batch drivers in `scripts/`.
Runtimes below are for a single workstation-class GPU node.

## Paper artifact map

| Paper element | Script | Result file |
|---|---|---|
| Table 4 (ordering spectrum) | `attack_spectrum.py`, `b2_spectrum_5s.py` | `attack_spectrum_5s.npz`, `dataset_generalization.npz` |
| Table 5 (connectivity baselines) | `gf_check.py` | `gf_check.npz` |
| Table 6 (degree statistics) | `c2_tail.py` | `c2_tail.npz` |
| Table 8 (search budget) | `c1_beta.py` | `c1_beta.npz` |
| Table 9 (repair regimes) | `c4_repair.py`, `c6b_repair_ext.py`, `d6_mrng_repair.py` | `c4_repair.npz`, `c6b_repair_ext.npz`, `d6_mrng_repair.npz` |
| Table 10 (Qdrant spot check) | `d4b_qdrant_gist.py`, `d4c_qdrant_random.py` | logs |
| Table 11 (low-f regime) | `c7_lowf.py` | `c7_lowf.npz` |
| Table 12 (repair-free policies) | `c5_trigger_repair.py` | `c5_trigger_repair.npz` |
| Table 13 (frontier under repair) | `d1_theta.py` | `d1_theta.npz` |
| Table 14 (mixed insertion) | `d5_mixed.py`, `d5b_theta45.py` | `d5_mixed.npz`, `d5b_theta45.npz` |
| Figure 4 (knowledge decomposition) | `b10_nq_gold.py` | `nq_gold.npz` |
| Figure 8 (mass reparameterization) | `e1_mass_collapse.py` | `e1_mass_collapse.npz` |
| Section 3.5 (tombstone masking) | `d3_hnswlib.py` | `d3_hnswlib.npz` |
| Section 8 (age axis) | `c3_age.py` | `c3_age.npz` |

## Notes

- All randomized results are reported as mean and standard deviation over five
  seeds; repair and extension experiments use three seeds, as annotated in the
  paper. The Adaptive, K-core, and Traversal orderings are single-seed.
- Deletion is applied physically (nodes and incident edges removed), not through
  a tombstone path; see Section 3.5 for why this distinction matters.
