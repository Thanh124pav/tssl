# Overnight experiment plan — 2026-10-27 / 28

Goal. Produce evidence for a paper that argues VAOT/VASOT's three limitations
(T* near-one-hot, boundary-blind encoder, index-based structural prior) are
addressable by continuous-curve representations, **across multiple datasets
and multiple smoothing families**, so the result cannot be dismissed as a
cherry-picked single configuration.

## 1. Scope

### Variants

| Tag | Description | Hypothesis it attacks |
|---|---|---|
| `baseline` | paper Table 5 hparams, no change | reference |
| `E1_arc` | arc-length re-param, no warm-up | (C) index prior |
| `E2_bsx_b03` | B-spline on X, residual blend λ=0.3, warmup 5 ep | (A)(C) |
| `E2_bsx_b01` | B-spline on X, residual blend λ=0.1 (lighter) | (A)(C) |
| `E2_bsl_b03` | B-spline on logits, residual blend λ=0.3 | (A)(B) |
| `E2d_gauss2` | Gaussian conv σ=2 frames, blend λ=0.3 | sanity baseline |
| `E2d_gauss5` | Gaussian conv σ=5 frames, blend λ=0.3 | sanity baseline |
| `E3_arc_bsx` | arc-length + B-spline X λ=0.3 | (A)(B)(C) stacked |

### Datasets × epochs

| Dataset | n_cls | epochs |
|---|---:|---:|
| FSeval  | 12 | 30 |
| FS Mid  | 19 | 100 |
| Breakfast — coffee | 7 | 50 |
| Breakfast — cereals | 5 | 50 |
| Breakfast — scrambledegg | 12 | 50 |

5 datasets × 8 variants × 2 seeds = **80 runs**. Average ≈ 12 min/run on a
single H100 with 4-concurrent → ≈ 240 min = 4 h wall-clock. Fits inside the
overnight window with budget for retries.

### Diagnostics (post-training)

Compute on each resulting checkpoint:
- row entropy of segmentation T* (hypothesis A)
- boundary z-score of `||ΔX||` vs GT boundaries (hypothesis B)
- correlation of index distance with semantic label distance (hypothesis C)
- codebook singular-value spectrum (collapse detection)
- cost-matrix statistics (degeneracy detection)

## 2. Phases

| Phase | Task | Parallelism | Est. duration |
|---|---|---|---|
| 0 | code: NaN fix, blend, Gaussian, diagnose extensions | — | done |
| 1 | pre-run diagnostics on existing baselines (optional) | 1 | 10 min |
| 2 | **variant sweep** (80 runs) | 4-concurrent | 4 h |
| 3 | post-training diagnostics on all 80 checkpoints | 4-concurrent | 25 min |
| 4 | aggregate → SESSION_REPORT_OVERNIGHT.md | — | 5 min |
| 5 | commit + push to `tssl` | — | 1 min |

## 3. Statistical reporting

For each (variant × dataset) cell we have two seeds. The report shows mean ±
range, flags any cell where the two seeds disagree by more than 2 pp (likely
instability), and bolds variants that beat baseline by more than the range in
at least 4 of 5 datasets.

## 4. Related-work references (included inline in SESSION_REPORT_OVERNIGHT)

- CTE (Kukleva et al., ICCV 2019) — representation-then-K-means; collapse-prone
- TOT (Kumar et al., CVPR 2022) — temporal OT, suffers from action-order variation
- UFSA (Tran et al., 2023) — unified framework, still clock-time structural prior
- ASOT (Xu et al., CVPR 2024) — fused GW; **our baseline inherits all its limits**
- VAVA (Konin et al., ICCV 2023) — KOT with optimality prior
- LAV (Liu et al., 2022) — DTW, monotone-order assumption
- GTCC / TCC (Dwibedi, 2019) — frame-by-frame cycle consistency

The common thread across these is a **discrete, index-indexed representation
over sampled frames**, with structural priors defined on the sampling index.

## 5. Mathematical limitations we will formalise in the report

1. **CE under one-hot T*** degenerates to top-1 classification: `-Σ T*_ij log P_ij
   = -log P_{i, j*(i)}`. Encoder is rewarded only for argmax; no gradient on
   rank-2 and below → the model never learns to be "almost right".
2. **FGW structural prior has rank ≤ 2(Nr)+1 per row** when `C^x_{ik}` is a step
   function of `|i−k|`. Even with full D-dim embeddings, GW contribution to the
   gradient only "sees" a very low-rank slice of the pairwise structure.
3. **Sampling destroys semantic resolution**: uniform temporal sampling of
   `n_frames=256` from a 10 k-frame video means each sample represents a
   wildly non-uniform span of real content; the structural prior still assumes
   adjacency in sample index ≈ adjacency in semantic time.
4. **Over-smoothing kills discriminability**: if adjacent embeddings have
   `cos(X_i, X_{i+1}) = 1 − ε`, smoothing further pushes `cos → 1` and the
   segmentation OT cost `C_{ij} = 1 − X·A^T` becomes degenerate (equal cost
   to every cluster). A residual blend keeps per-frame discriminability.

Each of (1), (3), (4) is testable in our setup; (2) is a design observation we
note but do not expect to fix without redesigning the structural prior itself.
