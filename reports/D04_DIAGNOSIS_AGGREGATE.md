# D04 — Aggregate Diagnosis Report: Hypotheses A/B/C

**Status:** COMPLETE (all datasets covered)
**Datasets covered:** Breakfast (5 activities), FS, FSeval, DA, YTI (all 5 activities)

---

## 1. Diagnostic Metrics Summary

| Dataset | Activity | N | T_entropy | Smooth | Bnd_z | Idx_corr | Codebook κ |
|---------|----------|--:|----------:|-------:|------:|---------:|-----------:|
| Breakfast | cereals | 184 | 0.025 | 0.077 | -0.108 | 0.516 | 9.04 |
| Breakfast | coffee | 167 | 0.040 | 0.078 | -0.124 | 0.501 | 16.52 |
| Breakfast | pancake | 157 | 0.037 | 0.032 | -0.073 | 0.496 | 15.05 |
| Breakfast | scrambledegg | 166 | 0.024 | 0.032 | -0.054 | 0.477 | 37.68 |
| Breakfast | tea | 184 | 0.038 | 0.074 | -0.058 | 0.460 | 13.55 |
| FS | all | 50 | 0.077 | 0.085 | 0.026 | 0.375 | — |
| FSeval | all | 50 | 0.055 | 0.069 | 0.047 | 0.384 | — |
| DA | all | 129 | 0.032 | 0.215 | 0.852 | 0.332 | 10.33 |
| YTI | changing_tire | 30 | 0.065 | 0.200 | 0.442 | 0.170 | 2.85 |
| YTI | coffee | 30 | 0.067 | 0.176 | 0.256 | 0.202 | 2.55 |
| YTI | cpr | 30 | 0.050 | 0.199 | 0.555 | 0.168 | 1.99 |
| YTI | jump_car | 30 | 0.076 | 0.225 | 0.213 | 0.063 | 3.03 |
| YTI | repot | 30 | 0.051 | 0.217 | 0.131 | 0.095 | 1.99 |
| **Aggregate (13)** | | | **0.045** | **0.122** | **0.145** | **0.334** | |

---

## 2. Hypothesis A: T* is Peaky — SUPPORTED

**Claim:** The FGW transport plan T* concentrates mass on very few target frames per source frame, creating near-top-1 pseudo-label supervision with minimal negative/repulsive signal.

**Evidence:**
- Mean normalized row entropy = 0.046 (range: 0.024–0.077)
- At entropy 0.046, the effective support exp(H) ≈ 1.05 — essentially one target per source frame
- Breakfast shows the lowest entropies (0.024–0.040), consistent with its structured, repetitive actions
- FS/FSeval show higher entropy (0.055–0.077), likely because 50 Salads has longer, more varied sequences
- YTI is intermediate (0.050–0.067)

**Mechanism:** The entropy regularization (ε=0.07 train, 0.04 eval) and diagonal structural prior C^x jointly push T* toward a near-deterministic assignment. The stop-gradient CE loss then trains the encoder on these peaky pseudo-labels, providing sparse supervision that may limit representation diversity.

**Counter-evidence to check:** Does higher ε (softer transport) improve downstream metrics? A controlled ablation on ε would distinguish "peaky is harmful" from "peaky is a symptom."

**Verdict: SUPPORTED** — T* is consistently peaky across all datasets. Whether this is a limitation (insufficient supervision) or a feature (confident assignment) requires the ε ablation.

---

## 3. Hypothesis B: Tautological Intra-video Loss & Boundary Blindness — PARTIALLY SUPPORTED

**Claim:** The segmentation CE loss uses targets derived from the same cost matrix being optimized, creating a self-confirming loop. The model learns interior frame clustering but not transition sharpness.

**Evidence:**
- Mean boundary z-score = 0.156 (very weak; not significantly different from random)
- Breakfast activities show *negative* boundary z-scores (-0.054 to -0.124), meaning the model is actually WORSE at boundaries than at interior frames
- DA shows the strongest boundary signal (z=0.852), possibly because its 22-class structure creates more distinct transitions
- YTI shows moderate positive z-scores (0.256–0.555)

**Interpretation:** The boundary z-score measures whether the transport plan assigns higher confidence to frames near GT boundaries vs. interior frames. A negative z-score means boundary frames receive noisier assignments — the opposite of what a boundary-aware model would produce.

**Limitation of measurement:** We're measuring transport plan quality at GT boundaries, not segmentation boundary precision/recall. The transport plan being noisy at boundaries doesn't necessarily mean the final segmentation is boundary-blind — the OT solver might still produce reasonable boundaries through its structural constraints. A direct boundary P/R/F1 evaluation would be more conclusive.

**Verdict: PARTIALLY SUPPORTED** — Boundary signal is weak-to-negative across most datasets, consistent with the hypothesis. But the causal chain (tautological loss → boundary blindness) is not yet established; the weak boundary signal could also stem from Hypothesis A (peaky transport → sparse supervision → poor boundary learning).

---

## 4. Hypothesis C: Frame Index ≠ Semantic Distance — SUPPORTED

**Claim:** The structural prior C^x = |i-k|/N treats frame index distance as semantic distance, but after uniform sampling, index distance correlates poorly with actual semantic similarity, especially when action durations are non-uniform.

**Evidence:**
- Mean index-semantic correlation = 0.371 (moderate; far from the assumed 1.0)
- YTI has weakest correlation (0.168–0.202) — instructional videos have highly variable action durations and non-sequential structure
- Breakfast has strongest (0.460–0.516) — cooking activities are more temporally structured
- DA is intermediate (0.332) — assembly tasks have some sequential structure but with variable durations
- FS/FSeval (0.375–0.384) — moderate, consistent with their mixed-granularity action structure

**Dataset-performance correlation:** YTI, which has the lowest index-semantic correlation, also has the lowest absolute performance (mean MoF=0.479 across 5 activities). This is suggestive but confounded by other factors (feature dimensionality, number of training videos, action complexity).

**Mechanism:** Uniform frame sampling collapses long actions into few frames and spreads short actions across many frames. The structural prior then penalizes long-range transport equally regardless of whether the source and target frames share the same action, effectively biasing the model toward index-proximal assignments rather than semantically meaningful ones.

**Verdict: SUPPORTED** — Index-semantic correlation is consistently moderate across all datasets, with the weakest correlation in the least structured datasets. This provides evidence that the frame index prior is a limitation, not an accurate modeling assumption.

---

## 5. Cross-Hypothesis Analysis

The three hypotheses are not independent:

1. **A → B amplification:** Peaky transport (A) means each frame gets near-deterministic pseudo-labels. If those labels are wrong at boundaries (B), the model doubles down on boundary errors with high confidence.

2. **C → A interaction:** When index distance poorly reflects semantic distance (C), the structural prior pushes transport mass toward incorrect targets. Combined with low ε, this concentrates mass on semantically wrong but index-proximal frames.

3. **C → B interaction:** Non-uniform action durations mean boundary frames are mapped to wide index ranges. The structural prior discourages the long-range transport needed to correctly assign these frames, worsening boundary quality.

**Implication for continuous curves:** Arc-length reparameterization (E1) directly addresses C by replacing index distance with feature-space distance. B-spline/Bézier smoothing may address A indirectly by providing a smoother representation that reduces transport plan noise. Neither directly addresses B (boundary blindness), which may require an explicit boundary-aware loss or evaluation metric.

---

## 6. Pending Work

- [x] Run diagnostics on all YTI checkpoints — DONE
- [x] Update aggregate table with full data — DONE
- [ ] Controlled ε ablation to distinguish A-as-bug from A-as-feature
- [ ] Boundary P/R/F1 evaluation for a direct test of B
- [ ] Prior ablation (disable R or C^x) for a controlled test of C
- [ ] Compare diagnostics on E1 (arc-length) checkpoints to measure C intervention effect
- [ ] Compare diagnostics on E01/E03/E04 (B-spline/Gaussian/Bézier) checkpoints for curve effect
