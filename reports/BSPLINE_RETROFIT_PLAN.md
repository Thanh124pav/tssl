# B-spline Retrofit for VAOT/VASOT — Implementation Plan

**Context.** VAOT/VASOT (Ali et al., ICCV 2025) represent each video as `X ∈ R^{N×D}`
— D-dim embedding per sampled frame. The FGW/ASOT objective (Eq. 3 in paper) uses
a step-function structural prior `C^x_{ik} = 1/r·𝟙[|i-k| ≤ N·r]` over integer
frame indices and a banded temporal prior `R_{ij} = |i/N − j/M|`.

**Hypothesis (user).** Replacing the discrete frame-indexed representation with a
continuous B-spline curve `X(u): [0,1] → R^D` (knots + control points, after
Han et al. "B-spline Policy", 2025) should fix three measured limitations:

| Limitation | Diagnostic value (FS Mid, 100ep, r=0.02) | How B-spline helps |
|---|---|---|
| (A) **T* sub-representation** — `T*` nearly one-hot (row entropy / log K = 0.077) | 0.077 of 1.000 | Continuous curve lets us use **dense gradient** everywhere on the curve (not just at sampled frames); or OT between measures-on-curves distributes mass more smoothly. |
| (B) **Intra-video loss misses boundaries** — `||X_{i+1}-X_i||` at GT boundary ≈ interior | boundary z-score ≈ 0.03 | Explicit **curvature-based boundary signal**: `||d²X/du²||` can be added as auxiliary target (large at semantic transitions, small inside actions). |
| (C) **Frame-index structural prior is weak** — corr(`\|i-k\|/N`, `GT differs`) = 0.375 | 0.375 Pearson | **Adaptive knot placement**: knots concentrate where curvature is high (= boundaries) → `C^x` becomes semantic-aware, not uniform in index. |

## 1. Representation design

For a video with `N` sampled frames and embeddings `X ∈ R^{N×D}`, fit a cubic
B-spline

```
X̂(u) = Σ_{i=0}^{K-1} N_{i,3}(u) · c_i,   u ∈ [0,1]
```

parameterised by:
- `U = [u_0=0, u_1, …, u_{K+3}=1]` — knot vector with `K+4` entries (clamped on both ends);
- `C = [c_0, …, c_{K-1}] ∈ R^{K×D}` — control points.

With `K << N` (e.g. `K = N/8`), this gives a smooth, locally-supported, differentiable curve.

### Two fitting modes

- **Mode A (retrofit, post-hoc fit per batch).** Keep MLP encoder unchanged. After
  forward pass that produces `X ∈ R^{N×D}`, run FITPACK-style adaptive fitting
  (Alg. 1 BSP paper) to recover `(U, C)`, then **re-sample** `X̂` at the
  original frame indices (or at adaptively chosen `u`) before OT.
  - Pros: no retraining of encoder; drop-in; cheap to test.
  - Cons: fitting is not differentiable end-to-end → gradients from OT go to `X`
    (unchanged), not into `(U, C)` directly. We only exploit smoothness, not
    adaptive-knot-aware OT.

- **Mode B (native, encoder outputs curve).** Replace the per-frame MLP with a
  sequence encoder (e.g. 2-layer Transformer encoder over the raw features)
  that outputs `K` control points + knot deltas. OT is then applied to points
  sampled from the fitted curve.
  - Pros: end-to-end; adaptive knots learned.
  - Cons: architectural change, more params; need to re-tune hparams.

Plan: **start with Mode A** to measure the ceiling of just-smoothness.
Decide on Mode B only if Mode A improves metrics ≥ 1 pp.

## 2. Fitting procedure (Mode A)

Follows BSP paper Algorithm 1 (adaptive FITPACK) with cubic basis (`p=3`):

```python
def fit_bspline(X, tol=1e-2, k_max=None):
    """X: (N, D) tensor. Returns (U, C, X_hat) with X_hat ∈ R^{N×D}."""
    N, D = X.shape
    k_max = k_max or max(8, N // 8)
    u_samples = torch.linspace(0, 1, N, device=X.device)
    U = torch.tensor([0.0, 1.0], device=X.device)
    while True:
        # Fit degree-3 B-spline with current knot vector U using least squares.
        C = lsq_fit(u_samples, X, U, p=3)
        X_hat = evaluate(U, C, u_samples)                       # (N, D)
        err = (X_hat - X).norm(dim=-1)                           # (N,)
        if err.max() < tol or len(U) >= k_max:
            return U, C, X_hat
        # Insert a knot at the point of max error (FITPACK criterion).
        u_new = u_samples[err.argmax()]
        U = torch.cat([U, u_new[None]]).sort()[0]
```

Use `torch` + a small cubic-B-spline evaluation (closed-form, differentiable in
`C` though NOT in `U` because knot positions are discrete decisions). For
dim D=40 and K≈32 this fit is <1ms on H100.

## 3. Where B-spline plugs into the training loss

VASOT training loss (Eq. 7 paper):

```
L_joint = w_align · L_xy(P^{xy}, T^{xy*}) + w_seg · (L_xa(P^{xa}, T^{xa*}) + L_ya(P^{ya}, T^{ya*}))
```

- `P^{xy}_{ij} = softmax(X Y^T / τ)` and `T^{xy*}` from `align_ot(cost=C(X,Y), rho, …)`.
- `C^x_{ik}` is the structural prior over X.

### Mode-A changes (minimal)

Replace the following in `train.py`:

| Line | Original | B-spline version |
|---|---|---|
| 97–100 (features_X normalize) | `X = F.normalize(mlp(X_raw))` | `X_disc = F.normalize(mlp(X_raw))` then `U_X, C_X, X = fit_bspline(X_disc, tol=0.05)` |
| 97–100 (features_Y) | same | same |
| `segmentation_ot.temporal_prior(T, K, ρ, dev)` and `C^x_{ik}` inside `align_ot`/`segment_ot` | step function in `|i-k|` | step function in `|u_i - u_k|` using the **adaptive `u`** anchored to knot positions |

The second row is the crux: ASOT's `C^x_{ik}` depends on integer indices.
We need to pass a float position `u_i ∈ [0,1]` for each frame — these come
from the adaptive knot assignment (every frame between knots gets the knot's u,
or we re-parameterise with arc-length `u_i = cumsum(||dX||) / total`).

### Arc-length re-parameterisation (recommended)

Instead of uniform `u_i = i/N`, define

```
u_i = Σ_{k<i} ||X_{k+1} - X_k||  / Σ_k ||X_{k+1} - X_k||
```

so **slow-moving (static) regions get compressed and transition regions get
expanded** in u-space. This is cheap and directly addresses limitation (C):
the structural prior `|u_i - u_k|` now encodes semantic proximity, not index
proximity. **This change alone, without full B-spline fitting, is the first
experiment to run** — it is the smallest possible intervention and lets us
isolate how much of the gain comes from reparameterisation vs. the smooth
curve itself.

## 4. Experiment ladder (lowest → highest cost)

| # | Name | What changes | Hours of code | GPU-hours |
|---|---|---|---|---|
| E0 | Baseline (current, paper hparams) | — | 0 | done |
| E1 | **Arc-length reparam only** | compute `u_i` from `||ΔX||`; use `|u_i - u_k|` in `C^x` and `R` | 2 | 2 (FS+FSeval+1 Breakfast activity) |
| E2 | **Post-hoc B-spline fit (Mode A)** | fit `X̂` per video, OT on `X̂`, uniform `u` | 4 | 2 |
| E3 | **E1 + E2 combined** | B-spline + arc-length u | 1 | 2 |
| E4 | **Curvature auxiliary loss** | add `λ · (−correlation(||d²X̂||, GT_boundary))` self-supervised proxy via `|ΔX|` percentile | 3 | 2 |
| E5 | **Native B-spline encoder (Mode B)** | Transformer → `(U, C)`; new hparams | 2 days | 8 |

Each run reports: alignment Acc@{0.1, 0.5, 1.0}, Progress, τ, AP@{5,10,15},
and segmentation MoF/F1/mIoU — same as paper Tables 2 & 3. Minimum one seed
per setting on FS Mid + FSeval (fast, 15 min + 5 min). Breakfast only after
promising on FS.

## 5. Metrics to watch during experiments

- `T* row entropy / log K` — **should rise** with B-spline (more diffuse mass).
- `||X̂_{i+1} - X̂_i||` at GT boundaries vs. interior — **boundary z-score should rise**.
- Hungarian confusion matrix — **off-diagonal should concentrate** at semantically
  similar classes (e.g. `cut_tomato` vs `cut_cucumber`) rather than scatter.
- OT solver iterations to converge — **should decrease** if `(C, R, C^x)` are better aligned.

## 6. Risks and mitigations

- **Smoothing kills transition sharpness.** If B-spline tolerance is too loose,
  action boundaries disappear. Mitigation: tie tolerance to feature variance;
  use adaptive knots.
- **Non-differentiable knot insertion.** Mode A knots are chosen by argmax of
  error → no gradient. Acceptable for retrofit. For Mode B use Gumbel-Softmax
  over candidate knot positions or output knot deltas as positive scalars via
  softplus + normalisation.
- **Arc-length re-param can be unstable at initialisation.** When encoder is
  random, `||ΔX||` is random too. Mitigation: warm up with uniform `u` for the
  first 5 epochs, then switch to arc-length.

## 7. Minimal first commit (E1, this week)

File: `src/arc_length_reparam.py` with

```python
def arc_u(X, mask):
    """X: (B,T,D); mask: (B,T). Returns u: (B,T) in [0,1]."""
    diffs = (X[:, 1:] - X[:, :-1]).norm(dim=-1) * mask[:, 1:].float()  # (B, T-1)
    cum = diffs.cumsum(dim=1)
    total = cum[:, -1:].clamp_min(1e-9)
    u = torch.cat([torch.zeros_like(cum[:, :1]), cum], dim=1) / total
    return u
```

Modify `segmentation_ot.temporal_prior(T, K, rho, device)` to accept a per-frame
`u` tensor instead of using `torch.arange(T) / T`. Modify ASOT's `C^x`
construction similarly. Train FS Mid, 100 epochs, r=0.02, seed 0. If MoF
rises by ≥1 pp without hurting Acc@1 we go to E2.

---

**References**
- Ali et al., ICCV 2025 (VAOT/VASOT), arXiv:2503.16832
- Han et al., 2026 (B-spline Policy), arXiv:2607.09648
- Diagnostic numbers in `reproductions/tssl/outputs/diag_FS_r0.02/diagnostics.json`
  and `diag_FSeval_r0.02/diagnostics.json`
