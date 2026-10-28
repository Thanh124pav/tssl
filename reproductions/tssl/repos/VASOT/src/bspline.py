"""E2 — cubic B-spline retrofit with adaptive knot insertion.

Fits X ∈ R^{N×D} with a cubic (degree 3) B-spline X̂(u) = Σ_i N_{i,3}(u) c_i,
where the knot vector U is grown greedily:

    repeat
        fit control points C via least-squares with the current U
        pick u* = argmax_i ||X_i − X̂(u_i)||
        insert u* into U (clamped at endpoints)
    until max error < tol  or  |U| >= k_max

Because adjacent in-segment frames are near-collinear in embedding space,
residuals are small inside segments and large near action boundaries, so
adaptive insertion naturally **concentrates knots at the transitions**.

Design decisions
----------------
- Clamped endpoint knots (multiplicity p+1 = 4 at 0 and 1): the curve passes
  through c_0 at u=0 and c_{K-1} at u=1.
- Least-squares fit over the control points — differentiable wrt X (so
  gradients from downstream OT reach the encoder); non-differentiable wrt
  knot positions (that is fine for the retrofit mode).
- Per-video loop: knot insertion is data-dependent control flow, so batches
  (B=2 at train time) are processed in Python for-loop. Cost on H100 for
  N=256, D=40, K_max=32 is <1 ms per video.

Interfaces
----------
fit_bspline(X: (N,D), mask: (N,), tol, k_max) -> (U, C)
evaluate_bspline(U, C, u_query) -> (len(u_query), D)
smooth_X(X, mask, u_target, tol, k_max) -> X_hat (N, D) with X_hat sampled at u_target
"""
from __future__ import annotations

from typing import Tuple

import torch

_EPS = 1e-9


def _cox_deboor(u: torch.Tensor, U: torch.Tensor, p: int = 3) -> torch.Tensor:
    """Vectorised Cox-de Boor basis evaluation.

    u : (M,)   query positions in [0, 1]
    U : (nK,)  knot vector (clamped at endpoints, non-decreasing)
    p : spline degree

    Returns B : (M, nK - p - 1) — the value of each basis function at each query.
    """
    M = u.shape[0]
    nK = U.shape[0]
    K = nK - p - 1
    device = u.device
    # Degree 0: step functions
    u_m = u.unsqueeze(1)                            # (M, 1)
    U_l = U[:-1].unsqueeze(0)                       # (1, nK-1)
    U_r = U[1:].unsqueeze(0)                        # (1, nK-1)
    B = ((u_m >= U_l) & (u_m < U_r)).to(u.dtype)    # (M, nK-1)
    # Fix the right end so queries at u=1 belong to the last non-zero interval.
    at_right = u >= U[-1] - _EPS
    if at_right.any():
        # Find the last interval whose U_l == U_r != 1 is non-empty.
        # Simplest: for every query at the right endpoint, set its last-nonzero column.
        # The last interval where U_l < U_r corresponds to column `last_nontrivial`.
        last_nontrivial = (U_l < U_r).nonzero(as_tuple=False)[-1, 1].item()
        B[at_right, :] = 0
        B[at_right, last_nontrivial] = 1
    # Recursion up to degree p
    for d in range(1, p + 1):
        n_curr = nK - d - 1
        U_i = U[:n_curr].unsqueeze(0)              # (1, n_curr)
        U_id = U[d:d + n_curr].unsqueeze(0)
        U_i1 = U[1:1 + n_curr].unsqueeze(0)
        U_id1 = U[d + 1:d + 1 + n_curr].unsqueeze(0)
        denom_l = (U_id - U_i).clamp_min(_EPS)
        denom_r = (U_id1 - U_i1).clamp_min(_EPS)
        term_l = (u_m - U_i) / denom_l
        term_r = (U_id1 - u_m) / denom_r
        # Zero coefficients where original interval had zero length.
        term_l = torch.where(U_id > U_i, term_l, torch.zeros_like(term_l))
        term_r = torch.where(U_id1 > U_i1, term_r, torch.zeros_like(term_r))
        B = term_l * B[:, :n_curr] + term_r * B[:, 1:1 + n_curr]
    return B  # (M, K)


def _fit_control_points(u: torch.Tensor, X: torch.Tensor, U: torch.Tensor, p: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Least-squares fit of control points for the given knot vector.

    u : (N,) ascending query positions
    X : (N, D)
    U : (nK,) knot vector
    Returns (C, X_hat) where C: (K, D), X_hat: (N, D)
    """
    B = _cox_deboor(u, U, p=p)                      # (N, K)
    K = B.shape[1]
    # A clamped basis evaluates to the first/last control point at the two
    # endpoints, but an unconstrained least-squares fit does not make those
    # controls equal the first/last observations.  Fix both endpoint controls
    # and solve only for the interior controls so the fitted curve preserves
    # the observed sequence endpoints.
    C = torch.empty(K, X.shape[1], device=X.device, dtype=X.dtype)
    C[0] = X[0]
    C[-1] = X[-1]
    B_inner = B[:, 1:-1]
    target = X - B[:, :1] * C[0] - B[:, -1:] * C[-1]
    # Prefer the Moore-Penrose pseudo-inverse via SVD: stable even when the
    # basis is rank-deficient (which happens near duplicate knots).
    try:
        C[1:-1] = torch.linalg.pinv(B_inner) @ target
    except RuntimeError:
        C[1:-1] = torch.linalg.lstsq(B_inner, target).solution[:K - 2]
    if torch.isnan(C).any() or torch.isinf(C).any():
        # Fall back to a Tikhonov-regularised normal equation.
        BtB = B_inner.T @ B_inner + 1e-4 * torch.eye(K - 2, device=B.device, dtype=B.dtype)
        C[1:-1] = torch.linalg.solve(BtB, B_inner.T @ target)
    X_hat = B @ C
    return C, X_hat


def fit_bspline(X: torch.Tensor, mask: torch.Tensor | None = None,
                tol: float = 0.03, k_max: int | None = None, p: int = 3,
                u: torch.Tensor | None = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adaptive cubic B-spline fit on a single video.

    X : (N, D) frame embeddings (unit-norm after model's normalize).
    mask : (N,) float/bool, invalid frames excluded from error computation.
    tol : max per-frame reconstruction error before stopping.
    k_max : max number of control points; defaults to N // 8.
    u : (N,) query positions in [0,1]. If None, uniform in [0,1].

    Returns (U, C): final knot vector and control points.
    """
    N, D = X.shape
    device = X.device
    k_max = k_max or max(8, N // 8)
    if u is None:
        u = torch.linspace(0.0, 1.0, N, device=device, dtype=X.dtype)
    # Initial clamped knot vector = [0]*(p+1) + [1]*(p+1) → K = 4 control points for p=3.
    U = torch.cat([torch.zeros(p + 1, device=device, dtype=X.dtype),
                   torch.ones(p + 1, device=device, dtype=X.dtype)])
    # Internal (interior) knots: start empty.
    interior = torch.empty(0, device=device, dtype=X.dtype)
    m = mask.to(X.dtype) if mask is not None else torch.ones(N, device=device, dtype=X.dtype)

    while True:
        full_U = torch.cat([torch.zeros(p + 1, device=device, dtype=X.dtype),
                            interior,
                            torch.ones(p + 1, device=device, dtype=X.dtype)])
        C, X_hat = _fit_control_points(u, X, full_U, p=p)
        err = (X - X_hat).norm(dim=-1) * m
        # Candidate for insertion: argmax error, but only among valid frames
        # not at the endpoints (knots are already clamped there).
        valid = m.clone()
        valid[0] = 0; valid[-1] = 0
        err_masked = err * valid
        max_err = err_masked.max().item()
        K = full_U.shape[0] - p - 1
        if max_err < tol or K >= k_max:
            return full_U, C
        # Pick an interior u-position to insert — the max-error vertex, skipping
        # candidates that are too close to any existing knot (including the
        # clamped endpoints at 0 and 1).  min_gap scales with k_max so the final
        # knot vector always stays numerically well-separated.
        min_gap = max(1.0 / (4 * k_max), 1.0 / (2 * N))
        all_knots = torch.cat([torch.zeros(1, device=device, dtype=X.dtype),
                                interior,
                                torch.ones(1, device=device, dtype=X.dtype)])
        err_order = torch.argsort(err_masked, descending=True)
        inserted = False
        for i_cand in err_order.tolist():
            if err_masked[i_cand].item() < tol:
                break  # remaining candidates all below tolerance
            u_cand = u[i_cand].item()
            if (all_knots - u_cand).abs().min().item() >= min_gap:
                interior = torch.cat([interior,
                                       torch.tensor([u_cand], device=device, dtype=X.dtype)]).sort()[0]
                inserted = True
                break
        if not inserted:
            return full_U, C


def evaluate_bspline(U: torch.Tensor, C: torch.Tensor, u_query: torch.Tensor, p: int = 3) -> torch.Tensor:
    """Evaluate X̂(u_query) given knots U and control points C."""
    B = _cox_deboor(u_query, U, p=p)
    return B @ C


def smooth_X(X: torch.Tensor, mask: torch.Tensor | None = None,
             u_target: torch.Tensor | None = None,
             tol: float = 0.03, k_max: int | None = None, p: int = 3,
             return_knots: bool = False):
    """Batched convenience wrapper for retrofitting X -> X̂.

    X : (B, N, D)
    mask : (B, N) or None
    u_target : (B, N) query positions, or None (uniform).
    Returns X_hat : (B, N, D)   and, if return_knots, list of (U, C) per batch.
    """
    Bn, N, D = X.shape
    device = X.device
    if u_target is None:
        u_target = torch.linspace(0.0, 1.0, N, device=device, dtype=X.dtype).unsqueeze(0).expand(Bn, N)
    out = torch.empty_like(X)
    knots = []
    for b in range(Bn):
        mb = mask[b] if mask is not None else None
        U, Cb = fit_bspline(X[b], mb, tol=tol, k_max=k_max, p=p)
        out[b] = evaluate_bspline(U, Cb, u_target[b], p=p)
        if return_knots:
            knots.append((U, Cb))
    if return_knots:
        return out, knots
    return out


# ---------------- Diagnostic helpers ----------------

def knot_density_at_boundaries(U: torch.Tensor, gt_boundaries_u: torch.Tensor, window: float = 0.05) -> float:
    """Fraction of interior knots that fall within `window` of any GT boundary u.

    U : full knot vector (clamped p+1 at ends)
    gt_boundaries_u : (B,) u-positions of GT action boundaries.
    """
    # Interior knots only.
    interior = U[(U > 0) & (U < 1)]
    if interior.numel() == 0 or gt_boundaries_u.numel() == 0:
        return 0.0
    d = (interior.unsqueeze(1) - gt_boundaries_u.unsqueeze(0)).abs().min(dim=1).values
    return (d < window).float().mean().item()
