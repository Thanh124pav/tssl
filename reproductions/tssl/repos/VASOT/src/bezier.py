"""E04 — piecewise cubic Bézier spline with C1 continuity.

Fits X ∈ R^{N×D} with a piecewise cubic Bézier curve composed of S segments,
each defined by 4 control points (start, cp1, cp2, end). Adjacent segments
share the endpoint and its tangent direction for C1 continuity.

Segment boundaries are placed at uniform u-intervals by default, or at
positions of largest reconstruction error (adaptive mode). The control points
are fit via least-squares with C1 constraints at joints.

Compared to B-spline:
- Explicit local control: each segment's shape depends only on its 4 points
- C1 (not C2) continuity — trades global smoothness for sharper transitions
- Simpler numerics: no Cox-de Boor recursion, just Bernstein polynomials

Interfaces
----------
fit_bezier(X, mask, n_segments, adaptive, tol) -> BezierSpline
evaluate_bezier(spline, u_query) -> (len(u_query), D)
smooth_X_bezier(X, mask, u_target, n_segments, ...) -> X_hat (N, D)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class BezierSpline:
    """Piecewise cubic Bézier representation."""
    breakpoints: torch.Tensor   # (S+1,) u-values of segment boundaries
    controls: torch.Tensor      # (S, 4, D) control points per segment


def _bernstein(t: torch.Tensor) -> torch.Tensor:
    """Cubic Bernstein basis values at parameter t ∈ [0,1].

    t: (M,) → returns (M, 4) with columns [B_0, B_1, B_2, B_3].
    """
    t2 = t * t
    t3 = t2 * t
    omt = 1.0 - t
    omt2 = omt * omt
    omt3 = omt2 * omt
    return torch.stack([omt3, 3 * omt2 * t, 3 * omt * t2, t3], dim=-1)


def _local_param(u: torch.Tensor, breakpoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map global u to (segment_index, local_t ∈ [0,1]).

    u: (M,), breakpoints: (S+1,) sorted.
    Returns seg_idx: (M,) long, t: (M,) float.
    """
    S = breakpoints.shape[0] - 1
    seg_idx = torch.searchsorted(breakpoints[1:], u).clamp(0, S - 1)
    u_lo = breakpoints[seg_idx]
    u_hi = breakpoints[seg_idx + 1]
    span = (u_hi - u_lo).clamp_min(1e-8)
    t = ((u - u_lo) / span).clamp(0.0, 1.0)
    return seg_idx, t


def evaluate_bezier(spline: BezierSpline, u_query: torch.Tensor) -> torch.Tensor:
    """Evaluate the piecewise Bézier at query positions.

    u_query: (M,) in [0, 1].
    Returns: (M, D).
    """
    seg_idx, t = _local_param(u_query, spline.breakpoints)
    B = _bernstein(t)                                # (M, 4)
    cp = spline.controls[seg_idx]                    # (M, 4, D)
    return (B.unsqueeze(-1) * cp).sum(dim=1)         # (M, D)


def _fit_segment_ls(u_seg: torch.Tensor, X_seg: torch.Tensor,
                    p0: torch.Tensor, p3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit interior control points (p1, p2) for one segment with fixed endpoints.

    u_seg: (n,) local parameters in [0,1]
    X_seg: (n, D) target points
    p0, p3: (D,) fixed endpoints
    Returns p1, p2: (D,) each.
    """
    B = _bernstein(u_seg)                            # (n, 4)
    target = X_seg - B[:, 0:1] * p0 - B[:, 3:4] * p3   # (n, D)
    A = B[:, 1:3]                                    # (n, 2)
    try:
        sol = torch.linalg.lstsq(A, target).solution  # (2, D)
    except RuntimeError:
        AtA = A.T @ A + 1e-4 * torch.eye(2, device=A.device, dtype=A.dtype)
        sol = torch.linalg.solve(AtA, A.T @ target)
    p1 = sol[0]
    p2 = sol[1]
    return p1, p2


def fit_bezier(X: torch.Tensor, mask: Optional[torch.Tensor] = None,
               n_segments: int = 4, adaptive: bool = True,
               tol: float = 0.03, max_segments: int = 32,
               u: Optional[torch.Tensor] = None) -> BezierSpline:
    """Fit a piecewise cubic Bézier to a single video's embeddings.

    X: (N, D) frame embeddings.
    mask: (N,) valid frames.
    n_segments: initial number of segments.
    adaptive: if True, split the worst segment until tol is met or max_segments reached.
    tol: max per-frame reconstruction error for adaptive stopping.
    u: (N,) parameter positions in [0,1]; uniform if None.

    Returns a BezierSpline.
    """
    N, D = X.shape
    device = X.device
    dtype = X.dtype
    n_segments = min(n_segments, N // 2)

    if u is None:
        u = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype)
    m = mask.to(dtype) if mask is not None else torch.ones(N, device=device, dtype=dtype)

    breakpoints = torch.linspace(0.0, 1.0, n_segments + 1, device=device, dtype=dtype)

    for _iter in range(max_segments - n_segments + 1):
        spline = _fit_with_breakpoints(u, X, breakpoints, m)
        if not adaptive:
            return spline

        X_hat = evaluate_bezier(spline, u)
        err = (X - X_hat).norm(dim=-1) * m
        max_err = err.max().item()

        if max_err < tol or breakpoints.shape[0] - 1 >= max_segments:
            return spline

        S = breakpoints.shape[0] - 1
        seg_idx, _ = _local_param(u, breakpoints)
        seg_errors = torch.zeros(S, device=device, dtype=dtype)
        for s in range(S):
            in_seg = seg_idx == s
            if in_seg.any():
                seg_errors[s] = err[in_seg].max()

        worst = seg_errors.argmax().item()
        mid = (breakpoints[worst] + breakpoints[worst + 1]) / 2.0
        breakpoints = torch.cat([breakpoints[:worst + 1],
                                  torch.tensor([mid], device=device, dtype=dtype),
                                  breakpoints[worst + 1:]]).sort()[0]

    return _fit_with_breakpoints(u, X, breakpoints, m)


def _fit_with_breakpoints(u: torch.Tensor, X: torch.Tensor,
                          breakpoints: torch.Tensor, m: torch.Tensor) -> BezierSpline:
    """Fit piecewise Bézier with given breakpoints, enforcing C1 at joints."""
    N, D = X.shape
    S = breakpoints.shape[0] - 1
    device = X.device
    dtype = X.dtype

    seg_idx, t_local = _local_param(u, breakpoints)
    controls = torch.zeros(S, 4, D, device=device, dtype=dtype)

    # Pass 1: assign endpoints — the curve passes through X at breakpoint positions
    for s in range(S + 1):
        bp_u = breakpoints[s].item()
        dist = (u - bp_u).abs()
        dist = dist + (1.0 - m) * 1e6  # mask invalid
        nearest = dist.argmin().item()
        if s < S:
            controls[s, 0] = X[nearest]
        if s > 0:
            controls[s - 1, 3] = X[nearest]

    # Pass 2: fit interior control points per segment
    for s in range(S):
        in_seg = (seg_idx == s) & (m > 0.5)
        if in_seg.sum() < 2:
            controls[s, 1] = (2 * controls[s, 0] + controls[s, 3]) / 3
            controls[s, 2] = (controls[s, 0] + 2 * controls[s, 3]) / 3
        else:
            p1, p2 = _fit_segment_ls(t_local[in_seg], X[in_seg],
                                     controls[s, 0], controls[s, 3])
            controls[s, 1] = p1
            controls[s, 2] = p2

    # Pass 3: enforce C1 continuity at interior joints
    for s in range(1, S):
        tangent_left = controls[s - 1, 3] - controls[s - 1, 2]
        tangent_right = controls[s, 1] - controls[s, 0]
        avg_tangent = (tangent_left + tangent_right) / 2.0
        controls[s - 1, 2] = controls[s - 1, 3] - avg_tangent
        controls[s, 1] = controls[s, 0] + avg_tangent

    return BezierSpline(breakpoints=breakpoints, controls=controls)


def smooth_X_bezier(X: torch.Tensor, mask: Optional[torch.Tensor] = None,
                    u_target: Optional[torch.Tensor] = None,
                    n_segments: int = 4, adaptive: bool = True,
                    tol: float = 0.03, max_segments: int = 32,
                    return_splines: bool = False):
    """Batched convenience wrapper for Bézier retrofitting.

    X: (B, N, D)
    mask: (B, N) or None
    u_target: (B, N) query positions, or None (uniform).
    Returns X_hat: (B, N, D) and optionally list of BezierSpline per batch.
    """
    Bn, N, D = X.shape
    device = X.device
    dtype = X.dtype
    if u_target is None:
        u_target = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype).unsqueeze(0).expand(Bn, N)
    out = torch.empty_like(X)
    splines: List[BezierSpline] = []
    for b in range(Bn):
        mb = mask[b] if mask is not None else None
        spline = fit_bezier(X[b], mb, n_segments=n_segments, adaptive=adaptive,
                           tol=tol, max_segments=max_segments)
        out[b] = evaluate_bezier(spline, u_target[b])
        if return_splines:
            splines.append(spline)
    if return_splines:
        return out, splines
    return out
