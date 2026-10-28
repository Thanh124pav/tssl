"""E1 (arc-length reparameterisation) utilities.

Given per-frame embeddings X ∈ R^{B,T,D}, we build a monotone warp
    u_i = cumsum(||ΔX||) / total_arc_length  ∈ [0,1]
and resample X at uniform u-positions, yielding X_warped ∈ R^{B,T,D} in which
the structural prior |i-k|/N approximates semantic distance rather than index
distance (hypothesis C in the diagnostic report).

The warp is a differentiable linear interpolation along the time axis, so
gradients flow through to the encoder.  GT labels and masks are warped the
same way (nearest-neighbor for labels, linear ≥ 0.5 threshold for masks).

Un-warping goes the other direction: given predictions in u-uniform space,
map them back to the original frame indices via the inverse CDF of u(i).
We use nearest-neighbor on the segment id for evaluation purposes.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def arc_u(X: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6, min_step: float = 1e-4) -> torch.Tensor:
    """Compute per-frame arc-length parameter u ∈ [0,1].

    X:    (B, T, D)
    mask: (B, T) bool/float; invalid frames contribute zero step.
    Returns u: (B, T) with u[:, 0]=0, u[:, T-1]=1 (if mask allows).
    """
    B, T, _ = X.shape
    diffs = (X[:, 1:] - X[:, :-1]).norm(dim=-1)                     # (B, T-1)
    if mask is not None:
        m = mask.float()
        step_mask = m[:, 1:] * m[:, :-1]                            # pair valid
        diffs = diffs * step_mask
    diffs = diffs.clamp_min(min_step)                               # avoid total=0 degeneracy
    cum = diffs.cumsum(dim=1)                                       # (B, T-1)
    total = cum[:, -1:].clamp_min(eps)
    u = torch.cat([torch.zeros_like(cum[:, :1]), cum], dim=1) / total
    return u                                                        # (B, T)


def resample_uniform_u(X: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Linear-interpolate X at uniform u-positions {0, 1/(T-1), …, 1}.

    X: (B, T, D); u: (B, T) monotone, u[:,0]=0, u[:,-1]=1.
    Returns X_warped: (B, T, D).
    """
    B, T, D = X.shape
    device = X.device
    u_target = torch.linspace(0.0, 1.0, T, device=device).unsqueeze(0).expand(B, T)   # (B, T)

    # For each target u*, find the segment (left, right) in u such that u[left] <= u* <= u[right].
    # Use searchsorted on each batch independently.
    out = torch.empty_like(X)
    for b in range(B):
        ub = u[b]                                                   # (T,)
        tgt = u_target[b]                                           # (T,)
        idx_right = torch.searchsorted(ub, tgt).clamp(1, T - 1)     # (T,)
        idx_left = idx_right - 1
        u_l = ub[idx_left]
        u_r = ub[idx_right]
        w = (tgt - u_l) / (u_r - u_l).clamp_min(1e-8)
        out[b] = (1 - w.unsqueeze(-1)) * X[b][idx_left] + w.unsqueeze(-1) * X[b][idx_right]
    return out


def resample_labels_nn(gt: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor resample integer labels at uniform u-positions."""
    B, T = gt.shape
    device = gt.device
    u_target = torch.linspace(0.0, 1.0, T, device=device).unsqueeze(0).expand(B, T)
    out = torch.empty_like(gt)
    for b in range(B):
        ub = u[b]
        tgt = u_target[b]
        idx_right = torch.searchsorted(ub, tgt).clamp(1, T - 1)
        idx_left = idx_right - 1
        d_l = (tgt - ub[idx_left]).abs()
        d_r = (ub[idx_right] - tgt).abs()
        choose_right = (d_r < d_l)
        picked = torch.where(choose_right, idx_right, idx_left)
        out[b] = gt[b][picked]
    return out


def resample_mask(mask: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor resample a boolean/0-1 mask."""
    return resample_labels_nn(mask.long(), u).to(mask.dtype)


def inverse_warp_predictions(seg: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Map segment predictions back from uniform-u space to original frame indices.

    seg: (B, T) integer segment ids defined on uniform u-grid.
    u:   (B, T) original-frame u positions.
    Returns seg_orig: (B, T) ids for the original frame positions.
    """
    B, T = seg.shape
    device = seg.device
    u_src = torch.linspace(0.0, 1.0, T, device=device).unsqueeze(0).expand(B, T)   # uniform grid where seg lives
    out = torch.empty_like(seg)
    for b in range(B):
        ub = u[b]                                                   # original-frame u (monotone)
        # For each original frame i, find the warped position closest to u[i].
        idx_right = torch.searchsorted(u_src[b], ub).clamp(1, T - 1)
        idx_left = idx_right - 1
        d_l = (ub - u_src[b][idx_left]).abs()
        d_r = (u_src[b][idx_right] - ub).abs()
        picked = torch.where(d_r < d_l, idx_right, idx_left)
        out[b] = seg[b][picked]
    return out
