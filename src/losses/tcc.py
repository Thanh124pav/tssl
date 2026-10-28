"""Minimal Temporal Cycle-Consistency loss (regression variant).

Follows the "cycle-back regression" from Dwibedi et al., CVPR 2019:
  1.  For each embedding u_i in U (T x D), compute soft-nn in V -> \tilde v_i.
  2.  Compute soft-nn of \tilde v_i back in U -> distribution beta_i over U.
  3.  Regression target is i. Loss = normalized squared error of beta_i's mean.

Reference impl: google-research/tcc/tcc/losses.py (we only need the regression
variant for a smoke test).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _soft_nn(u: torch.Tensor, v: torch.Tensor, tau: float) -> torch.Tensor:
    """Return weights (|u|, |v|) with softmax over -||u-v||^2 / tau."""
    # ||u-v||^2 = u^2 + v^2 - 2 u.v
    u2 = (u * u).sum(-1, keepdim=True)              # (T, 1)
    v2 = (v * v).sum(-1, keepdim=True).transpose(0, 1)  # (1, T)
    dist = u2 + v2 - 2 * u @ v.T
    return F.softmax(-dist / tau, dim=-1)


def tcc_cycle_regression_loss(u: torch.Tensor, v: torch.Tensor,
                              tau: float = 0.1,
                              variance_lambda: float = 1.0) -> torch.Tensor:
    """u, v: (T, D) frame embeddings for two videos."""
    T = u.shape[0]
    assert v.shape[0] == T, "same num frames required for toy test"
    alpha = _soft_nn(u, v, tau)              # (T, T)
    v_tilde = alpha @ v                      # (T, D)
    beta = _soft_nn(v_tilde, u, tau)         # (T, T)
    idx = torch.arange(T, device=u.device, dtype=u.dtype)
    mu = (beta * idx[None, :]).sum(-1)       # (T,) predicted return index
    sigma2 = (beta * (idx[None, :] - mu[:, None]) ** 2).sum(-1) + 1e-6
    err = (mu - idx) ** 2
    loss = (err / sigma2 + variance_lambda * torch.log(sigma2)).mean()
    return loss
