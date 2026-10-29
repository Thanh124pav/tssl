"""Action-space bridge between segmentation and video alignment.

The segmentation branch produces a posterior q(a | x_i) for each frame.
This module turns those posteriors into a bounded frame-to-frame similarity
which can (1) augment the detached OT alignment cost and (2) define a
differentiable alignment prediction supervised by the detached OT plan.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def action_similarity(q_x: torch.Tensor, q_y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity between action posteriors, shape ``(B, N, M)``.

    Cosine is used instead of a raw dot product so uncertain/diffuse and
    confident posteriors have comparable scale. Inputs remain differentiable.
    """
    q_x = F.normalize(q_x.clamp_min(0.0), dim=-1, eps=eps)
    q_y = F.normalize(q_y.clamp_min(0.0), dim=-1, eps=eps)
    return (q_x @ q_y.transpose(1, 2)).clamp(0.0, 1.0)


def action_alignment_distribution(
    q_x: torch.Tensor,
    q_y: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Row-normalized frame alignment distribution induced by action space."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(action_similarity(q_x, q_y) / temperature, dim=-1)


def action_alignment_ce(
    target_transport: torch.Tensor,
    q_x: torch.Tensor,
    q_y: torch.Tensor,
    mask_x: torch.Tensor,
    mask_y: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """CE from a stop-gradient frame transport to action-induced alignment."""
    pred = action_alignment_distribution(q_x, q_y, temperature=temperature)
    valid = mask_x.unsqueeze(2) * mask_y.unsqueeze(1)
    return -((target_transport.detach() * torch.log(pred + eps)) * valid).sum(dim=2).mean()
