"""Unit tests for the action-space segmentation/alignment bridge."""

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from action_bridge import (  # noqa: E402
    action_alignment_ce,
    action_alignment_distribution,
    action_similarity,
)
from train import VideoSSL  # noqa: E402


def test_action_similarity_recovers_one_hot_identity():
    q = torch.eye(3).unsqueeze(0)
    sim = action_similarity(q, q)
    assert torch.allclose(sim, torch.eye(3).unsqueeze(0), atol=1e-6)
    assert sim.min() >= 0 and sim.max() <= 1


def test_action_alignment_distribution_is_row_stochastic():
    torch.manual_seed(3)
    q_x = torch.softmax(torch.randn(2, 5, 4), dim=-1)
    q_y = torch.softmax(torch.randn(2, 7, 4), dim=-1)
    pred = action_alignment_distribution(q_x, q_y, temperature=0.2)
    assert pred.shape == (2, 5, 7)
    assert torch.allclose(pred.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_action_alignment_ce_propagates_to_both_action_trajectories():
    torch.manual_seed(5)
    logits_x = torch.randn(1, 4, 3, requires_grad=True)
    logits_y = torch.randn(1, 6, 3, requires_grad=True)
    q_x = torch.softmax(logits_x, dim=-1)
    q_y = torch.softmax(logits_y, dim=-1)
    target = torch.softmax(torch.randn(1, 4, 6), dim=-1)
    loss = action_alignment_ce(
        target, q_x, q_y, torch.ones(1, 4), torch.ones(1, 6), temperature=0.2
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits_x.grad is not None and logits_x.grad.abs().sum() > 0
    assert logits_y.grad is not None and logits_y.grad.abs().sum() > 0


def test_action_bridge_defaults_are_backward_compatible():
    model = VideoSSL()
    assert model.action_bridge_cost_weight == 0.0
    assert model.action_bridge_loss_weight == 0.0
