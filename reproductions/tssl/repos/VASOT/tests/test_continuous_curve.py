"""CPU smoke tests for the continuous-curve extensions."""

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from arc_reparam import arc_u, inverse_warp_predictions, resample_uniform_u
from bspline import evaluate_bspline, fit_bspline, smooth_X
from gauss_smooth import gaussian_smooth


def test_arc_parameter_is_monotone_and_invertible_for_constant_labels():
    x = torch.tensor([[[0.0], [0.1], [0.2], [1.0], [1.1]]])
    mask = torch.ones(1, 5)
    u = arc_u(x, mask)

    assert torch.all(u[:, 1:] >= u[:, :-1])
    assert torch.allclose(u[:, 0], torch.zeros(1))
    assert torch.allclose(u[:, -1], torch.ones(1))

    warped = resample_uniform_u(x, u)
    assert warped.shape == x.shape
    labels = torch.full((1, 5), 3, dtype=torch.long)
    assert torch.equal(inverse_warp_predictions(labels, u), labels)


def test_bspline_fit_is_finite_and_preserves_endpoints():
    u = torch.linspace(0.0, 1.0, 32)
    x = torch.stack((u, u.square(), torch.sin(torch.pi * u)), dim=-1)
    knots, controls = fit_bspline(x, tol=0.02, k_max=12)
    fitted = evaluate_bspline(knots, controls, u)

    assert fitted.shape == x.shape
    assert torch.isfinite(fitted).all()
    assert torch.allclose(fitted[[0, -1]], x[[0, -1]], atol=1e-4)
    assert (fitted - x).norm(dim=-1).max() < 0.08


def test_bspline_and_gaussian_paths_propagate_gradients():
    torch.manual_seed(7)
    x = torch.randn(2, 24, 4, requires_grad=True)
    mask = torch.ones(2, 24)
    result = smooth_X(x, mask, tol=0.2, k_max=8) + gaussian_smooth(x, sigma=2)
    result.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
