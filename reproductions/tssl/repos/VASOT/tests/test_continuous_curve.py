"""CPU smoke tests for the continuous-curve extensions."""

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from arc_reparam import arc_u, inverse_warp_predictions, resample_uniform_u
from bezier import BezierSpline, evaluate_bezier, fit_bezier, smooth_X_bezier
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


def test_bezier_fit_is_finite_and_preserves_endpoints():
    u = torch.linspace(0.0, 1.0, 32)
    x = torch.stack((u, u.square(), torch.sin(torch.pi * u)), dim=-1)
    spline = fit_bezier(x, tol=0.02, n_segments=4, adaptive=True, max_segments=12)
    fitted = evaluate_bezier(spline, u)

    assert fitted.shape == x.shape
    assert torch.isfinite(fitted).all()
    assert torch.allclose(fitted[[0, -1]], x[[0, -1]], atol=1e-3)
    assert (fitted - x).norm(dim=-1).max() < 0.1


def test_bezier_c1_continuity_at_joints():
    """Interior joints should have matching tangent directions from both sides."""
    torch.manual_seed(42)
    x = torch.randn(64, 4)
    spline = fit_bezier(x, n_segments=4, adaptive=False)
    S = spline.controls.shape[0]
    for s in range(1, S):
        tangent_left = spline.controls[s - 1, 3] - spline.controls[s - 1, 2]
        tangent_right = spline.controls[s, 1] - spline.controls[s, 0]
        assert torch.allclose(tangent_left, tangent_right, atol=1e-5), \
            f"C1 broken at joint {s}: left={tangent_left}, right={tangent_right}"


def test_bezier_short_sequence():
    """Bézier should handle very short sequences without crashing."""
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    spline = fit_bezier(x, n_segments=1, adaptive=False)
    fitted = evaluate_bezier(spline, torch.linspace(0, 1, 3))
    assert torch.isfinite(fitted).all()


def test_bezier_identity_blend():
    """With very high tolerance, output should still be finite."""
    torch.manual_seed(0)
    x = torch.randn(2, 24, 4)
    mask = torch.ones(2, 24)
    result = smooth_X_bezier(x, mask, n_segments=2, adaptive=False, tol=100.0)
    assert result.shape == x.shape
    assert torch.isfinite(result).all()


def test_bspline_and_gaussian_paths_propagate_gradients():
    torch.manual_seed(7)
    x = torch.randn(2, 24, 4, requires_grad=True)
    mask = torch.ones(2, 24)
    result = smooth_X(x, mask, tol=0.2, k_max=8) + gaussian_smooth(x, sigma=2)
    result.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_bezier_propagates_gradients():
    torch.manual_seed(7)
    x = torch.randn(2, 24, 4, requires_grad=True)
    mask = torch.ones(2, 24)
    result = smooth_X_bezier(x, mask, n_segments=4, adaptive=False, tol=0.1)
    result.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
