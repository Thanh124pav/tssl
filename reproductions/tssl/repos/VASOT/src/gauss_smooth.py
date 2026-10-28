"""Gaussian smoothing baseline for the "smooth-the-discrete-representation" family.

Simple conv1d with a Gaussian kernel along the time axis. No knot adaptivity,
no reparameterisation — this is the trivial baseline to compare B-spline and
arc-length variants against. If Gaussian also helps, the gain is not specific
to spline structure.
"""
import torch
import torch.nn.functional as F


def gaussian_kernel(sigma: float, device, dtype) -> torch.Tensor:
    kernel_size = max(3, int(6 * sigma) | 1)                 # odd size covering ±3σ
    half = kernel_size // 2
    idx = torch.arange(-half, half + 1, device=device, dtype=dtype)
    k = torch.exp(-(idx ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    return k, half


def gaussian_smooth(X: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """X: (B, T, D).  Returns smoothed X: (B, T, D)."""
    B, T, D = X.shape
    k, half = gaussian_kernel(sigma, X.device, X.dtype)
    kernel = k.view(1, 1, -1)
    x = X.transpose(1, 2).reshape(B * D, 1, T)
    y = F.conv1d(x, kernel, padding=half)
    return y.reshape(B, D, T).transpose(1, 2)
