"""Regression checks for the temporal-prior ablation controls."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from train import VideoSSL  # noqa: E402


def test_default_rho_is_backward_compatible():
    model = VideoSSL(rho=0.15)
    assert model.rho_seg == 0.15
    assert model.rho_align == 0.15


def test_rho_overrides_are_independent():
    seg0 = VideoSSL(rho=0.15, rho_seg=0.0)
    align0 = VideoSSL(rho=0.15, rho_align=0.0)
    assert (seg0.rho_seg, seg0.rho_align) == (0.0, 0.15)
    assert (align0.rho_seg, align0.rho_align) == (0.15, 0.0)
