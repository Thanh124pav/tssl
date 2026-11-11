"""Label-free regularisers that spread the embedding cloud, one per prior work.

Each takes unit-norm frame embeddings (N x D, the valid frames of one or more
videos) and returns a scalar to MINIMISE. None uses ground truth.

  vc         VICReg variance + covariance (Bardes et al., 2022)
  unif       uniformity on the sphere (Wang & Isola, 2020)
  koleo      Kozachenko-Leonenko nearest-neighbour spread (Sablayrolles et al., 2019; DINOv2)
  ldreg      local intrinsic dimensionality (Huang et al., LDReg, 2024)
  koleo_res  koleo on prototype residuals, neighbours inside the same pseudo-action
             (VLAD-style residual + intra-normalisation)
  tneg       intra-video temporal contrast: Gaussian positives around each frame,
             every other frame of the video is a negative (TCN / CARL-style)
"""
import torch
import torch.nn.functional as F

NAMES = ('vc', 'unif', 'koleo', 'ldreg', 'koleo_res', 'tneg')


def _sqdist(z):
    """Pairwise squared distance of unit vectors, +inf on the diagonal."""
    d2 = (2.0 - 2.0 * z @ z.T).clamp_min(0)
    return d2 + torch.diag(torch.full((len(z),), float('inf'), device=z.device, dtype=z.dtype))


def vc(z):
    """Variance hinge + off-diagonal covariance (VICReg's 25 : 1 ratio).
    z is rescaled by sqrt(D): a uniform cloud on the sphere has unit variance
    per dimension, so the hinge at 1 is reachable."""
    d = z.shape[-1]
    x = (z - z.mean(0, keepdim=True)) * d ** 0.5
    std = (x.var(0, unbiased=False) + 1e-4).sqrt()
    cov = x.T @ x / len(x)
    off = cov - torch.diag(cov.diag())
    return F.relu(1.0 - std).mean() + 0.04 * off.pow(2).sum() / d


def unif(z, t=2.0):
    return torch.pdist(z).pow(2).mul(-t).exp().mean().log()


def koleo(z, same=None, eps=1e-6):
    """-mean log(distance to the nearest neighbour). `same` (N x N bool)
    restricts the neighbour search; rows without an allowed neighbour are dropped."""
    d2 = _sqdist(z)
    if same is not None:
        d2 = d2.masked_fill(~same, float('inf'))
    nn = d2.min(dim=1).values
    ok = torch.isfinite(nn)
    if not ok.any():
        return z.sum() * 0
    return -0.5 * (nn[ok] + eps).log().mean()


def ldreg(z, k=20, eps=1e-6):
    """-mean log LID, LID by the MLE over the k nearest neighbours:
    LID_i = -1 / mean_j log(d_ij / d_ik)."""
    k = min(k, len(z) - 1)
    if k < 2:
        return z.sum() * 0
    d2 = _sqdist(z).topk(k, dim=1, largest=False).values.clamp_min(eps)   # ascending
    inv_lid = 0.5 * (d2[:, -1:].log() - d2.log()).mean(dim=1)             # mean log(d_k / d_j) >= 0
    return inv_lid.clamp_min(eps).log().mean()


def koleo_res(z, assign, protos):
    """koleo on normalised residuals z - prototype(assigned action), nearest
    neighbour searched among frames of the same pseudo-action."""
    res = F.normalize(z - protos[assign], dim=-1)
    return koleo(res, same=assign[:, None] == assign[None, :])


def tneg(z, temp=0.1, sigma=4.0):
    """One video, frames in temporal order. Target over the other frames is a
    Gaussian in the frame-index gap; the softmax makes distant frames negatives."""
    n = len(z)
    if n < 3:
        return z.sum() * 0
    idx = torch.arange(n, device=z.device, dtype=z.dtype)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    target = torch.exp(-(idx[:, None] - idx[None, :]).pow(2) / (2 * sigma ** 2)).masked_fill(eye, 0)
    target = target / target.sum(-1, keepdim=True)
    logp = (z @ z.T / temp).masked_fill(eye, -1e4).log_softmax(-1)
    return -(target * logp).sum(-1).mean()


def spread_loss(name, zs, assigns=None, protos=None, temp=0.1):
    """zs: list of (N_v x D) valid-frame embeddings, one per video, in temporal
    order. assigns: matching list of pseudo-action ids (koleo_res only)."""
    if name == 'tneg':
        return torch.stack([tneg(z, temp=temp) for z in zs]).mean()
    z = torch.cat(zs, dim=0)
    if name == 'vc':
        return vc(z)
    if name == 'unif':
        return unif(z)
    if name == 'koleo':
        return koleo(z)
    if name == 'ldreg':
        return ldreg(z)
    if name == 'koleo_res':
        return koleo_res(z, torch.cat(assigns, dim=0), protos)
    raise ValueError(name)


def csls_offset(sim, mask_x, k=10):
    """Density of each Y frame as seen from X: mean of its k largest similarities
    to valid X frames (B x N_y). In a row-softmax over Y the X-side CSLS term is
    constant per row and cancels, so only this column term matters."""
    s = sim.masked_fill(~mask_x.bool().unsqueeze(2), -1e4)
    k = min(k, s.shape[1])
    return s.topk(k, dim=1).values.mean(dim=1)
