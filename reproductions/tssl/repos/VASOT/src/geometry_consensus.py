"""Anti-collapse geometry and soft multi-video pseudo-label consensus.

No ground-truth labels or frame positions are used as supervision. Temporal
distance only selects candidate negatives; action-posterior disagreement
protects repeated actions from unconditional repulsion.
"""

import itertools

import torch
import torch.nn.functional as F


def valid_frames(z, mask, stride=4):
    return F.normalize(z[0, mask[0].bool()][::stride], dim=-1)


def geometry_loss(views, masks, posteriors=None, stride=4, variance_floor=0.10,
                  far_fraction=0.25, cosine_margin=0.3):
    """VICReg-like variance/covariance plus gated far-frame repulsion.

    For unit-normalized D-dimensional vectors, standard deviation is about
    1/sqrt(D), so the variance floor is scaled for this embedding space.
    """
    sampled = [valid_frames(z, m, stride) for z, m in zip(views, masks)]
    all_z = torch.cat(sampled, dim=0)
    if len(all_z) < 3:
        return all_z.sum() * 0, all_z.sum() * 0, all_z.sum() * 0
    centered = all_z - all_z.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-6)
    variance = F.relu(variance_floor - std).mean()
    covariance = (centered.T @ centered) / len(centered)
    covariance = covariance - torch.diag_embed(covariance.diag())
    decorrelation = covariance.square().sum() / all_z.shape[-1]

    repulsion_terms = []
    if posteriors is not None:
        for z, mask, q in zip(views, masks, posteriors):
            zz = valid_frames(z, mask, stride)
            qq = q[0, mask[0].bool()][::stride].detach()
            qq = qq / qq.sum(-1, keepdim=True).clamp_min(1e-8)
            n = len(zz)
            if n < 4:
                continue
            ii, jj = torch.triu_indices(n, n, offset=max(1, int(n * far_fraction)), device=z.device)
            # Continuous disagreement weight; same-action repetitions get little weight.
            disagreement = (1 - (qq[ii] * qq[jj]).sum(-1)).clamp_min(0)
            cosine = (zz[ii] * zz[jj]).sum(-1)
            repulsion_terms.append((disagreement * F.relu(cosine - cosine_margin).square()).sum()
                                    / disagreement.sum().clamp_min(1e-8))
    repulsion = torch.stack(repulsion_terms).mean() if repulsion_terms else all_z.sum() * 0
    return variance, decorrelation, repulsion


def spectot_temporal_affinity_loss(views, masks, stride=4, window=2,
                                  nonlocal_scale=64.0, negative_weight=0.1):
    """SpecT-OT TAR objective on valid subsampled frames (separate ablation).

    This is the paper's label-free local attraction and distant repulsion
    applied to VASOT embeddings. It is deliberately kept separate from the
    pseudo-label-gated repulsion above, to expose repeated-action conflicts.
    """
    terms = []
    for view, mask in zip(views, masks):
        z = valid_frames(view, mask, stride)
        n = len(z)
        if n < 3:
            continue
        d = (torch.arange(n, device=z.device)[:, None]
             - torch.arange(n, device=z.device)[None, :]).abs()
        local = (d <= window) & (d > 0)
        distant = d > window
        affinity = ((1 + z @ z.T) / 2).clamp(1e-6, 1 - 1e-6)
        local_w = 1 / (d.square() + 1)
        distant_w = 1 - torch.exp(-(d - window).clamp_min(0).square() / nonlocal_scale)
        positive = -(local_w * local * affinity.log()).sum() / (local_w * local).sum().clamp_min(1e-8)
        negative = -(distant_w * distant * (1 - affinity).log()).sum() / (distant_w * distant).sum().clamp_min(1e-8)
        terms.append(positive + negative_weight * negative)
    return torch.stack(terms).mean() if terms else views[0].sum() * 0


def soft_consensus(views, masks, posteriors, stride=4, temperature=0.1,
                   blend=0.5, cycle_weight=1.0, reconstruction_weight=0.1):
    """Soft reciprocal and multi-path consistency, plus label propagation.

    P_ab is row-stochastic frame correspondence. Every other video supplies
    a direct and an indirect path. A reciprocal mismatch pays -log(return
    probability); triangle paths pay KL(direct || indirect). Detached path
    agreement weights the transferred OT targets, without a binary gate.
    Returns a differentiable consistency loss and detached refined targets.
    """
    if len(views) < 3 or not 0 <= blend <= 1:
        raise ValueError("soft consensus needs >=3 videos and blend in [0, 1]")
    z = [valid_frames(v, m, stride) for v, m in zip(views, masks)]
    q = [p[0, m[0].bool()][::stride] for p, m in zip(posteriors, masks)]
    q = [p / p.sum(-1, keepdim=True).clamp_min(1e-8) for p in q]
    n_views = len(z)
    P = {(a, b): (z[a] @ z[b].T / temperature).softmax(-1)
         for a, b in itertools.permutations(range(n_views), 2)}
    losses = []
    transferred = []
    confidence = []
    for a, b in itertools.permutations(range(n_views), 2):
        pa, pb = P[a, b], P[b, a]
        reciprocal = (pa * pb.T).sum(-1).clamp_min(1e-8)
        losses.append(-reciprocal.log().mean())
        # Reconstruct X embeddings after the soft X→Y→X walk. Detaching the
        # target prevents the target itself moving to satisfy reconstruction.
        reconstructed = pa @ (pb @ z[a])
        losses.append(reconstruction_weight *
                      (1 - F.cosine_similarity(reconstructed, z[a].detach(), dim=-1)).mean())
        path_scores = []
        for c in range(n_views):
            if c in (a, b):
                continue
            indirect = P[a, c] @ P[c, b]
            # Bounded discrepancy; avoids treating every non-mutual edge as
            # a hard negative and works for one-to-many correspondences.
            path_scores.append((pa - indirect).abs().mean(-1))
        agreement = (1 - torch.stack(path_scores).mean(0)).clamp(0, 1)
        losses.append(cycle_weight * torch.stack(path_scores).mean())
        # Stop gradients through pseudo-label construction and reliability.
        weight = (reciprocal.detach() * agreement.detach()).clamp_min(1e-8)
        transferred.append((a, (pa.detach() @ q[b].detach()) * weight[:, None]))
        confidence.append((a, weight))
    refined = [p.clone() for p in posteriors]
    for a in range(n_views):
        votes = torch.stack([v for idx, v in transferred if idx == a]).sum(0)
        weights = torch.stack([v for idx, v in confidence if idx == a]).sum(0)
        consensus = votes / weights[:, None].clamp_min(1e-8)
        indices = torch.where(masks[a][0].bool())[0][::stride]
        base_mass = refined[a][0, indices].sum(-1, keepdim=True)
        local_blend = blend * (weights / (n_views - 1)).clamp(0, 1)
        target = (1 - local_blend[:, None]) * q[a].detach() + local_blend[:, None] * consensus
        refined[a][0, indices] = target * base_mass
    return torch.stack(losses).mean(), refined
