"""Ray geometry objectives: impose the proposal's design goals during training.

Design goals, as stated by the user:
  (1) frames that align across videos should lie on one line through the action
      centre, so that the line generalises correspondence to a new video;
  (2) the lines of the correspondence groups inside one segment should be
      concurrent, and that concurrency point should update the action centre.

Two things make the direct translation of (1) into a loss unusable, both
measured on this project's own checkpoints:

  * `clusters` cannot serve as the geometric centre. The segmentation cost is
    `1 - z a_k^T`, which wants unit-norm codebook rows, while a centre that
    frames radiate from must sit inside the cloud. A06 measured the conflict:
    the radius around the prototype is LESS constant than matched noise (ratio
    1.14) while around the group centroid it is 0.894. So this module keeps a
    separate, unconstrained centre `c_k`, initialised at the action centroid,
    and leaves `clusters` to the segmentation cost.
  * The degeneracy is CENTRE ESCAPE, and it is triggered by goal (2), not by
    goal (1). Measured on free points (scripts/test_ray_geometry.py):

        centre fixed            loss 2e-8, |c| 0.00, radius 3.6, between-group 0.027
        centre free (goal 2)    loss 3e-8, |c| 4.27, radius 6.3, between-group 0.866
        centre free + radius    loss 2e-8, |c| 0.22, radius 1.0, between-group 0.018

    With the centre free to move, the optimiser pushes it far from the cloud; at
    a distance every direction converges on the direction from the centre to the
    cloud, so every pair looks collinear and the loss reaches zero carrying no
    information at all. A radius constraint, not a contrastive term, is what
    blocks this. An earlier version of this file claimed goal (1) alone
    collapses every frame onto one ray; the test falsified that -- the ray loss
    is separable across groups, so each group independently aligns to whichever
    ray is nearest its initialisation (between-group cosine 0.027).

    The group-separation term is kept because it directly bounds the
    between-group similarity the escape exploits, but the necessary guards are
    the radius regulariser and the centre containment below.

A03 adds one more constraint: over one median action the drift moves 0.293
while the jitter radius is 0.418 (SNR 0.70), so a direction computed from a raw
frame embedding is dominated by noise. Directions here are computed from the
temporally smoothed (drift) component.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def smooth_drift(z: torch.Tensor, window: int) -> torch.Tensor:
    """Moving average along time, re-normalised. z: (B, T, D)."""
    if window <= 1:
        return z
    b, t, d = z.shape
    k = torch.ones(d, 1, window, device=z.device, dtype=z.dtype) / window
    s = F.conv1d(z.transpose(1, 2), k, padding=window // 2, groups=d).transpose(1, 2)
    return F.normalize(s[:, :t], dim=-1)


def directions(points: torch.Tensor, centre: torch.Tensor, eps: float = 1e-4):
    """Unit direction from `centre` to each point, plus the radius.

    `eps` is a floor on the radius, not a numerical afterthought: if a point sits
    on the centre its direction is undefined and must not dominate the loss.
    """
    v = points - centre
    r = v.norm(dim=-1, keepdim=True)
    u = v / r.clamp_min(eps)
    return u, r.squeeze(-1)


def best_fit_lines(groups: torch.Tensor, mask: torch.Tensor):
    """Best-fit line per group. groups: (G, V, D), mask: (G, V) bool.

    Returns (mu, d, var_frac): a point on the line, a unit direction, and the
    fraction of the group's variance the line explains, which is the honest
    measure of whether a line describes the group at all.
    """
    w = mask.float().unsqueeze(-1)
    n = w.sum(dim=1).clamp_min(1.0)
    mu = (groups * w).sum(dim=1) / n
    x = (groups - mu.unsqueeze(1)) * w
    # power iteration is enough for the leading direction and keeps this cheap
    d = F.normalize(torch.randn(groups.shape[0], groups.shape[-1],
                                device=groups.device, dtype=groups.dtype), dim=-1)
    for _ in range(12):
        d = F.normalize(torch.einsum('gvd,gv->gd', x, torch.einsum('gvd,gd->gv', x, d)), dim=-1)
    proj = torch.einsum('gvd,gd->gv', x, d)
    total = (x ** 2).sum(dim=(1, 2)).clamp_min(1e-12)
    var_frac = (proj ** 2).sum(dim=1) / total
    return mu, d, var_frac


def least_squares_common_point(mu: torch.Tensor, d: torch.Tensor, weight: torch.Tensor,
                               damping: float = 1e-3):
    """Point minimising the weighted sum of squared distances to the lines.

    Returns (point, mean_residual, conditioning). `conditioning` is the smallest
    eigenvalue of the normal matrix: when the line directions nearly agree the
    intersection is undetermined and the caller must skip the update rather than
    follow an ill-posed solution.
    """
    dim = mu.shape[-1]
    eye = torch.eye(dim, device=mu.device, dtype=mu.dtype)
    P = eye.unsqueeze(0) - torch.einsum('gi,gj->gij', d, d)
    w = weight.view(-1, 1, 1)
    A = (P * w).sum(dim=0) + damping * eye
    b = torch.einsum('gij,gj->i', P * w, mu)
    x = torch.linalg.solve(A, b)
    v = x.unsqueeze(0) - mu
    res = (v - torch.einsum('gi,gi->g', v, d).unsqueeze(-1) * d).norm(dim=-1)
    cond = torch.linalg.eigvalsh(A).min()
    return x, (res * weight).sum() / weight.sum().clamp_min(1e-12), cond


# --------------------------------------------------------------------------
# correspondence groups from transport, with a cycle-consistency filter
# --------------------------------------------------------------------------

def build_groups(plans_fwd, plans_bwd, n_ref, stride=4, cycle_tol=4,
                 min_members=2, min_consistent_frac=0.5):
    """Groups anchored on a reference video.

    plans_fwd[v]: (N_ref,) argmax index in video v for each reference frame.
    plans_bwd[v]: (N_v,) argmax index back in the reference video.

    A02 measured that this transport is not cycle-consistent (X->Y->X returns
    within 4 frames for 46% of frames), so unfiltered groups would be built from
    roughly half-wrong members. Every member must survive the round trip, and a
    group is kept only if enough of its members do. The yield is returned so it
    can be logged rather than assumed.
    """
    idx = list(range(0, n_ref, stride))
    members, masks = [], []
    for i in idx:
        row, ok = [], []
        for v in plans_fwd:
            j = int(plans_fwd[v][i])
            back = int(plans_bwd[v][j])
            row.append(j)
            ok.append(abs(back - i) <= cycle_tol)
        if sum(ok) + 1 >= min_members and (sum(ok) / max(len(ok), 1)) >= min_consistent_frac:
            members.append(row)
            masks.append(ok)
    return idx, members, masks


# --------------------------------------------------------------------------
# the objectives
# --------------------------------------------------------------------------

def ray_alignment_loss(group_points, group_mask, centre, eps=1e-4):
    """Goal (1): members of one group share a ray from the action centre.

    Mean over groups of the transport-weighted (1 - u_a . u_b) over member pairs.
    Written as a pairwise term rather than a distance-to-line so that it cannot
    be reduced by shrinking the radius.
    """
    u, r = directions(group_points, centre.unsqueeze(0).unsqueeze(0), eps)
    m = group_mask.float()
    sim = torch.einsum('gvd,gwd->gvw', u, u)
    pair = m.unsqueeze(2) * m.unsqueeze(1)
    pair = pair * (1.0 - torch.eye(pair.shape[-1], device=pair.device)).unsqueeze(0)
    denom = pair.sum().clamp_min(1.0)
    return ((1.0 - sim) * pair).sum() / denom, u, r


def group_separation_loss(group_dirs, group_mask, temperature=0.2, margin=0.3):
    """Structural companion to goal (1): different groups of the SAME action must
    not share a direction, or (1) collapses every frame onto one ray.

    InfoNCE over group mean directions: the positives are a group's own members,
    the negatives are the other groups of the same action. A hinge on the
    between-group similarity is added because InfoNCE alone tolerates a
    near-degenerate configuration when the group count is small.
    """
    m = group_mask.float().unsqueeze(-1)
    mean_dir = F.normalize((group_dirs * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0), dim=-1)
    g = mean_dir.shape[0]
    if g < 2:
        z = group_dirs.sum() * 0.0
        return z, z, mean_dir
    pos = torch.einsum('gvd,gd->gv', group_dirs, mean_dir)
    pos = (pos * group_mask.float()).sum(dim=1) / group_mask.float().sum(dim=1).clamp_min(1.0)
    between = mean_dir @ mean_dir.T
    off = ~torch.eye(g, dtype=torch.bool, device=mean_dir.device)
    logits = torch.cat([(pos / temperature).unsqueeze(1),
                        (between[off].view(g, g - 1) / temperature)], dim=1)
    nce = F.cross_entropy(logits, torch.zeros(g, dtype=torch.long, device=mean_dir.device))
    hinge = F.relu(between[off] - (1.0 - margin)).pow(2).mean()
    return nce, hinge, mean_dir


def centre_separation_loss(centres, margin=0.5):
    """Keep the free geometric centres from merging. G00 measured that no
    collapse happens with the current codebook, so this is a guard on the NEW
    parameter, not a fix for an existing problem."""
    k = centres.shape[0]
    if k < 2:
        return centres.sum() * 0.0
    dist = torch.cdist(centres.unsqueeze(0), centres.unsqueeze(0))[0]
    off = ~torch.eye(k, dtype=torch.bool, device=centres.device)
    return F.relu(margin - dist[off]).pow(2).mean()


def radius_regularisation_loss(radius, target=1.0, mask=None):
    """Keep ||z - c_k|| near a target. This is the guard that blocks centre
    escape: without it the optimiser satisfies the ray loss by moving the centre
    far away, where all directions agree and carry nothing (see the module
    docstring for the measurement). `target` should be the action's own measured
    mean radius rather than a global constant when that is available."""
    err = (radius - target) ** 2
    if mask is not None:
        m = mask.float()
        return (err * m).sum() / m.sum().clamp_min(1.0)
    return err.mean()


@torch.no_grad()
def concurrency_centre_update(centres, k, group_points, group_mask, weight,
                              momentum=0.9, min_conditioning=1e-2, max_step=0.2,
                              anchor=None, max_anchor_offset=0.5):
    """Goal (2): re-estimate the action centre as the concurrency point of the
    group lines, in closed form, then move the centre part of the way there.

    This is a Lloyd-style step rather than a gradient, because G00 measured that
    the gradient-trained prototypes stay at cosine 0.872 from their own random
    initialisation: SGD does not move this parameter. A01 then showed the
    initialisation materially changes the outcome, so the centre is a real lever
    that gradient descent is failing to use.

    Three guards, each reported in the diagnostics: the step is skipped when the
    lines are too nearly parallel for an intersection to be defined; the step
    length is capped; and the new centre is kept within `max_anchor_offset` of
    `anchor` (the action's frame centroid). The last one is not cosmetic -- with
    an unconstrained centre this update is exactly the degeneracy measured in the
    module docstring, where the centre escapes the cloud and the ray loss becomes
    vacuous.
    """
    out = {"updated": 0.0, "n_lines": float(group_points.shape[0])}
    if group_points.shape[0] < 3:
        out["reason"] = "too_few_lines"
        return out
    mu, d, var_frac = best_fit_lines(group_points, group_mask)
    out["line_variance_fraction"] = float(var_frac.mean())
    point, residual, cond = least_squares_common_point(mu, d, weight)
    out["common_point_residual"] = float(residual)
    out["conditioning"] = float(cond)
    if not torch.isfinite(point).all() or float(cond) < min_conditioning:
        out["reason"] = "ill_conditioned"
        return out
    step = point - centres[k]
    norm = step.norm()
    if float(norm) > max_step:
        step = step * (max_step / norm.clamp_min(1e-12))
    candidate = centres[k] + (1.0 - momentum) * step
    if anchor is not None:
        offset = candidate - anchor
        off_norm = offset.norm()
        out["anchor_offset"] = float(off_norm)
        if float(off_norm) > max_anchor_offset:
            candidate = anchor + offset * (max_anchor_offset / off_norm.clamp_min(1e-12))
            out["clamped_to_anchor"] = 1.0
    out["centre_displacement"] = float((candidate - centres[k]).norm())
    centres[k] = candidate
    out["updated"] = 1.0
    return out

# --------------------------------------------------------------------------
# sphere variant: frames of an action live on a sphere around its centre
# --------------------------------------------------------------------------
#
# Instead of "corresponding frames share a ray from the centre, radius free",
# this says "frames of action k lie at radius rho_k from c_k, and corresponding
# frames share an angular position". Two reasons it is the better-supported of
# the two, both measured:
#
#   * It blocks the centre-escape degeneracy STRUCTURALLY. The escape works by
#     letting the radius grow while the centre leaves the cloud; pinning the
#     radius was the only thing that stopped it (module docstring). Here the
#     radius constraint is the hypothesis, not a guard bolted on.
#   * A06 measured the between-video std of the mean radius at 0.059 against a
#     within-group radius std of 0.144. The radius varies LESS between videos
#     than inside one, so "radius encodes video identity" (the ray model's
#     reading, and G02's original `r_{v,k}`) is not supported, while "radius is
#     roughly common per action" is cheap to impose.
#
# The similarity metric changes with the hypothesis: on a sphere the natural
# frame-to-frame similarity is angular (u_i . u_j), and the natural
# frame-to-action cost is distance to the shell, |‖z - c_k‖ - rho_k|.


def shell_cost(z, centres, radii, eps=1e-4):
    """Frame-to-action cost under the sphere hypothesis: distance to the shell.

    z: (B, T, D), centres: (K, D), radii: (K,). Returns (B, T, K).
    This is the replacement for `1 - z a_k^T` that the hypothesis implies.
    """
    d = (z.unsqueeze(2) - centres.view(1, 1, *centres.shape)).norm(dim=-1)
    return (d - radii.view(1, 1, -1)).abs().clamp_min(0.0) + eps * 0.0


def sphere_radius_loss(z, centres, radii, posterior, eps=1e-4):
    """The sphere constraint itself: each frame sits at its action's radius,
    weighted by the (detached) action posterior. This is what makes the centre
    escape impossible rather than merely penalised."""
    d = (z.unsqueeze(2) - centres.view(1, 1, *centres.shape)).norm(dim=-1)
    err = (d - radii.view(1, 1, -1)) ** 2
    w = posterior
    return (err * w).sum() / w.sum().clamp_min(1e-6)


@torch.no_grad()
def update_radii(radii, z, centres, posterior, momentum=0.9):
    """EMA of the posterior-weighted mean radius per action. Learning rho_k by
    gradient competes with the loss that uses it; an EMA of the observed radius
    is both stabler and interpretable, and it keeps rho_k inside the data."""
    d = (z.unsqueeze(2) - centres.view(1, 1, *centres.shape)).norm(dim=-1)
    w = posterior
    num = (d * w).sum(dim=(0, 1))
    den = w.sum(dim=(0, 1)).clamp_min(1e-6)
    obs = num / den
    seen = den > 1e-3
    radii[seen] = momentum * radii[seen] + (1.0 - momentum) * obs[seen]
    return {"radius_mean": float(radii.mean()), "radius_min": float(radii.min()),
            "actions_seen": float(seen.float().mean())}


def sphere_alignment_loss(group_points, group_mask, centre, radius, eps=1e-4,
                          radius_weight=1.0):
    """Goal (1) under the sphere hypothesis: members of a correspondence group
    share an angular position AND sit at the action radius.

    Returns (loss, u, r) so the caller can log the geometry it is imposing.
    """
    ang, u, r = ray_alignment_loss(group_points, group_mask, centre, eps)
    m = group_mask.float()
    rad = (((r - radius) ** 2) * m).sum() / m.sum().clamp_min(1.0)
    return ang + radius_weight * rad, u, r

