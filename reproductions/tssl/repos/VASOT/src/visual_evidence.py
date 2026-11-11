"""Visual affinity preservation and conservative cross-video label transfer.

Raw features are the fixed, pre-extracted visual evidence. No frame index,
temporal prior, action label, or GT boundary enters either function.
"""

import torch
import torch.nn.functional as F


def _valid_views(raw, embedding, mask):
    if raw.ndim != 3 or embedding.ndim != 3 or raw.shape[0] != 1:
        raise ValueError("visual evidence objectives currently require batch size 1")
    indices = torch.where(mask[0].bool())[0]
    if len(indices) == 0:
        return raw.new_zeros((0, raw.shape[-1])), embedding.new_zeros((0, embedding.shape[-1])), indices
    visual = raw[0, indices].detach()
    visual = visual - visual.mean(dim=0, keepdim=True)
    latent = embedding[0, indices]
    latent = latent - latent.mean(dim=0, keepdim=True)
    return F.normalize(visual, dim=-1), F.normalize(latent, dim=-1), indices


def _weighted_kl(teacher_sim, student_sim, teacher_temp, student_temp, min_ratio):
    if teacher_sim.shape[1] < 2 or teacher_sim.shape[0] == 0:
        return student_sim.sum() * 0
    teacher = torch.softmax(teacher_sim / teacher_temp, dim=-1)
    log_student = torch.log_softmax(student_sim / student_temp, dim=-1)
    row_kl = (teacher * (teacher.clamp_min(1e-12).log() - log_student)).sum(-1)
    confidence = (teacher.max(dim=-1).values - min_ratio / teacher.shape[1]).clamp_min(0)
    if confidence.sum() <= 1e-8:
        return student_sim.sum() * 0
    return (row_kl * confidence / confidence.sum()).sum()


def visual_affinity_loss(raw_x, raw_y, z_x, z_y, mask_x, mask_y,
                         stride=4, teacher_temp=0.07, student_temp=0.1,
                         min_ratio=1.5):
    """Preserve confident raw cross-video visual neighbours in the embedding.

    The stop-gradient teacher is a distribution over frames in the other video;
    its confidence weights suppress rows with no distinct visual neighbour.
    Both directions contribute, and padding is removed before subsampling.
    """
    if stride < 1 or teacher_temp <= 0 or student_temp <= 0 or min_ratio < 1:
        raise ValueError("invalid visual affinity hyperparameters")
    rx, zx, _ = _valid_views(raw_x, z_x, mask_x)
    ry, zy, _ = _valid_views(raw_y, z_y, mask_y)
    rx, zx = rx[::stride], zx[::stride]
    ry, zy = ry[::stride], zy[::stride]
    if min(len(rx), len(ry)) < 2:
        return (z_x.sum() + z_y.sum()) * 0
    teacher_sim = rx @ ry.T
    student_sim = zx @ zy.T
    return 0.5 * (
        _weighted_kl(teacher_sim, student_sim, teacher_temp, student_temp, min_ratio)
        + _weighted_kl(teacher_sim.T, student_sim.T, teacher_temp, student_temp, min_ratio)
    )


@torch.no_grad()
def consistent_frame_pairs(raw_x, raw_y, raw_z, z_x, z_y, z_z,
                           mask_x, mask_y, mask_z, raw_topk=8, cycle_tolerance=4,
                           z_raw_support="either"):
    """Find X↔Y mutual visual matches confirmed through a third video Z.

    A learned-embedding match must be reciprocal, appear in the raw visual
    top-k list in both directions, and reach approximately the same Z frame
    by X→Z and X→Y→Z. At least one Z leg must have raw top-k support by
    default; ``both`` is stricter and ``none`` is an audit ablation.
    Returns indices in the original padded X/Y tensors and coverage stats.
    """
    if raw_topk < 1 or cycle_tolerance < 0:
        raise ValueError("invalid match hyperparameters")
    if z_raw_support not in ("both", "either", "none"):
        raise ValueError("z_raw_support must be both, either, or none")
    rx, zx, ix = _valid_views(raw_x, z_x, mask_x)
    ry, zy, iy = _valid_views(raw_y, z_y, mask_y)
    rz, zz, _ = _valid_views(raw_z, z_z, mask_z)
    empty = ix[:0], iy[:0], {"mutual": 0, "raw_supported": 0,
                             "cycle_supported": 0, "coverage_x": 0.0}
    if min(len(ix), len(iy), len(rz)) < 2:
        return empty

    sxy, sxz, syz = zx @ zy.T, zx @ zz.T, zy @ zz.T
    rxy, rxz, ryz = rx @ ry.T, rx @ rz.T, ry @ rz.T
    j = sxy.argmax(dim=1)
    i_back = sxy.argmax(dim=0)
    src = torch.arange(len(ix), device=ix.device)
    mutual = i_back[j] == src

    def in_topk(sim, row, col):
        return (sim.topk(min(raw_topk, sim.shape[1]), dim=1).indices[row] == col[:, None]).any(dim=1)

    supported = mutual & in_topk(rxy, src, j) & in_topk(rxy.T, j, src)
    k_xz = sxz.argmax(dim=1)
    k_yz = syz.argmax(dim=1)
    cycle = ((k_xz - k_yz[j]).abs() <= cycle_tolerance)
    if z_raw_support != "none":
        xz_supported = in_topk(rxz, src, k_xz)
        yz_supported = in_topk(ryz, j, k_yz[j])
        cycle &= (xz_supported & yz_supported) if z_raw_support == "both" else (
            xz_supported | yz_supported)
    keep = supported & cycle
    src = src[keep]
    target = j[keep]
    stats = {"mutual": int(mutual.sum()), "raw_supported": int(supported.sum()),
             "cycle_supported": int(keep.sum()), "coverage_x": float(keep.float().mean())}
    return ix[src], iy[target], stats


@torch.no_grad()
def refine_action_pseudolabels(q_x, q_y, indices_x, indices_y, strength=0.5):
    """Symmetrically soften OT action targets at confirmed frame matches.

    Original row masses are preserved so the existing CE scale is unchanged.
    Unmatched rows are bit-identical to the input targets.
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    if len(indices_x) == 0 or strength == 0:
        return q_x, q_y
    px = q_x[0, indices_x]
    py = q_y[0, indices_y]
    mass_x = px.sum(-1, keepdim=True)
    mass_y = py.sum(-1, keepdim=True)
    nx = px / mass_x.clamp_min(1e-12)
    ny = py / mass_y.clamp_min(1e-12)
    consensus = 0.5 * (nx + ny)
    out_x, out_y = q_x.clone(), q_y.clone()
    out_x[0, indices_x] = ((1 - strength) * nx + strength * consensus) * mass_x
    out_y[0, indices_y] = ((1 - strength) * ny + strength * consensus) * mass_y
    return out_x, out_y
