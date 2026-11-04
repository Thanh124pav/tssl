"""Dense-matrix restatement of the VASOT alignment solver.

`alignment_ot.align_ot` hard-codes the structure matrices as conv1d filters
over frame indices. To swap in a different C^X / C^Y without changing the
optimiser, this module reimplements the same mirror-descent loop with explicit
matrices. `index_structure()` rebuilds exactly the matrices the conv filters
represent, and `check_equivalence()` verifies that the dense loop reproduces
the public solver on those matrices, so any difference in a later experiment is
attributable to the structure and not to the solver.
"""

import torch

import diag_common  # noqa: F401  (puts the VASOT src on sys.path)
import alignment_ot


def index_structure(n, m, radius, device):
    """The C^X, C^Y the public conv filters implement."""
    rx = int(n * radius)
    ry = int(m * radius)
    ix = torch.arange(n, device=device)
    iy = torch.arange(m, device=device)
    dx = (ix[:, None] - ix[None, :]).abs()
    dy = (iy[:, None] - iy[None, :]).abs()
    Cx = torch.where((dx >= 1) & (dx <= rx), torch.full_like(dx, 1.0, dtype=torch.float),
                     torch.zeros_like(dx, dtype=torch.float)) / radius
    Cy = torch.where((dy >= 1) & (dy <= ry), torch.zeros_like(dy, dtype=torch.float),
                     torch.ones_like(dy, dtype=torch.float))
    return Cx, Cy


def _grad_fgw(T, cost, alpha, Cx, Cy):
    return alpha * (Cx @ T @ Cy) + (1.0 - alpha) * cost


def _objective(T, cost, Cx, Cy, eps, alpha, ub_frames, ub_actions,
               lambda_frames, lambda_actions, mask_X, mask_Y):
    dev = cost.device
    B, N, K = cost.shape
    nnz_X = mask_X.sum(dim=1)
    nnz_Y = mask_Y.sum(dim=1)
    T_mask = T * mask_X.unsqueeze(2) * mask_Y.unsqueeze(1)
    fgw_obj = (_grad_fgw(T_mask, cost, alpha, Cx, Cy) * T_mask).sum(dim=(1, 2))
    dy = torch.ones((B, K), device=dev) / nnz_Y[:, None]
    dx = torch.ones((B, N), device=dev) / nnz_X[:, None]
    ub = torch.zeros(B, device=dev)
    if ub_frames:
        ub += alignment_ot.kld(T_mask.sum(dim=2), dx) * lambda_frames
    if ub_actions:
        ub += alignment_ot.kld(T_mask.sum(dim=1), dy) * lambda_actions
    entr = -eps * alignment_ot.entropy(T)
    return 0.5 * fgw_obj + ub + entr


def align_ot_dense(cost_matrix, Cx, Cy, mask_X=None, mask_Y=None,
                   eps=0.07, alpha=0.3, ub_frames=False, ub_actions=True,
                   lambda_frames=0.1, lambda_actions=0.05,
                   n_iters=(25, 1), stable_thres=7., step_size=None):
    """Same mirror-descent loop as alignment_ot.align_ot, dense structure."""
    dev = cost_matrix.device
    B, N, K = cost_matrix.shape
    if mask_X is None:
        mask_X = torch.full((B, N - 1), 1, dtype=bool, device=dev)
    if mask_Y is None:
        mask_Y = torch.full((B, K - 1), 1, dtype=bool, device=dev)
    mask_X = torch.cat((torch.ones(1, 1, dtype=torch.bool, device=dev), mask_X), dim=1)
    mask_Y = torch.cat((torch.ones(1, 1, dtype=torch.bool, device=dev), mask_Y), dim=1)
    nnz_X = mask_X.sum(dim=1)
    nnz_Y = mask_Y.sum(dim=1)
    dy = torch.ones((B, K, 1), device=dev) / nnz_Y[:, None, None]
    dx = torch.ones((B, N, 1), device=dev) / nnz_X[:, None, None]
    T = dx * dy.transpose(1, 2)
    T = T * mask_X.unsqueeze(2) * mask_Y.unsqueeze(1)

    obj_trace = []
    it = 0
    while True:
        with torch.no_grad():
            obj_trace.append(_objective(T, cost_matrix, Cx, Cy, eps, alpha, ub_frames,
                                        ub_actions, lambda_frames, lambda_actions,
                                        mask_X, mask_Y))
        if it >= n_iters[0]:
            break
        fgw_cost = _grad_fgw(T, cost_matrix, alpha, Cx, Cy)
        grad = fgw_cost - alignment_ot.grad_entropy(T, eps)
        if ub_frames:
            grad = grad + alignment_ot.grad_kld(T, dx, lambda_frames, 2)
        if ub_actions:
            grad = grad + alignment_ot.grad_kld(T, dy.transpose(1, 2), lambda_actions, 1)
        if it == 0 and step_size is None:
            step_size = 4. / grad.max().item()
        T = T * torch.exp(-step_size * grad)
        if not ub_frames and not ub_actions:
            T = alignment_ot.project_to_polytope_KL(fgw_cost, mask_X, mask_Y, eps, dx, dy,
                                                    n_iters=n_iters[1], stable_thres=stable_thres)
        elif not ub_frames:
            T = torch.nan_to_num(T / T.sum(dim=2, keepdim=True), posinf=0., neginf=0.) * dx
        elif not ub_actions:
            T = torch.nan_to_num(T / T.sum(dim=1, keepdim=True), posinf=0., neginf=0.) * dy.transpose(1, 2)
        it += 1
    T = T * nnz_X[:, None, None]
    return T[:, 1:, 1:], torch.cat(obj_trace)


def check_equivalence(n=64, m=64, radius=0.04, seed=0, device="cpu", **kwargs):
    """Max abs difference between the public conv solver and this dense loop."""
    g = torch.Generator().manual_seed(seed)
    cost = torch.rand(1, n + 1, m + 1, generator=g).to(device)
    mask_X = torch.ones(1, n, dtype=torch.bool, device=device)
    mask_Y = torch.ones(1, m, dtype=torch.bool, device=device)
    params = dict(eps=0.07, alpha=0.3, ub_frames=False, ub_actions=True,
                  lambda_frames=0.1, lambda_actions=0.05, n_iters=(25, 1))
    params.update(kwargs)
    ref, ref_trace = alignment_ot.align_ot(cost, mask_X, mask_Y, radius=radius, **params)
    Cx, Cy = index_structure(n + 1, m + 1, radius, device)
    got, got_trace = align_ot_dense(cost, Cx, Cy, mask_X, mask_Y, **params)
    return {"max_abs_plan_diff": float((ref - got).abs().max()),
            "max_rel_plan_diff": float(((ref - got).abs() / ref.abs().clamp_min(1e-12)).max()),
            "max_abs_objective_diff": float((ref_trace - got_trace).abs().max())}


if __name__ == "__main__":
    import json
    print(json.dumps(check_equivalence(), indent=2))
