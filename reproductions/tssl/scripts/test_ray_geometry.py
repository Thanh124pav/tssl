"""Invariants for the ray-geometry training objectives.

Each test fixes a synthetic configuration whose answer is known, so the losses
are shown to impose the intended geometry rather than asserted to. Test 4 is the
important one: it demonstrates the degeneracy of goal (1) taken alone, which is
the reason the group-separation term is structural.

    python reproductions/tssl/scripts/test_ray_geometry.py
"""

import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "repos" / "VASOT" / "src"
sys.path.insert(0, str(SRC))

import ray_geometry as rg  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        RESULTS.append((True, name, fn()))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((False, name, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def make_ray_groups(n_groups=6, n_members=5, d=40, noise=0.0, seed=0, shared_dir=False):
    """Groups that lie exactly on rays from a known centre."""
    g = torch.Generator().manual_seed(seed)
    centre = torch.randn(d, generator=g) * 0.3
    base = F.normalize(torch.randn(d, generator=g), dim=0)
    pts, mask = [], []
    for gi in range(n_groups):
        dirn = base if shared_dir else F.normalize(torch.randn(d, generator=g), dim=0)
        radii = torch.rand(n_members, generator=g) * 0.8 + 0.6
        p = centre.unsqueeze(0) + radii.unsqueeze(1) * dirn.unsqueeze(0)
        if noise:
            p = p + torch.randn(p.shape, generator=g) * noise
        pts.append(p)
        mask.append(torch.ones(n_members, dtype=torch.bool))
    return torch.stack(pts), torch.stack(mask), centre


def t_ray_loss_zero_on_rays():
    pts, mask, centre = make_ray_groups(noise=0.0)
    loss, u, r = rg.ray_alignment_loss(pts, mask, centre)
    assert float(loss) < 1e-5, f"exact rays should give 0, got {float(loss)}"
    pts_n, mask_n, _ = make_ray_groups(noise=0.25, seed=3)
    loss_n, _, _ = rg.ray_alignment_loss(pts_n, mask_n, centre)
    assert float(loss_n) > 10 * max(float(loss), 1e-9), "noisy groups must cost more"
    return f"exact rays {float(loss):.2e}; perturbed groups {float(loss_n):.4f}"


def t_ray_loss_optimises_to_rays():
    """Gradient descent on the loss must actually make members collinear."""
    torch.manual_seed(0)
    d = 40
    centre = torch.zeros(d)
    pts = (F.normalize(torch.randn(6, 5, d), dim=-1) * 1.2).requires_grad_(True)
    mask = torch.ones(6, 5, dtype=torch.bool)
    opt = torch.optim.Adam([pts], lr=0.05)
    before = float(rg.ray_alignment_loss(pts, mask, centre)[0])
    for _ in range(300):
        opt.zero_grad()
        loss, _, _ = rg.ray_alignment_loss(pts, mask, centre)
        loss.backward()
        opt.step()
    after = float(rg.ray_alignment_loss(pts, mask, centre)[0])
    u, _ = rg.directions(pts.detach(), centre.view(1, 1, -1))
    within = torch.einsum('gvd,gwd->gvw', u, u)
    off = ~torch.eye(5, dtype=torch.bool).unsqueeze(0).expand(6, -1, -1)
    assert after < 0.02, f"loss did not converge: {before:.4f} -> {after:.4f}"
    assert float(within[off].mean()) > 0.98, "members did not become collinear"
    return f"loss {before:.4f} -> {after:.4f}; within-group direction cosine {float(within[off].mean()):.4f}"


def t_separation_loss_discriminates():
    distinct, mask, centre = make_ray_groups(shared_dir=False, seed=1)
    shared, _, _ = make_ray_groups(shared_dir=True, seed=1)
    u_d, _ = rg.directions(distinct, centre.view(1, 1, -1))
    u_s, _ = rg.directions(shared, centre.view(1, 1, -1))
    nce_d, hinge_d, _ = rg.group_separation_loss(u_d, mask)
    nce_s, hinge_s, _ = rg.group_separation_loss(u_s, mask)
    assert float(nce_s) > float(nce_d), f"shared directions must cost more: {float(nce_s)} vs {float(nce_d)}"
    assert float(hinge_s) > 10 * float(hinge_d) + 1e-6, "hinge must fire on shared directions"
    return (f"distinct groups nce={float(nce_d):.4f} hinge={float(hinge_d):.4f}; "
            f"shared nce={float(nce_s):.4f} hinge={float(hinge_s):.4f}")


def t_centre_escape_is_the_degeneracy():
    """THE design-critical test, and it corrected the design.

    The first version of this test asserted that goal (1) alone collapses every
    frame onto one ray. That is false: the ray loss is separable across groups,
    so each group aligns to whichever ray is nearest its initialisation and the
    between-group cosine stays at 0.03. The real degeneracy appears as soon as
    goal (2) lets the centre move: the optimiser pushes the centre out of the
    cloud, where every direction converges on the direction from the centre to
    the cloud, so the loss reaches zero while carrying no information. Pinning
    the radius blocks it.
    """
    def run(free_centre, pin_radius, steps=600, d=40, n_groups=8):
        torch.manual_seed(1)
        pts = (F.normalize(torch.randn(n_groups, 5, d), dim=-1) * 1.2).requires_grad_(True)
        mask = torch.ones(n_groups, 5, dtype=torch.bool)
        centre = torch.zeros(d, requires_grad=free_centre)
        opt = torch.optim.Adam([pts] + ([centre] if free_centre else []), lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            loss, u, r = rg.ray_alignment_loss(pts, mask, centre)
            if pin_radius:
                loss = loss + rg.radius_regularisation_loss(r, 1.0)
            loss.backward()
            opt.step()
        loss, u, r = rg.ray_alignment_loss(pts, mask, centre)
        md = F.normalize(u.mean(dim=1), dim=-1)
        bet = md @ md.T
        off = ~torch.eye(n_groups, dtype=torch.bool)
        return {"loss": float(loss), "centre_norm": float(centre.norm()),
                "radius": float(r.mean()), "between": float(bet[off].mean())}

    fixed = run(False, False)
    free = run(True, False)
    pinned = run(True, True)
    # goal (1) alone does NOT collapse the groups
    assert fixed["between"] < 0.2, f"ray loss alone unexpectedly coupled the groups: {fixed}"
    # goal (2) without a radius guard does
    assert free["between"] > 0.7, f"centre escape did not reproduce: {free}"
    assert free["centre_norm"] > 2.0, f"the centre did not escape: {free}"
    # and the radius regulariser blocks it
    assert pinned["between"] < 0.2, f"radius pin failed to block the escape: {pinned}"
    assert pinned["centre_norm"] < 1.0, f"centre still escaped with the pin: {pinned}"
    assert abs(pinned["radius"] - 1.0) < 0.05, f"radius not held at the target: {pinned}"
    return ("centre fixed: between %.3f |c| %.2f | centre FREE: between %.3f |c| %.2f r %.1f "
            "| free+radius pin: between %.3f |c| %.2f r %.3f") % (
        fixed["between"], fixed["centre_norm"], free["between"], free["centre_norm"],
        free["radius"], pinned["between"], pinned["centre_norm"], pinned["radius"])


def t_centre_update_respects_the_anchor():
    """The closed-form centre update must stay near the action's frame centroid,
    which is the containment form of the guard above."""
    torch.manual_seed(7)
    d = 40
    pts, mask, _ = make_ray_groups(n_groups=6, n_members=5, d=d, noise=0.02, seed=11)
    anchor = pts.mean(dim=(0, 1))
    centres = torch.zeros(2, d)
    far = anchor + F.normalize(torch.randn(d), dim=0) * 10.0
    info = rg.concurrency_centre_update(centres, 0, pts + far.view(1, 1, -1), mask,
                                        torch.ones(pts.shape[0]), momentum=0.0,
                                        anchor=anchor, max_anchor_offset=0.5)
    off = float((centres[0] - anchor).norm())
    assert info.get("clamped_to_anchor") == 1.0, f"the anchor clamp did not fire: {info}"
    assert off <= 0.5 + 1e-4, f"centre escaped the anchor by {off}"
    return f"clamp fired, centre held {off:.4f} from the anchor (limit 0.5)"


def t_common_point_recovers_centre():
    torch.manual_seed(2)
    d = 12
    centre = torch.randn(d) * 0.4
    dirs = F.normalize(torch.randn(7, d), dim=-1)
    mu = centre.unsqueeze(0) + torch.rand(7, 1) * dirs  # points on lines through centre
    w = torch.ones(7)
    pt, res, cond = rg.least_squares_common_point(mu, dirs, w)
    err = float((pt - centre).norm())
    assert err < 1e-3, f"common point missed the true centre by {err}"
    assert float(res) < 1e-3, f"residual should vanish, got {float(res)}"
    # near-parallel lines must be reported as ill-conditioned
    par = F.normalize(dirs[:1].expand(7, -1) + torch.randn(7, d) * 0.01, dim=-1)
    mu2 = centre.unsqueeze(0) + torch.rand(7, 1) * par
    _, _, cond2 = rg.least_squares_common_point(mu2, par, w)
    assert float(cond2) < float(cond) / 10, f"parallel lines not flagged: {float(cond2)} vs {float(cond)}"
    return (f"centre recovered to {err:.2e}, residual {float(res):.2e}, "
            f"conditioning {float(cond):.3f} vs parallel {float(cond2):.4f}")


def t_centre_update_moves_and_guards():
    torch.manual_seed(3)
    d = 40
    true_centre = torch.randn(d) * 0.3
    pts, mask, _ = make_ray_groups(n_groups=6, n_members=5, d=d, noise=0.02, seed=5)
    pts = pts - pts.mean(dim=(0, 1)) + true_centre  # recentre the synthetic rays
    centres = torch.zeros(3, d)
    start = float((centres[1] - true_centre).norm())
    info = None
    for _ in range(40):
        info = rg.concurrency_centre_update(centres, 1, pts, mask, torch.ones(pts.shape[0]),
                                            momentum=0.7)
    end = float((centres[1] - true_centre).norm())
    assert info["updated"] == 1.0, f"update was skipped: {info}"
    assert end < start, f"centre moved away from the truth: {start:.4f} -> {end:.4f}"
    # the guard must fire on parallel lines
    flat = torch.stack([pts[0] + 0.001 * i for i in range(6)])
    guard = rg.concurrency_centre_update(centres.clone(), 1, flat,
                                         torch.ones(6, 5, dtype=torch.bool), torch.ones(6))
    return (f"distance to the true centre {start:.4f} -> {end:.4f}; "
            f"parallel-line guard: updated={guard['updated']} reason={guard.get('reason')}")


def t_smooth_drift():
    torch.manual_seed(4)
    z = F.normalize(torch.randn(1, 128, 16), dim=-1)
    s = rg.smooth_drift(z, 15)
    assert s.shape == z.shape, f"shape changed: {s.shape}"
    lag1_raw = (z[0, 1:] * z[0, :-1]).sum(-1).mean()
    lag1_sm = (s[0, 1:] * s[0, :-1]).sum(-1).mean()
    assert float(lag1_sm) > float(lag1_raw) + 0.3, "smoothing did not raise autocorrelation"
    assert float(rg.smooth_drift(z, 1).sub(z).abs().max()) == 0.0, "window 1 must be identity"
    return f"lag-1 cosine {float(lag1_raw):.3f} -> {float(lag1_sm):.3f}; window 1 is the identity"


def t_group_builder_filters_cycle_breaks():
    n = 32
    good = {1: list(range(n)), 2: list(range(n))}
    back_good = {1: list(range(n)), 2: list(range(n))}
    idx, mem, mask = rg.build_groups(good, back_good, n, stride=4, cycle_tol=2)
    assert len(mem) == len(idx) and all(all(m) for m in mask), "consistent plans must all pass"
    bad_back = {1: [(j + 20) % n for j in range(n)], 2: list(range(n))}
    idx2, mem2, mask2 = rg.build_groups(good, bad_back, n, stride=4, cycle_tol=2,
                                        min_consistent_frac=0.9)
    assert len(mem2) == 0, f"cycle-inconsistent members were not filtered: {len(mem2)} kept"
    idx3, mem3, mask3 = rg.build_groups(good, bad_back, n, stride=4, cycle_tol=2,
                                        min_consistent_frac=0.4)
    assert len(mem3) == len(idx3) and all(m[0] is False for m in mask3), "mask must mark the break"
    return (f"consistent: {len(mem)}/{len(idx)} groups kept; one video broken: "
            f"{len(mem2)} kept at frac>=0.9, {len(mem3)} kept at frac>=0.4 with the member masked")


def t_sphere_blocks_escape_without_a_hand_set_target():
    """The design claim for the sphere variant: with the radius as part of the
    HYPOTHESIS (and rho learned by EMA from the data, not hand-set), the centre
    cannot escape. Compare against the ray loss with a free centre, which does
    escape."""
    def run(mode, steps=600, d=40, n_groups=8):
        torch.manual_seed(1)
        pts = (F.normalize(torch.randn(n_groups, 5, d), dim=-1) * 1.2).requires_grad_(True)
        mask = torch.ones(n_groups, 5, dtype=torch.bool)
        centre = torch.zeros(d, requires_grad=True)
        radius = torch.tensor(1.2)
        opt = torch.optim.Adam([pts, centre], lr=0.05)
        for step in range(steps):
            opt.zero_grad()
            if mode == "ray":
                loss, u, r = rg.ray_alignment_loss(pts, mask, centre)
            else:
                loss, u, r = rg.sphere_alignment_loss(pts, mask, centre, radius)
            loss.backward()
            opt.step()
            if mode == "sphere" and step % 20 == 0:
                with torch.no_grad():  # rho follows the data, as in training
                    _, _, rr = rg.ray_alignment_loss(pts, mask, centre)
                    radius = 0.9 * radius + 0.1 * rr.mean()
        loss, u, r = rg.ray_alignment_loss(pts, mask, centre)
        md = F.normalize(u.mean(dim=1), dim=-1)
        bet = md @ md.T
        off = ~torch.eye(n_groups, dtype=torch.bool)
        return {"centre_norm": float(centre.norm()), "radius": float(r.mean()),
                "between": float(bet[off].mean()),
                "within": float(torch.einsum('gvd,gwd->gvw', u, u)[
                    ~torch.eye(5, dtype=torch.bool).unsqueeze(0).expand(n_groups, -1, -1)].mean())}

    ray = run("ray")
    sph = run("sphere")
    assert ray["between"] > 0.7, f"the ray escape did not reproduce: {ray}"
    assert sph["between"] < 0.25, f"sphere variant still collapsed: {sph}"
    assert sph["centre_norm"] < 1.0, f"sphere variant let the centre escape: {sph}"
    assert sph["within"] > 0.9, f"sphere variant lost within-group alignment: {sph}"
    return ("ray + free centre: |c| %.2f between %.3f | sphere + free centre: "
            "|c| %.2f r %.2f between %.3f within %.3f") % (
        ray["centre_norm"], ray["between"], sph["centre_norm"], sph["radius"],
        sph["between"], sph["within"])


def t_shell_cost_and_radius_ema():
    """The metric the sphere hypothesis implies, its EMA, and the condition under
    which it can discriminate at all.

    Measured on the real checkpoint (A09 section on the sphere variant): the
    shell cost is a WORSE action cost than cosine (one-to-one 0.356 vs 0.411)
    because the distance from a frame to every centre is nearly the same -- the
    per-frame spread of ||z - c_k|| across k is 0.175 against a mean of 0.760,
    and the radii differ by std 0.038 around 0.424. So the test below asserts
    both halves: the cost is exact and does discriminate when the centres are
    well separated, and it stops discriminating as they close in. That is why
    the training integration ramps the shell cost in and adds centre-separation
    and radius-spread terms rather than swapping the metric at step 0.
    """
    torch.manual_seed(5)
    d, K, T = 16, 4, 60
    dirs = F.normalize(torch.randn(T, d), dim=-1)

    def setup(centre_scale):
        centres = F.normalize(torch.randn(K, d), dim=-1) * centre_scale
        radii = torch.full((K,), 1.0)
        z = (centres[1].unsqueeze(0) + 1.3 * dirs).unsqueeze(0)
        return centres, radii, z

    # well separated centres: the shell cost is exact and picks the right action
    centres, radii, z = setup(3.0)
    cost = rg.shell_cost(z, centres, radii)
    assert cost.shape == (1, T, K), f"bad shape {cost.shape}"
    assert abs(float(cost[0, :, 1].mean()) - 0.3) < 1e-3, float(cost[0, :, 1].mean())
    picked = float((cost[0].argmin(dim=-1) == 1).float().mean())
    assert picked > 0.99, f"separated centres: shell cost picked action 1 only {picked:.2f}"

    # centres as close as the real ones relative to the radius: it stops working
    centres_c, radii_c, z_c = setup(0.2)
    picked_close = float((rg.shell_cost(z_c, centres_c, radii_c)[0].argmin(dim=-1) == 1)
                         .float().mean())
    assert picked_close < 0.9, (f"the known failure mode did not reproduce: "
                                f"{picked_close:.2f}")

    # the EMA reaches the true radius and leaves unseen actions alone
    post = torch.zeros(1, T, K)
    post[0, :, 1] = 1.0
    for _ in range(80):
        rg.update_radii(radii, z, centres, post, momentum=0.8)
    assert abs(float(radii[1]) - 1.3) < 1e-2, f"EMA missed the radius: {float(radii[1])}"
    assert abs(float(radii[0]) - 1.0) < 1e-6, "unseen actions must not move"
    assert float(rg.shell_cost(z, centres, radii)[0, :, 1].mean()) < 1e-2, \
        "after the EMA the shell cost should vanish on the fitted action"
    return (f"separated centres: picks the action {picked:.2f} of the time; centres at "
            f"0.2: {picked_close:.2f}; rho_1 -> {float(radii[1]):.4f} (true 1.3)")


def t_windowed_encoder():
    """A13's windowed encoder must be exactly the old module at window 0, must
    stack the right frames with edge padding, and must handle both the (B, T, D)
    and the flattened (B*T, D) call conventions the training step uses."""
    import torch.nn as nn
    sys.path.insert(0, str(SRC))
    from train import WindowedMLP
    torch.manual_seed(0)
    T, D, H = 24, 6, 5
    x = torch.randn(1, T, D)

    # window 0 is the identity wrapper
    inner0 = nn.Linear(D, H)
    w0 = WindowedMLP(inner0, 0)
    assert torch.allclose(w0(x), inner0(x)), "window 0 changed the output"
    assert torch.allclose(w0(x[0]), inner0(x[0])), "window 0 broke the flat convention"

    # window w concatenates 2w+1 frames, edge-padded, in order
    w = 3
    inner = nn.Linear(D * (2 * w + 1), H)
    wm = WindowedMLP(inner, w)
    out = wm(x)
    assert out.shape == (1, T, H), f"bad shape {out.shape}"
    for t in (0, 1, T // 2, T - 1):
        idx = [min(max(t + o, 0), T - 1) for o in range(-w, w + 1)]
        manual = inner(torch.cat([x[0, i] for i in idx]))
        assert torch.allclose(out[0, t], manual, atol=1e-6), f"window content wrong at t={t}"
    # the flattened convention must agree with the batched one
    assert torch.allclose(wm(x[0]), out[0], atol=1e-6), "flat and batched disagree"
    # edge padding must repeat the edge frame, not wrap around
    idx0 = [0] * w + list(range(0, w + 1))
    assert torch.allclose(out[0, 0], inner(torch.cat([x[0, i] for i in idx0])), atol=1e-6), \
        "left edge is not edge-padded"
    return (f"window 0 is bit-identical; window {w} stacks {2*w+1} frames in order with "
            f"edge padding; flat and batched conventions agree")

def t_template_alignment_is_cycle_consistent():
    """A14's defining property: factoring the correspondence through a shared
    template makes composition EXACT, not approximate. A02 measured the
    published pairwise plan disagreeing with its own composition by 18.5
    frames; on the rows where the template defines a correspondence this must
    be zero by construction.

    Two things this test pins down, both found by it rather than assumed:
    exactness holds only for HARD slot assignments, and only on rows whose slot
    has a counterpart in the target video.
    """
    sys.path.insert(0, str(SRC))
    from train import VideoSSL
    torch.manual_seed(0)
    L = 12
    model = VideoSSL.__new__(VideoSSL)
    nX, nY, nZ = 60, 56, 64
    plans = {}
    for name, n in (('x', nX), ('y', nY), ('z', nZ)):
        raw = torch.rand(1, n, L) + 0.05
        plans[name] = raw / raw.sum(dim=-1, keepdim=True) / n

    P_xy, v_xy = VideoSSL.induced_alignment(model, plans['x'], plans['y'])
    P_yz, _ = VideoSSL.induced_alignment(model, plans['y'], plans['z'])
    P_xz, v_xz = VideoSSL.induced_alignment(model, plans['x'], plans['z'])
    assert torch.allclose(P_xy.sum(-1)[v_xy], torch.ones(int(v_xy.sum())), atol=1e-5)

    rows = (v_xy & v_xz)[0]
    err = float(((P_xy @ P_yz) - P_xz)[0][rows].abs().max())
    assert err < 1e-5, f"composition is not exact on defined rows: {err}"

    P_yx, _ = VideoSSL.induced_alignment(model, plans['y'], plans['x'])
    P_xx, v_xx = VideoSSL.induced_alignment(model, plans['x'], plans['x'])
    rt_rows = (v_xy & v_xx)[0]
    rt = float(((P_xy @ P_yx) - P_xx)[0][rt_rows].abs().max())
    assert rt < 1e-5, f"round trip is not exact: {rt}"

    # the soft variant is NOT exact: that is why hard is the default
    S_xy, _ = VideoSSL.induced_alignment(model, plans['x'], plans['y'], hard=False)
    S_yz, _ = VideoSSL.induced_alignment(model, plans['y'], plans['z'], hard=False)
    S_xz, _ = VideoSSL.induced_alignment(model, plans['x'], plans['z'], hard=False)
    soft_err = float(((S_xy @ S_yz) - S_xz).abs().max())
    assert soft_err > 100 * max(err, 1e-9), f"soft should not be exact, got {soft_err}"

    # an unfactored pairwise surrogate has no such property
    g = torch.Generator().manual_seed(1)
    def rand_plan(a, b):
        m = torch.rand(1, a, b, generator=g)
        return m / m.sum(-1, keepdim=True)
    base = float((rand_plan(nX, nY) @ rand_plan(nY, nZ) - rand_plan(nX, nZ)).abs().max())

    # empty slots must be reported invalid, not filled with an invented target
    sparse = torch.zeros(1, 8, L)
    sparse[0, :, 0] = 1.0                      # every Y frame in slot 0
    _, v_sparse = VideoSSL.induced_alignment(model, plans['x'], sparse)
    assert not bool(v_sparse.all()), "rows with no counterpart were not flagged invalid"
    frac = float(v_sparse.float().mean())

    return (f"hard: exact to {err:.1e} on defined rows, round trip {rt:.1e}; soft off by "
            f"{soft_err:.4f}; unfactored surrogate {base:.3f}; empty-slot guard leaves "
            f"{frac:.2f} of rows defined")

def main():
    for name, fn in [
        ("ray loss is zero exactly on rays", t_ray_loss_zero_on_rays),
        ("ray loss optimises to collinear groups", t_ray_loss_optimises_to_rays),
        ("separation loss discriminates", t_separation_loss_discriminates),
        ("centre escape is the real degeneracy", t_centre_escape_is_the_degeneracy),
        ("centre update respects the anchor", t_centre_update_respects_the_anchor),
        ("sphere blocks escape structurally", t_sphere_blocks_escape_without_a_hand_set_target),
        ("shell cost and radius EMA", t_shell_cost_and_radius_ema),
        ("A13 windowed encoder", t_windowed_encoder),
        ("A14 template alignment is cycle-consistent", t_template_alignment_is_cycle_consistent),
        ("common point recovers a known centre", t_common_point_recovers_centre),
        ("centre update moves and guards", t_centre_update_moves_and_guards),
        ("smooth drift raises autocorrelation", t_smooth_drift),
        ("group builder filters cycle breaks", t_group_builder_filters_cycle_breaks),
    ]:
        check(name, fn)
    w = max(len(n) for _, n, _ in RESULTS)
    failed = 0
    for ok, name, detail in RESULTS:
        if ok:
            print(f"PASS  {name:<{w}}  {detail}")
        else:
            failed += 1
            print(f"FAIL  {name:<{w}}\n{detail}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} invariants hold")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
