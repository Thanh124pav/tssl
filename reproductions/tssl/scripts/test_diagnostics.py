"""Invariant tests for the Phase G / Phase N diagnostic code.

Every check here is a property that must hold regardless of what the
experiments conclude, so a failing experiment still leaves trustworthy
measurement machinery behind. Run:

    python reproductions/tssl/scripts/test_diagnostics.py

Exit code 0 means all invariants hold; the report prints one line per check.
"""

import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import diag_common as dc
import fgw_dense
import alignment_ot

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail))
    except Exception as exc:  # noqa: BLE001
        RESULTS.append((False, name, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


# --------------------------------------------------------------------------
# warp machinery
# --------------------------------------------------------------------------

def t_warp_monotone():
    worst = None
    for sev in (0.0, 0.25, 0.5, 1.0, 2.0):
        for seed in range(5):
            src, meta = dc.monotone_warp(256, 5, sev, seed=seed)
            d = np.diff(src)
            assert (d >= -1e-9).all(), f"warp not monotone at sev={sev} seed={seed}"
            assert abs(src[0]) < 1e-6 and abs(src[-1] - 255) < 1e-6, "warp must span the video"
            worst = min(d.min(), worst) if worst is not None else d.min()
    return f"monotone over 25 warps, min step {worst:.2e}, endpoints exact"


def t_warp_identity():
    src, _ = dc.monotone_warp(128, 5, 0.0, seed=3)
    err = np.abs(src - np.arange(128)).max()
    assert err < 1e-6, f"severity 0 is not the identity (max err {err})"
    return f"severity 0 == identity, max err {err:.2e}"


def t_resample_identity():
    raw = torch.randn(64, 10)
    out = dc.resample_features(raw, np.arange(64, dtype=float))
    err = (raw - out).abs().max().item()
    assert err < 1e-6, f"resampling at integer positions changed the data ({err})"
    half = dc.resample_features(raw, np.arange(64, dtype=float) + 0.5)
    mid = (raw[:-1] + raw[1:]) / 2
    err2 = (half[:-1] - mid).abs().max().item()
    assert err2 < 1e-5, f"half-frame resample is not the midpoint ({err2})"
    return f"integer resample exact ({err:.2e}); half-frame == midpoint ({err2:.2e})"


def t_relative_phase():
    gt = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    ph = dc.relative_phase(gt)
    assert abs(ph[0]) < 1e-9 and abs(ph[2] - 1) < 1e-9, "phase must run 0..1 inside a segment"
    assert abs(ph[3]) < 1e-9, "phase must restart at a new segment"
    segs = dc.gt_segments(gt)
    assert segs == [(0, 3, 0), (3, 5, 1), (5, 9, 2)], f"bad segmentation {segs}"
    d = dc.signed_boundary_distance(gt)
    assert d[3] == 0 and d[2] == -1 and d[4] == 1, f"bad signed distance {d[:6]}"
    return f"segments {segs}, phase and signed distance correct"


# --------------------------------------------------------------------------
# solver machinery
# --------------------------------------------------------------------------

def t_dense_equivalence():
    out = fgw_dense.check_equivalence(n=64, m=64, radius=0.04)
    assert out["max_abs_plan_diff"] < 1e-5, out
    assert out["max_abs_objective_diff"] < 1e-5, out
    return (f"dense vs public conv solver: plan {out['max_abs_plan_diff']:.2e}, "
            f"objective {out['max_abs_objective_diff']:.2e}")


def t_index_structure_matches_filters():
    n = m = 48
    radius = 0.08
    Cx, Cy = fgw_dense.index_structure(n, m, radius, "cpu")
    T = torch.rand(1, n, m)
    conv = alignment_ot.mult_Cv(alignment_ot.construct_Cv_filter(n, radius, "cpu"),
                                alignment_ot.mult_Ca(alignment_ot.construct_Ca_filter(m, radius, "cpu"), T))
    dense = Cx @ T @ Cy
    err = (conv - dense).abs().max().item()
    assert err < 1e-3, f"dense structure does not match the conv filters ({err})"
    return f"Cx T Cy matches the conv filters, max err {err:.2e}"


def t_plan_stats():
    one_hot = torch.zeros(1, 8, 16)
    one_hot[0, torch.arange(8), torch.arange(8)] = 1.0
    s = dc.plan_stats(one_hot)
    assert abs(s["top1_mass"] - 1.0) < 1e-6 and abs(s["effective_support"] - 1.0) < 1e-6, s
    uniform = torch.full((1, 8, 16), 1.0 / 16)
    u = dc.plan_stats(uniform)
    assert abs(u["effective_support"] - 16) < 1e-3, u
    assert abs(u["row_entropy_norm"] - 1.0) < 1e-6, u
    return ("one-hot -> support 1.000 / top1 1.000; uniform over 16 -> support "
            f"{u['effective_support']:.3f} / entropy norm {u['row_entropy_norm']:.3f}")


def t_virtual_column_placement():
    """The published training loop prepends the virtual ROW but APPENDS the
    virtual COLUMN, while align_ot drops index 0 on both axes. Two identical
    videos must align to the identity; this records that they do not under the
    training padding."""
    torch.manual_seed(0)
    n = 64
    z = F.normalize(torch.randn(1, n, 8), dim=-1)
    cost = 1.0 - z @ z.transpose(1, 2)
    mask = torch.ones(1, n, dtype=torch.bool)
    out = {}
    for pad in ("train_py", "diagnostics"):
        c = torch.cat((torch.ones(1, 1, n) * 0.5, cost), dim=1)
        if pad == "train_py":
            c = torch.cat((c, torch.ones(1, n + 1, 1) * 0.5), dim=2)
        else:
            c = torch.cat((torch.ones(1, n + 1, 1) * 0.5, c), dim=2)
        T, _ = alignment_ot.align_ot(c, mask, mask, eps=0.07, alpha=0.3, radius=0.04,
                                     ub_frames=False, ub_actions=True,
                                     lambda_frames=0.05, lambda_actions=0.01, n_iters=(25, 1))
        shift = (T[0].argmax(-1) - torch.arange(n)).float()
        out[pad] = (float(shift.mean()), float((shift == 0).float().mean()))
    assert out["diagnostics"][1] > 0.99, f"diagnostic padding broken: {out}"
    assert out["train_py"][0] < -0.5, f"expected the -1 shift to reproduce: {out}"
    return (f"diagnostics padding: identity recovered ({out['diagnostics'][1]:.3f} exact); "
            f"train.py padding: mean shift {out['train_py'][0]:+.3f} frames "
            f"({out['train_py'][1]:.3f} exact) -- upstream off-by-one reproduced")


# --------------------------------------------------------------------------
# scoring machinery
# --------------------------------------------------------------------------

def t_ranking_metrics():
    from eval_g01_angular_probe import ranking_metrics
    score = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 1.0]])
    pos = np.array([[True, False, False], [False, True, False]])
    cand = np.ones_like(pos, dtype=bool)
    m = ranking_metrics(score, pos, cand)
    assert m["auroc"] == 1.0 and m["recall_at_1"] == 1.0, m
    flipped = ranking_metrics(-score, pos, cand)
    assert flipped["auroc"] == 0.0 and flipped["recall_at_1"] == 0.0, flipped
    return f"perfect scores -> AUROC 1.0/R@1 1.0; inverted -> {flipped['auroc']}/{flipped['recall_at_1']}"


def t_centre_controls_are_deterministic():
    from eval_g01_angular_probe import centers
    torch.manual_seed(0)
    clusters = F.normalize(torch.randn(6, 12), dim=-1)
    post = torch.softmax(torch.randn(20, 6), dim=-1)
    for mode in ("hard", "soft", "shuffled", "random"):
        a = centers(post, clusters, mode)
        b = centers(post, clusters, mode)
        assert torch.allclose(a, b), f"centre mode {mode} is not deterministic"
    # the shuffled control must be a wrong but CONSISTENT mapping: the same
    # posterior row must map to the same centre in two different calls
    p2 = post.clone()
    assert torch.allclose(centers(post, clusters, "shuffled"), centers(p2, clusters, "shuffled"))
    hard = centers(post, clusters, "hard")
    shuf = centers(post, clusters, "shuffled")
    assert not torch.allclose(hard, shuf), "shuffled control must differ from hard centres"
    return "hard/soft/shuffled/random centres are deterministic and consistent across calls"


def t_oracle_plan():
    from eval_nc1_warp_stress import oracle_plan, dtw_path
    n = 32
    src = np.arange(n, dtype=float)
    T, j = oracle_plan(src, n, "cpu")
    assert (j == np.arange(n)).all(), "identity warp must give the identity oracle"
    assert abs(float(T.sum()) - 1.0) < 1e-6, "oracle plan must carry total mass 1"
    cost = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    path = dtw_path(cost)
    err = np.nanmax(np.abs(path - np.arange(n)))
    assert err < 1e-9, f"DTW on a diagonal-minimal cost must be the identity ({err})"
    return f"oracle plan identity + unit mass; DTW identity on a diagonal cost (err {err:.1e})"


def t_warp_costs_zero_at_identity():
    """The cost decomposition must assign exactly zero to a correct, unwarped
    correspondence - otherwise every warp number is measured from a wrong zero."""
    from eval_nc1_warp_stress import decompose, oracle_plan

    class FakeModel:
        alpha_train = 0.3
        radius_gw = 0.04
    n = 48
    torch.manual_seed(1)
    z = F.normalize(torch.randn(1, n, 8), dim=-1)
    T, _ = oracle_plan(np.arange(n, dtype=float), n, "cpu")
    terms = decompose(FakeModel(), z, z, T, 0.1)
    for k, v in terms.items():
        assert abs(v) < 1e-6, f"{k} is {v}, expected 0 for an identity correspondence"
    return "all cost terms are exactly 0 for the identity correspondence"


def t_velocity_features():
    from eval_nb1_transition import velocity
    z = torch.arange(10, dtype=torch.float).unsqueeze(-1) * torch.ones(1, 4)
    v = velocity(z, 2, directed=True)
    assert torch.allclose(v[5], F.normalize(z[5] - z[3], dim=-1)), "directed velocity wrong"
    assert v.shape == z.shape, "directed velocity must keep the feature width"
    vs = velocity(z, 2, directed=False)
    assert vs.shape[-1] == 2 * z.shape[-1], "symmetric velocity must concatenate both sides"
    return "directed velocity = z_i - z_{i-w}; symmetric concatenates both sides"


def t_band_structure_reduces_to_index():
    """A band-limited structure built on perfectly similar frames must reduce to
    the index structure, up to the mean-entry rescaling. Otherwise the N-A1
    comparison is not one-factor-at-a-time."""
    from eval_na1_dynamics_cost import build_structure
    n, radius = 40, 0.1
    z = torch.ones(n, 6)  # every pair perfectly similar -> similarity == 1 in band
    Cx, Cy = build_structure("band_feature", z, radius, scale_to=None)
    Ix, Iy = fgw_dense.index_structure(n, n, radius, "cpu")
    band = (Ix > 0).float()
    assert torch.allclose((Cx > 0).float(), band), "band support differs from the index band"
    assert torch.allclose(Cy, 1.0 - band, atol=1e-6), "Cy must be 0 inside the band when similarity is 1"
    scaled = Cx * (float(Ix.mean()) / float(Cx.mean()))
    assert torch.allclose(scaled, Ix, atol=1e-4), "rescaled band structure != index structure"
    return "band_feature on identical frames == index structure after rescaling"


def t_na1_structure_conventions():
    """Cx must be a within-X similarity (non-negative, zero diagonal) and Cy a
    within-Y dissimilarity in [0, 1] with a diagonal of 1, for every variant."""
    from eval_na1_dynamics_cost import build_structure
    torch.manual_seed(0)
    n = 32
    z = F.normalize(torch.randn(n, 8), dim=-1)
    checked = []
    for kind in ("index", "feature", "velocity", "shuffled", "random",
                 "band_feature", "band_feature_shuffled", "band_velocity", "band_shuffled"):
        Cx, Cy = build_structure(kind, z, 0.1, scale_to=None, seed=0)
        assert (Cx >= -1e-6).all(), f"{kind}: Cx must be non-negative"
        assert float(Cx.diagonal().abs().max()) < 1e-6, f"{kind}: Cx diagonal must be 0"
        assert float(Cy.min()) >= -1e-6 and float(Cy.max()) <= 1 + 1e-6, f"{kind}: Cy outside [0,1]"
        assert torch.allclose(Cy.diagonal(), torch.ones(n), atol=1e-6), f"{kind}: Cy diagonal must be 1"
        a, b = build_structure(kind, z, 0.1, scale_to=None, seed=0)[0], \
               build_structure(kind, z, 0.1, scale_to=None, seed=0)[0]
        assert torch.allclose(a, b), f"{kind} is not deterministic for a fixed seed"
        checked.append(kind)
    return f"{len(checked)} structures obey the Cx/Cy convention and are seed-deterministic"


def t_conic_fitter_recovers_known_shapes():
    """The A06 fitter must FIND a conic when one is there, and must not find one
    in matched-covariance noise. Without this, a null result from A06 would be
    uninterpretable."""
    from eval_a06_conic_structure import group_test
    rng = np.random.default_rng(0)
    n, d = 60, 40
    emb = rng.normal(size=(2, d))  # two ambient directions to embed the 2D shape in
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    def lift(xy, noise=0.01):
        return xy @ emb + rng.normal(scale=noise, size=(xy.shape[0], d))

    t = rng.uniform(0, 2 * np.pi, n)
    circle = lift(np.stack([np.cos(t), np.sin(t)], axis=1))
    ell = lift(np.stack([2.0 * np.cos(t), 0.6 * np.sin(t)], axis=1))
    u = rng.uniform(-1, 1, n)
    line = lift(np.stack([u, 0.5 * u], axis=1))
    blob = rng.normal(size=(n, d)) * 0.5

    got, got_abs = {}, {}
    for name, Z in (("circle", circle), ("ellipse", ell), ("line", line), ("blob", blob)):
        rows = [group_test(Z, np.random.default_rng(i), 16) for i in range(5)]
        got[name] = {m: float(np.mean([r[f"{m}_ratio_real_over_null"] for r in rows]))
                     for m in ("line", "circle", "conic")}
        got_abs[name] = {m: float(np.mean([r[f"{m}_heldout_residual"] for r in rows]))
                         for m in ("line", "circle", "conic")}
    # Discriminating power lives in the circle and conic models:
    assert got["circle"]["circle"] < 0.2, f"circle not detected: {got['circle']}"
    assert got["ellipse"]["conic"] < 0.2, f"ellipse not detected by the conic: {got['ellipse']}"
    assert got["ellipse"]["conic"] < 0.1 * got["ellipse"]["circle"], \
        f"only the conic should fit an ellipse: {got['ellipse']}"
    assert got["circle"]["line"] > 0.8, f"a line must NOT fit a circle: {got['circle']}"
    assert got["blob"]["circle"] > 0.8, f"noise must not look like a circle: {got['blob']}"
    assert got["blob"]["conic"] > 0.5, f"noise must not look like a conic: {got['blob']}"
    # The LINE model's ratio is deliberately not asserted low. A Gaussian null that
    # copies an elongated group's covariance is itself nearly a line, so the
    # matched-covariance null is conservative there and the ratio saturates near
    # 0.5 for a true line AND for noise (measured: 0.505 vs 0.498). For the line
    # model only the absolute residual discriminates, so that is what is checked.
    assert got_abs["line"]["line"] < 0.05, f"line absolute residual too large: {got_abs['line']}"
    assert got_abs["blob"]["line"] > 5 * got_abs["line"]["line"], "line residual must separate from noise"
    return ("circle->circle %.3f, ellipse->conic %.3f (circle %.3f), line->line %.3f, "
            "noise->conic %.3f (all ratios real/null)") % (
        got["circle"]["circle"], got["ellipse"]["conic"], got["ellipse"]["circle"],
        got["line"]["line"], got["blob"]["conic"])




def t_quadric_fitter_recovers_known_shapes():
    """The quadric arm of A06 must detect a non-planar quadric (a 3D sphere /
    ellipsoid shell embedded in 40 dimensions) and must not fire on noise."""
    from eval_a06_conic_structure import quadric_test
    rng = np.random.default_rng(1)
    n, d = 120, 40
    emb = rng.normal(size=(5, d))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    shell = np.concatenate([v * np.array([1.0, 1.0, 1.0]), np.zeros((n, 2))], axis=1) @ emb
    shell = shell + rng.normal(scale=0.01, size=(n, d))
    ellipsoid = np.concatenate([v * np.array([2.0, 1.0, 0.5]), np.zeros((n, 2))], axis=1) @ emb
    ellipsoid = ellipsoid + rng.normal(scale=0.01, size=(n, d))
    blob = rng.normal(size=(n, d)) * 0.5
    got = {}
    for name, Z in (("shell", shell), ("ellipsoid", ellipsoid), ("blob", blob)):
        rows = [quadric_test(Z, np.random.default_rng(i)) for i in range(5)]
        got[name] = float(np.mean([r["quadric3d_ratio_real_over_null"] for r in rows]))
    assert got["shell"] < 0.3, f"3D sphere shell not detected: {got}"
    assert got["ellipsoid"] < 0.3, f"ellipsoid not detected: {got}"
    assert got["blob"] > 2 * got["shell"], f"noise must separate from a shell: {got}"
    return (f"sphere shell {got['shell']:.3f}, ellipsoid {got['ellipsoid']:.3f}, "
            f"noise {got['blob']:.3f} (quadric3d ratio real/null)")



def main():
    for name, fn in [
        ("warp: monotone and spanning", t_warp_monotone),
        ("warp: severity 0 is the identity", t_warp_identity),
        ("warp: resampling is exact / interpolates", t_resample_identity),
        ("labels: segments, phase, boundary distance", t_relative_phase),
        ("solver: dense == public conv solver", t_dense_equivalence),
        ("solver: dense structure == conv filters", t_index_structure_matches_filters),
        ("stats: plan_stats on known plans", t_plan_stats),
        ("solver: virtual column placement", t_virtual_column_placement),
        ("scoring: ranking metrics on known input", t_ranking_metrics),
        ("scoring: centre controls deterministic", t_centre_controls_are_deterministic),
        ("N-C1: oracle plan and DTW sanity", t_oracle_plan),
        ("N-C1: zero cost at identity", t_warp_costs_zero_at_identity),
        ("N-B1: velocity features", t_velocity_features),
        ("N-A1: band structure reduces to index", t_band_structure_reduces_to_index),
        ("N-A1: structure conventions hold", t_na1_structure_conventions),
        ("A06: conic fitter recovers known shapes", t_conic_fitter_recovers_known_shapes),
        ("A06: quadric fitter recovers known shapes", t_quadric_fitter_recovers_known_shapes),
    ]:
        check(name, fn)

    width = max(len(n) for _, n, _ in RESULTS)
    failed = 0
    for ok, name, detail in RESULTS:
        if ok:
            print(f"PASS  {name:<{width}}  {detail}")
        else:
            failed += 1
            print(f"FAIL  {name:<{width}}\n{detail}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} invariants hold")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
