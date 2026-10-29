#!/usr/bin/env python3
"""E05: Collect and compare all E-phase experiment results on FSeval.

Reads manifests from artifacts/runs/E0* and baseline runs,
produces a comparison table for the scale gate decision.

Scale gate: improve >= 1pp mean over 2 seeds, or improve diagnostic
without losing >1pp task metric.
"""
import json
import os
import sys
from pathlib import Path

ARTIFACTS = Path("/mnt/data/thanhpv/tssl/reproductions/tssl/artifacts/runs")

# Known results from old logs (no manifests in new system)
KNOWN_RESULTS = {
    "baseline_s0": {"test_mof_full": 0.4510, "test_f1_full": 0.4832, "test_miou_full": 0.2553, "seed": 0},
    "arc_length_s0": {"test_mof_full": 0.4850, "test_f1_full": 0.0, "test_miou_full": 0.0, "seed": 0},
    "arc_length_s1": {"test_mof_full": 0.4448, "test_f1_full": 0.0, "test_miou_full": 0.0, "seed": 1},
}


def load_manifest(run_dir):
    mf = run_dir / "manifest.json"
    if not mf.exists():
        return None
    with open(mf) as f:
        return json.load(f)


def load_diagnostics(run_dir):
    dj = run_dir / "diagnostics" / "diagnostics.json"
    if not dj.exists():
        return None
    with open(dj) as f:
        return json.load(f).get("summary", {})


def main():
    # Collect all E-phase and baseline results
    results = {}

    # Known results from old logs
    for name, metrics in KNOWN_RESULTS.items():
        results[name] = {"metrics": metrics, "diagnostics": None, "status": "known"}

    # Scan artifacts
    for d in sorted(ARTIFACTS.iterdir()):
        if not d.name.startswith("E0") and not d.name.startswith("sweep_baseline"):
            continue
        m = load_manifest(d)
        if m is None:
            continue
        run_id = m["run_id"]
        status = m.get("status", "unknown")
        t = m.get("test_metrics", {})
        diag = load_diagnostics(d)
        results[run_id] = {
            "metrics": t,
            "diagnostics": diag,
            "status": status,
            "wandb": m.get("wandb_run_id", ""),
        }

    # Print comparison table
    print("\n" + "=" * 100)
    print("E05 COMPARISON MATRIX: FSeval Continuous Curve Experiments")
    print("=" * 100)
    print(f"\n{'Run ID':40s} {'Seed':>4s} {'MoF':>8s} {'F1':>8s} {'mIoU':>8s} {'T_ent':>7s} {'Bnd_z':>7s} {'Status':>10s}")
    print("-" * 100)

    # Group by variant
    variants = {}
    for run_id, data in sorted(results.items()):
        t = data["metrics"]
        mof = t.get("test_mof_full", 0)
        f1 = t.get("test_f1_full", 0)
        miou = t.get("test_miou_full", 0)

        # Extract seed from run_id
        seed = t.get("seed", "?")
        if "_s0" in run_id:
            seed = 0
        elif "_s1" in run_id:
            seed = 1

        # Extract variant
        rid_l = run_id.lower()
        if "baseline" in rid_l:
            variant = "baseline"
        elif "bspline" in rid_l:
            variant = "bspline"
        elif "gauss" in rid_l:
            variant = "gaussian"
        elif "bezier" in rid_l:
            variant = "bezier"
        elif "arc" in rid_l:
            variant = "arc"
        elif "e07a_eps0p15" in rid_l:
            variant = "eps0p15"
        elif "e07a_eps0p30" in rid_l:
            variant = "eps0p30"
        elif "e07b_rho0p05" in rid_l:
            variant = "rho0p05"
        elif "e07b_rho0p0_" in rid_l:
            variant = "rho0p0"
        elif "e07c_r0p12" in rid_l:
            variant = "r0p12"
        elif "e07c_r0p50" in rid_l:
            variant = "r0p50"
        else:
            variant = "other"

        diag = data.get("diagnostics") or {}
        t_ent = diag.get("T_entropy_mean_norm", 0)
        bnd_z = diag.get("boundary_z_mean", 0)

        status = data["status"]
        if mof > 0:
            print(f"{run_id:40s} {seed:>4} {mof:8.4f} {f1:8.4f} {miou:8.4f} {t_ent:7.4f} {bnd_z:7.4f} {status:>10s}")

            if variant not in variants:
                variants[variant] = []
            variants[variant].append({"mof": mof, "f1": f1, "miou": miou, "seed": seed})

    # Summary by variant
    print("\n" + "=" * 100)
    print("VARIANT SUMMARY (mean over seeds)")
    print("=" * 100)
    print(f"\n{'Variant':15s} {'N_seeds':>7s} {'MoF':>8s} {'F1':>8s} {'mIoU':>8s} {'Δ MoF':>8s} {'Δ F1':>8s} {'Δ mIoU':>8s}")
    print("-" * 80)

    baseline_mof = 0
    baseline_f1 = 0
    baseline_miou = 0

    all_variants = [
        "baseline", "arc", "bspline", "gaussian", "bezier",
        "eps0p15", "eps0p30",
        "rho0p05", "rho0p0", "r0p12", "r0p50",
    ]
    for variant in all_variants:
        if variant not in variants:
            continue
        runs = variants[variant]
        n = len(runs)
        mean_mof = sum(r["mof"] for r in runs) / n
        mean_f1 = sum(r["f1"] for r in runs) / n
        mean_miou = sum(r["miou"] for r in runs) / n

        if variant == "baseline":
            baseline_mof = mean_mof
            baseline_f1 = mean_f1
            baseline_miou = mean_miou
            d_mof = d_f1 = d_miou = 0
        else:
            d_mof = mean_mof - baseline_mof
            d_f1 = mean_f1 - baseline_f1
            d_miou = mean_miou - baseline_miou

        d_str = lambda x: f"{x:+8.4f}" if x != 0 else f"{'—':>8s}"
        print(f"{variant:15s} {n:>7d} {mean_mof:8.4f} {mean_f1:8.4f} {mean_miou:8.4f} {d_str(d_mof)} {d_str(d_f1)} {d_str(d_miou)}")

    # Scale gate check
    print("\n" + "=" * 100)
    print("SCALE GATE CHECK (need ≥1pp improvement over baseline mean)")
    print("=" * 100)
    for variant in all_variants:
        if variant == "baseline" or variant not in variants:
            continue
        runs = variants[variant]
        n = len(runs)
        mean_mof = sum(r["mof"] for r in runs) / n
        d_mof = (mean_mof - baseline_mof) * 100
        passed = d_mof >= 1.0
        status = "PASS" if passed else "FAIL"
        print(f"  {variant:15s} (n={n}): Δ MoF = {d_mof:+.2f} pp → {status}")


if __name__ == "__main__":
    main()
