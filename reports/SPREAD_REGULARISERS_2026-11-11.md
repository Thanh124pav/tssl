# Seven prior-work "spread the embedding" ideas on VASOT: 3-seed results

**Date:** 2026-11-11. Server runs (Slurm, 1× H100, env `acvcl`), follow-up to `SERVER_RESULTS_2026-11-11.md`.
**One line:** none of the seven ideas, alone or combined, improves segmentation and alignment together. VICReg is the only one with a segmentation gain (+2.0 MoF at weight 1.0, +2.5 at 3.0) and it loses Kendall τ in proportion to its weight (−3.0, −9.2). Temporal negatives and VICReg + `local` give the largest AP@K gains so far (+1.6, +2.3). CSLS harms no metric and is small; with target blur and `local` it is the best alignment configuration (τ +2.2, AP@5 +1.1, both on 7/7 benchmarks). VICReg at 1.0 on top of that configuration is the nearest to a joint gain: MoF +1.1, AP@5 +2.0, τ −0.4.

## 1. Protocol

Same as `SERVER_RESULTS_2026-11-11.md` §1: 7 benchmarks (FSeval, DA, YTI × 5 activities), seeds 0/1/2, paired difference to the baseline of the same seed, averaged over seeds and then over benchmarks. 147 training runs, 0 failures.

Each idea was run at **one weight, with no sweep**. The weight was set from the gradient norm of the regulariser relative to the alignment loss on baseline checkpoints (`scripts/calib_spread.py`), so that every regulariser enters at roughly 0.3–0.6× the alignment gradient. The table therefore compares the ideas at a similar strength; it does not show the best each idea can do.

## 2. Arms

Code: `repos/VASOT/src/spread_reg.py`, flags `--spread-reg`, `--spread-weight`, `--spread-warmup-epochs`, `--align-csls` of `train.py`. All regularisers act on the unit-norm frame embeddings of the valid frames of a batch; none uses labels.

| Idea | Arm | Loss | Weight | Gradient ratio at weight 1 (FSeval / DA / YTI coffee) |
|---|---|---|---:|---|
| variance + decorrelation (VICReg) | `vcreg` | hinge on per-dimension std + off-diagonal covariance | 3.0 | 0.15 / 0.12 / 0.14 |
| uniformity on the sphere (Wang & Isola) | `unif` | log mean exp(−2‖zᵢ−zⱼ‖²) | 1.0 | 0.66 / 0.36 / 0.43 |
| push the nearest neighbour away (KoLeo) | `koleo` | −mean log(nearest-neighbour distance) | 0.2 | 1.33 / 2.94 / 2.93 |
| raise local intrinsic dimension (LDReg) | `ldreg` | −mean log LID, MLE over 20 neighbours | 0.1 | 6.4 / 1.9 / 3.7 |
| density-normalised similarity (CSLS) | `csls` | alignment logits minus ½ × mean top-10 similarity of each target frame; OT cost unchanged | – | – |
| residual to the centroid | `koleo_res` | KoLeo on normalised `z − prototype`, neighbours inside the same pseudo-action, from epoch 5 | 0.2 | 1.34 / 3.03 / 2.96 |
| temporal negatives (TCN / CARL-style) | `tneg` | intra-video contrast, Gaussian positives (σ = 4 frames), all other frames negatives | 0.5 | 1.04 / 0.93 / 0.70 |

The contrastive version of "residual to the centroid" is the existing `local` arm (`SERVER_RESULTS_2026-11-11.md`: AP@5 +0.9 on 7/7, MoF +0.5); `koleo_res` is the nearest-neighbour version.

## 3. Results (mean paired Δ vs baseline over 7 benchmarks, 3 seeds)

| Arm | MoF | F1 | mIoU | SVM Acc | Kendall τ | AP@5 | AP@10 | AP@15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vcreg` | **+2.5** (6/7) | +1.1 (4/7) | +1.3 (6/7) | +0.8 (5/7) | **−9.2** (0/7) | +1.1 (5/7) | +1.2 (5/7) | +1.3 (6/7) |
| `unif` | +0.6 (3/7) | −0.0 (3/7) | +0.6 (5/7) | +1.4 (4/7) | **−8.2** (0/7) | +1.4 (5/7) | +1.4 (5/7) | +1.3 (5/7) |
| `koleo` | +0.4 (5/7) | +0.2 (3/7) | −0.1 (4/7) | +0.1 (4/7) | −1.3 (2/7) | +0.3 (6/7) | +0.4 (5/7) | +0.4 (5/7) |
| `ldreg` | +0.4 (4/7) | +0.4 (4/7) | −0.3 (1/7) | +0.3 (2/7) | +0.8 (5/7) | −1.0 (0/7) | −0.9 (1/7) | −0.9 (0/7) |
| `csls` | +0.4 (5/7) | +0.4 (6/7) | +0.4 (5/7) | +0.1 (4/7) | +0.2 (4/7) | +0.4 (7/7) | +0.5 (7/7) | +0.5 (7/7) |
| `koleo_res` | +0.4 (6/7) | +0.5 (5/7) | +0.2 (5/7) | −0.0 (3/7) | −1.0 (2/7) | −0.1 (2/7) | −0.1 (2/7) | −0.1 (2/7) |
| `tneg` | +0.9 (4/7) | +0.1 (4/7) | +0.2 (3/7) | **+1.3** (7/7) | −1.2 (1/7) | **+1.6** (6/7) | **+1.7** (6/7) | **+1.7** (6/7) |

For comparison, from the previous report: `hub` MoF +2.9, τ −2.8, AP@5 −2.2; `local` MoF +0.5, τ +0.1, AP@5 +0.9; `blur6_local` MoF +0.2, τ +2.0, AP@5 +0.8.

## 4. Reading

1. **VICReg is a second source of segmentation gain, with a different trade from the hub.** MoF +2.5 against the hub's +2.9. Unlike the hub it does not lower SVM accuracy or AP@K (+0.8, +1.1); it lowers τ three times as much (−9.2 against −2.8). The MoF gain comes from three YTI activities (coffee +6.1, cpr +7.2, repot +6.8, each 3/3 seeds); FSeval and DA are unchanged (+0.0, +0.3) and changing_tire loses 3.4 (0/3 seeds).
2. **Both global spreaders lose τ on every benchmark and every YTI seed.** `vcreg`: FSeval −2.9, DA −0.9, YTI −8.9 to −15.0. `unif`: −3.0 to −13.3. This is the benchmark pattern of decentering (`dc`: τ −16.4, concentrated on YTI) at about half the size, and without the collapse in segmentation and AP@K. The cause was not investigated.
3. **Uniformity has the τ cost without the segmentation gain** (MoF +0.6 on 3/7; FSeval −2.8, 0/3 seeds). Its SVM-accuracy mean (+1.4) is carried by cpr (+7.4).
4. **Temporal negatives are the strongest alignment-retrieval arm so far.** SVM accuracy up on 7/7 benchmarks, AP@5/10/15 +1.6/+1.7/+1.7 on 6/7 (the `local` arm: +0.9), MoF +0.9. τ falls by 1.2 (1/7): up on DA (+1.8, 3/3 seeds), down on cpr (−3.4) and repot (−3.8), 0/3 seeds each.
5. **CSLS is safe and small.** AP@5/10/15 up on 7/7 benchmarks (+0.4 to +0.5), τ +0.2, no metric down on average. Its MoF mean is one benchmark: DA +3.6 (3/3 seeds); the other six average −0.1.
6. **LDReg trades AP@K for τ.** τ +0.8 (5/7; jump_car +1.5 and FSeval +0.4 on 3/3 seeds), AP@5 down on 7/7 (−1.0), mIoU down on 6/7.
7. **KoLeo, global or on centroid residuals, shows no clear effect** at this weight (all means within ±0.5 except τ −1.3 / −1.0). One exception: `koleo` on repot, MoF +3.1 (3/3 seeds).
8. **jump_car loses AP@K under every regulariser except CSLS** (AP@5 −1.0 to −3.5; 0/3 seeds for `vcreg`, `unif`, `koleo`, `koleo_res`, `tneg`).

Current picture, with the previous report: segmentation gain comes from `hub` or `vcreg`, AP@K gain from `tneg` or `local`, τ gain from target blur (and weakly `ldreg`). Every single arm that raises MoF by more than 1 lowers τ.

## 5. Combinations (3 seeds)

Six follow-up arms, same protocol (126 runs, 0 failures). `local` = `--local-weight 1.0 --local-warmup-epochs 5`, `blur6` = `--align-target-blur 6`.

| Arm | MoF | F1 | mIoU | SVM Acc | Kendall τ | AP@5 | AP@10 | AP@15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vcreg1` (VICReg at weight 1.0) | **+2.0** (6/7) | +0.7 (5/7) | +1.0 (6/7) | +1.1 (6/7) | −3.0 (0/7) | **+1.6** (6/7) | +1.7 (6/7) | +1.7 (6/7) |
| `vcreg_blur6` (weight 3.0 + blur) | +2.2 (4/7) | +0.0 (3/7) | +0.6 (5/7) | +0.7 (6/7) | −5.9 (0/7) | +1.1 (5/7) | +1.2 (5/7) | +1.2 (5/7) |
| `tneg_blur6` | +0.6 (4/7) | +0.3 (5/7) | −0.2 (3/7) | +1.0 (5/7) | +0.4 (5/7) | +1.4 (7/7) | +1.4 (5/7) | +1.4 (5/7) |
| `csls_local` | +0.8 (6/7) | +0.7 (5/7) | +0.7 (5/7) | +0.4 (5/7) | +0.4 (6/7) | +1.2 (7/7) | +1.2 (7/7) | +1.1 (7/7) |
| `csls_blur6_local` | +0.3 (4/7) | +0.8 (6/7) | +0.2 (5/7) | +0.8 (6/7) | **+2.2** (7/7) | +1.1 (7/7) | +1.1 (7/7) | +1.0 (7/7) |
| `ldreg_local` | −0.1 (3/7) | +0.5 (4/7) | −0.6 (1/7) | +0.9 (4/7) | +1.3 (6/7) | +0.1 (3/7) | +0.2 (5/7) | +0.1 (4/7) |

1. **Lowering the VICReg weight from 3.0 to 1.0 keeps most of the MoF gain (+2.5 → +2.0) and cuts the τ loss to a third (−9.2 → −3.0).** `vcreg1` has the hub's τ cost (hub: −2.8) with positive SVM accuracy and AP@K where the hub is negative (−1.9, −2.2). τ is still down on 7/7 benchmarks, most on repot (−7.1) and cpr (−5.6). The MoF gain comes from DA (+3.9) and three YTI activities (coffee +2.8, cpr +3.8, repot +4.8); changing_tire loses 2.7 (0/3 seeds).
2. **Target blur does not repair VICReg's τ** (`vcreg_blur6`: −5.9, 0/7).
3. **`csls_blur6_local` is the best alignment configuration so far**: τ up on 7/7 benchmarks, AP@5/10/15 up on 7/7, SVM accuracy +0.8, no segmentation loss. It is a small step over `blur6_local` (τ +2.0, AP@5 +0.8).
4. **`csls_local` has no negative column**, all gains under 1.3. Its MoF mean leans on DA (+4.1, 3/3 seeds).
5. **Blur removes `tneg`'s τ loss** (−1.2 → +0.4) and keeps its AP@5 (+1.4, 7/7); FSeval loses 2.7 MoF (0/3 seeds).
6. `local` restores `ldreg`'s AP@K (−1.0 → +0.1); mIoU is still down on 6/7.

## 6. VICReg weight, and VICReg next to the alignment configuration (3 seeds)

84 runs, 0 failures. `vcregW` = `--spread-reg vc --spread-weight W` (`vcreg` = 3.0, `vcreg1` = 1.0, `vcreg03` = 0.3).

| Arm | MoF | F1 | mIoU | SVM Acc | Kendall τ | AP@5 | AP@10 | AP@15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vcreg` (3.0) | +2.5 (6/7) | +1.1 (4/7) | +1.3 (6/7) | +0.8 (5/7) | −9.2 (0/7) | +1.1 (5/7) | +1.2 (5/7) | +1.3 (6/7) |
| `vcreg1` (1.0) | +2.0 (6/7) | +0.7 (5/7) | +1.0 (6/7) | +1.1 (6/7) | −3.0 (0/7) | +1.6 (6/7) | +1.7 (6/7) | +1.7 (6/7) |
| `vcreg03` (0.3) | +0.1 (2/7) | −0.1 (3/7) | −0.2 (3/7) | +0.0 (4/7) | −0.6 (1/7) | +0.6 (6/7) | +0.5 (6/7) | +0.5 (6/7) |
| `vcreg1_local` | +1.9 (6/7) | +1.1 (5/7) | +0.8 (5/7) | +1.6 (6/7) | −3.2 (1/7) | **+2.3** (6/7) | +2.3 (6/7) | +2.3 (6/7) |
| `vcreg1_csls_blur6_local` | +1.1 (5/7) | +0.1 (4/7) | +0.1 (3/7) | **+1.8** (6/7) | −0.4 (4/7) | +2.0 (6/7) | +2.1 (6/7) | +2.0 (7/7) |
| `vcreg03_csls_blur6_local` | −0.0 (4/7) | −0.1 (4/7) | −0.4 (4/7) | +0.3 (4/7) | +1.6 (7/7) | +1.4 (7/7) | +1.4 (7/7) | +1.4 (7/7) |

1. **The segmentation gain and the τ loss of VICReg move together with the weight.** 0.3 / 1.0 / 3.0: MoF +0.1 / +2.0 / +2.5, τ −0.6 / −3.0 / −9.2. At 0.3 there is no segmentation gain left. No weight gives MoF without τ loss.
2. **`vcreg1_local` has the largest AP@K gain of any arm** (+2.3) with the MoF of `vcreg1` (+1.9) and its τ loss (−3.2).
3. **`vcreg1_csls_blur6_local` is the closest to a joint improvement**: MoF +1.1 (5/7), SVM accuracy +1.8, AP@5 +2.0, τ −0.4 (4/7). It is not one: F1 and mIoU are unchanged (+0.1), τ falls on cpr (−1.8, 0/3 seeds) and repot (−3.4, 0/3), and the MoF gain is half that of `vcreg1`. The blur and the CSLS term give back τ and take away MoF.
4. `vcreg03_csls_blur6_local` behaves like `csls_blur6_local` (τ +1.6 on 7/7, AP@5 +1.4 on 7/7, no segmentation change): VICReg at 0.3 adds nothing to it.

Summary over both reports: three arms sit on the trade-off frontier. `hub_local` or `vcreg1_local` for segmentation (MoF +2.9 / +1.9, τ −2.0 / −3.2; `vcreg1_local` also raises AP@5 by 2.3 where `hub_local` loses 0.7), `csls_blur6_local` for alignment (τ +2.2, AP@5 +1.1, MoF +0.3), `vcreg1_csls_blur6_local` in between. No arm raises MoF by more than 1 without lowering τ on at least three benchmarks.

## 7. Not done

- Weight sweeps: each idea has one weight, except VICReg (§6).
- Why the global spreaders lower τ on YTI.
- Breakfast, Pouring, Penn Action for these arms.

## 8. Reproduce

```bash
source reproductions/env.sh; export TSSL_PYTHON=$ACVCL_ENV/bin/python
cd reproductions/tssl/scripts
$TSSL_PYTHON calib_spread.py --ckpt ../artifacts/runs/current/mrt_fseval_baseline_s0/checkpoints/last.ckpt   # gradient ratios
ARMS="vcreg unif koleo ldreg csls koleo_res tneg" SEEDS="0 1 2" JOBS=13 bash run_arm_queue.sh
ARMS="tneg_blur6 csls_local csls_blur6_local vcreg1 vcreg_blur6 ldreg_local" SEEDS="0 1 2" JOBS=13 bash run_arm_queue.sh
ARMS="vcreg03 vcreg1_local vcreg1_csls_blur6_local vcreg03_csls_blur6_local" SEEDS="0 1 2" JOBS=13 bash run_arm_queue.sh
JOBS=3 $TSSL_PYTHON run_va_new.py vcreg unif koleo ldreg csls koleo_res tneg
$TSSL_PYTHON collect_mrt_results.py vcreg unif koleo ldreg csls koleo_res tneg
```

---

# Appendix: per-benchmark tables

Cell format: mean over seeds (paired Δ vs baseline, seeds with Δ > 0 / seeds). `raw` = input features without training.

## A. The seven single arms

### Segmentation: MoF (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | vcreg | unif | koleo | ldreg | csls | koleo_res | tneg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 49.1 (n=3) | 49.1 (+0.0, 1/3) | 46.3 (-2.8, 0/3) | 47.7 (-1.4, 1/3) | 48.2 (-0.8, 2/3) | 48.1 (-1.0, 1/3) | 49.5 (+0.4, 2/3) | 48.9 (-0.2, 2/3) |
| da | 60.0 (n=3) | 60.3 (+0.3, 1/3) | 58.4 (-1.7, 1/3) | 60.5 (+0.5, 2/3) | 60.7 (+0.6, 1/3) | 63.6 (+3.6, 3/3) | 59.1 (-0.9, 0/3) | 62.2 (+2.2, 2/3) |
| yti_changing_tire | 48.6 (n=3) | 45.2 (-3.4, 0/3) | 45.7 (-2.9, 0/3) | 48.9 (+0.3, 1/3) | 48.1 (-0.5, 1/3) | 48.2 (-0.4, 0/3) | 48.6 (+0.0, 2/3) | 46.7 (-1.9, 0/3) |
| yti_coffee | 56.4 (n=3) | 62.5 (+6.1, 3/3) | 60.5 (+4.1, 3/3) | 56.6 (+0.2, 2/3) | 57.3 (+1.0, 1/3) | 56.7 (+0.3, 2/3) | 56.8 (+0.4, 2/3) | 58.5 (+2.1, 2/3) |
| yti_cpr | 47.6 (n=3) | 54.8 (+7.2, 3/3) | 51.4 (+3.9, 2/3) | 47.8 (+0.2, 1/3) | 49.7 (+2.1, 2/3) | 47.8 (+0.2, 3/3) | 48.6 (+1.0, 2/3) | 49.7 (+2.2, 2/3) |
| yti_jump_car | 33.8 (n=3) | 34.5 (+0.7, 2/3) | 33.5 (-0.2, 2/3) | 33.4 (-0.4, 1/3) | 34.3 (+0.5, 1/3) | 33.8 (+0.0, 1/3) | 34.0 (+0.2, 1/3) | 33.2 (-0.6, 1/3) |
| yti_repot | 39.2 (n=3) | 46.0 (+6.8, 3/3) | 43.1 (+3.9, 3/3) | 42.3 (+3.1, 3/3) | 38.8 (-0.4, 1/3) | 39.4 (+0.2, 3/3) | 40.7 (+1.5, 3/3) | 41.9 (+2.7, 3/3) |
| **mean delta** |  | +2.5, 6/7 bench | +0.6, 3/7 bench | +0.4, 5/7 bench | +0.4, 4/7 bench | +0.4, 5/7 bench | +0.4, 6/7 bench | +0.9, 4/7 bench |

### Segmentation: F1 (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | vcreg | unif | koleo | ldreg | csls | koleo_res | tneg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 51.4 (n=3) | 50.8 (-0.6, 1/3) | 48.1 (-3.3, 0/3) | 49.5 (-1.9, 1/3) | 51.0 (-0.4, 2/3) | 50.3 (-1.1, 1/3) | 52.1 (+0.7, 3/3) | 49.7 (-1.7, 0/3) |
| da | 64.3 (n=3) | 64.0 (-0.4, 1/3) | 62.9 (-1.4, 1/3) | 63.9 (-0.4, 1/3) | 65.3 (+0.9, 1/3) | 67.2 (+2.9, 2/3) | 64.2 (-0.2, 1/3) | 64.8 (+0.5, 2/3) |
| yti_changing_tire | 31.2 (n=3) | 30.1 (-1.1, 1/3) | 30.5 (-0.7, 1/3) | 30.9 (-0.2, 1/3) | 31.0 (-0.2, 1/3) | 31.7 (+0.5, 2/3) | 32.1 (+0.9, 2/3) | 30.1 (-1.1, 1/3) |
| yti_coffee | 40.3 (n=3) | 41.5 (+1.2, 3/3) | 41.2 (+0.9, 3/3) | 40.6 (+0.3, 2/3) | 39.1 (-1.3, 1/3) | 40.4 (+0.0, 2/3) | 40.3 (-0.1, 2/3) | 41.2 (+0.9, 3/3) |
| yti_cpr | 43.2 (n=3) | 48.4 (+5.3, 3/3) | 47.0 (+3.8, 2/3) | 44.6 (+1.4, 2/3) | 45.4 (+2.2, 2/3) | 43.6 (+0.4, 3/3) | 44.5 (+1.4, 3/3) | 44.3 (+1.1, 2/3) |
| yti_jump_car | 26.2 (n=3) | 26.4 (+0.2, 2/3) | 25.7 (-0.5, 1/3) | 26.1 (-0.1, 1/3) | 27.1 (+0.9, 2/3) | 26.2 (+0.0, 1/3) | 26.6 (+0.4, 3/3) | 25.7 (-0.4, 2/3) |
| yti_repot | 25.6 (n=3) | 28.6 (+3.0, 3/3) | 26.7 (+1.1, 2/3) | 28.3 (+2.7, 2/3) | 26.2 (+0.6, 2/3) | 25.7 (+0.1, 3/3) | 25.9 (+0.3, 2/3) | 27.0 (+1.4, 2/3) |
| **mean delta** |  | +1.1, 4/7 bench | -0.0, 3/7 bench | +0.2, 3/7 bench | +0.4, 4/7 bench | +0.4, 6/7 bench | +0.5, 5/7 bench | +0.1, 4/7 bench |

### Segmentation: mIoU (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | vcreg | unif | koleo | ldreg | csls | koleo_res | tneg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 28.6 (n=3) | 29.2 (+0.6, 1/3) | 28.8 (+0.2, 2/3) | 26.5 (-2.0, 0/3) | 27.3 (-1.2, 2/3) | 27.6 (-1.0, 1/3) | 29.0 (+0.4, 2/3) | 27.5 (-1.1, 1/3) |
| da | 41.3 (n=3) | 41.8 (+0.5, 2/3) | 40.1 (-1.2, 1/3) | 41.7 (+0.4, 1/3) | 42.5 (+1.2, 1/3) | 44.5 (+3.2, 2/3) | 40.7 (-0.6, 0/3) | 42.8 (+1.5, 3/3) |
| yti_changing_tire | 22.7 (n=3) | 22.3 (-0.4, 2/3) | 21.5 (-1.1, 0/3) | 23.0 (+0.3, 2/3) | 22.4 (-0.2, 1/3) | 22.5 (-0.2, 1/3) | 22.9 (+0.2, 2/3) | 21.5 (-1.2, 0/3) |
| yti_coffee | 25.3 (n=3) | 28.3 (+3.0, 3/3) | 28.3 (+3.0, 3/3) | 26.0 (+0.6, 3/3) | 25.2 (-0.1, 1/3) | 25.7 (+0.3, 3/3) | 26.1 (+0.8, 2/3) | 27.2 (+1.9, 3/3) |
| yti_cpr | 22.4 (n=3) | 24.2 (+1.8, 2/3) | 24.3 (+1.9, 2/3) | 21.8 (-0.6, 1/3) | 21.9 (-0.5, 1/3) | 22.7 (+0.2, 3/3) | 22.2 (-0.3, 1/3) | 22.0 (-0.5, 2/3) |
| yti_jump_car | 16.0 (n=3) | 17.3 (+1.3, 2/3) | 16.4 (+0.5, 2/3) | 15.7 (-0.3, 1/3) | 15.7 (-0.2, 1/3) | 16.0 (+0.0, 2/3) | 16.2 (+0.3, 2/3) | 15.7 (-0.3, 1/3) |
| yti_repot | 17.1 (n=3) | 19.5 (+2.5, 3/3) | 18.3 (+1.3, 2/3) | 18.1 (+1.1, 3/3) | 16.2 (-0.9, 0/3) | 17.2 (+0.1, 3/3) | 17.7 (+0.6, 3/3) | 18.4 (+1.3, 3/3) |
| **mean delta** |  | +1.3, 6/7 bench | +0.6, 5/7 bench | -0.1, 4/7 bench | -0.3, 1/7 bench | +0.4, 5/7 bench | +0.2, 5/7 bench | +0.2, 3/7 bench |

### Alignment: acc (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | vcreg | unif | koleo | ldreg | csls | koleo_res | tneg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 65.5 (n=3) | 63.4 (-1.7, 0/2) | 67.6 (+2.1, 3/3) | 66.9 (+1.4, 3/3) | 65.6 (+0.1, 2/3) | 64.7 (-0.8, 0/3) | 65.9 (+0.4, 3/3) | 65.7 (+0.2, 2/3) | 67.0 (+1.4, 3/3) |
| da | 83.0 (n=3) | 82.6 (-0.6, 0/2) | 84.1 (+1.2, 3/3) | 82.3 (-0.7, 1/3) | 84.0 (+1.0, 3/3) | 82.5 (-0.4, 1/3) | 83.1 (+0.2, 2/3) | 82.8 (-0.1, 1/3) | 83.8 (+0.8, 3/3) |
| yti_changing_tire | 53.0 (n=3) | 44.4 (-8.3, 0/2) | 53.2 (+0.3, 2/3) | 55.4 (+2.4, 2/3) | 52.8 (-0.2, 1/3) | 52.0 (-1.0, 0/3) | 52.8 (-0.1, 1/3) | 52.9 (-0.0, 2/3) | 55.1 (+2.1, 2/3) |
| yti_coffee | 61.6 (n=3) | 58.1 (-4.8, 0/2) | 63.6 (+2.0, 3/3) | 63.7 (+2.2, 2/3) | 62.5 (+0.9, 2/3) | 60.8 (-0.7, 0/3) | 61.8 (+0.2, 2/3) | 62.4 (+0.8, 2/3) | 62.7 (+1.1, 3/3) |
| yti_cpr | 65.8 (n=3) | 70.5 (+7.1, 2/2) | 69.1 (+3.4, 3/3) | 73.2 (+7.4, 3/3) | 67.0 (+1.3, 2/3) | 69.5 (+3.8, 2/3) | 65.7 (-0.0, 1/3) | 67.6 (+1.8, 2/3) | 67.9 (+2.2, 1/3) |
| yti_jump_car | 34.4 (n=3) | 30.9 (-3.5, 0/2) | 33.7 (-0.7, 2/3) | 34.0 (-0.4, 2/3) | 33.6 (-0.8, 0/3) | 36.1 (+1.7, 2/3) | 34.4 (+0.0, 2/3) | 34.0 (-0.4, 1/3) | 34.9 (+0.5, 1/3) |
| yti_repot | 45.5 (n=3) | 42.3 (-3.8, 0/2) | 42.8 (-2.7, 0/3) | 43.3 (-2.3, 0/3) | 44.0 (-1.6, 1/3) | 44.8 (-0.7, 1/3) | 45.5 (-0.1, 2/3) | 43.2 (-2.4, 0/3) | 46.6 (+1.1, 2/3) |
| **mean delta** |  | -2.2, 1/7 bench | +0.8, 5/7 bench | +1.4, 4/7 bench | +0.1, 4/7 bench | +0.3, 2/7 bench | +0.1, 4/7 bench | -0.0, 3/7 bench | +1.3, 7/7 bench |

### Alignment: tau (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | vcreg | unif | koleo | ldreg | csls | koleo_res | tneg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 31.9 (n=3) | 7.8 (-24.4, 0/2) | 29.0 (-2.9, 0/3) | 27.4 (-4.5, 0/3) | 31.9 (+0.0, 2/3) | 32.3 (+0.4, 3/3) | 32.6 (+0.7, 3/3) | 32.0 (+0.1, 2/3) | 30.7 (-1.2, 0/3) |
| da | 89.5 (n=3) | 38.8 (-50.5, 0/2) | 88.6 (-0.9, 1/3) | 86.5 (-3.0, 0/3) | 88.8 (-0.7, 1/3) | 89.0 (-0.5, 1/3) | 89.6 (+0.1, 1/3) | 89.2 (-0.3, 1/3) | 91.3 (+1.8, 3/3) |
| yti_changing_tire | 64.8 (n=3) | 13.1 (-49.3, 0/2) | 54.4 (-10.4, 0/3) | 57.2 (-7.5, 0/3) | 63.9 (-0.9, 0/3) | 65.4 (+0.6, 2/3) | 64.8 (-0.0, 2/3) | 63.9 (-0.9, 0/3) | 64.2 (-0.6, 1/3) |
| yti_coffee | 71.0 (n=3) | 12.6 (-58.1, 0/2) | 62.1 (-8.9, 0/3) | 62.5 (-8.4, 0/3) | 69.0 (-2.0, 0/3) | 70.7 (-0.3, 1/3) | 70.9 (-0.1, 1/3) | 69.7 (-1.2, 0/3) | 70.6 (-0.3, 1/3) |
| yti_cpr | 51.8 (n=3) | 9.5 (-42.2, 0/2) | 40.2 (-11.6, 0/3) | 41.4 (-10.4, 0/3) | 51.8 (+0.0, 1/3) | 52.8 (+1.1, 2/3) | 51.6 (-0.2, 0/3) | 51.8 (+0.0, 2/3) | 48.4 (-3.4, 0/3) |
| yti_jump_car | 56.2 (n=3) | 3.7 (-51.5, 0/2) | 41.6 (-14.6, 0/3) | 46.1 (-10.1, 0/3) | 54.4 (-1.9, 1/3) | 57.7 (+1.5, 3/3) | 56.8 (+0.5, 3/3) | 54.2 (-2.1, 0/3) | 55.1 (-1.1, 1/3) |
| yti_repot | 68.3 (n=3) | 5.3 (-63.9, 0/2) | 53.3 (-15.0, 0/3) | 55.1 (-13.3, 0/3) | 65.0 (-3.3, 0/3) | 70.9 (+2.6, 2/3) | 68.6 (+0.3, 3/3) | 65.9 (-2.4, 0/3) | 64.5 (-3.8, 0/3) |
| **mean delta** |  | -48.5, 0/7 bench | -9.2, 0/7 bench | -8.2, 0/7 bench | -1.3, 2/7 bench | +0.8, 5/7 bench | +0.2, 4/7 bench | -1.0, 2/7 bench | -1.2, 1/7 bench |

### Alignment: ap@5 (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | vcreg | unif | koleo | ldreg | csls | koleo_res | tneg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 55.5 (n=3) | 49.9 (-5.2, 0/2) | 57.6 (+2.1, 3/3) | 56.6 (+1.1, 3/3) | 55.9 (+0.4, 3/3) | 54.9 (-0.7, 0/3) | 56.4 (+0.9, 3/3) | 55.9 (+0.4, 3/3) | 56.9 (+1.4, 3/3) |
| da | 77.8 (n=3) | 79.1 (+1.1, 2/2) | 79.3 (+1.5, 3/3) | 77.1 (-0.7, 1/3) | 79.0 (+1.2, 3/3) | 77.1 (-0.7, 1/3) | 78.0 (+0.2, 2/3) | 77.6 (-0.2, 1/3) | 79.0 (+1.2, 3/3) |
| yti_changing_tire | 49.0 (n=3) | 36.3 (-12.4, 0/2) | 50.4 (+1.4, 2/3) | 51.9 (+2.9, 3/3) | 49.7 (+0.7, 2/3) | 47.3 (-1.7, 1/3) | 49.8 (+0.8, 3/3) | 48.4 (-0.6, 1/3) | 52.2 (+3.2, 3/3) |
| yti_coffee | 56.1 (n=3) | 44.5 (-12.3, 0/2) | 58.7 (+2.6, 3/3) | 59.3 (+3.2, 3/3) | 56.5 (+0.4, 2/3) | 55.9 (-0.2, 1/3) | 56.1 (+0.0, 1/3) | 57.1 (+1.0, 3/3) | 57.6 (+1.5, 3/3) |
| yti_cpr | 59.0 (n=3) | 61.7 (+3.0, 2/2) | 62.8 (+3.8, 3/3) | 63.6 (+4.6, 3/3) | 59.1 (+0.1, 1/3) | 58.9 (-0.1, 1/3) | 59.4 (+0.4, 2/3) | 58.9 (-0.1, 1/3) | 60.9 (+2.0, 2/3) |
| yti_jump_car | 33.9 (n=3) | 26.4 (-7.8, 0/2) | 30.3 (-3.5, 0/3) | 32.0 (-1.9, 0/3) | 31.6 (-2.3, 0/3) | 32.8 (-1.0, 1/3) | 34.3 (+0.5, 3/3) | 32.6 (-1.2, 0/3) | 32.5 (-1.4, 0/3) |
| yti_repot | 38.2 (n=3) | 33.8 (-4.2, 0/2) | 38.1 (-0.1, 1/3) | 38.9 (+0.7, 2/3) | 40.1 (+2.0, 2/3) | 35.8 (-2.3, 0/3) | 38.3 (+0.2, 2/3) | 38.1 (-0.1, 2/3) | 41.6 (+3.4, 3/3) |
| **mean delta** |  | -5.4, 2/7 bench | +1.1, 5/7 bench | +1.4, 5/7 bench | +0.3, 6/7 bench | -1.0, 0/7 bench | +0.4, 7/7 bench | -0.1, 2/7 bench | +1.6, 6/7 bench |

### Alignment: ap@10 (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | vcreg | unif | koleo | ldreg | csls | koleo_res | tneg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 54.7 (n=3) | 47.7 (-6.5, 0/2) | 56.9 (+2.2, 3/3) | 55.7 (+1.0, 3/3) | 55.1 (+0.4, 3/3) | 54.0 (-0.7, 0/3) | 55.7 (+1.0, 3/3) | 55.2 (+0.4, 3/3) | 56.3 (+1.6, 3/3) |
| da | 77.0 (n=3) | 75.6 (-1.6, 0/2) | 78.5 (+1.5, 3/3) | 76.2 (-0.8, 1/3) | 78.1 (+1.2, 3/3) | 76.3 (-0.7, 1/3) | 77.2 (+0.3, 2/3) | 76.8 (-0.2, 1/3) | 78.2 (+1.3, 3/3) |
| yti_changing_tire | 48.3 (n=3) | 35.0 (-12.9, 0/2) | 49.6 (+1.4, 2/3) | 51.3 (+3.0, 3/3) | 49.2 (+0.9, 2/3) | 46.9 (-1.4, 1/3) | 49.1 (+0.9, 3/3) | 47.7 (-0.6, 1/3) | 51.6 (+3.3, 3/3) |
| yti_coffee | 55.7 (n=3) | 43.5 (-13.0, 0/2) | 58.4 (+2.7, 3/3) | 58.6 (+2.9, 3/3) | 56.1 (+0.4, 2/3) | 55.8 (+0.1, 2/3) | 55.9 (+0.2, 3/3) | 56.6 (+0.9, 3/3) | 57.2 (+1.5, 3/3) |
| yti_cpr | 58.6 (n=3) | 60.8 (+2.5, 2/2) | 62.7 (+4.1, 3/3) | 63.1 (+4.6, 3/3) | 58.5 (-0.0, 1/3) | 58.3 (-0.3, 1/3) | 59.0 (+0.5, 3/3) | 58.3 (-0.3, 1/3) | 61.0 (+2.4, 2/3) |
| yti_jump_car | 33.2 (n=3) | 24.8 (-8.8, 0/2) | 30.1 (-3.1, 0/3) | 31.6 (-1.6, 0/3) | 31.2 (-2.0, 0/3) | 32.3 (-0.9, 1/3) | 33.5 (+0.4, 3/3) | 32.3 (-0.9, 0/3) | 31.9 (-1.2, 0/3) |
| yti_repot | 37.8 (n=3) | 33.3 (-4.2, 0/2) | 37.7 (-0.1, 1/3) | 38.5 (+0.7, 2/3) | 39.5 (+1.7, 1/3) | 35.5 (-2.3, 0/3) | 37.9 (+0.2, 3/3) | 37.6 (-0.2, 1/3) | 41.0 (+3.2, 3/3) |
| **mean delta** |  | -6.3, 1/7 bench | +1.2, 5/7 bench | +1.4, 5/7 bench | +0.4, 5/7 bench | -0.9, 1/7 bench | +0.5, 7/7 bench | -0.1, 2/7 bench | +1.7, 6/7 bench |

### Alignment: ap@15 (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | vcreg | unif | koleo | ldreg | csls | koleo_res | tneg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 54.4 (n=3) | 46.6 (-7.2, 0/2) | 56.4 (+2.1, 3/3) | 55.2 (+0.8, 2/3) | 54.7 (+0.3, 2/3) | 53.6 (-0.7, 0/3) | 55.3 (+1.0, 3/3) | 54.7 (+0.4, 3/3) | 55.9 (+1.5, 3/3) |
| da | 76.4 (n=3) | 72.5 (-4.1, 0/2) | 77.8 (+1.4, 3/3) | 75.5 (-0.9, 1/3) | 77.5 (+1.1, 3/3) | 75.8 (-0.7, 1/3) | 76.7 (+0.3, 2/3) | 76.2 (-0.3, 1/3) | 77.7 (+1.3, 3/3) |
| yti_changing_tire | 47.7 (n=3) | 33.9 (-13.5, 0/2) | 49.1 (+1.4, 2/3) | 50.7 (+3.0, 3/3) | 48.9 (+1.1, 2/3) | 46.4 (-1.3, 1/3) | 48.7 (+0.9, 3/3) | 47.2 (-0.6, 1/3) | 51.1 (+3.4, 3/3) |
| yti_coffee | 55.5 (n=3) | 42.8 (-13.4, 0/2) | 58.0 (+2.5, 3/3) | 58.2 (+2.7, 3/3) | 55.9 (+0.4, 2/3) | 55.4 (-0.1, 2/3) | 55.8 (+0.3, 3/3) | 56.2 (+0.7, 3/3) | 56.9 (+1.4, 3/3) |
| yti_cpr | 58.3 (n=3) | 60.4 (+2.4, 2/2) | 62.4 (+4.2, 3/3) | 62.8 (+4.5, 3/3) | 58.2 (-0.0, 1/3) | 57.9 (-0.4, 1/3) | 58.7 (+0.4, 2/3) | 58.0 (-0.2, 1/3) | 60.8 (+2.5, 2/3) |
| yti_jump_car | 32.7 (n=3) | 23.7 (-9.5, 0/2) | 29.9 (-2.7, 0/3) | 31.2 (-1.5, 0/3) | 30.5 (-2.1, 0/3) | 32.0 (-0.7, 0/3) | 33.0 (+0.3, 3/3) | 31.8 (-0.8, 0/3) | 31.4 (-1.2, 0/3) |
| yti_repot | 37.5 (n=3) | 32.8 (-4.2, 0/2) | 37.6 (+0.1, 1/3) | 38.2 (+0.8, 2/3) | 39.3 (+1.9, 2/3) | 35.4 (-2.1, 0/3) | 37.7 (+0.2, 3/3) | 37.4 (-0.1, 1/3) | 40.6 (+3.2, 3/3) |
| **mean delta** |  | -7.1, 1/7 bench | +1.3, 6/7 bench | +1.3, 5/7 bench | +0.4, 5/7 bench | -0.9, 0/7 bench | +0.5, 7/7 bench | -0.1, 2/7 bench | +1.7, 6/7 bench |

## B. Combinations (§5)

### Segmentation: MoF (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | tneg_blur6 | csls_local | csls_blur6_local | vcreg1 | vcreg_blur6 | ldreg_local |
|---|---:|---:|---:|---:|---:|---:|---:|
| fseval | 49.1 (n=3) | 46.4 (-2.7, 0/3) | 49.1 (+0.0, 1/3) | 48.2 (-0.8, 1/3) | 49.9 (+0.8, 2/3) | 47.4 (-1.7, 1/3) | 48.6 (-0.5, 2/3) |
| da | 60.0 (n=3) | 63.1 (+3.1, 2/3) | 64.1 (+4.1, 3/3) | 62.3 (+2.3, 3/3) | 64.0 (+3.9, 3/3) | 59.8 (-0.3, 2/3) | 60.4 (+0.4, 1/3) |
| yti_changing_tire | 48.6 (n=3) | 47.5 (-1.1, 0/3) | 48.3 (-0.3, 1/3) | 48.4 (-0.2, 0/3) | 45.9 (-2.7, 0/3) | 46.4 (-2.2, 1/3) | 48.2 (-0.4, 1/3) |
| yti_coffee | 56.4 (n=3) | 57.9 (+1.5, 1/3) | 57.6 (+1.2, 2/3) | 56.8 (+0.4, 2/3) | 59.2 (+2.8, 3/3) | 62.3 (+6.0, 3/3) | 56.1 (-0.3, 1/3) |
| yti_cpr | 47.6 (n=3) | 48.6 (+1.0, 1/3) | 47.8 (+0.3, 2/3) | 47.1 (-0.5, 1/3) | 51.4 (+3.8, 2/3) | 53.1 (+5.5, 3/3) | 49.0 (+1.5, 1/3) |
| yti_jump_car | 33.8 (n=3) | 33.1 (-0.7, 1/3) | 34.1 (+0.3, 3/3) | 34.3 (+0.5, 2/3) | 34.5 (+0.7, 2/3) | 34.4 (+0.6, 2/3) | 34.1 (+0.3, 1/3) |
| yti_repot | 39.2 (n=3) | 42.3 (+3.1, 3/3) | 39.5 (+0.3, 3/3) | 39.4 (+0.1, 2/3) | 44.0 (+4.8, 3/3) | 46.5 (+7.3, 3/3) | 37.9 (-1.3, 1/3) |
| **mean delta** |  | +0.6, 4/7 bench | +0.8, 6/7 bench | +0.3, 4/7 bench | +2.0, 6/7 bench | +2.2, 4/7 bench | -0.1, 3/7 bench |

### Segmentation: F1 (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | tneg_blur6 | csls_local | csls_blur6_local | vcreg1 | vcreg_blur6 | ldreg_local |
|---|---:|---:|---:|---:|---:|---:|---:|
| fseval | 51.4 (n=3) | 48.9 (-2.5, 1/3) | 50.8 (-0.6, 2/3) | 51.5 (+0.1, 1/3) | 51.0 (-0.4, 2/3) | 47.9 (-3.4, 1/3) | 51.3 (-0.1, 2/3) |
| da | 64.3 (n=3) | 66.9 (+2.6, 2/3) | 68.0 (+3.7, 2/3) | 66.9 (+2.6, 3/3) | 66.4 (+2.1, 2/3) | 63.3 (-1.0, 1/3) | 65.3 (+0.9, 1/3) |
| yti_changing_tire | 31.2 (n=3) | 31.2 (+0.0, 1/3) | 30.5 (-0.7, 1/3) | 31.0 (-0.2, 2/3) | 30.0 (-1.2, 1/3) | 31.0 (-0.2, 1/3) | 31.6 (+0.4, 2/3) |
| yti_coffee | 40.3 (n=3) | 41.0 (+0.6, 2/3) | 41.0 (+0.6, 2/3) | 40.6 (+0.2, 2/3) | 41.0 (+0.6, 1/3) | 41.6 (+1.2, 2/3) | 38.9 (-1.4, 0/3) |
| yti_cpr | 43.2 (n=3) | 43.5 (+0.4, 1/3) | 43.5 (+0.4, 2/3) | 44.7 (+1.6, 2/3) | 45.9 (+2.7, 2/3) | 46.5 (+3.3, 3/3) | 45.4 (+2.2, 1/3) |
| yti_jump_car | 26.2 (n=3) | 25.9 (-0.3, 1/3) | 26.6 (+0.4, 3/3) | 26.6 (+0.5, 2/3) | 26.7 (+0.5, 2/3) | 26.2 (-0.0, 2/3) | 27.5 (+1.3, 3/3) |
| yti_repot | 25.6 (n=3) | 26.7 (+1.1, 2/3) | 26.3 (+0.7, 3/3) | 26.3 (+0.7, 3/3) | 26.3 (+0.7, 1/3) | 26.0 (+0.4, 2/3) | 25.5 (-0.0, 1/3) |
| **mean delta** |  | +0.3, 5/7 bench | +0.7, 5/7 bench | +0.8, 6/7 bench | +0.7, 5/7 bench | +0.0, 3/7 bench | +0.5, 4/7 bench |

### Segmentation: mIoU (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | tneg_blur6 | csls_local | csls_blur6_local | vcreg1 | vcreg_blur6 | ldreg_local |
|---|---:|---:|---:|---:|---:|---:|---:|
| fseval | 28.6 (n=3) | 24.4 (-4.1, 0/3) | 27.9 (-0.7, 1/3) | 26.7 (-1.8, 1/3) | 28.9 (+0.4, 2/3) | 26.0 (-2.5, 1/3) | 27.3 (-1.2, 2/3) |
| da | 41.3 (n=3) | 43.2 (+1.9, 2/3) | 45.0 (+3.7, 3/3) | 42.9 (+1.6, 3/3) | 44.4 (+3.1, 3/3) | 40.5 (-0.8, 1/3) | 41.9 (+0.6, 1/3) |
| yti_changing_tire | 22.7 (n=3) | 22.2 (-0.5, 0/3) | 22.4 (-0.3, 2/3) | 22.6 (-0.1, 1/3) | 21.2 (-1.5, 0/3) | 22.7 (+0.0, 2/3) | 22.2 (-0.5, 0/3) |
| yti_coffee | 25.3 (n=3) | 26.2 (+0.9, 3/3) | 26.2 (+0.9, 2/3) | 25.9 (+0.6, 2/3) | 26.9 (+1.6, 3/3) | 27.9 (+2.6, 3/3) | 24.6 (-0.7, 0/3) |
| yti_cpr | 22.4 (n=3) | 21.9 (-0.5, 1/3) | 22.9 (+0.5, 3/3) | 22.8 (+0.4, 3/3) | 23.1 (+0.6, 2/3) | 23.0 (+0.6, 2/3) | 21.8 (-0.6, 1/3) |
| yti_jump_car | 16.0 (n=3) | 15.4 (-0.5, 0/3) | 16.3 (+0.3, 3/3) | 16.2 (+0.3, 2/3) | 16.4 (+0.4, 2/3) | 17.2 (+1.2, 2/3) | 15.6 (-0.4, 1/3) |
| yti_repot | 17.1 (n=3) | 18.3 (+1.2, 3/3) | 17.4 (+0.3, 3/3) | 17.3 (+0.2, 3/3) | 19.2 (+2.2, 3/3) | 19.9 (+2.8, 3/3) | 15.9 (-1.2, 0/3) |
| **mean delta** |  | -0.2, 3/7 bench | +0.7, 5/7 bench | +0.2, 5/7 bench | +1.0, 6/7 bench | +0.6, 5/7 bench | -0.6, 1/7 bench |

### Alignment: acc (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | tneg_blur6 | csls_local | csls_blur6_local | vcreg1 | vcreg_blur6 | ldreg_local |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 65.5 (n=3) | 63.4 (-1.7, 0/2) | 66.2 (+0.7, 3/3) | 66.3 (+0.8, 3/3) | 65.6 (+0.1, 1/3) | 67.3 (+1.8, 3/3) | 65.9 (+0.4, 2/3) | 65.2 (-0.3, 1/3) |
| da | 83.0 (n=3) | 82.6 (-0.6, 0/2) | 82.9 (-0.1, 2/3) | 83.4 (+0.4, 2/3) | 83.2 (+0.3, 1/3) | 84.0 (+1.1, 3/3) | 83.8 (+0.8, 2/3) | 82.5 (-0.4, 1/3) |
| yti_changing_tire | 53.0 (n=3) | 44.4 (-8.3, 0/2) | 56.0 (+3.1, 3/3) | 53.6 (+0.6, 2/3) | 55.2 (+2.3, 3/3) | 53.8 (+0.9, 3/3) | 54.9 (+2.0, 3/3) | 53.8 (+0.9, 2/3) |
| yti_coffee | 61.6 (n=3) | 58.1 (-4.8, 0/2) | 61.2 (-0.4, 1/3) | 62.8 (+1.3, 3/3) | 62.0 (+0.4, 2/3) | 63.1 (+1.5, 3/3) | 62.9 (+1.3, 3/3) | 62.5 (+0.9, 3/3) |
| yti_cpr | 65.8 (n=3) | 70.5 (+7.1, 2/2) | 67.6 (+1.9, 1/3) | 65.3 (-0.4, 1/3) | 67.9 (+2.2, 2/3) | 69.9 (+4.1, 3/3) | 68.4 (+2.6, 3/3) | 68.6 (+2.9, 2/3) |
| yti_jump_car | 34.4 (n=3) | 30.9 (-3.5, 0/2) | 35.9 (+1.5, 3/3) | 35.7 (+1.3, 3/3) | 35.6 (+1.2, 2/3) | 35.7 (+1.3, 2/3) | 35.0 (+0.6, 1/3) | 37.1 (+2.7, 2/3) |
| yti_repot | 45.5 (n=3) | 42.3 (-3.8, 0/2) | 46.2 (+0.6, 2/3) | 44.1 (-1.5, 1/3) | 44.9 (-0.7, 0/3) | 42.8 (-2.7, 1/3) | 42.8 (-2.8, 1/3) | 45.1 (-0.4, 1/3) |
| **mean delta** |  | -2.2, 1/7 bench | +1.0, 5/7 bench | +0.4, 5/7 bench | +0.8, 6/7 bench | +1.1, 6/7 bench | +0.7, 6/7 bench | +0.9, 4/7 bench |

### Alignment: tau (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | tneg_blur6 | csls_local | csls_blur6_local | vcreg1 | vcreg_blur6 | ldreg_local |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 31.9 (n=3) | 7.8 (-24.4, 0/2) | 31.9 (+0.0, 2/3) | 32.3 (+0.5, 2/3) | 32.8 (+1.0, 3/3) | 30.6 (-1.2, 0/3) | 30.0 (-1.9, 0/3) | 31.9 (+0.0, 2/3) |
| da | 89.5 (n=3) | 38.8 (-50.5, 0/2) | 90.2 (+0.7, 2/3) | 89.6 (+0.1, 1/3) | 91.3 (+1.8, 3/3) | 89.2 (-0.3, 1/3) | 89.4 (-0.1, 1/3) | 88.9 (-0.6, 1/3) |
| yti_changing_tire | 64.8 (n=3) | 13.1 (-49.3, 0/2) | 66.5 (+1.7, 3/3) | 66.2 (+1.4, 3/3) | 69.1 (+4.4, 3/3) | 61.9 (-2.9, 0/3) | 57.9 (-6.9, 0/3) | 67.0 (+2.3, 2/3) |
| yti_coffee | 71.0 (n=3) | 12.6 (-58.1, 0/2) | 72.1 (+1.1, 3/3) | 71.4 (+0.4, 2/3) | 72.6 (+1.6, 3/3) | 70.3 (-0.7, 0/3) | 66.3 (-4.7, 0/3) | 71.4 (+0.4, 2/3) |
| yti_cpr | 51.8 (n=3) | 9.5 (-42.2, 0/2) | 51.6 (-0.2, 1/3) | 51.1 (-0.7, 0/3) | 52.9 (+1.2, 2/3) | 46.2 (-5.6, 0/3) | 43.7 (-8.1, 0/3) | 53.6 (+1.8, 2/3) |
| yti_jump_car | 56.2 (n=3) | 3.7 (-51.5, 0/2) | 57.0 (+0.7, 2/3) | 56.4 (+0.2, 3/3) | 59.2 (+2.9, 3/3) | 53.1 (-3.1, 0/3) | 48.2 (-8.0, 0/3) | 57.4 (+1.2, 2/3) |
| yti_repot | 68.3 (n=3) | 5.3 (-63.9, 0/2) | 66.8 (-1.6, 0/3) | 69.4 (+1.0, 3/3) | 71.0 (+2.7, 3/3) | 61.3 (-7.1, 0/3) | 57.0 (-11.4, 0/3) | 72.3 (+4.0, 3/3) |
| **mean delta** |  | -48.5, 0/7 bench | +0.4, 5/7 bench | +0.4, 6/7 bench | +2.2, 7/7 bench | -3.0, 0/7 bench | -5.9, 0/7 bench | +1.3, 6/7 bench |

### Alignment: ap@5 (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | tneg_blur6 | csls_local | csls_blur6_local | vcreg1 | vcreg_blur6 | ldreg_local |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 55.5 (n=3) | 49.9 (-5.2, 0/2) | 56.0 (+0.5, 3/3) | 56.9 (+1.4, 3/3) | 56.0 (+0.4, 3/3) | 57.6 (+2.1, 3/3) | 55.4 (-0.2, 1/3) | 55.5 (-0.1, 2/3) |
| da | 77.8 (n=3) | 79.1 (+1.1, 2/2) | 77.9 (+0.1, 1/3) | 78.7 (+0.9, 3/3) | 79.0 (+1.2, 3/3) | 78.9 (+1.1, 2/3) | 79.7 (+1.9, 3/3) | 77.2 (-0.6, 1/3) |
| yti_changing_tire | 49.0 (n=3) | 36.3 (-12.4, 0/2) | 52.6 (+3.5, 3/3) | 50.7 (+1.7, 3/3) | 51.3 (+2.2, 3/3) | 50.8 (+1.8, 3/3) | 51.1 (+2.1, 3/3) | 49.3 (+0.3, 1/3) |
| yti_coffee | 56.1 (n=3) | 44.5 (-12.3, 0/2) | 56.9 (+0.8, 1/3) | 57.9 (+1.8, 3/3) | 57.6 (+1.4, 3/3) | 57.7 (+1.6, 3/3) | 58.2 (+2.1, 3/3) | 58.4 (+2.3, 3/3) |
| yti_cpr | 59.0 (n=3) | 61.7 (+3.0, 2/2) | 60.7 (+1.8, 3/3) | 59.8 (+0.9, 3/3) | 59.9 (+1.0, 3/3) | 62.5 (+3.5, 2/3) | 61.9 (+2.9, 3/3) | 59.7 (+0.8, 1/3) |
| yti_jump_car | 33.9 (n=3) | 26.4 (-7.8, 0/2) | 33.9 (+0.0, 2/3) | 35.8 (+1.9, 3/3) | 35.3 (+1.5, 3/3) | 33.3 (-0.6, 0/3) | 31.2 (-2.7, 0/3) | 33.7 (-0.2, 2/3) |
| yti_repot | 38.2 (n=3) | 33.8 (-4.2, 0/2) | 41.2 (+3.0, 3/3) | 38.3 (+0.1, 2/3) | 38.3 (+0.1, 2/3) | 39.9 (+1.7, 2/3) | 40.0 (+1.8, 3/3) | 36.6 (-1.6, 0/3) |
| **mean delta** |  | -5.4, 2/7 bench | +1.4, 7/7 bench | +1.2, 7/7 bench | +1.1, 7/7 bench | +1.6, 6/7 bench | +1.1, 5/7 bench | +0.1, 3/7 bench |

### Alignment: ap@10 (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | tneg_blur6 | csls_local | csls_blur6_local | vcreg1 | vcreg_blur6 | ldreg_local |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 54.7 (n=3) | 47.7 (-6.5, 0/2) | 55.3 (+0.5, 3/3) | 56.2 (+1.5, 3/3) | 55.4 (+0.7, 3/3) | 57.0 (+2.3, 3/3) | 54.4 (-0.3, 1/3) | 54.8 (+0.1, 2/3) |
| da | 77.0 (n=3) | 75.6 (-1.6, 0/2) | 76.9 (-0.1, 1/3) | 77.9 (+0.9, 3/3) | 78.0 (+1.0, 3/3) | 78.2 (+1.2, 3/3) | 78.5 (+1.5, 3/3) | 76.5 (-0.5, 1/3) |
| yti_changing_tire | 48.3 (n=3) | 35.0 (-12.9, 0/2) | 51.9 (+3.6, 3/3) | 50.0 (+1.7, 3/3) | 50.5 (+2.3, 3/3) | 50.1 (+1.8, 3/3) | 50.5 (+2.2, 3/3) | 48.6 (+0.3, 1/3) |
| yti_coffee | 55.7 (n=3) | 43.5 (-13.0, 0/2) | 56.6 (+0.9, 2/3) | 57.4 (+1.8, 3/3) | 57.1 (+1.4, 3/3) | 57.5 (+1.8, 3/3) | 57.8 (+2.1, 3/3) | 57.8 (+2.1, 3/3) |
| yti_cpr | 58.6 (n=3) | 60.8 (+2.5, 2/2) | 60.3 (+1.8, 2/3) | 59.4 (+0.8, 3/3) | 59.6 (+1.0, 3/3) | 62.2 (+3.6, 3/3) | 61.8 (+3.2, 3/3) | 59.2 (+0.6, 1/3) |
| yti_jump_car | 33.2 (n=3) | 24.8 (-8.8, 0/2) | 33.1 (-0.1, 1/3) | 34.6 (+1.5, 3/3) | 34.5 (+1.4, 3/3) | 32.8 (-0.3, 1/3) | 30.8 (-2.3, 0/3) | 33.3 (+0.1, 2/3) |
| yti_repot | 37.8 (n=3) | 33.3 (-4.2, 0/2) | 40.7 (+3.0, 3/3) | 37.9 (+0.1, 2/3) | 37.8 (+0.0, 2/3) | 39.3 (+1.5, 2/3) | 39.6 (+1.8, 2/3) | 36.2 (-1.5, 0/3) |
| **mean delta** |  | -6.3, 1/7 bench | +1.4, 5/7 bench | +1.2, 7/7 bench | +1.1, 7/7 bench | +1.7, 6/7 bench | +1.2, 5/7 bench | +0.2, 5/7 bench |

### Alignment: ap@15 (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | tneg_blur6 | csls_local | csls_blur6_local | vcreg1 | vcreg_blur6 | ldreg_local |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 54.4 (n=3) | 46.6 (-7.2, 0/2) | 54.8 (+0.5, 3/3) | 55.8 (+1.5, 3/3) | 55.0 (+0.7, 3/3) | 56.6 (+2.2, 3/3) | 53.7 (-0.6, 0/3) | 54.3 (-0.0, 2/3) |
| da | 76.4 (n=3) | 72.5 (-4.1, 0/2) | 76.2 (-0.2, 1/3) | 77.4 (+1.0, 3/3) | 77.3 (+0.9, 3/3) | 77.6 (+1.2, 3/3) | 77.6 (+1.2, 3/3) | 75.9 (-0.5, 1/3) |
| yti_changing_tire | 47.7 (n=3) | 33.9 (-13.5, 0/2) | 51.4 (+3.6, 3/3) | 49.5 (+1.7, 3/3) | 50.0 (+2.3, 3/3) | 49.6 (+1.8, 3/3) | 50.1 (+2.3, 3/3) | 48.1 (+0.4, 1/3) |
| yti_coffee | 55.5 (n=3) | 42.8 (-13.4, 0/2) | 56.2 (+0.7, 2/3) | 57.1 (+1.6, 3/3) | 56.8 (+1.3, 3/3) | 57.1 (+1.6, 3/3) | 57.5 (+2.0, 3/3) | 57.5 (+2.0, 3/3) |
| yti_cpr | 58.3 (n=3) | 60.4 (+2.4, 2/2) | 60.2 (+1.9, 3/3) | 59.0 (+0.8, 3/3) | 59.0 (+0.7, 3/3) | 61.9 (+3.6, 2/3) | 61.6 (+3.3, 3/3) | 58.6 (+0.3, 1/3) |
| yti_jump_car | 32.7 (n=3) | 23.7 (-9.5, 0/2) | 32.5 (-0.2, 1/3) | 33.9 (+1.2, 3/3) | 33.9 (+1.3, 3/3) | 32.4 (-0.3, 1/3) | 30.8 (-1.9, 1/3) | 32.7 (+0.0, 1/3) |
| yti_repot | 37.5 (n=3) | 32.8 (-4.2, 0/2) | 40.6 (+3.1, 3/3) | 37.6 (+0.2, 2/3) | 37.6 (+0.1, 2/3) | 39.1 (+1.7, 2/3) | 39.3 (+1.8, 2/3) | 35.9 (-1.5, 0/3) |
| **mean delta** |  | -7.1, 1/7 bench | +1.4, 5/7 bench | +1.1, 7/7 bench | +1.0, 7/7 bench | +1.7, 6/7 bench | +1.2, 5/7 bench | +0.1, 4/7 bench |

## C. VICReg weight and combinations (§6)

### Segmentation: MoF (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | vcreg | vcreg1 | vcreg03 | vcreg1_local | vcreg1_csls_blur6_local | vcreg03_csls_blur6_local |
|---|---:|---:|---:|---:|---:|---:|---:|
| fseval | 49.1 (n=3) | 49.1 (+0.0, 1/3) | 49.9 (+0.8, 2/3) | 48.7 (-0.3, 1/3) | 49.3 (+0.3, 2/3) | 48.5 (-0.6, 1/3) | 47.7 (-1.3, 1/3) |
| da | 60.0 (n=3) | 60.3 (+0.3, 1/3) | 64.0 (+3.9, 3/3) | 59.6 (-0.4, 2/3) | 63.7 (+3.6, 3/3) | 60.8 (+0.8, 2/3) | 60.7 (+0.7, 2/3) |
| yti_changing_tire | 48.6 (n=3) | 45.2 (-3.4, 0/3) | 45.9 (-2.7, 0/3) | 47.7 (-0.9, 0/3) | 46.3 (-2.4, 0/3) | 46.9 (-1.7, 0/3) | 48.2 (-0.4, 1/3) |
| yti_coffee | 56.4 (n=3) | 62.5 (+6.1, 3/3) | 59.2 (+2.8, 3/3) | 57.9 (+1.5, 3/3) | 59.0 (+2.6, 3/3) | 58.4 (+2.1, 2/3) | 57.1 (+0.7, 1/3) |
| yti_cpr | 47.6 (n=3) | 54.8 (+7.2, 3/3) | 51.4 (+3.8, 2/3) | 47.5 (-0.0, 1/3) | 52.4 (+4.8, 2/3) | 50.5 (+2.9, 2/3) | 46.8 (-0.7, 1/3) |
| yti_jump_car | 33.8 (n=3) | 34.5 (+0.7, 2/3) | 34.5 (+0.7, 2/3) | 33.5 (-0.3, 1/3) | 34.5 (+0.7, 2/3) | 34.0 (+0.2, 2/3) | 34.0 (+0.2, 2/3) |
| yti_repot | 39.2 (n=3) | 46.0 (+6.8, 3/3) | 44.0 (+4.8, 3/3) | 40.7 (+1.5, 3/3) | 43.1 (+3.9, 3/3) | 43.0 (+3.8, 3/3) | 40.1 (+0.8, 3/3) |
| **mean delta** |  | +2.5, 6/7 bench | +2.0, 6/7 bench | +0.1, 2/7 bench | +1.9, 6/7 bench | +1.1, 5/7 bench | -0.0, 4/7 bench |

### Segmentation: F1 (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | vcreg | vcreg1 | vcreg03 | vcreg1_local | vcreg1_csls_blur6_local | vcreg03_csls_blur6_local |
|---|---:|---:|---:|---:|---:|---:|---:|
| fseval | 51.4 (n=3) | 50.8 (-0.6, 1/3) | 51.0 (-0.4, 2/3) | 50.2 (-1.2, 1/3) | 50.7 (-0.7, 2/3) | 49.6 (-1.8, 1/3) | 50.2 (-1.2, 2/3) |
| da | 64.3 (n=3) | 64.0 (-0.4, 1/3) | 66.4 (+2.1, 2/3) | 61.3 (-3.1, 0/3) | 66.2 (+1.9, 2/3) | 63.1 (-1.3, 2/3) | 63.9 (-0.4, 1/3) |
| yti_changing_tire | 31.2 (n=3) | 30.1 (-1.1, 1/3) | 30.0 (-1.2, 1/3) | 31.0 (-0.2, 1/3) | 30.0 (-1.1, 1/3) | 30.8 (-0.4, 1/3) | 30.5 (-0.7, 1/3) |
| yti_coffee | 40.3 (n=3) | 41.5 (+1.2, 3/3) | 41.0 (+0.6, 1/3) | 40.7 (+0.3, 2/3) | 41.9 (+1.6, 3/3) | 41.7 (+1.3, 2/3) | 40.8 (+0.4, 3/3) |
| yti_cpr | 43.2 (n=3) | 48.4 (+5.3, 3/3) | 45.9 (+2.7, 2/3) | 46.1 (+2.9, 2/3) | 46.3 (+3.1, 2/3) | 45.2 (+2.0, 2/3) | 43.6 (+0.4, 2/3) |
| yti_jump_car | 26.2 (n=3) | 26.4 (+0.2, 2/3) | 26.7 (+0.5, 2/3) | 25.8 (-0.4, 1/3) | 26.7 (+0.5, 2/3) | 26.3 (+0.2, 2/3) | 26.4 (+0.2, 2/3) |
| yti_repot | 25.6 (n=3) | 28.6 (+3.0, 3/3) | 26.3 (+0.7, 1/3) | 26.5 (+0.9, 2/3) | 28.1 (+2.5, 2/3) | 26.4 (+0.8, 2/3) | 26.3 (+0.7, 2/3) |
| **mean delta** |  | +1.1, 4/7 bench | +0.7, 5/7 bench | -0.1, 3/7 bench | +1.1, 5/7 bench | +0.1, 4/7 bench | -0.1, 4/7 bench |

### Segmentation: mIoU (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | vcreg | vcreg1 | vcreg03 | vcreg1_local | vcreg1_csls_blur6_local | vcreg03_csls_blur6_local |
|---|---:|---:|---:|---:|---:|---:|---:|
| fseval | 28.6 (n=3) | 29.2 (+0.6, 1/3) | 28.9 (+0.4, 2/3) | 27.4 (-1.1, 2/3) | 28.4 (-0.1, 2/3) | 26.3 (-2.3, 1/3) | 25.2 (-3.4, 1/3) |
| da | 41.3 (n=3) | 41.8 (+0.5, 2/3) | 44.4 (+3.1, 3/3) | 40.3 (-1.0, 0/3) | 44.5 (+3.2, 3/3) | 41.5 (+0.2, 2/3) | 41.1 (-0.2, 1/3) |
| yti_changing_tire | 22.7 (n=3) | 22.3 (-0.4, 2/3) | 21.2 (-1.5, 0/3) | 22.1 (-0.6, 1/3) | 21.0 (-1.7, 0/3) | 22.1 (-0.6, 0/3) | 22.0 (-0.6, 0/3) |
| yti_coffee | 25.3 (n=3) | 28.3 (+3.0, 3/3) | 26.9 (+1.6, 3/3) | 25.9 (+0.6, 3/3) | 27.1 (+1.8, 3/3) | 27.0 (+1.6, 3/3) | 25.9 (+0.6, 3/3) |
| yti_cpr | 22.4 (n=3) | 24.2 (+1.8, 2/3) | 23.1 (+0.6, 2/3) | 23.1 (+0.6, 3/3) | 22.8 (+0.4, 2/3) | 22.3 (-0.1, 2/3) | 22.5 (+0.0, 1/3) |
| yti_jump_car | 16.0 (n=3) | 17.3 (+1.3, 2/3) | 16.4 (+0.4, 2/3) | 15.6 (-0.3, 1/3) | 16.3 (+0.3, 2/3) | 15.9 (-0.0, 2/3) | 16.1 (+0.1, 2/3) |
| yti_repot | 17.1 (n=3) | 19.5 (+2.5, 3/3) | 19.2 (+2.2, 3/3) | 17.6 (+0.5, 3/3) | 18.6 (+1.5, 3/3) | 19.0 (+1.9, 3/3) | 17.5 (+0.5, 3/3) |
| **mean delta** |  | +1.3, 6/7 bench | +1.0, 6/7 bench | -0.2, 3/7 bench | +0.8, 5/7 bench | +0.1, 3/7 bench | -0.4, 4/7 bench |

### Alignment: acc (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | vcreg | vcreg1 | vcreg03 | vcreg1_local | vcreg1_csls_blur6_local | vcreg03_csls_blur6_local |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 65.5 (n=3) | 63.4 (-1.7, 0/2) | 67.6 (+2.1, 3/3) | 67.3 (+1.8, 3/3) | 66.2 (+0.7, 3/3) | 67.2 (+1.7, 3/3) | 66.2 (+0.7, 2/3) | 65.5 (-0.0, 1/3) |
| da | 83.0 (n=3) | 82.6 (-0.6, 0/2) | 84.1 (+1.2, 3/3) | 84.0 (+1.1, 3/3) | 83.2 (+0.3, 2/3) | 83.8 (+0.9, 3/3) | 84.3 (+1.3, 3/3) | 84.0 (+1.1, 3/3) |
| yti_changing_tire | 53.0 (n=3) | 44.4 (-8.3, 0/2) | 53.2 (+0.3, 2/3) | 53.8 (+0.9, 3/3) | 52.8 (-0.2, 1/3) | 55.1 (+2.1, 3/3) | 56.4 (+3.4, 3/3) | 54.9 (+2.0, 3/3) |
| yti_coffee | 61.6 (n=3) | 58.1 (-4.8, 0/2) | 63.6 (+2.0, 3/3) | 63.1 (+1.5, 3/3) | 62.1 (+0.5, 3/3) | 64.9 (+3.3, 3/3) | 64.8 (+3.2, 3/3) | 62.3 (+0.7, 2/3) |
| yti_cpr | 65.8 (n=3) | 70.5 (+7.1, 2/2) | 69.1 (+3.4, 3/3) | 69.9 (+4.1, 3/3) | 65.9 (+0.2, 1/3) | 69.2 (+3.4, 3/3) | 69.4 (+3.7, 3/3) | 65.1 (-0.6, 1/3) |
| yti_jump_car | 34.4 (n=3) | 30.9 (-3.5, 0/2) | 33.7 (-0.7, 2/3) | 35.7 (+1.3, 2/3) | 33.8 (-0.6, 0/3) | 36.0 (+1.6, 2/3) | 36.0 (+1.6, 2/3) | 34.7 (+0.3, 1/3) |
| yti_repot | 45.5 (n=3) | 42.3 (-3.8, 0/2) | 42.8 (-2.7, 0/3) | 42.8 (-2.7, 1/3) | 44.7 (-0.8, 1/3) | 43.8 (-1.7, 1/3) | 44.0 (-1.6, 1/3) | 44.5 (-1.1, 1/3) |
| **mean delta** |  | -2.2, 1/7 bench | +0.8, 5/7 bench | +1.1, 6/7 bench | +0.0, 4/7 bench | +1.6, 6/7 bench | +1.8, 6/7 bench | +0.3, 4/7 bench |

### Alignment: tau (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | vcreg | vcreg1 | vcreg03 | vcreg1_local | vcreg1_csls_blur6_local | vcreg03_csls_blur6_local |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 31.9 (n=3) | 7.8 (-24.4, 0/2) | 29.0 (-2.9, 0/3) | 30.6 (-1.2, 0/3) | 31.9 (+0.0, 2/3) | 30.7 (-1.2, 0/3) | 32.8 (+0.9, 3/3) | 33.3 (+1.4, 3/3) |
| da | 89.5 (n=3) | 38.8 (-50.5, 0/2) | 88.6 (-0.9, 1/3) | 89.2 (-0.3, 1/3) | 89.5 (-0.0, 1/3) | 89.5 (+0.0, 1/3) | 90.6 (+1.1, 3/3) | 91.6 (+2.1, 3/3) |
| yti_changing_tire | 64.8 (n=3) | 13.1 (-49.3, 0/2) | 54.4 (-10.4, 0/3) | 61.9 (-2.9, 0/3) | 63.3 (-1.5, 0/3) | 62.6 (-2.2, 0/3) | 66.0 (+1.2, 3/3) | 67.8 (+3.0, 3/3) |
| yti_coffee | 71.0 (n=3) | 12.6 (-58.1, 0/2) | 62.1 (-8.9, 0/3) | 70.3 (-0.7, 0/3) | 70.6 (-0.3, 1/3) | 70.7 (-0.3, 2/3) | 71.8 (+0.8, 3/3) | 72.2 (+1.2, 3/3) |
| yti_cpr | 51.8 (n=3) | 9.5 (-42.2, 0/2) | 40.2 (-11.6, 0/3) | 46.2 (-5.6, 0/3) | 50.8 (-0.9, 1/3) | 44.3 (-7.5, 0/3) | 50.0 (-1.8, 0/3) | 52.1 (+0.4, 3/3) |
| yti_jump_car | 56.2 (n=3) | 3.7 (-51.5, 0/2) | 41.6 (-14.6, 0/3) | 53.1 (-3.1, 0/3) | 56.0 (-0.2, 1/3) | 51.8 (-4.4, 0/3) | 54.8 (-1.4, 1/3) | 58.0 (+1.8, 3/3) |
| yti_repot | 68.3 (n=3) | 5.3 (-63.9, 0/2) | 53.3 (-15.0, 0/3) | 61.3 (-7.1, 0/3) | 67.3 (-1.0, 0/3) | 61.5 (-6.8, 0/3) | 64.9 (-3.4, 0/3) | 69.7 (+1.4, 2/3) |
| **mean delta** |  | -48.5, 0/7 bench | -9.2, 0/7 bench | -3.0, 0/7 bench | -0.6, 1/7 bench | -3.2, 1/7 bench | -0.4, 4/7 bench | +1.6, 7/7 bench |

### Alignment: ap@5 (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | vcreg | vcreg1 | vcreg03 | vcreg1_local | vcreg1_csls_blur6_local | vcreg03_csls_blur6_local |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 55.5 (n=3) | 49.9 (-5.2, 0/2) | 57.6 (+2.1, 3/3) | 57.6 (+2.1, 3/3) | 56.2 (+0.7, 3/3) | 57.9 (+2.4, 3/3) | 57.4 (+1.9, 3/3) | 56.6 (+1.0, 3/3) |
| da | 77.8 (n=3) | 79.1 (+1.1, 2/2) | 79.3 (+1.5, 3/3) | 78.9 (+1.1, 2/3) | 78.0 (+0.2, 2/3) | 79.0 (+1.2, 3/3) | 80.0 (+2.2, 3/3) | 79.8 (+2.0, 3/3) |
| yti_changing_tire | 49.0 (n=3) | 36.3 (-12.4, 0/2) | 50.4 (+1.4, 2/3) | 50.8 (+1.8, 3/3) | 49.4 (+0.4, 2/3) | 52.2 (+3.2, 3/3) | 53.0 (+4.0, 3/3) | 51.6 (+2.5, 3/3) |
| yti_coffee | 56.1 (n=3) | 44.5 (-12.3, 0/2) | 58.7 (+2.6, 3/3) | 57.7 (+1.6, 3/3) | 56.5 (+0.4, 2/3) | 59.3 (+3.2, 3/3) | 58.9 (+2.8, 3/3) | 57.3 (+1.2, 3/3) |
| yti_cpr | 59.0 (n=3) | 61.7 (+3.0, 2/2) | 62.8 (+3.8, 3/3) | 62.5 (+3.5, 2/3) | 60.5 (+1.5, 3/3) | 62.8 (+3.8, 3/3) | 60.2 (+1.2, 2/3) | 60.6 (+1.7, 3/3) |
| yti_jump_car | 33.9 (n=3) | 26.4 (-7.8, 0/2) | 30.3 (-3.5, 0/3) | 33.3 (-0.6, 0/3) | 33.7 (-0.1, 1/3) | 33.2 (-0.7, 1/3) | 33.3 (-0.5, 1/3) | 35.0 (+1.1, 3/3) |
| yti_repot | 38.2 (n=3) | 33.8 (-4.2, 0/2) | 38.1 (-0.1, 1/3) | 39.9 (+1.7, 2/3) | 39.0 (+0.8, 1/3) | 40.9 (+2.7, 2/3) | 40.7 (+2.5, 3/3) | 38.7 (+0.5, 1/3) |
| **mean delta** |  | -5.4, 2/7 bench | +1.1, 5/7 bench | +1.6, 6/7 bench | +0.6, 6/7 bench | +2.3, 6/7 bench | +2.0, 6/7 bench | +1.4, 7/7 bench |

### Alignment: ap@10 (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | vcreg | vcreg1 | vcreg03 | vcreg1_local | vcreg1_csls_blur6_local | vcreg03_csls_blur6_local |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 54.7 (n=3) | 47.7 (-6.5, 0/2) | 56.9 (+2.2, 3/3) | 57.0 (+2.3, 3/3) | 55.5 (+0.8, 3/3) | 57.2 (+2.5, 3/3) | 56.7 (+1.9, 3/3) | 55.9 (+1.2, 3/3) |
| da | 77.0 (n=3) | 75.6 (-1.6, 0/2) | 78.5 (+1.5, 3/3) | 78.2 (+1.2, 3/3) | 77.1 (+0.1, 2/3) | 78.2 (+1.2, 3/3) | 79.0 (+2.0, 3/3) | 78.8 (+1.8, 3/3) |
| yti_changing_tire | 48.3 (n=3) | 35.0 (-12.9, 0/2) | 49.6 (+1.4, 2/3) | 50.1 (+1.8, 3/3) | 48.7 (+0.4, 2/3) | 51.6 (+3.3, 3/3) | 52.6 (+4.3, 3/3) | 51.0 (+2.8, 3/3) |
| yti_coffee | 55.7 (n=3) | 43.5 (-13.0, 0/2) | 58.4 (+2.7, 3/3) | 57.5 (+1.8, 3/3) | 56.4 (+0.7, 3/3) | 58.8 (+3.1, 3/3) | 58.4 (+2.7, 3/3) | 57.0 (+1.3, 3/3) |
| yti_cpr | 58.6 (n=3) | 60.8 (+2.5, 2/2) | 62.7 (+4.1, 3/3) | 62.2 (+3.6, 3/3) | 60.0 (+1.4, 3/3) | 62.5 (+3.9, 3/3) | 60.0 (+1.4, 3/3) | 60.1 (+1.5, 3/3) |
| yti_jump_car | 33.2 (n=3) | 24.8 (-8.8, 0/2) | 30.1 (-3.1, 0/3) | 32.8 (-0.3, 1/3) | 32.8 (-0.4, 1/3) | 32.8 (-0.3, 1/3) | 33.1 (-0.1, 1/3) | 34.2 (+1.0, 3/3) |
| yti_repot | 37.8 (n=3) | 33.3 (-4.2, 0/2) | 37.7 (-0.1, 1/3) | 39.3 (+1.5, 2/3) | 38.5 (+0.7, 2/3) | 40.4 (+2.6, 3/3) | 40.1 (+2.3, 3/3) | 38.3 (+0.5, 2/3) |
| **mean delta** |  | -6.3, 1/7 bench | +1.2, 5/7 bench | +1.7, 6/7 bench | +0.5, 6/7 bench | +2.3, 6/7 bench | +2.1, 6/7 bench | +1.4, 7/7 bench |

### Alignment: ap@15 (mean over seeds | paired delta vs baseline, wins/seeds)

| bench | baseline | raw | vcreg | vcreg1 | vcreg03 | vcreg1_local | vcreg1_csls_blur6_local | vcreg03_csls_blur6_local |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fseval | 54.4 (n=3) | 46.6 (-7.2, 0/2) | 56.4 (+2.1, 3/3) | 56.6 (+2.2, 3/3) | 55.0 (+0.7, 3/3) | 56.9 (+2.5, 3/3) | 56.2 (+1.8, 3/3) | 55.5 (+1.1, 3/3) |
| da | 76.4 (n=3) | 72.5 (-4.1, 0/2) | 77.8 (+1.4, 3/3) | 77.6 (+1.2, 3/3) | 76.5 (+0.1, 2/3) | 77.6 (+1.2, 3/3) | 78.2 (+1.8, 3/3) | 78.1 (+1.7, 3/3) |
| yti_changing_tire | 47.7 (n=3) | 33.9 (-13.5, 0/2) | 49.1 (+1.4, 2/3) | 49.6 (+1.8, 3/3) | 48.2 (+0.5, 2/3) | 51.2 (+3.5, 3/3) | 52.0 (+4.2, 3/3) | 50.5 (+2.7, 3/3) |
| yti_coffee | 55.5 (n=3) | 42.8 (-13.4, 0/2) | 58.0 (+2.5, 3/3) | 57.1 (+1.6, 3/3) | 56.3 (+0.8, 3/3) | 58.6 (+3.1, 3/3) | 58.0 (+2.5, 3/3) | 56.8 (+1.3, 3/3) |
| yti_cpr | 58.3 (n=3) | 60.4 (+2.4, 2/2) | 62.4 (+4.2, 3/3) | 61.9 (+3.6, 2/3) | 59.6 (+1.4, 3/3) | 62.2 (+3.9, 3/3) | 59.8 (+1.5, 3/3) | 59.7 (+1.5, 3/3) |
| yti_jump_car | 32.7 (n=3) | 23.7 (-9.5, 0/2) | 29.9 (-2.7, 0/3) | 32.4 (-0.3, 1/3) | 32.2 (-0.4, 1/3) | 32.2 (-0.4, 1/3) | 32.7 (+0.1, 1/3) | 33.6 (+0.9, 3/3) |
| yti_repot | 37.5 (n=3) | 32.8 (-4.2, 0/2) | 37.6 (+0.1, 1/3) | 39.1 (+1.7, 2/3) | 38.1 (+0.7, 2/3) | 40.0 (+2.5, 3/3) | 39.7 (+2.3, 3/3) | 38.1 (+0.7, 2/3) |
| **mean delta** |  | -7.1, 1/7 bench | +1.3, 6/7 bench | +1.7, 6/7 bench | +0.5, 6/7 bench | +2.3, 6/7 bench | +2.0, 7/7 bench | +1.4, 7/7 bench |
