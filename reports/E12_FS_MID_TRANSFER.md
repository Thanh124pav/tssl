# E12 — Does the segmentation-prior change transfer to FS Mid?

Date: 2026-10-29. E11c established a +8.69 percentage-point mean MoF gain
for `rho_seg=0` on FSeval over three paired seeds. E12 advances exactly that
intervention to the next pre-registered E00 rung, 50 Salads Mid (`FS`, 19
classes). Baseline and intervention use the frozen FS Mid/README config:
30 epochs, `rho=0.15`, `lambda_actions_train=0.15`, radius 0.04, and the same
seed, data, evaluation, and other hyperparameters within each pair. Only
`rho_seg` changes to zero; `rho_align` remains 0.15. Four runs completed,
all four `last.ckpt` state dictionaries are finite.

| Seed | Setting | MoF | Segment F1 | mIoU |
|---:|---|---:|---:|---:|
| 0 | baseline | 0.4774 | 0.4095 | 0.2842 |
| 0 | `rho_seg=0` | 0.4732 | 0.4279 | 0.2723 |
| 1 | baseline | 0.4070 | 0.3706 | 0.2360 |
| 1 | `rho_seg=0` | 0.4147 | 0.3510 | 0.2269 |
| mean, N=2 | baseline | 0.4422 | 0.3901 | 0.2601 |
| mean, N=2 | `rho_seg=0` | 0.4440 | 0.3895 | 0.2496 |

Paired MoF effects are **−0.42 pp** (seed 0) and **+0.77 pp** (seed 1),
mean **+0.18 pp**. This misses the E00 task-metric scale gate of +1.0 pp.
Mean segment F1 is approximately unchanged (−0.06 pp); mIoU falls by
1.06 pp and is negative on both seeds. No pre-registered diagnostic gate
was measured as passing. **Scale: SKIP DA/YTI** for `rho_seg=0`.

The baseline seed-0 MoF/F1/mIoU (47.74/40.95/28.42%) match the earlier FS
Mid reproduction, providing a protocol sanity check. The negative transfer
does not refute that an index-based segmentation prior can harm particular
videos. It narrows the actionable conclusion: removing that prior helps
FSeval's 12-class evaluation mapping, but is not robust even to the 19-class
mapping of the same underlying 50 Salads videos. Combined with E08's DA/YTI
failures for removing both priors, this does not support a general-purpose
`rho=0` fix for limitation C.

Raw runs: `reproductions/tssl/artifacts/runs/E12_FS_mid_{baseline,seg0}_s{0,1}/`.
Script: `reproductions/tssl/scripts/run_e12_fs_mid_rho_seg0.sh`.
Log: `outputs/logs/E12_fs_mid_rho_seg0.log`.
