# REPRODUCTION_STATUS

Live status of the ACVCL experimental testbed. Updated as components come up.

## 1. Hardware / software

| Resource | Value | Note |
|---|---|---|
| GPU | 1× **NVIDIA H100 80 GB** carved into a **MIG 3g.40gb slice** (~42 GB usable) | not the full card; MIG mode disables NVIDIA EGL for CUDA-less GL rendering → MuJoCo must use `MUJOCO_GL=osmesa` |
| CPU | 4 cores (cgroup-pinned on a 128-core Xeon 8462Y+ host) | preprocessing bottleneck; cap `num_workers<=2` |
| RAM | 2 TiB visible (host-level) | effectively unbounded |
| Disk | 10 TB mount, **876 GB free (92 % used)** | budget-sensitive; skip 200+ GB datasets |
| Driver / CUDA | 570.133.20 / runtime 12.8, toolkit 12.2 on-host | PyTorch cu121 wheel used |
| Python | 3.10.12 | no conda; use venvs |
| ffmpeg | not system-installed | fallback: `imageio_ffmpeg` bundled binary |
| GL / Vulkan | `libEGL`, `libGL`, Vulkan icd.d present | software rendering fine; MIG blocks NV EGL |

## 2. Network policy (self-probed)

Allow-list behavior. Reachable hosts: `github.com`, `*.githubusercontent.com`,
`api.github.com`, `codeload.github.com`, `huggingface.co`, `pypi.org`,
`zenodo.org`, `carla.org`, `storage.googleapis.com` (public GCS). **Blocked**:
`*.ens.fr`, `*.inria.fr`, `*.eecs.yorku.ca`, `youcook2.eecs.umich.edu`,
`*.github.io`, `web.archive.org`, `drive.google.com` (DNS-only IPv6 unroutable),
`media.withyoutube.com`, `nuscenes.org`, `tau.ac.il`, `sjtu.edu.cn`.

Implication: **anything hosted only on university or Google-Drive mirrors is
unreachable from here**. HuggingFace Hub and public GitHub are the viable paths.

## 3. Python environments

| venv | Purpose | Status |
|---|---|---|
| `.venv/` (python 3.10) | main — torch cu121, transformers, decord, HF, hydra, gdown | ✅ |
| `.venv-mujoco/` | mujoco + dm_control + robosuite + metaworld | ✅ (robosuite 1.5.2 has a joint-type assertion with mujoco 3.14 — see sim table) |
| `.venv-maniskill/` | ManiSkill 3.0.1 / SAPIEN | ⚠ installed, but SAPIEN **cannot init Vulkan** in this container (`ErrorIncompatibleDriver`); even CPU-sim path forces Vulkan via `_setup_scene`. Hard-blocked. |
| `.venv-carla/` | CARLA | **not created** — CARLA requires its own server + Vulkan drivers; infeasible in this container |

### Key package versions

main venv: torch 2.4.1+cu121, torchvision 0.19.1+cu121, transformers 4.57.6,
decord 0.6, av 17.1, imageio 2.38, datasets 5.0.1, huggingface_hub 1.33.0,
timm 1.0.30, hydra-core 1.3.7, numpy 2.2.6.

mujoco venv: mujoco 3.14.0, dm_control (latest), gymnasium 1.4.0,
robosuite 1.5.2, metaworld 3.0.0, numpy<2.

## 4. Simulators

| Simulator | Status | Notes |
|---|---|---|
| **DMC — walker/stand**           | ✅ pass | 84×84 RGB via osmesa, 0.9 s |
| **DMC — reacher/easy**           | ✅ pass | 0.3 s |
| **DMC — cartpole/swingup**       | ✅ pass | 0.3 s |
| **MetaWorld — reach-v3**         | ✅ pass | MT1 API with `set_task` + rgb_array |
| **MetaWorld — pick-place-v3**    | ✅ pass | |
| **robosuite — Lift (Panda)**     | ✗ fail  | `AssertionError` in `get_joint_qpos_addr`: robosuite 1.5.2 adds composite robot models with free/ball joints; `binding_utils.py:525` restricts to hinge/slide. `robosuite_models` already installed but didn't help. Fix would need downgrading robosuite to 1.4 or patching the assert. |
| **RoboCasa — kitchen task**      | ⛔ deferred | downstream of robosuite; same blocker. |
| **ManiSkill — PushCube/PickCube**| ✗ fail  | SAPIEN `vk::createInstanceUnique: ErrorIncompatibleDriver` — Vulkan driver absent/unsupported in container. Even `sim_backend='cpu'` triggers Vulkan via `_setup_scene`. |
| **CARLA — any map**              | ⛔ deferred | needs 15 GB server binary + port 2000 + Vulkan driver; container-hostile. |

Rendering: `MUJOCO_GL=osmesa` + `PYOPENGL_PLATFORM=osmesa` works on this MIG
slice. GPU EGL is blocked by MIG mode (expected per NVIDIA MIG docs).

## 5. Datasets

| Dataset | Role | State | Path / Source | Issue |
|---|---|---|---|---|
| COIN annotations           | COIN task metadata                | ✅ full                 | `data/coin/COIN.json` (11 827 videos, 180 tasks) | — |
| YouCook2 splits + captions | YouCook2 cross-dataset validation | ✅ metadata-only        | `data/youcook2/merve__YouCook2` + `lmms-eval__YouCook2/data/*.parquet` | annotations json at UMich blocked |
| HowTo100M loader           | ref metadata                      | ✅                      | `data/howto100m/HuggingFaceM4__howto100m/` | raw videos stay on YouTube |
| CrossTask                  | TCC/CARL/LAV eval                 | ⛔ not downloadable      | — | upstream hosts (ENS, INRIA, YorkU) all blocked; no HF/Zenodo mirror |
| Pouring (Sermanet)         | TCC/CARL/LAV Exp 1 primary        | ✅ full 475 MB          | `data/pouring/sermanet_pouring/` (21 MOVs + 798 frame jpgs) | second attempt succeeded after disabling xet |
| YouCook2 InternVideo feats | Exp 7 real-data pair comparison   | ✅ 436 videos / 89 tasks | `data/youcook2/internvideo_features/internvideo_MM_L14_features/` | InternVideo-MM-L14 768-D @ 8 fps |
| Penn Action (subset)       | CARL eval                         | ⚠ not downloaded        | `urzsi/Penn-Action-Lossless-MKV-Test-Subset` (4.45 GB) | pending; disk budget OK |
| FineGym (skeleton)         | CARL eval                         | ⚠ not downloaded        | `Lozumi/FineGym-skeleton` (9.75 GB) | skeleton features; RGB subactions subset only |
| IKEA ASM                   | ref dataset                       | ⛔ skipped               | — | 206 GB on HF — exceeds our disk budget |
| SYNCR (metadata)           | SYNCR Exp 5                       | ✅ meta + SFT json      | `data/syncr/` | large tars (habitat 10 GB, kubric 38 GB) deferred |
| MDS-VQA (YouTube-UGC/SFV)  | MDS-VQA Exp 4                     | ⛔ unreachable           | — | hosted at `media.withyoutube.com` (blocked) |
| VideoSetDiff               | VideoSetDiff Exp 6                | ⛔ no release            | — | official repo is a website scaffold only; no code, no data |
| ActiveAD (nuScenes/CARLA)  | ActiveAD Exp (deferred)           | ⛔ out of scope          | — | repo is README-only; nuScenes requires auth + 400 GB |

## 6. Baseline repositories (`third_party/`)

| Baseline | Repo | Code? | Data? | Trainable here? |
|---|---|---|---|---|
| TCC            | `google-research/tcc` (sparse) + `June01/tcc_Temporal_Cycle_Consistency_Loss.pytorch` | ✅ | ⚠ needs Pouring/PennAction (CrossTask blocked) | ✅ once Pouring finishes |
| LAV            | `hadjisma/VideoAlignment` | ✅ | needs Penn Action etc. | ✅ when data ready |
| CARL           | `minghchen/CARL_code` | ✅ (+ data_prep scripts) | needs Pouring/PennAction/FineGym | ✅ when data ready |
| MDS-VQA        | `Multimedia-Analytics-Laboratory/MDS-VQA` | ✅ | ⛔ datasets unreachable | ⛔ |
| ActiveAD       | `Thinklab-SJTU/ActiveAD` | **README-only** | — | ⛔ |
| VideoSetDiff   | `qiuyue1993/videosetdiff` | website scaffold, no model code | — | ⛔ |
| SYNCR          | `SaraGhazanfari/SYNCR` | data-generation scripts | ✅ (benchmark on HF) | 🟡 eval possible, generation needs CLEVRER/Kubric/Habitat (deferred) |

## 7. Experiments

| ID | Baseline | Dataset | Simulator | venv | Setup | Train | Eval | Result | Issue |
|---|---|---|---|---|---|---|---|---|---|
| 1 | TCC (minimal port) | Pouring (sermanet, 11 train / 10 val MOVs) | — | main | ✅ | ✅ 100 iters | ✅ cycle-cons | **val_cycle 0.125** (vs 0.153 at init) | TCC regression σ² term is noisy on tiny dataset; see §9 |
| 2 | LAV              | Pouring                              | — | main   | code only | — | — | — | not run; the Soft-DTW variant can be run against the TCC pipeline next |
| 3 | CARL (ref only)  | Pouring                              | — | main   | ⚠ legacy deps (torch 1.6 + tf-gpu 2.3 incompatible with H100 / CUDA 12) | — | — | — | would need a port; we use their dataset prep + the ported TCC head instead |
| 4 | MDS-VQA          | YouTube-UGC / SFV                    | — | main   | ⛔ blocked | — | — | — | datasets unreachable |
| 5 | SYNCR eval       | HF SYNCR test metadata               | — | main   | ✅ metadata only | — | — | — | 10 GB habitat tar needed for actual eval; LoRA SFT deferred |
| 6 | VideoSetDiff     | —                                    | — | main   | ⛔ no code | — | — | — | official repo is website scaffold |
| 7 | **Pair-strategy** controlled (YouCook2 InternVideo) | YouCook2 InternVideo-MM-L14 (436 videos / 89 tasks) | — | main | ✅ | ✅ 300 iters/selector (4s on H100) | ✅ held-out same-task cycle-cons | **sim > hard > random**: 0.131 / 0.091 / 0.030 | ran on real precomputed features |

### Exp 1 (TCC-port on Pouring) — detail

- Model: frozen `facebook/dinov2-small` → 2-layer Conv1d temporal head (384 → 256 → 128).
- Data: 11 train MOVs (Sermanet Pouring), 10 val MOVs (sampled 24 frames each at 224×224).
- Loss: TCC cycle-back regression (my port in `src/losses/tcc.py`).
- 100 iters @ 1 pair/iter, Adam lr 3e-4, bf16 autograd on MIG 3g.40gb slice.
- Runtime: ~500 s (~5 s/iter — bottlenecked by DINOv2 forward on 48 frames + 4 CPU video decode).
- Val cycle consistency: 0.153 → 0.125 (small dataset, high variance; σ² denominator in the regression loss amplifies instability — reproducible but noisy).

### Exp 7 — detail

Shared across runs: 2-layer Transformer temporal head (256 hidden, 4 heads), lr 1e-3, Adam, 48 frames, 2 pairs/iter, 300 iters, bf16. Held-out set = 16 same-task pairs chosen with a fixed seed never used in training.

| selector    | loss@start | loss@end | **final cycle** | runtime |
|-------------|-----------:|---------:|----------------:|--------:|
| random      | ~1e7 (σ² term blow-up) | 6.25 | **0.030** | 4 s |
| similarity  | ~2.2e7 | 6.24 | **0.131** | 4 s |
| hard_pair   | ~2.8e6 | 41.80 | **0.091** | 4 s |

Interpretation (preliminary): selecting *similar* videos within a task yields a cleaner training signal than random pairing on this feature-space; hard pairs are informative but noisier at this scale. Matches the hypothesis motivating ACVCL.

## 8. Scripts added

| path | purpose |
|---|---|
| `scripts/env_check.py`              | torch/CUDA/decord/pyav/DINOv2 health |
| `scripts/smoke_test_sims.py`        | DMC / robosuite / MetaWorld smoke tests (osmesa) |
| `scripts/inspect_dataset.py`        | per-dataset sanity report |
| `scripts/smoke_test_pairs.py`       | COIN surrogate pair-selection pipeline |
| `scripts/prepare_pouring.py`        | sermanet/pouring MOV → CARL train/val/test.pkl |
| `scripts/tcc_train_pouring.py`      | **Exp 1** — TCC-port training on Pouring |
| `scripts/exp7_pair_strategy_youcook2.py` | **Exp 7** — real controlled pair-strategy on InternVideo features |

## 9. Open issues / deferrals

1. **CrossTask** — only hosted on blocked ENS / INRIA / YorkU URLs. No HF or
   Zenodo mirror found. Options: user-side upload, or re-extract DINOv2
   features from the CrossTask YouTube IDs (needs YouTube reach, not tested).
2. **CARL original repo** is Python 3.7 + torch 1.6 + tf-gpu 2.3 (CUDA 10.1).
   None of that works on an H100 CUDA 12 stack. Would need a full port —
   we instead reuse CARL's `dataset_preparation` scripts and ported only
   the TCC loss into `src/losses/tcc.py`.
3. **TCC regression loss** on tiny datasets (Pouring 11 videos) is numerically
   noisy — the `log σ²` term dominates whenever the softmax collapses on
   a single frame. For the next pass we should switch to the TCC
   *classification* variant or add σ clamping. Not critical for pipeline
   validation.
4. **robosuite 1.5.2 + mujoco 3.14** `get_joint_qpos_addr` assert. Fix:
   downgrade robosuite to 1.4 (next time we need it).
5. **ManiSkill / SAPIEN** needs Vulkan which this container cannot provide
   (`ErrorIncompatibleDriver`). No fix without changing container image.
6. **CARLA / ActiveAD / VideoSetDiff / MDS-VQA** data or server infeasible,
   or no public code — treated as references only.
7. **SYNCR eval**: downloading `habitat_data.tar` (10 GB) + `kubric_data.tar`
   (38 GB) would use 5 % of our disk budget. Deferred until we actually need
   to run multimodal eval; the SFT JSON (`sft_all8_answer_only.json`,
   20 MB) is already in `data/syncr/train/`.

---

# Part II — Baseline paper reproductions

Separate workstream from the ACVCL testbed above; the Part I material is
unchanged. Work lives under `reproductions/{data_selection,visual_debiasing,robot_learning}/`,
external code under `third_party/` (untracked), environments in
[`ENVIRONMENTS.md`](ENVIRONMENTS.md), logging in [`WANDB_SETUP.md`](WANDB_SETUP.md).

Cluster context: Slurm job 60385 on `worker-mig-3g40gb-0`, 16 CPU, 64 GB RAM,
1× H100 MIG 3g.40gb, 10 h walltime. All experiments run **sequentially** — one
heavy job on the slice at a time, per `RULE.md` §2.

## II.0 Summary — what ran, what it produced

One 10 h allocation, seven papers attempted, all runs sequential on one MIG
slice. W&B project `acvcl-reproduction` (entity `hust_edu_vn`) holds every run.

| # | paper | dataset | headline reproduced number | label |
|---|---|---|---|---|
| 1 | **GLISTER** (AAAI'21) | CIFAR-10 @ 10 % | 91.30 % test acc | faithful |
| 1 | **GradMatch** (ICML'21), PB variant | CIFAR-10 @ 10 % | 92.27 % test acc | faithful |
| — | Random subset baseline | CIFAR-10 @ 10 % | 75.49 % test acc | faithful |
| 2 | **Correct-N-Contrast** (ICML'22) | Waterbirds | 88.72 % worst-group (val, best-of-evals) | partial — reproducibly SIGKILLed |
| 3 | **Learning from Failure** (NeurIPS'20) | Colored MNIST 99 % | 70.20 % unbiased acc vs 46.38 % ERM | modified |
| 4 | **Let Samples Speak** (CVPR'25) | Waterbirds | 87.38 % worst-group / 95.74 % mean | partial |
| 5 | **ACT** | aloha_sim_insertion_human | 10 % task success @ 15 k steps | partial |
| 6 | **Diffusion Policy** | pusht | 40 % task success @ 15 k steps | partial |
| 7 | **SOAR** (ICCV'23) | UCF101 | not started | blocked — no allocation time left |

The strongest results are the data-selection pair, where both methods beat a
random subset of the same size by ~16 pp and the whole comparison ran at the
papers' own published configs. The debiasing group all produced the qualitative
effect each paper claims. The two robot policies trained and evaluated in
simulation but at 15 % of the reference step budget, so their success rates are
lower bounds, not reproductions.

Nothing here should be read as matching a published figure unless this document
says so explicitly — see §II.3 for the per-row caveats and §II.5 for four bugs
in the published code that had to be fixed before anything would run.

## II.1 Reproduction fidelity labels

| label | meaning |
|---|---|
| **faithful** | official config + official data + official metric, unmodified |
| **modified** | official config, but something had to change (noted per row) |
| **partial** | ran, but fewer epochs/seeds or a subset of the paper's grid |
| **smoke** | pipeline verified end-to-end, result not meaningful |
| **blocked** | could not run; reason recorded |

## II.2 Dataset provenance

Every original host below was checked from the compute node first. The
approved sources are HF / Google Drive / GitHub / PyPI.

| dataset | original host | status | source used | size |
|---|---|---|---|---|
| CIFAR-10 | `cs.toronto.edu` | **blocked** (000) | HF `uoft-cs/cifar10` | ~150 MB |
| Waterbirds | `nlp.stanford.edu` | **blocked** (000) | HF `grodino/waterbirds` | 412 MB → 473 MB on disk |
| MNIST (→ Colored MNIST) | `yann.lecun.com` | **blocked** (000) | `ossci-datasets.s3.amazonaws.com` (torchvision mirror 2) | 64 MB |
| Tiny ImageNet | `cs231n.stanford.edu` | **blocked** (000) | HF `zh-plus/tiny-imagenet` (not yet pulled) | ~240 MB |
| Food-101 | — | available | HF `ethz/food101` (not yet pulled) | ~5 GB |
| CnC pretrained ERM | Google Drive | available | `gdown` folder, Waterbirds files only | 283 MB |

Verification done before use, not assumed:

- CIFAR-10: 50 000 train / 10 000 test, 10 classes, exactly 5 000 and 1 000
  per class, 32×32 RGB.
- Waterbirds: 4 795 / 1 199 / 5 794 and per-group counts
  `(landbird,land)=3498, (landbird,water)=184, (waterbird,land)=56, (waterbird,water)=1057`
  on train — identical to the published 95 %-correlation split. The HF parquet
  carries the original JPEG bytes and filenames, so
  `reproductions/visual_debiasing/prep_waterbirds.py` rebuilds the
  `waterbird_complete95_forest2water2/` tree byte-for-byte and refuses to write
  if the group counts disagree.
- Colored MNIST: 60 000 train at **99.00 %** bias-aligned (skew 0.01) and
  10 000 validation at **10.00 %** — matches LfF's protocol.

Waterbirds is built once and symlinked into both `correct-n-contrast` and
`nsf_debiasing`; MNIST is shared between CnC and LfF. No dataset is duplicated.

## II.3 Results

CIFAR-10 / ResNet18 / fraction 0.1, official CORDS configs (SGD lr 0.05,
momentum 0.9, nesterov, wd 5e-4, cosine `T_max`=300, 300 epochs, batch 128).

| method | select_every | test acc (final / best) | wall | selection overhead | GPU peak | label |
|---|---|---|---|---|---|---|
| **GLISTER** | 20 | 91.30 % / 91.30 % | 884 s | 146 s (16.5 %) | 4.2 GB | faithful |
| **GradMatchPB** | 1 | **92.27 % / 92.36 %** | 2 564 s | 1 825 s (71.2 %) | 4.2 GB | faithful |
| Random (baseline) | 1 | 75.49 % / 75.59 % | 738 s | 0.4 s (0.06 %) | 4.1 GB | faithful |

Both selection methods beat random subset selection by a wide margin at the
same 10 % budget — **+15.8 pp** (GLISTER) and **+16.8 pp** (GradMatchPB) — which
is the qualitative claim of both papers, and GLISTER's 91.3 % sits in the
~90–91 % band the AAAI'21 paper and the CORDS benchmark report for this setting.

The cost side is worth recording separately: the two methods are nearly tied on
accuracy (0.97 pp apart) but not on price. GradMatchPB's shipped config
re-selects **every** epoch (299 selections, 71 % of wall time) against GLISTER's
every 20th (15 selections, 17 %), so GradMatchPB spends 2.9× the wall clock for
its ~1 pp edge. Random costs essentially nothing to select and loses ~16 pp.

Not run: the full-data (100 %) reference and AdaptiveRandom, and the Tiny
ImageNet / Food-101 adaptations — deprioritised to reach the other five papers
within one 10 h allocation. CORDS has no Food-101 branch, so that one needs a
new dataset branch rather than just a config.

### Visual debiasing

| paper | dataset | metric | reproduced | label |
|---|---|---|---|---|
| **Learning from Failure** (NeurIPS'20) | Colored MNIST, skew 0.01 / sev 4 | unbiased val acc | **70.20 %** (best 71.11 %) | modified |
| ERM baseline for the above | same | unbiased val acc | **46.38 %** | modified |
| **Let Samples Speak** (CVPR'25) — full `ssc.py` | Waterbirds | worst-group / mean | **87.38 % / 95.74 %** | partial |
| its ERM pre-training stage | Waterbirds | worst-group / mean | **74.92 % / 89.94 %** | partial |
| **Correct-N-Contrast** (ICML'22) | Waterbirds | worst-group (val) | **88.72 % best observed** | partial |

LfF reproduces the paper's central claim cleanly: the biased model collapses
onto the colour shortcut (100 % on bias-aligned, 8.4 % on bias-conflicting) and
the debiased model lands at 70.2 % unbiased — **+23.8 pp over its own ERM
baseline**. It is labelled *modified* only because it runs on torch 2.4 rather
than the paper's torch 1.4.

**Let Samples Speak** is the cleanest of the three on Waterbirds: the full
`ssc.py` pipeline ran to completion (exit 0) and lifts worst-group accuracy
from its own ERM base's 74.92 % to **87.38 %, +12.46 pp**, while mean accuracy
rises 89.94 % → 95.74 %. Per-group: 96.72 / 98.14 / 92.21 / 87.38 %. Note the
worst group after debiasing is (waterbird, water) — a *majority* group — not the
rare (waterbird, land) group, which reached 92.21 %: the method traded majority
headroom for minority robustness, which is the intended behaviour.

It is labelled **partial** for one specific reason: its ERM base was trained
30 epochs instead of the official 100, so the 87.38 % rests on a shorter
pre-training run than the paper's. The paper reports >20 pp worst-group gain
over ERM; we measure +12.46 pp over *our* 30-epoch ERM, which is not the same
comparison and should not be read as agreeing or disagreeing with it.

**Correct-N-Contrast** did not finish, and the failure is reproducible in a way
worth recording. Its first attempt died on the repo bug below (§II.5). After
that was fixed, two further attempts were **SIGKILLed (exit 137)** — not the
wall-clock bound, which returns 124 and which neither run reached:

| attempt | `batch_factor` | elapsed before kill | eval passes reached | exit |
|---|---|---|---|---|
| 2 | 32 (official) | ~56 min | 17 | 137 |
| 3 | 12 (reduced per `RULE.md` §2) | ~56 min | 17 | 137 |

Cutting `batch_factor` to a third changed **nothing** — same signal, same
elapsed time, same stage. So the cause is not the contrastive batch size, and
reducing it further is not the fix: something allocates a large, roughly
fixed amount at the stage reached after 17 evaluation passes and runs into the
job's 64 GB ceiling. Diagnosing which allocation needs another ~1 h run with
memory instrumentation, which the allocation had no room for. Recorded as a
**reproducible blocker**, not as solved.

Consequently 88.72 % is the best
worst-group accuracy seen across evaluation passes, not a completed run's
selected checkpoint, and it is the **validation** split — CnC's
`evaluate_model` is called with `[train_loader, val_loader]` and keeps the last
loader — whereas the paper's headline 88.5 % is test worst-group. Two
independent runs both peaked at exactly 88.72 %, which is reassuring, but
"best over evals" is a more generous selection rule than the paper's and the
splits differ, so **this is not a reproduction of the published number**.

### Robot learning

Both trained from **RGB observations** (the datasets' camera features), in the
separate `lerobot` env, with lerobot's own policy defaults; only the step count
was cut — from lerobot's 100 k default to 15 k — to fit the allocation. Both are
therefore **partial** by construction, at 15 % of the reference budget.

| policy | dataset | steps (of 100 k) | train loss | task success | eval | GPU peak | label |
|---|---|---|---|---|---|---|---|
| **ACT** | `lerobot/aloha_sim_insertion_human` | 15 000 (4.8 ep) | 0.1147 (l1 0.0953, kld 0.0019) | **10 %** | 10 ep, AlohaInsertion-v0 | 2.1 GB | partial |
| **Diffusion Policy** (+EMA) | `lerobot/pusht` | 15 000 (37.4 ep) | 0.0084 | **40 %** | 10 ep, PushT | 6.0 GB | partial |

Success is `eval/pc_success` from lerobot's own simulator rollouts, so these are
real policy evaluations, not training proxies. `gym_aloha` and `gym_pusht` both
run here under `MUJOCO_GL=osmesa`, because MIG mode disables NVIDIA EGL.

**These numbers must not be read as reproducing the published ones.** At 15 % of
the training budget they are lower bounds on what the full recipes reach, and
the published figures for these two settings were not re-verified from the
papers during this session, so no claim of agreement is made either way. The
honest statement is: the pipelines train, evaluate in simulation, and produce
plausible partial-budget success rates (DP's 40 % on PushT after 37 epochs,
ACT's 10 % on the harder insertion task after 4.8 epochs).

## II.4 Deviations from the published setups

1. **Data sources** — the three blocked hosts above forced HF mirrors. Content
   was verified identical (counts, per-class/per-group sizes, and for
   Waterbirds the original bytes), so this changes provenance, not data.
2. **CORDS loader shim** — `torchvision.datasets.CIFAR10` would still try to
   reach `cs.toronto.edu` and fails its md5 check on any locally rebuilt copy,
   so `reproductions/data_selection/cords_compat.py` substitutes an
   HF-backed class with the same `.targets` / `transform` semantics.
   Training code, config and splits are untouched.
3. **CORDS `print_args`** — adding `subtrn_loss` to the official print list
   crashes `train_sl.py` (`subtrn_losses[-1]` on an empty list), so each
   config's own `print_args` is used unchanged.
4. **nsf_debiasing `train_sup.sh` is not used.** It picks a GPU by scanning
   `nvidia-smi` output, overrides `CUDA_VISIBLE_DEVICES`, and launches
   `nohup … &` detached processes — all three conflict with `RULE.md` §2/§4 on
   a shared Slurm node. `train_supervised.py` is invoked directly with the
   same hyperparameters the script would have passed. Its
   `--label_filename` flag is also dropped: that argument belongs to
   `ssc.py`'s parser, not `train_supervised.py`'s.
5. **LfF runs on modern torch.** The repo pins torch 1.4 / numpy 1.18 /
   sklearn 0.21, none of which support an H100. It runs unmodified on torch
   2.4.1 apart from import shims (`wand`, needed only by the Corrupted
   CIFAR-10 protocol we do not use) and `server_user`'s paths, which the repo
   marks as user configuration. Numerical results may therefore differ
   slightly from the paper's torch 1.4 runs → labelled **modified**.
6. **`setuptools<81`** is pinned in the env because `sacred` (LfF's config
   system) imports `pkg_resources`.
7. **`mmcv` is stubbed, not installed.** `nsf_debiasing/ssc_common.py` imports
   mmcv (absent from its README requirements) and uses exactly one symbol from
   it, `mmcv.ProgressBar`, in three places. A tqdm-style stand-in with the same
   API is in `reproductions/visual_debiasing/nsf_compat.py`; installing the
   OpenMMLab package, which builds CUDA ops against a pinned torch, would have
   been disproportionate.
8. **lerobot's torch is downgraded to 2.8.0+cu128.** `pip install lerobot`
   pulls torch 2.11, whose wheel needs a newer driver than this node has
   (`CUDA initialization: The NVIDIA driver on your system is too old
   (found version 12080)`), leaving CUDA unavailable. 2.8.0 satisfies lerobot's
   `torch>=2.7,<2.12` and sees the MIG slice.
9. **lerobot CLI**: `--policy.push_to_hub=false` is required (the default True
   demands a `repo_id` and fails `cfg.validate()`), and the eval cadence flag
   is `--env_eval_freq`, not `--eval_freq`. torchcodec 0.11 is ABI-incompatible
   with torch 2.8 and lerobot falls back to the `pyav` decoder on its own; this
   is logged as a warning and needs no intervention.

## II.5 Bugs found in the published code

Not environment problems — defects in the repos as published, which had to be
fixed for the code to run at all. All are in `third_party/` (untracked), and
each is a minimal change, not a rewrite.

1. **correct-n-contrast — `summarize_acc` return value misused (4 sites).**
   `utils/logging.py:summarize_acc` returns a tuple `(average, worst_group)`,
   but `evaluate.py:30`, `evaluate.py:88`, `train.py:103` and `train.py:141`
   assign it to a scalar and then compare it against a float
   (`args.max_robust_acc`, initialised to `-1`). Any run reaching an evaluation
   pass dies with `TypeError: '>' not supported between instances of 'tuple'
   and 'int'` — which is what killed the first CnC attempt after 12 minutes of
   training. Fixed by unpacking both values at each site.
2. **correct-n-contrast — two absolute paths from the authors' machine.**
   `datasets/__init__.py` pointed `root_dir` at
   `../slice-and-dice-smol/datasets/data/Waterbirds/` (the README's own
   commented alternative was correct), and `resnet.py:13` loaded ImageNet
   ResNet50 weights from
   `/dfs/scratch1/mzhang/projects/.../resnet50-19c8e357.pth`. The second is
   just torchvision's standard checkpoint, re-fetched from
   `download.pytorch.org` and verified (267 keys, `fc` 1000×2048).
3. **LfF — cross-device tensor indexing.** `train.py:251` builds
   `aligned_mask`/`skewed_mask` on CUDA (from `attr`, moved to device at
   line 178) and indexes `loss_per_sample_b` / `loss_weight`, which stay on
   CPU. torch 1.4 tolerated this; torch 2.x raises
   `RuntimeError: indices should be either on cpu or on the same device as the
   indexed tensor`. The block is logging-only, so moving the masks to CPU
   changes no training math.
4. **nsf_debiasing — `train_sup.sh` is unsafe on a shared scheduler** (see
   deviation 4 above) and passes `--label_filename` to `train_supervised.py`,
   which has no such argument — it belongs to `ssc.py`'s parser.
