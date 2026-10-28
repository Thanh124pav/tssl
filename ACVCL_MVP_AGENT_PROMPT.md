# ACVCL MVP — Autonomous Implementation and Overnight Experiment Prompt

**Target agents:** Codex CLI / Claude Code  
**Repository:** https://github.com/Thanh124pav/acvcl  
**Working directory (verify on compute node):** /mnt/data/thanhpv/acvcl  
**Research target:** CVPR — visual interaction representations, complementary RGB-video selection, and downstream robot imitation learning  
**Compute target:** One *allocated full* NVIDIA H100 80GB; approximately 12 hours of experimental runtime **after** environment/data/checkpoint downloads and setup. This is a goal, not a guarantee.

---

## 0. Role and non-negotiable task

Act as an autonomous senior PyTorch / Computer Vision / Robot Learning research engineer. **Extend the existing ACVCL repository rather than starting a new project.** Inspect its real code, data, checkpoints, environments, prior experiments, and logs; reuse working components; implement the smallest scientifically meaningful ACVCL MVP; execute training and held-out evaluation; and write an accurate results report.

**Do not stop at producing a plan, scaffolding, or training scripts.** After safety checks, actually smoke-test, run experiments, evaluate policies, and analyze completed results. Do not fabricate experiments, scores, available data, or successful execution.

First priority is a valid end-to-end result. A smaller, properly controlled experiment is better than six broken or non-comparable runs. Never overwrite old reproductions or conflate existing results with new ACVCL validation.

Before changing code, inspect the latest repository commit and read:
- **RULE.md** — mandatory cluster and shared-resource safety; overrides this prompt where stricter.
- **REPRODUCTION_STATUS.md** — what was and was not reproduced, including partial-budget results.
- **ENVIRONMENTS.md**, **SETUP.md**, **SETUP_STATUS.md** — environment history; some entries are stale.
- **WANDB_SETUP.md**, **reproductions/env.sh**, **reproductions/robot_learning/env_lerobot.sh**.
- **reproductions/robot_learning/run_act_dp.sh**, **scripts/exp7_pair_strategy_youcook2.py**, **src/selectors/**, **src/losses/tcc.py**, and the simulator smoke tests.

Keep a concise audit of what exists, what is missing, and what will be reused. Do not repeat paper reproductions already completed.

## 1. Safety, hardware and environment (mandatory preflight)

The server is shared and SLURM-managed. Follow **RULE.md** exactly, including its stop conditions.

1. Verify hostname, active SLURM_JOB_ID, allocated CPU/RAM/GPU/walltime and CUDA_VISIBLE_DEVICES before substantive work. **No agents, builds, dataset downloads, inference, training, preprocessing, or heavy operations on login nodes.** Work only in a valid authorized compute allocation. Do not start GPU work without one.
2. The user intends to use a **full H100 80GB**, but the existing reproduction reports were collected on an **H100 MIG 3g.40GB** slice. **Verify actual allocated device and usable VRAM.** Never change CUDA_VISIBLE_DEVICES to escape the allocation; do not assume full H100 merely from hardware model names.
3. Respect site-specific partition and GPU GRES permissions. Do not invent an SLURM GPU resource string or submit overlapping reservations. Use the existing approved allocation where possible.
4. Keep all large datasets, weights, generated videos, checkpoints, caches and reports under authorized /mnt/data user-owned locations, not $HOME. Follow the environment scripts' cache mappings; do not change another project's environment.
5. Avoid heavy or persistent use of /tmp and /dev/shm. Use verified job-local scratch or authorized project storage and bounded temporary directories; never indiscriminately clean any shared path.
6. Cap torch, OpenMP, BLAS, DataLoader and simulator worker counts to the CPU allocation. Control file count, write frequency, W&B traffic and checkpoint retention; monitor disk/VRAM and job walltime.
7. Never terminate other users' jobs/processes, disable permissions, bypass quotas, or launch detached/uncontrolled agents. Do not allocate a GPU merely to keep a remote-control agent online.
8. Estimate size and confirm destination before downloading/extracting large datasets or checkpoints; obtain approval for unusual or uncertain storage use as prescribed in RULE.md.
9. State a short preflight summary: verified job context, resource limits, output/cache/temp paths, maximum expected data footprint, and cleanup plan. If required facts are unknown, stop the unsafe action and report the minimal blocker.

Existing Conda installation has been documented at **/mnt/data/thanhpv/miniconda3**. Reuse the existing **acvcl** and **lerobot** environments if valid. The document ENVIRONMENTS.md previously claimed the lerobot environment was not yet created, but later experiments show that LeRobot ran successfully; inspect the current filesystem rather than trusting stale snapshots. Do not create redundant environments or downgrade working packages globally. If a new environment is truly necessary, document why and minimize the count.

Software rendering **MUJOCO_GL=osmesa** and **PYOPENGL_PLATFORM=osmesa** was necessary on the previous MIG allocation; test what works on the *current* allocation rather than assuming EGL support. The repository already contains working environment wrappers.

## 2. Research hypothesis and strict scientific scope

Working hypothesis:

> Visual appearance alone may be insufficient to characterize object interactions. We hypothesize that selectively incorporating complementary interaction videos can improve the learning of contact and post-contact dynamics representations, thereby enhancing downstream robot policy learning.

Test **two distinct hypotheses**:

- **H1 — Representation benefit:** RGB supplementary interaction clips plus contact/post-contact auxiliary training can improve downstream policy learning beyond policy-only training.
- **H2 — Selection benefit:** Choosing *complementary* interaction clips improves policy learning beyond random, visual-similarity, and motion-similarity selection when support budget, optimization, data, and backbone are controlled.

H1 does **not** prove H2. Contact prediction improvement does **not** automatically prove policy improvement.

Do not claim complete tactile/force inference from RGB. Focus on **contact events, participating objects, contact location where observable, and post-contact motion/outcomes**. Do not require VLA, human-to-robot transfer, cross-embodiment generalization, or spurious-correlation suppression for the first MVP. Those may be future extensions only when supported by experiments.

The supplemental pool is **RGB as model input, without action labels for imitation loss**. When training on simulator-generated videos, contact/outcome ground truth may be generated from privileged simulator states **as auxiliary supervision**; report clearly that this is a *simulation-supervised RGB-input MVP*, not proof that truly unlabeled internet videos work without annotations. Optionally test RGB pseudo-labels later; do not silently misrepresent privileged training labels.

## 3. Existing code and results that MUST be reused

### 3.1 Working LeRobot pipelines

Existing script: **reproductions/robot_learning/run_act_dp.sh** and wrapper **reproductions/robot_learning/env_lerobot.sh**.

Prior partial-budget checks:
- **ACT** on **lerobot/aloha_sim_insertion_human**: 15,000 steps, **10% success on 10 rollout episodes**.
- **Diffusion Policy** on **lerobot/pusht**: 15,000 steps, **40% success on 10 rollout episodes**.

They train and evaluate in real simulator rollouts, but **are not full reproduction of published performances**. Check actual checkpoint files and compatibility before attempting resume. Prefer LeRobot ACT as primary policy; retain Diffusion Policy only as optional secondary comparison if time remains. Do **not** needlessly replace this stack with an incompatible ACT repo or retrain a policy for a different embodiment and call it pretrained transfer.

### 3.2 Visual features and temporal models

Existing implementation already used **facebook/dinov2-small**, a trainable lightweight temporal head, and precomputed InternVideo-MM-L14 features. Reuse extraction/caching patterns, model modules, and tensor conventions as practical.

Existing components:
- **src/losses/tcc.py**, used in minimal temporal alignment experiments.
- **scripts/tcc_train_pouring.py** with DINOv2-small and a temporal head.
- **scripts/exp7_pair_strategy_youcook2.py** with Transformer temporal head and real precomputed video features.

TCC is **optional**, not the ACVCL core objective. Do not divert overnight efforts into reproducing LAV/CARL/TCC in full.

### 3.3 Existing selectors

Source files:
- **src/selectors/base.py**
- **src/selectors/random.py**
- **src/selectors/similarity.py**
- **src/selectors/hard_pair.py**

Reuse and adapt their simple baselines; add a new contact/outcome-conditioned selector rather than rewriting all selection machinery.

Important current limitation: the selector protocol chooses global **(video_a, video_b)** pairs. The proposed ACVCL coordinator needs a **query-batch-to-support-K** interface. Write a small adapter or a new interface while retaining the old baselines for reproducibility.

### 3.4 Known experimental issues — do not repeat

In **scripts/exp7_pair_strategy_youcook2.py**, the hard-pair experiment initializes **HardPairSel(feats, feats)**, so cosine similarity minus itself is identically zero (up to floating-point noise). It is not an effective prediction-disagreement selector. Also, the validation pairs are assembled from video IDs that remain available to training selectors; thus validation and training can share videos. Fix these issues if reusing that experiment, and always split video IDs/episodes *before* building candidate indices.

Historic Exp 7 reports held-out same-task cycle values of **random 0.030, hard pair 0.091, similarity 0.131**. Treat these only as preliminary results from an imperfect selection experiment, **not evidence of ACVCL downstream policy benefit**. Preserve historical outputs; do not overwrite them or claim their hard-pair ranking establishes complementarity.

Existing GLISTER / GradMatch, debiasing, TCC and other reproductions are useful references but **not directly equivalent** to action-free video retrieval for policy improvement. Do not spend the 12-hour budget rerunning them.

## 4. Minimum end-to-end ACVCL architecture

Implement three connected components with testable interfaces.

### A. Policy: LeRobot ACT

- Input: RGB observations and robot proprioception; action labels only for policy demonstrations.
- Output: action chunks; use the existing LeRobot ACT loss exactly as defined by that installed version, including its regularizers.
- Prefer a **pretrained DINOv2 ViT-S/14 frozen backbone** feeding a trainable shared projection/temporal adapter that is **actually used by the ACT action head**. Validate LeRobot API compatibility before editing internals. If reusing ACT's native visual encoder is a substantially safer integration, do so and describe the deviation clearly.
- Do not assume a pretrained DINOv2 embedding can be dropped into ACT without adjusting image normalization, projection, and shape expectations.
- The contact/outcome head and policy must share trainable features. Prove auxiliary gradients reach these shared parameters.
- LoRA is optional, not first-line: start with **frozen pretrained features + trainable shared adapter** because this permits efficient caching. Later, a controlled DINOv2 LoRA comparison (e.g., rank 8/16 on a subset of attention layers) is allowed only if primary comparisons finish and additional trainable-backbone adaptation is justified.

### B. Coordinator

Given a policy training query sample/interaction, retrieve **up to K=3** supplemental RGB interaction clips from a training-only pool, with an initial allocation of one clip per group:

1. **Appearance variation:** similar interaction role/object relevance with altered object shape/size/gripper appearance (only if supported by data).
2. **Contact/outcome agreement:** comparable pre-contact interaction, similar contact and similar post-contact outcome.
3. **Contact/outcome disagreement:** comparable pre-contact interaction and contact, **different post-contact outcome** (e.g., grasp succeeds versus slips).

If a category is empty, log the condition and use a documented deterministic fallback. Do not fabricate examples or claim all categories were present.

Compute object-centric visual features on masked/cropped gripper/object regions. Reuse available segmentation, object detector, or simulator masks in a clearly distinguished **oracle preprocessing** experiment. For the **RGB-only coordinator** headline, the runtime selection score may use only evidence computed from RGB/video (e.g., detector outputs, estimated contact events, object tracks, estimated outcomes). It must **not** read simulator contact flags, 3D positions, hidden state, or action labels as coordinator inputs. An oracle selector can be studied as a separately labeled upper bound.

Precompute object features, motion tracks, interaction descriptors and retrieval index; avoid detector passes over every pool clip every optimizer step. Begin with deterministic selection. Record selected clip IDs, category, score, fallback and latency. Exclude same episode, duplicates, near duplicates and held-out evaluation episodes.

### C. Auxiliary contact/outcome predictor

Input: RGB-derived features from a video **prefix ending at time t**. Targets:
- **Binary gripper-object (or object-pair) contact** at a well-defined observed/predicted time.
- **Future object displacement** at fixed horizon t + delta, relative to t, preferably in calibrated 2D image coordinates for MVP.

Optional if stable: contact pair ID, projected **2D contact heatmap**, coarse outcome class (lift / slip / static), or rotation. Do **not** require full 3D contact surface/force reconstruction in v0.

Crucial: do not feed frames from t+delta into the predictor that is tasked with predicting state at t+delta. The coordinator may use observed outcomes in **recorded training-corpus** clips for offline matching; the deployed policy at inference receives **only current/past observations**. Do not mix the coordinator's offline observations with inference-time policy features.

### Training losses

Let:

L_total = L_ACT(demonstrations) + lambda_c * L_contact(eligible annotated clips) + lambda_o * L_outcome(eligible annotated clips).

- **L_ACT** only uses policy demos with ground-truth actions.
- **L_contact / L_outcome** may use labeled policy demos and selected supplemental RGB clips with legitimate simulator-derived or validated pseudo-labels.
- For unannotated clips, do not invent labels; exclude those auxiliary terms or use a clearly described self-supervised target with separate experiment labels.
- Normalize losses across valid samples and gradient-accumulation microbatches. Ensure matching *effective* support budget K across E2–E5.
- Validate nonzero, finite gradients on the **shared adapter** and confirm a one-step auxiliary update changes the same representation feeding ACT. No detached/cached-tensor shortcut may bypass the shared trainable adapter.

## 5. Simulator and dataset decision tree — favor working components

Do **not** make RoboMimic Lift mandatory for the first overnight result. The historical repository status reports:
- **MetaWorld reach-v3 and pick-place-v3**: simulator smoke tests passed.
- **DMC walker/reacher/cartpole**: passed, but manipulation suitability limited.
- **robosuite Lift (Panda)**: failed with a joint-type assertion under then-current robosuite 1.5.2 / mujoco 3.14.
- **ManiSkill / SAPIEN**: failed due to Vulkan driver incompatibility.
- **LeRobot ALOHA and PushT**: train and simulator-eval loops have already run with osmesa.

Select the shortest **working** path to a contact-annotated manipulation task:

**Option A (preferred if ready):** Existing LeRobot ALOHA/PushT policy/evaluation loop; inspect underlying simulator access for contact annotations and consistent action-labeled demonstrations. ALOHA insertion may be challenging (prior 10% success), so first establish that a baseline learns. PushT is a viable contact-sensitive fallback but must be implemented and reported using its own policy/action environment, not falsely labeled as RoboMimic Lift.

**Option B:** MetaWorld pick-place-v3 (known simulation smoke-test pass); build a minimal policy-demo dataset and a LeRobot-compatible adapter *only if* data generation and contact labeling are practical in the actual environment. Verify whether contact IDs/locations are exposed through MuJoCo; a successful render test is **not** proof of accessible labels or high-quality demonstrations.

**Option C:** Fix robosuite compatibility and use RoboMimic Lift if fixes are quick, safe, tested, and enough data are already present. Avoid an open-ended dependency investigation that consumes the training window.

**Stop-gap if none is integration-ready:** Construct a small **controlled contact/outcome learning proof-of-concept** on generated simulation data, evaluate representation and selection quality with leakage-free splits, and clearly report that **policy-level H1/H2 remain untested**. Do not pretend proxy losses are robot policy performance.

Before picking a route, verify actual file paths and dataset counts on the compute node. Reuse pre-existing checkpoints/videos/weights. Do not download hundreds of GB of new data. The earlier server reports mention restricted downloads and low free disk; recheck both conditions.

Recommended starting targets **if data availability supports them**:
- ~50–100 action-labeled policy demonstration trajectories.
- ~500–1,000 separate supplementary RGB interaction clips.
- Fixed held-out simulator initial states / episodes.
- Video prefix 4–8 frames at 224x224; contact/outcome labels and pair identities.
- Include both successful and failed contacts; if failure examples are scarce, generate controlled variations (object position/size, gripper alignment and interactions) instead of mislabeling arbitrary videos.

**Task validity check:** initial contact/motion conditions must be comparable across agreement vs disagreement groups. Different outcomes due merely to different executed actions are not by themselves useful physics contrast.

## 6. Supervision generation and consistency checks

When using MuJoCo or a compatible simulator:
- Inspect contact records for named gripper and object collision geometries; map geom IDs to **semantic object pairs**, not raw unordered integer IDs.
- Record simulation contact time, object pose at t and t+delta, contact flags, and camera metadata if available.
- If using 2D contact points, project documented 3D world contact coordinates into the actual camera view and verify pixel coordinates visually. If calibration is unavailable, omit this optional target rather than creating invalid labels.
- Keep privileged simulator annotations in dataset metadata with explicit provenance and **do not pass them into an RGB-only coordinator**.
- Check class imbalance, false contacts, object tracking ID switches, missing observations, and future horizon boundary conditions.
- Split entire trajectories/episodes (and generated variants of the same episode/seed) before training/retrieval index creation. Hold out test scenes/episode IDs completely; guard against duplicate frames/clips across splits.

A contact predictor that sees the entire video while being scored on its final outcome is cheating. A selector that uses oracle labels must be identified as **oracle**, not RGB-only.

## 7. Efficient training settings on one full H100

Provisional settings (tune based on real throughput, not arbitrary adherence):
- Pretrained DINOv2 ViT-S/14, frozen with feature caching when compatible.
- Shared trainable adapter and small temporal head (e.g., 256 hidden width).
- LeRobot ACT; standard official behavior-cloning objective.
- BF16 autocast where stable; AdamW.
- Effective batch size around 32; smaller microbatches plus accumulation as needed.
- RGB 224x224 and 4–8 frames for supplementary clips.
- **K=3**; approximately 5,000 steps per run only as a starting estimate.
- First 100–200 steps as smoke-test/profiling, then adjust steps/learning schedule.

Do not expect all run times to scale linearly with GPU size: simulator CPU rollouts, image decoding, metadata indexing, and software OpenGL may dominate. Cache frozen vision features for policy and pool where feasible, without disconnecting gradients through the trainable adapter. Do not precompute features from a component that you intend to LoRA-tune.

To save time, create **one shared policy-only initialization checkpoint theta_0**, never exposed to the supplementary pool, then fine-tune experimental branches from *exactly* theta_0 with equal update budgets. The E0 continuation is policy-only. Report this as **matched-initialization fine-tuning**, not from-scratch comparison. Existing ALOHA ACT checkpoint may replace theta_0 only if checkpoint, environment, action space, observation shapes and preprocessing are identical; confirm by loading and evaluating it.

## 8. Required experiment matrix and priority

| ID | Condition | Purpose |
| --- | --- | --- |
| E0 | ACT-only, same continuation budget | Policy baseline |
| E1 | ACT + predictor on policy demos only | Test auxiliary loss without extra pool |
| E2 | ACT + predictor + random K support videos | Test additional video data |
| E3 | ACT + predictor + visual cosine-nearest K | Test appearance-based retrieval |
| E4 | ACT + predictor + motion-similarity K | Test motion-only retrieval |
| E5 | ACT + predictor + three-category contact/outcome complementary K | Main ACVCL v0 |

**Overnight priority: E0, E2, E3, E5**, followed by E1 and E4 if time allows. However, **E1 is conceptually important for isolating H1**; if time becomes tight, record explicitly which H1 comparisons could not be separated. Run at least one genuine held-out evaluation after every completed condition before declaring success.

For all E2–E5, match:
- Policy demos and supplemental pool IDs/splits.
- Same number of **selected support clips per training update**, including fallback accounting.
- ACT architecture, frozen backbone, shared adapter, predictor, loss weights and targets.
- Shared initialization, optimizer/update count, action training steps and augmentation.
- Evaluation seeds/initial states and rollout protocol.
- Training time and extra preprocessing separately, since equal steps are not equal wall-clock compute.

Run **1 seed first** to screen methods, then add independent seeds to promising conditions if time remains. Do not equate 20–30 evaluation episodes or one random seed with statistically conclusive gains. When possible use paired rollout seeds and provide uncertainty/CI.

The three categories themselves are working candidate heuristics, **not established novelty**. If E5 wins, separately ablate appearance variation, outcome agreement, outcome disagreement and mixed K selection. Do not announce a learning-utility or causal claim from these heuristics alone.

## 9. Evaluation requirements

**Primary:** simulator robot **task success rate on held-out rollout episodes**, not training action reconstruction or contact F1.

**Secondary:** binary contact F1 / precision / recall (with prevalence), future object displacement MAE/RMSE, ACT validation loss, retrieval validity, category coverage, GPU peak VRAM, wall time, preprocessing overhead and selection latency.

Where supported, evaluate generalization to altered object initial positions, appearance or sizes with a documented split. Do not call an unverified shift OOD.

Log and inspect a few qualitative selected video examples: query clip, selected candidate IDs, before/after frames, inferred contact/outcome reason, and failure cases. Ensure captions match actual generated visualizations.

Use the following interpretation safeguards:

- Auxiliary predictor improves but policy stays unchanged: auxiliary representation objective worked; **H1 downstream transfer not established**.
- E2 > E0 but E5 ≈ E2: extra videos help; **H2 selection not established**.
- E5 > E2/E3/E4 under matched conditions: **preliminary support for H2**, not CVPR-level proof from one seed/task.
- Baseline policy all near 0%: cannot test selection benefit; simplify task or improve shared policy initialization.
- Baselines near 100%: task may be saturated and uninformative.
- Only contact/dynamics metrics without rollout success: explicitly a **representation-only pilot**, not validated policy learning.
- Oracle selection > random but RGB selection does not: privileged information result, not deployment-ready evidence.

## 10. Practical 12-hour execution budget (after setup and downloads)

Treat the clock as a **hard planning budget**, not a guarantee:
1. **Hour 0–1:** verify active allocation, inspect datasets/checkpoints, run one end-to-end smoke test, benchmark policy and simulator throughput, verify gradient pathways.
2. **Hour 1–2:** finish bounded feature cache, simulator contact label validation, and shared baseline checkpoint if missing.
3. **Hour 2–6:** primary E0/E2/E3/E5 comparable training and minimal evaluations, checkpointing each successfully completed condition.
4. **Hour 6–9:** finish/retry important primary experiments and, if feasible, E1/E4 or a second seed.
5. **Hour 9–11:** held-out policy evaluations, support-selection diagnostics and result verification.
6. **Hour 11–12:** aggregate metrics, tables, plots, checkpoints and concise scientific conclusions.

Re-budget after throughput measurements. Use one GPU and sequential experiments unless a measured, authorized arrangement demonstrably improves completion. Do not run multiple simultaneous GPU-heavy jobs by default. If a job walltime is shorter than 12 hours, use compliant resume/checkpoint steps without evading cluster scheduling.

## 11. Implementation quality and correctness gates

Before a full run, automated tests must pass for:
- Dataset shapes, image normalization and action chunk alignment.
- Contact/object pair mapping; simulation-to-RGB label provenance.
- No train/validation/test **episode** overlap and no self-retrieval.
- No future frames leaking into temporal predictors.
- No privileged simulator states/actions used by the RGB-only selector.
- Shared trainable representation receives auxiliary gradients.
- Correct gradient accumulation and effective K sample weighting.
- Checkpoint save/load and deterministic selection for fixed seeds.
- Rollout evaluation actually reports simulator success, not a proxy.
- Degenerate hard-pair score is either fixed or excluded from claims.
- Baselines and E5 have matched initialization, update counts and support budgets.

Do not abandon the integration to build an unnecessary general framework. Prefer a minimal modular implementation under **src/** and a small number of scripts/configs for data generation, training, comparison and evaluation. Keep legacy scripts intact; new code may import from them where safe. Avoid brittle monkey-patching of LeRobot internals unless there is no maintainable alternative, and document changes.

## 12. W&B and reproducibility

The existing API key is stored **only** at **/home/thanhpv/wandb.txt**. Source the existing environment wrappers where appropriate; never print, copy or commit the key, including inside logs, scripts, prompt echoes or error traces.

For new experiments use the W&B project **acvcl-mvp** under the already configured entity if permissions allow; preserve the historical project **acvcl-reproduction**. If the new project/entity is unavailable, use a clearly named run group in an authorized project and document the deviation. Offline/local logging is a fallback if network/auth is unavailable.

Log compactly:
- Exact repo commit, config, seed, dataset/splits, simulator version, model/checkpoint provenance.
- E0–E5 condition, actual K/category counts, score/retrieval latency.
- ACT/auxiliary loss components, learning rate, step, train throughput.
- Held-out policy successes/episodes, predictor metrics and uncertainty.
- Peak allocated GPU memory, job time, storage/preprocessing overhead.

No bulk dataset or large checkpoint upload to W&B unless explicitly authorized. Save output artifacts on approved HPS storage and use a bounded retention policy.

## 13. Deliverables

Produce and verify:
1. New, working source modules (without deleting or overwriting reproduction work).
2. Minimal dataset preparation/annotation and video index scripts, with provenance.
3. Reproducible config files, commands and documented environment requirements.
4. **At least E0/E2/E3/E5 real experimental runs** when feasible; otherwise clearly mark completed subset and why the rest failed.
5. Trained checkpoints and matched-seed rollout evaluation logs.
6. W&B run references (if created) and an experiment manifest.
7. Comparison table: method, support budget K, train steps, contact F1, displacement error, task success numerator/denominator, success %, wall time, peak VRAM.
8. Loss/metric plots, example selected support clips and failure analysis.
9. A concise report at **reports/acvcl_mvp_results.md** distinguishing measured facts, engineering issues, inconclusive findings and next experiments.
10. A short README section showing exact rerun commands and the required SLURM execution context.

**Do not commit** local credentials, temporary caches, datasets, videos, model weights, W&B state, or enormous training logs. Existing reproduced-paper artifacts must remain intact.

## 14. Stretch goal — scientific novelty, only after the core MVP works

After establishing H1/H2 comparisons, consider a model-conditioned selection score that incorporates not only contact/outcome category relevance but also **marginal training utility** of a candidate for the shared representation/policy. A cheap first-order gradient alignment on trainable adapter parameters is an optional exploratory proxy; it can reflect redundancy and auxiliary-policy conflict, and is **not** by itself proven downstream utility.

Relevant inspirations to inspect if needed: VRB (CVPR 2023) for contact/post-contact visual targets; Robo-ABC (ECCV 2024) for contact affordance correspondence; STRAP (ICLR 2025) for temporally aligned retrieval; GradMatch (ICML 2021) and DataMIL (ICLR 2026) for gradient/utility-aware data selection. **These are related work / possible techniques, not direct drop-in equivalents or already-reproduced ACVCL baselines.** Verify actual code and input/supervision assumptions before citing or adapting anything.

Do **not** trade away valid evaluation for a speculative sophisticated selector.

## 15. Agent execution instructions

1. Read repo documents and inspect actual state; report a compact reuse/missing-components table.
2. Perform the RULE.md cluster/resource preflight. **Never** execute compute on the login node.
3. Select the fastest valid environment/data/policy route using the decision tree above.
4. Implement a true end-to-end smoke path: one action-labeled policy batch, one supplementary RGB clip, contact/outcome target, both losses, shared gradients, and one simulator rollout.
5. Build leakage-free train/pool/validation/test splits and cached retrieval features.
6. Benchmark throughput; commit to a realistic within-allocation experiment schedule.
7. Run primary matched experiments and evaluate them honestly.
8. Summarize actual evidence for H1 and H2; produce report, reproducibility commands and artifacts.

Proceed autonomously on ordinary coding/debug choices, but ask for intervention when permissions, cluster rules, missing access or high-risk storage/resource actions require approval. A scientifically honest partial result is preferable to an invented successful experiment.

**Definition of MVP success:** A functioning ACT-based robot policy, a contact/post-contact auxiliary learner with proven shared-gradient connection, a valid RGB-support retrieval/coordinator, and held-out policy evaluations comparing ACVCL with suitable matched baselines. A failed hypothesis is still a valid result if the test is correct.

**Start execution now, inside the existing authorized compute allocation; do not stop after a plan.**
