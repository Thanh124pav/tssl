# SETUP — Active Cross-Video Consensus Learning (ACVCL)

## 1. Research Context

This project studies **Active Cross-Video Consensus Learning (ACVCL)**.

The target setting is representation learning from **cheap, noisy, and heterogeneous Internet videos**.

Existing cross-video learning methods typically rely on:
- random pairing;
- same-class or same-task pairing;
- similarity-based grouping;
- or video sets that are constructed in advance.

The long-term goal of this project is to develop a framework that **dynamically selects which video pairs or sets should be cross-learned based on the current model state**, and then performs cross-video consensus learning on the selected groups.

At this stage, **DO NOT implement the full proposed ACVCL method yet**.

The current goals are:

1. Set up the environment.
2. Prepare the relevant baseline codebases.
3. Download the required datasets and metadata.
4. Build a minimal codebase that can run cross-video learning smoke tests.
5. Validate the GPU, dataloader, feature, and dataset pipelines.

---

## 2. Compute Constraints

Current machine:

- 1 NVIDIA H100 GPU.
- Only 4 CPU cores.
- Do not pretrain a large video foundation model from scratch.
- Avoid CPU-heavy preprocessing whenever possible.
- Prefer pretrained and frozen visual encoders.
- For early smoke tests, prefer precomputed features over raw-video decoding.

Design everything to be **GPU-heavy, compute-efficient, and CPU-light**.

Do not set up robotics packages yet.

---

## 3. Python Environment and Libraries

Create a dedicated environment:

```bash
conda create -n acvcl python=3.10 -y
conda activate acvcl
```

Install the core dependencies:

```bash
pip install -U pip setuptools wheel

pip install \
    torch torchvision \
    transformers accelerate \
    timm einops \
    numpy scipy pandas \
    scikit-learn \
    opencv-python-headless \
    av decord \
    imageio imageio-ffmpeg \
    h5py \
    tqdm \
    wandb \
    hydra-core omegaconf \
    matplotlib \
    datasets huggingface_hub
```

If a video package causes dependency conflicts, keep the environment simple and document the issue instead of forcing every package to coexist.

Do not install robotics packages such as:
- robosuite;
- robomimic;
- ManiSkill;
- RoboCasa;
- LeRobot.

After installation, run:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_bf16_supported())
```

Also verify:
- `ffmpeg` is available;
- PyAV or decord can decode a sample video;
- Hugging Face models can be loaded;
- `wandb` imports correctly;
- CUDA memory is visible and usable.

---

## 4. Baseline Codebases to Reuse or Reference

Do not merge all external repositories into the main codebase.

Clone or inspect the following baselines and reuse only the necessary components.

### A. Cross-Video Learning Baselines

#### TCC — Temporal Cycle-Consistency Learning

Use it to study:
- cross-video temporal correspondence;
- pair sampling;
- cycle-consistency objectives;
- temporal alignment losses.

#### LAV — Learning by Aligning Videos in Time

Use it to study:
- cross-video temporal alignment;
- Soft-DTW-style alignment;
- video-pair sampling;
- alignment losses.

If an implementation is outdated or dependency-heavy, **do not force the original repository to run**.

Instead:
- inspect the implementation;
- port the relevant modules into a minimal modern PyTorch implementation.

### B. Recent Cross-Video References

Look for official code if available.

#### VideoSetDiff — ICCV 2025

Use it to study:
- how similar video sets are constructed;
- how multi-video representations are processed;
- static video-set construction baselines.

#### SYNCR — 2026

Use it to study:
- structured cross-video learning;
- cross-video supervision;
- temporal and relational multi-video tasks.

These repositories are primarily references at this stage.

Do not run large models if the compute or data requirements are excessive.

### C. Active-Selection References

Inspect implementations or pseudocode for:
- MDS-VQA, CVPR 2026;
- ActiveAD, CVPR 2026;
- active pair mining;
- hard-pair mining.

We mainly need ideas for:
- uncertainty;
- diversity;
- model-dependent utility;
- candidate ranking.

Do not reproduce their entire downstream tasks.

---

## 5. Main Project Structure

Create the following project structure:

```text
acvcl/
├── README.md
├── requirements.txt
├── configs/
│   ├── dataset/
│   ├── model/
│   └── experiment/
│
├── src/
│   ├── datasets/
│   │   ├── crosstask.py
│   │   ├── coin.py
│   │   └── video_utils.py
│   │
│   ├── models/
│   │   ├── encoder.py
│   │   ├── temporal_head.py
│   │   └── consensus.py
│   │
│   ├── selectors/
│   │   ├── random.py
│   │   ├── similarity.py
│   │   └── hard_pair.py
│   │
│   ├── losses/
│   │   ├── tcc.py
│   │   └── alignment.py
│   │
│   └── utils/
│
├── scripts/
│   ├── download_data.sh
│   ├── inspect_dataset.py
│   ├── extract_features.py
│   ├── smoke_test_pairs.py
│   └── train_toy_consensus.py
│
└── data/
    ├── crosstask/
    ├── coin/
    ├── youcook2/
    └── howto100m/
```

Do not over-engineer.

The code should be modular, readable, and easy to extend with additional selectors later.

---

## 6. Datasets to Prepare

### Dataset 1 — CrossTask

This is the highest-priority dataset.

Download:
- annotations;
- task definitions;
- split metadata;
- precomputed visual/video features if still available.

Prefer **precomputed features** over raw video initially.

Reasons:
- multiple executions of the same instructional task;
- suitable for cross-video temporal alignment;
- avoids repeated CPU-heavy video decoding.

After download, implement:

```bash
python scripts/inspect_dataset.py dataset=crosstask
```

The script should report:
- number of tasks;
- number of videos;
- videos per task;
- feature shapes;
- annotation structure;
- example pairs from the same task.

### Dataset 2 — COIN

Download first:
- annotations;
- taxonomy;
- metadata;
- video IDs.

Do not download the full raw-video dataset immediately.

If the downloader still works, download only a small subset:

```text
5–10 tasks
100–500 videos
```

Use this subset to validate the raw-video pipeline.

### Dataset 3 — YouCook2

Download:
- annotations;
- train/validation splits;
- precomputed features if available.

Do not download all raw videos yet.

Use YouCook2 as a second instructional-video dataset for cross-dataset validation.

### Dataset 4 — HowTo100M

**Do not download the full dataset.**

Only download:
- metadata;
- task/category information if available;
- evaluation metadata.

The purpose is only to prepare the pipeline for future experiments on large-scale noisy Internet videos.

---

## 7. Pretrained Visual Encoder

Do not train a large encoder from scratch.

Prepare at least one frozen visual encoder.

Preferred options:
- DINOv2;
- CLIP;
- SigLIP.

For the MVP, use DINOv2 or CLIP.

Target pipeline:

```text
video
→ sample 8–16 frames
→ frozen image encoder
→ frame features
→ temporal module
```

If a dataset already provides precomputed features, bypass the visual encoder.

For the temporal module, use only:
- mean pooling, or
- a small 2-layer Transformer.

Do not use a large video foundation model yet.

---

## 8. Baseline Selectors to Implement Immediately

Implement three simple selectors.

### Random

```text
Randomly sample two videos from the same task or category.
```

### Similarity

Compute pooled video features:

```text
z_video = mean(frame_features)
```

Then select nearest neighbors using cosine similarity.

### Hard Pair

Select videos that:
- have reasonably high semantic similarity;
- but differ strongly in model representation or prediction.

Do not implement the proposed ACVCL selector yet.

Use a common selector API:

```python
pairs = selector.select(
    model=model,
    dataset=dataset,
    num_pairs=k,
)
```

This should allow future ACVCL selectors to be added without modifying the training pipeline.

---

## 9. Experiments and Smoke Tests to Run Now

### Test A — Dataset Sanity Check

On CrossTask:

- sample two videos from the same task;
- load features;
- inspect tensor shapes;
- visualize the temporal cosine-similarity matrix.

Save the figure.

### Test B — Random vs Similarity Pairing

On a small CrossTask subset:

- sample random pairs;
- sample nearest-neighbor pairs.

Measure:
- feature cosine similarity;
- temporal alignment score;
- TCC or alignment loss.

Goal:
determine whether pairing strategy changes the learning signal in a measurable way.

### Test C — Minimal TCC/LAV-Style Objective

Implement a minimal cross-video alignment objective.

Do not try to reproduce the original paper numbers.

Only verify that:

```text
CrossTask pair
→ temporal features
→ alignment/cycle loss
→ backward()
→ loss decreases
```

Run for a few hundred to a few thousand iterations.

### Test D — Static Pairing Comparison

Keep the same model and loss.

Compare:

```text
Random
vs
Similarity
vs
Hard Pair
```

Measure:
- training loss;
- validation alignment;
- frame retrieval if annotations allow it.

This is the most important early experiment before implementing ACVCL.

### Test E — Feature Extraction Pipeline

Download 10–20 raw videos from COIN.

Run:

```text
raw MP4
→ frame sampling
→ DINOv2/CLIP
→ save features
```

Benchmark:
- videos per second;
- GPU utilization;
- CPU utilization;
- storage per video.

Because the machine has only 4 CPU cores:
- start with `num_workers <= 2`;
- do not decode the same videos repeatedly during training;
- cache extracted features on disk.

---

## 10. Do Not Do Yet

Do not:
- download full HowTo100M;
- download full COIN if it is large;
- pretrain a large video model;
- train VLMs or VLAs;
- set up robotics simulation;
- set up ACT;
- set up Diffusion Policy;
- set up RoboCasa;
- run online RL;
- run offline RL;
- implement the full ACVCL method.

The current goal is only to build a **clean experimental testbed for active cross-video selection**.

---

## 11. Expected Deliverables

### `README.md`

Document:
- environment setup;
- datasets downloaded;
- dataset paths;
- baseline repositories inspected;
- how to run each smoke test.

### `SETUP_STATUS.md`

Create a short status report:

```text
GPU:
CUDA:
PyTorch:

Datasets:
[ ] CrossTask
[ ] COIN
[ ] YouCook2
[ ] HowTo100M metadata

Baselines:
[ ] TCC reference
[ ] LAV reference
[ ] VideoSetDiff reference
[ ] SYNCR reference

Smoke tests:
[ ] CUDA
[ ] video decoding
[ ] CrossTask loading
[ ] random pairing
[ ] similarity pairing
[ ] alignment loss backward
[ ] feature extraction
```

### Minimal Experiment Result

Produce a table:

```text
Pairing strategy | Train loss | Val alignment | Runtime
Random
Similarity
Hard Pair
```

If no standard validation metric is available yet, at least report:
- loss convergence;
- runtime;
- qualitative alignment examples.

---

## 12. Final Report

Prioritize **correctness and reproducibility**, not performance optimization.

If an old baseline repository has dependency conflicts, do not break the main environment to support it. Keep it under `third_party/` or inspect the code and port only the required modules.

At the end, report:

1. Which libraries were installed successfully.
2. Which datasets were downloaded successfully.
3. Which datasets or download URLs are unavailable or broken.
4. Which baseline repositories can run.
5. Which smoke tests passed or failed.
6. What issues remain for the next stage.
