# H100 Shared Cluster — Mandatory Agent Safety Rules

**Scope:** Every AI coding agent (Codex, Claude Code, subagents, scripts it launches) operating on this shared Slurm-managed H100 cluster.

**Priority:** These rules are mandatory. Cluster administrator instructions and site policies take precedence. If the intended action might violate either, **STOP and ask the user**. Do not infer permissions from technical access.

## 0. Non-negotiable operating principles

1. **Login nodes are for access and job control only.** On `login-restricted-*`, restrict activity to SSH, basic shell navigation, lightweight inspection, editing small files, and Slurm commands such as `sbatch`, `squeue`, `scancel`. **Never run agents, background agent services, training, inference, builds, downloads, data preprocessing, or compute-heavy scans on a login node.** An agent reading this rule on a login node must not continue executing there; its workload must be launched inside a properly allocated compute job.
2. **No compute without a Slurm allocation.** Any substantive work must execute on an allocated compute node through `sbatch`/`srun` and stay within that job's CPU, memory, GPU and time limits. Merely being able to run a command is not permission.
3. **Do not degrade a shared service.** Avoid unnecessary CPU, RAM, filesystem metadata I/O, network I/O, GPU reservation, or large numbers of processes. Never bypass quotas, cgroups, Slurm scheduling, or administrator safeguards.
4. **Fail closed.** When host type, storage destination, permission, quota, or intended resource use is uncertain, do not start resource-intensive work. Ask the user or administrator.

## 1. Detect execution context before work

Run only **lightweight** checks (safe on login nodes):

```bash
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-}"
printf 'SLURM_CPUS_PER_TASK=%s\n' "${SLURM_CPUS_PER_TASK:-}"
printf 'SLURM_MEM_PER_NODE=%s\n' "${SLURM_MEM_PER_NODE:-}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
```

**Required decision:**
- Hostname matches `login-restricted-*` → **STOP all agent work on this host.** Only provide the user with a safe Slurm submission command or edit a small job script if allowed. Do not start daemonized or persistent agents on login nodes.
- No valid Slurm allocation (`SLURM_JOB_ID` unset) → **DO NOT perform compute**; request or submit an appropriate allocation. Check both hostname and allocation; neither alone proves compliance.
- Inside a compute allocation → proceed only with checks below.

## 2. CPU, RAM, and GPU — enforce limits

- Every Slurm job must request explicit, justified resources: `--partition`, `--cpus-per-task`, `--mem`, `--time`, and `--gres` **only if GPUs are needed**. Avoid excessive allocations and overlapping jobs.
- **Never request a GPU merely to keep Codex/Claude Remote Control connected.** Use an approved CPU-only allocation where available; ask the administrator if none exists.
- On this cluster, GPU jobs may use H100 MIG slices (e.g. `--gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1`), but do not assume that a specific partition or resource is always available or allowed.
- Cap CPU-bound library threads and workers to allocated CPUs (including Python DataLoader workers, preprocessing pools, BLAS/OpenMP). `OMP_NUM_THREADS` alone does not restrict process counts.
- Respect Slurm memory allocations; do not rely on physical free RAM or swap as permission to use it. Reduce batch size/workers if close to the job limit.
- Never access GPUs outside those allocated to the job; respect `CUDA_VISIBLE_DEVICES` and do not override it.
- Set a finite `--time`. End jobs promptly when idle or complete; do not hold GPU allocations for dormant interactive sessions.
- Do not create uncontrolled subprocess trees, detached sessions, cron jobs, or background agents. Subagents inherit all these rules.
- Monitor resource usage using available Slurm accounting (`squeue`, `sstat`, `sacct`) and job-local GPU monitoring. Monitoring tools may not report every metric on every cluster.

## 3. Storage rules: `$HOME`, HPS, `/tmp`, `/dev/shm`

### `$HOME` (shared root filesystem — keep small)
- Keep only essential source code, small configuration files, Slurm scripts, and necessary tool metadata.
- **Do not store large datasets, checkpoints, model weights, training outputs, videos, massive logs, or growing caches in `$HOME`.**
- Redirect newly generated large outputs to an **authorized per-user location under `/mnt/data` (HPS)**. Confirm the directory and permissions before writing. Do not assume `/mnt/data/$USER` exists.
- Avoid creating duplicate venvs or leaving unused archives; clean only after verifying they are unused and safe to remove.

### `/mnt/data` (HPS — large persistent research data)
- Use for approved datasets, checkpoints, model artifacts, and substantial outputs.
- Do not assume unlimited space or unlimited I/O throughput. Keep data under your authorized directory; monitor growth.
- Avoid pathological small-file workloads or unbounded checkpoint creation; implement retention, sensible saving intervals, and aggregation where appropriate.
- Do not relocate live virtual environments or active caches without assessing hardcoded paths, locking, and running jobs.

### `/tmp` and `/dev/shm` (high-risk shared RAM on login nodes)
- **Never use `/tmp` or `/dev/shm` on login nodes to store models, datasets, caches, `CODEX_HOME`, remote-control state, or large temporary files.** The admin reports that `/tmp` on login nodes is RAM-backed and that files there directly consumed scarce login-node RAM.
- **Never move `CODEX_HOME` to `/dev/shm`.** Keep it in a small, persistent user-owned directory (typically `~/.codex`) unless an approved alternative is specified.
- Within compute jobs, use **administrator-approved job-local scratch** (e.g. `$SLURM_TMPDIR` if provided) for ephemeral scratch. Do not assume `/tmp` on compute nodes is disk-backed or unconstrained.
- Set `TMPDIR` for applications that honor it, but verify other programs' actual paths; some write to `/tmp` directly.
- Clean up only the job's **own specifically named temporary directory**, after checking the path and ensuring no running process needs it. A shell `trap` is helpful but does not guarantee cleanup after abrupt termination.
- Never remove or modify another user's files. Never indiscriminately clean `/tmp`, `/dev/shm`, shared caches, or global directories.

## 4. AI agents, sockets, remote control, and VS Code

- Run Codex/Claude Remote Control **inside a Slurm compute allocation**, not on login nodes. Pairing/authentication is not a substitute for compute allocation.
- Keep socket directories private to the current user (`0700`) and use a site-approved filesystem/location. **Do not evade socket ownership/security checks by changing permissions on shared `/tmp` or using world-writable unsafe directories.**
- Do not allow idle agents to run for days. Set job limits, terminate unneeded agent sessions, and verify corresponding Slurm jobs stop.
- Do not spawn repeated retries, redundant background agent instances, or orphan processes after failed remote-control startups.
- Avoid accumulating stale VS Code servers; shut down your own unused sessions safely. Never stop another user's services.
- Log output must be size-controlled; avoid verbose stdout/stderr or runaway session transcripts that silently fill shared storage.

## 5. Downloading, installing, and running experiments

- **Before any download, extraction, install, training, dataset conversion, or checkpoint-producing run**, estimate bytes/files, specify the destination, check authorized free space/quota, and get user approval if unusually large or uncertain.
- Run resource-heavy downloads, archive extraction, environment builds, and recursive scans only in a Slurm compute allocation and consistent with site rules.
- Point applicable caches (`HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `PIP_CACHE_DIR`, etc.) to approved storage when their size may grow substantially. Be aware this does not retroactively move existing caches.
- Do not spawn massive parallel jobs, unlimited dataloader workers, high-frequency checkpointing, or large temporary copies.
- Avoid duplicating large compressed and extracted data; confirm integrity before deleting the redundant copy.
- Do not perform cluster-wide scans, read other users' private directories, or run aggressive `du`/`find` against shared storage without necessity and permission.

## 6. Required preflight for every substantial action

Before launching, the agent must be able to answer **YES** to all relevant items:

1. Am I on a compute node in a valid Slurm allocation, not a login node?
2. Are CPU, RAM, GPU (if any), and walltime explicitly bounded and appropriate?
3. Are dataset, cache, checkpoint, output, logging, and temporary-file paths known and authorized?
4. Will large persistent files go to HPS rather than `$HOME`?
5. Am I avoiding login-node `/tmp` and `/dev/shm`, and preventing uncontrolled file growth?
6. Are worker counts, parallelism, logging volume, and checkpoint frequency bounded?
7. Is there a safe cleanup plan that touches **only my own files/jobs**?
8. If an operation could consume substantial resources, did I obtain approval for its scale and location?

**If any answer is NO or UNKNOWN, do not execute the action.** Explain the blocker and propose the smallest compliant alternative.

## 7. Safe stop and cleanup protocol

1. Identify **only your own** Slurm jobs (`squeue -u "$USER"`) and processes; do not mass-kill unrelated jobs.
2. Gracefully stop the relevant process or `scancel <specific_job_id>` when needed.
3. Confirm it has stopped before moving/removing its state or scratch files.
4. Inspect exact candidate paths and their contents; never run a destructive wildcard or `rm -rf` against an unchecked, empty, root, or shared path.
5. Delete only artifacts known to be disposable. Preserve valuable checkpoints and sessions until explicitly authorized.
6. Recheck storage and queue status with lightweight, targeted commands.

## 8. Stop conditions — immediate user escalation

**STOP and ask the user/admin** if you encounter any of these:
- Agent is running on a login node or cannot establish a legitimate Slurm allocation.
- Temporary files or caches are unexpectedly growing, or may fill RAM-backed `/tmp`/`/dev/shm`.
- An operation might exceed job memory/time, disk quota, or storage/metadata capacity.
- A GPU job is idle for an extended period without a justified purpose.
- An action would require weakening permissions, bypassing access controls, killing others' processes, or deleting shared data.
- Cluster rules conflict with the requested operation, or you do not know the approved directory/partition.

## 9. Admin notice grounding and updates

These rules reflect cluster messages provided by the user (October 2026):
- `$HOME` sits on shared root storage; move checkpoints and large research data to `/mnt/data` (HPS).
- Login nodes (`login-restricted-1` / `login-restricted-2`) are for SSH and job submission, **not computing or running AI agents**.
- Login-node `/tmp` is RAM-backed; accumulated files caused substantial RAM pressure and service disruption.
- Stale VS Code servers, agents running for days, and putting `CODEX_HOME` on `/dev/shm` were explicitly flagged.

These messages do **not** specify a universally safe `$HOME` quota, `/mnt/data` quota, or acceptable CPU/RAM/GPU numeric threshold. **Do not invent such thresholds.** Follow later administrator guidance if stricter.

---

**Agent acknowledgment:** Before running commands that create significant load or data, briefly state: (1) current host/allocation, (2) resource limits, (3) output/cache/tmp paths, (4) cleanup plan. If you cannot verify these, STOP.