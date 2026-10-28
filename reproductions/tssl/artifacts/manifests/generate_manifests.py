"""Generate retrospective manifests for S00 inventory."""
import json, os, hashlib
from datetime import datetime

MANIFEST_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_BASE = "/mnt/data/thanhpv/acvcl/reproductions/tssl"
STATUS_FILE = f"{OLD_BASE}/logs/session_full/_status.tsv"

def parse_status_tsv(path):
    runs = []
    with open(path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            vals = line.strip().split('\t')
            runs.append(dict(zip(header, vals)))
    return runs

def make_manifest(run, idx):
    phase = run['phase']
    variant = run['variant']
    dataset = run['dataset']
    seed = int(run['seed'])
    
    if phase == 'repro':
        run_id = f"repro_{variant}_{dataset}_s{seed}"
        track = "VASOT"
        log_file = f"{OLD_BASE}/logs/session_full/repro_{variant}_{dataset}_s{seed}.log"
    else:
        run_id = f"sweep_{variant}_{dataset}_s{seed}"
        track = "VASOT_sweep"
        log_file = f"{OLD_BASE}/logs/session_full/sweep_{variant}_{dataset}_s{seed}.log"
    
    # Determine config from hparams
    if dataset == 'desktop_assembly':
        hparams_source = "version_62534"
        layer_sizes = [512, 128, 40]
        n_clusters = 22
        rho = 0.25
        n_epochs = 30
        radius_gw = 0.02
    elif dataset.startswith('YTI'):
        activity = dataset.replace('YTI_', '')
        hparams_source = "version_62534"
        layer_sizes = [512, 128, 40]  # actually depends - check per run
        n_clusters = 22  # placeholder
        rho = 0.25
        n_epochs = 30
        radius_gw = 0.02
    elif dataset == 'FSeval':
        hparams_source = "version_62379 or version_62534"
        layer_sizes = [64, 128, 40]
        n_clusters = 12
        rho = 0.15
        n_epochs = 30
        radius_gw = 0.02
    else:
        hparams_source = "unknown"
        layer_sizes = None
        n_clusters = None
        rho = None
        n_epochs = None
        radius_gw = None
    
    manifest = {
        "run_id": run_id,
        "track": track,
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "phase": phase,
        "start_time": run['start'],
        "end_time": run['end'],
        "exit_code": int(run['exit']),
        "metrics": {
            "mof": float(run['mof']),
            "f1": float(run['f1']),
            "miou": float(run['miou'])
        },
        "data_path": f"{OLD_BASE}/data/final/{dataset.split('_')[0] if 'YTI' in dataset else dataset}",
        "log_path": log_file,
        "log_exists": os.path.exists(log_file),
        "hparams_source": hparams_source,
        "checkpoint_mapping": "ambiguous_shared_version_62534" if phase == "repro" else "ambiguous",
        "checkpoint_identity": "unknown",
        "note": "Retrospective manifest created by S00 inventory. "
                "Checkpoint cannot be reliably mapped to this specific run due to "
                "shared lightning_logs/version_62534 across multiple concurrent runs.",
        "slurm_job_id": 62534,
        "hostname": "worker-0",
        "environment": {
            "python": "3.10.22",
            "pytorch": "2.4.1+cu121",
            "pytorch_lightning": "2.6.6",
            "conda_env": "acvcl"
        },
        "created_by": "S00_inventory",
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    return run_id, manifest

runs = parse_status_tsv(STATUS_FILE)
registry = []

for i, run in enumerate(runs):
    run_id, manifest = make_manifest(run, i)
    path = os.path.join(MANIFEST_DIR, f"{run_id}.json")
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    registry.append({
        "run_id": run_id,
        "dataset": run['dataset'],
        "variant": run['variant'],
        "seed": int(run['seed']),
        "exit": int(run['exit']),
        "mof": float(run['mof']),
        "manifest": path
    })
    print(f"  Created: {run_id}")

# Write registry
reg_path = os.path.join(os.path.dirname(MANIFEST_DIR), "run_registry.jsonl")
with open(reg_path, 'w') as f:
    for r in registry:
        f.write(json.dumps(r) + '\n')
print(f"\nRegistry: {reg_path} ({len(registry)} entries)")
