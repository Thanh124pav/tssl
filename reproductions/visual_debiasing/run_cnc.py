"""Run Correct-N-Contrast with W&B logging.

CnC ships no W&B support and resolves every path relative to its own repo
root, so this wrapper chdir's into the repo, hooks the one function that
computes per-group accuracy (`utils.logging.summarize_acc`) and forwards the
average and worst-group numbers it returns. Everything else, including the
hyperparameters, stays as the official command line specifies.
"""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CNC = REPO / "third_party" / "correct-n-contrast"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default="acvcl-reproduction")
    parser.add_argument("--group", default="visual_debiasing")
    parser.add_argument("--tags", default="")
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("cnc_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cnc_argv = [a for a in args.cnc_args if a != "--"]

    os.chdir(CNC)
    sys.path.insert(0, str(CNC))

    import wandb

    if not args.no_wandb:
        wandb.init(
            project=args.project,
            name=args.run_name,
            group=args.group,
            tags=[t for t in args.tags.split(",") if t],
            config={"cnc_argv": " ".join(cnc_argv)},
            dir=os.environ.get("WANDB_DIR", str(REPO / "outputs" / "wandb")),
        )

    import utils.logging as cnc_logging

    original = cnc_logging.summarize_acc
    history = []

    def hooked_summarize_acc(correct_by_groups, total_by_groups, stdout=True):
        avg_acc, worst_acc = original(correct_by_groups, total_by_groups, stdout)
        caller = inspect.stack()[1].function
        record = {"caller": caller, "avg_acc": avg_acc, "worst_group_acc": worst_acc}
        history.append(record)
        if not args.no_wandb:
            import torch

            payload = {
                f"{caller}/avg_acc": avg_acc,
                f"{caller}/worst_group_acc": worst_acc,
            }
            if torch.cuda.is_available():
                payload["gpu_mem_peak_MB"] = round(
                    torch.cuda.max_memory_allocated() / 2**20, 1
                )
            wandb.log(payload)
        return avg_acc, worst_acc

    cnc_logging.summarize_acc = hooked_summarize_acc

    import train as cnc_train
    import train_supervised_contrast as cnc_main

    cnc_train.summarize_acc = hooked_summarize_acc
    cnc_main.summarize_acc = hooked_summarize_acc

    sys.argv = ["train_supervised_contrast.py"] + cnc_argv
    start = time.perf_counter()
    status = "completed"
    try:
        cnc_main.main()
    except SystemExit as exc:
        status = f"sys_exit:{exc.code}"
    wall_s = round(time.perf_counter() - start, 2)

    test_records = [r for r in history if r["caller"] == "test_model"]
    summary = {
        "run_name": args.run_name,
        "status": status,
        "wall_time_s": wall_s,
        "n_evaluations": len(history),
        "final_avg_acc": test_records[-1]["avg_acc"] if test_records else None,
        "final_worst_group_acc": test_records[-1]["worst_group_acc"] if test_records else None,
        "best_worst_group_acc": (
            max(r["worst_group_acc"] for r in test_records) if test_records else None
        ),
        "cnc_argv": " ".join(cnc_argv),
    }
    print(json.dumps(summary, indent=2))

    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "history": history}, indent=2))

    if not args.no_wandb:
        wandb.summary.update(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
