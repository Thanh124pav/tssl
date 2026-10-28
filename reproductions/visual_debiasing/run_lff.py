"""Run Learning from Failure (Colored MNIST) with W&B logging.

LfF logs through TensorBoard, so `SummaryWriter.add_scalar` is mirrored into
W&B: that single hook catches every metric the official training loop reports,
including the unbiased/skewed/aligned accuracy splits that are the paper's
headline numbers.

Invoked as a thin front-end to LfF's sacred CLI, e.g.

    python run_lff.py --run-name cmnist_lff --script train.py -- \
        with server_user colored_mnist skewed3 severity4
"""

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LFF = REPO / "third_party" / "LfF"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--script", default="train.py", choices=["train.py", "train_vanilla.py"])
    parser.add_argument("--project", default="acvcl-reproduction")
    parser.add_argument("--group", default="visual_debiasing")
    parser.add_argument("--tags", default="")
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("sacred_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    sacred_argv = [a for a in args.sacred_args if a != "--"]

    os.chdir(LFF)
    sys.path.insert(0, str(LFF))
    sys.path.insert(0, str(HERE))

    import lff_compat

    lff_compat.install()

    import wandb

    if not args.no_wandb:
        wandb.init(
            project=args.project,
            name=args.run_name,
            group=args.group,
            tags=[t for t in args.tags.split(",") if t],
            config={"lff_script": args.script, "sacred_argv": " ".join(sacred_argv)},
            dir=os.environ.get("WANDB_DIR", str(REPO / "outputs" / "wandb")),
        )

    from torch.utils.tensorboard import SummaryWriter

    original_add_scalar = SummaryWriter.add_scalar
    scalars = {}

    def hooked_add_scalar(self, tag, scalar_value, global_step=None, *rest, **kwargs):
        value = float(scalar_value)
        scalars.setdefault(tag, []).append((global_step, value))
        if not args.no_wandb:
            payload = {tag: value}
            if global_step is not None:
                payload["step"] = global_step
            wandb.log(payload)
        return original_add_scalar(self, tag, scalar_value, global_step, *rest, **kwargs)

    SummaryWriter.add_scalar = hooked_add_scalar

    sys.argv = [args.script] + sacred_argv

    start = time.perf_counter()
    status = "completed"
    try:
        # sacred's @ex.automain only fires under __main__, so a plain import
        # would load the module and train nothing.
        runpy.run_path(str(LFF / args.script), run_name="__main__")
    except SystemExit as exc:
        status = "completed" if exc.code in (0, None) else f"sys_exit:{exc.code}"
    wall_s = round(time.perf_counter() - start, 2)

    finals = {tag: values[-1][1] for tag, values in scalars.items()}
    bests = {
        f"best/{tag}": max(v for _, v in values)
        for tag, values in scalars.items()
        if "acc" in tag.lower()
    }
    summary = {
        "run_name": args.run_name,
        "script": args.script,
        "status": status,
        "wall_time_s": wall_s,
        "sacred_argv": " ".join(sacred_argv),
        "final": finals,
        **bests,
    }
    print(json.dumps(summary, indent=2))

    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "scalars": scalars}, indent=2))

    if not args.no_wandb:
        wandb.summary.update({k: v for k, v in summary.items() if k != "final"})
        wandb.summary.update({f"final/{k}": v for k, v in finals.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
