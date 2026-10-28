"""Run nsf_debiasing's ssc.py (the Let Samples Speak method) with the mmcv shim.

ssc.py reads its own hyperparameters from the dataset name, so the only
argument is the dataset. It loads the ERM checkpoint that train_supervised.py
produced, reached through third_party/nsf_debiasing/logs/waterbirds/erm_seed1.
"""

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NSF = REPO / "third_party" / "nsf_debiasing"

os.chdir(NSF)
sys.path.insert(0, str(NSF))
sys.path.insert(0, str(HERE))

import nsf_compat

nsf_compat.install()

dataset = sys.argv[1] if len(sys.argv) > 1 else "waterbirds"
sys.argv = ["ssc.py", dataset]
runpy.run_path(str(NSF / "ssc.py"), run_name="__main__")
