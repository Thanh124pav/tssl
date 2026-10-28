"""Import shim for the vendored nsf_debiasing repo.

`ssc_common.py` imports mmcv but uses exactly one thing from it,
`mmcv.ProgressBar`, in three places. mmcv is a heavy OpenMMLab package that
wants to build CUDA ops against a specific torch, so a tqdm-backed stand-in
with the same tiny API is used instead. It is not listed in the repo's README
requirements either.
"""

import sys
import types


class _ProgressBar:
    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.completed = 0
        self._file = file
        if start:
            self.start()

    def start(self):
        self.completed = 0

    def update(self, num_tasks=1):
        self.completed += num_tasks
        if self.task_num and self.completed % max(1, self.task_num // 10) == 0:
            pct = 100.0 * self.completed / self.task_num
            print(f"  [{self.completed}/{self.task_num}] {pct:.0f}%", file=self._file, flush=True)


def install():
    if "mmcv" in sys.modules:
        return
    mmcv = types.ModuleType("mmcv")
    mmcv.ProgressBar = _ProgressBar
    sys.modules["mmcv"] = mmcv
