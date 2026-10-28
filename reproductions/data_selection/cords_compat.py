"""Runtime shims that let the vendored CORDS repo import and load data here.

Two unrelated obstacles, both handled without editing third_party/cords:

1. `cords/.../SL/builder.py` imports `torchtext.data` and
   `sklearn.datasets.load_boston` at module level. torchtext pins torch 1.x
   (unusable on an H100) and load_boston was removed in scikit-learn 1.2, so
   the import fails before any CIFAR-10 code runs. Both are only needed by
   NLP/regression branches we do not use.
2. The cluster network policy blocks cs.toronto.edu and cs231n.stanford.edu,
   so torchvision's CIFAR-10 and CORDS' Tiny ImageNet downloads fail. The
   HuggingFace mirrors carry identical data, so the dataset classes below
   serve it through the same API that builder.py expects.
"""

import os
import sys
import types

import numpy as np
import torch
from PIL import Image

HF_CACHE = os.environ.get("HF_HOME", "/mnt/data/thanhpv/cache/huggingface")


def _stub_module(name, attrs=()):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for attr in attrs:
        setattr(mod, attr, object)
    sys.modules[name] = mod


def install_import_shims():
    # Only reached by MILO's global-ordering paths, which we do not run.
    # submodlib needs a C++ build that is not worth pulling in for that.
    _stub_module("sentence_transformers", ("SentenceTransformer", "util"))
    _stub_module("submodlib")

    if "torchtext" not in sys.modules:
        torchtext = types.ModuleType("torchtext")
        torchtext.data = types.ModuleType("torchtext.data")
        torchtext.data.Field = object
        torchtext.data.TabularDataset = object
        torchtext.data.BucketIterator = object
        sys.modules["torchtext"] = torchtext
        sys.modules["torchtext.data"] = torchtext.data

    import sklearn.datasets

    def _removed(*_args, **_kwargs):
        raise RuntimeError("load_boston was removed from scikit-learn >=1.2")

    # Assigning shadows sklearn's module __getattr__, which raises ImportError
    # on attribute access rather than AttributeError, so hasattr cannot test it.
    sklearn.datasets.load_boston = _removed


class HFImageDataset(torch.utils.data.Dataset):
    """torchvision-style image classification dataset backed by a HF mirror."""

    def __init__(self, repo, split, transform=None, image_key="img", preload=True):
        from datasets import load_dataset

        self._ds = load_dataset(repo, split=split, cache_dir=HF_CACHE)
        self.transform = transform
        self._image_key = image_key if image_key in self._ds.column_names else "image"
        label_key = "label" if "label" in self._ds.column_names else "fine_label"
        self._label_key = label_key
        self.targets = list(self._ds[label_key])
        feat = self._ds.features[label_key]
        self.classes = list(getattr(feat, "names", []))
        self._data = None
        if preload:
            arrs = [np.asarray(r.convert("RGB")) for r in self._ds[self._image_key]]
            self._data = np.stack(arrs)
            self._ds = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        index = int(index)
        if self._data is not None:
            img = Image.fromarray(self._data[index])
        else:
            img = self._ds[index][self._image_key].convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]


def _cifar10_factory(root=None, train=True, download=False, transform=None, **_kwargs):
    return HFImageDataset(
        "uoft-cs/cifar10",
        split="train" if train else "test",
        transform=transform,
        image_key="img",
        preload=True,
    )


def _tinyimagenet_factory(root=None, split="train", download=False, transform=None, **_kwargs):
    return HFImageDataset(
        "zh-plus/tiny-imagenet",
        split="train" if split == "train" else "valid",
        transform=transform,
        image_key="image",
        preload=True,
    )


def install_dataset_shims():
    import torchvision

    torchvision.datasets.CIFAR10 = _cifar10_factory

    from cords.utils.data.datasets.SL import builder

    builder.TinyImageNet = _tinyimagenet_factory


def install_all():
    install_import_shims()
    install_dataset_shims()
