"""Import shims for the vendored LfF repo.

LfF's dataset package imports the Corrupted CIFAR-10 protocol unconditionally,
which pulls in `wand` (an ImageMagick binding needing a system library we do
not have). Colored MNIST, the benchmark we reproduce, never touches it.
"""

import sys
import types


def install():
    if "wand" in sys.modules:
        return
    wand = types.ModuleType("wand")
    image = types.ModuleType("wand.image")
    color = types.ModuleType("wand.color")
    api = types.ModuleType("wand.api")

    class _Unavailable:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("wand/ImageMagick is not installed in this env")

    class _AnyAttr:
        """Accepts the ctypes argtypes assignments the protocol does on import."""

        def __getattr__(self, _name):
            value = _AnyAttr()
            object.__setattr__(self, _name, value)
            return value

    image.Image = _Unavailable
    color.Color = _Unavailable
    api.library = _AnyAttr()
    wand.image = image
    wand.color = color
    wand.api = api

    sys.modules["wand"] = wand
    sys.modules["wand.image"] = image
    sys.modules["wand.color"] = color
    sys.modules["wand.api"] = api
