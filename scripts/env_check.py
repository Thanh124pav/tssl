"""Verify runtime: torch/CUDA/bf16, ffmpeg, video decoder, HF model load."""
from __future__ import annotations
import os, sys, time, traceback

def _ok(name, val): print(f"  OK  {name:25s} {val}")
def _err(name, e):  print(f"  ERR {name:25s} {type(e).__name__}: {e}")

def check_torch():
    import torch
    _ok("torch.__version__", torch.__version__)
    _ok("cuda.is_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        _ok("device_name", torch.cuda.get_device_name(0))
        _ok("bf16_supported", torch.cuda.is_bf16_supported())
        free, total = torch.cuda.mem_get_info()
        _ok("gpu_mem", f"{free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
        # tiny matmul on GPU in bf16
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        t0 = time.time()
        y = (x @ x).float().sum().item()
        _ok("bf16 matmul", f"{(time.time()-t0)*1000:.1f} ms, sum={y:.1f}")

def check_ffmpeg():
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    _ok("imageio_ffmpeg", exe)

def check_video_decode():
    """Create a tiny mp4 then decode it with decord + pyav."""
    import numpy as np, imageio.v3 as iio, tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    frames = (np.random.rand(16, 64, 64, 3) * 255).astype("uint8")
    iio.imwrite(tmp.name, frames, fps=8, codec="libx264", macro_block_size=1)
    _ok("sample_mp4", f"{os.path.getsize(tmp.name)} bytes @ {tmp.name}")
    import decord
    v = decord.VideoReader(tmp.name)
    _ok("decord", f"{len(v)} frames, shape {v[0].shape}")
    import av
    container = av.open(tmp.name)
    n = sum(1 for _ in container.decode(video=0))
    _ok("pyav", f"{n} frames")
    os.unlink(tmp.name)

def check_hf_model():
    """Load a tiny vision backbone, run forward on GPU."""
    import torch
    from transformers import AutoImageProcessor, AutoModel
    name = "facebook/dinov2-small"
    _ok("hf_model_name", name)
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to("cuda").eval()
    import numpy as np
    img = np.random.rand(224, 224, 3).astype("float32")
    inp = proc(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp)
    _ok("dinov2 fwd", f"last_hidden_state {tuple(out.last_hidden_state.shape)}")

def main():
    checks = [
        ("torch/cuda", check_torch),
        ("ffmpeg", check_ffmpeg),
        ("video decoding", check_video_decode),
        ("hf vision model", check_hf_model),
    ]
    rc = 0
    for name, fn in checks:
        print(f"\n[{name}]")
        try:
            fn()
        except Exception as e:
            _err(name, e)
            traceback.print_exc()
            rc = 1
    sys.exit(rc)

if __name__ == "__main__":
    main()
