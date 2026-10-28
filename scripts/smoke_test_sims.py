"""Smoke-test the MuJoCo-based simulator stack.

Verifies: import -> make env -> reset -> step with random action
-> grab RGB observation when available -> close.

Uses software rendering (MUJOCO_GL=osmesa) because the GPU is partitioned
into a MIG slice; NVIDIA EGL rendering is disabled on MIG.
"""
from __future__ import annotations
import os, sys, time, traceback
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

def _ok(name, msg): print(f"  OK  {name:30s} {msg}")
def _err(name, e): print(f"  ERR {name:30s} {type(e).__name__}: {e}")

RESULTS = []

def run(name, fn):
    print(f"\n[{name}]")
    try:
        t0 = time.time()
        info = fn()
        dt = time.time() - t0
        RESULTS.append((name, "ok", f"{dt:.1f}s  {info}"))
        _ok(name, info)
    except Exception as e:
        RESULTS.append((name, "err", f"{type(e).__name__}: {e}"))
        _err(name, e)
        traceback.print_exc()


def t_dm_control_walker():
    from dm_control import suite
    env = suite.load("walker", "stand")
    ts = env.reset()
    pix = env.physics.render(height=84, width=84, camera_id=0)
    assert pix.shape == (84, 84, 3)
    for _ in range(5):
        env.step(env.action_spec().minimum * 0 + 0.1)
    return f"walker/stand pix={pix.shape} dtype={pix.dtype}"


def t_dm_control_reacher():
    from dm_control import suite
    env = suite.load("reacher", "easy")
    env.reset()
    pix = env.physics.render(84, 84, camera_id=0)
    for _ in range(3):
        env.step(env.action_spec().minimum * 0)
    return f"reacher/easy pix={pix.shape}"


def t_dm_control_cartpole():
    from dm_control import suite
    env = suite.load("cartpole", "swingup")
    env.reset()
    pix = env.physics.render(84, 84, camera_id=0)
    return f"cartpole/swingup pix={pix.shape}"


def t_robosuite_lift():
    import robosuite
    env = robosuite.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names="agentview",
        camera_heights=84, camera_widths=84,
        control_freq=20, horizon=16,
    )
    obs = env.reset()
    import numpy as np
    for _ in range(3):
        env.step(np.zeros(env.action_dim))
    img = obs.get("agentview_image")
    env.close()
    return f"Lift pix={None if img is None else img.shape}"


def _metaworld_env(prefix: str):
    import metaworld, random
    bench = metaworld.MT1(prefix, seed=0)
    env_cls = bench.train_classes[prefix]
    env = env_cls(render_mode="rgb_array", width=84, height=84)
    env.set_task(random.choice(bench.train_tasks))
    env.reset()
    for _ in range(3):
        env.step(env.action_space.sample())
    pix = env.render()
    return f"{prefix} pix={None if pix is None else pix.shape}"


def t_metaworld_reach():       return _metaworld_env("reach-v3")
def t_metaworld_pick_place():  return _metaworld_env("pick-place-v3")


def main():
    tests = [
        ("dmc:walker-stand", t_dm_control_walker),
        ("dmc:reacher-easy", t_dm_control_reacher),
        ("dmc:cartpole-swingup", t_dm_control_cartpole),
        ("robosuite:Lift", t_robosuite_lift),
        ("metaworld:reach-v3", t_metaworld_reach),
        ("metaworld:pick-place-v3", t_metaworld_pick_place),
    ]
    for n, fn in tests:
        run(n, fn)
    print("\n== summary ==")
    for name, status, msg in RESULTS:
        marker = "✓" if status == "ok" else "✗"
        print(f"  {marker} {name:30s} {msg}")


if __name__ == "__main__":
    main()
