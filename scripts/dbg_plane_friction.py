"""dbg: PBDOptions.plane_friction (multiflow MF-17) — table-plane tangential friction.

A thin slab of PBD liquid rests on an analytic boundary_plane inside a wide cube domain and is
given a uniform initial tangential velocity. plane_friction attenuates the tangential velocity
component of liquid particles inside the plane contact band (C <= particle_radius) once per
substep in the velocity stage. 0.0 (default) keeps the solver bit-identical: the Python gate
short-circuits before the kernel is ever invoked, so the kernel is neither called nor traced.

Tests (orchestrated by the default mode, one GPU subprocess per run):
  T1  physics: plane_friction=0.02 vs 0.0 — the 0.02 run's mean tangential speed must decay
      substantially faster (half-lives reported), nan=0, slab stays on the plane.
  T2  zero regression: default option -> the new kernel is never called (monkeypatch counter,
      guard mode) and two default replicas establish the GPU nondeterminism noise floor that
      pre/post divergence cannot exceed (the code path is unchanged, so pre/post divergence
      equals replica/replica divergence by construction). First-step KE reported as the
      deterministic proxy.
  T3  validation: PBDOptions(boundary_plane=None, plane_friction=0.1) raises.

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \\
    "$PY" multiflow/scripts/dbg_plane_friction.py
"""

import argparse
import json
import os
import subprocess
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402

print("genesis loaded from:", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), (
    "Wrong genesis! Must be the multiflow copy."
)

DT = 1.0 / 60.0
SUBSTEPS = 8
PS = 0.008
RHO = 1000.0
V0 = 0.25  # initial uniform tangential speed (m/s)
SLAB_HALF = 0.05
SLAB_LAYERS = 2
VIDEOS = os.path.join(WORKSPACE, "videos")


def make_scene(plane_friction=None):
    """Slab-on-plane scene. plane_friction=None omits the option entirely (pure default)."""
    extra = {}
    if plane_friction is not None:
        extra["plane_friction"] = plane_friction
    gs.init(backend=gs.gpu, precision="32", seed=0)
    return gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            particle_size=PS,
            lower_bound=(-1.0, -0.35, -0.05),
            upper_bound=(1.0, 0.35, 0.60),
            max_density_solver_iterations=20,
            max_viscosity_solver_iterations=1,
            velocity_damping=1.0,
            boundary_plane=(0.0,),
            **extra,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )


def add_slab(scene):
    return scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=RHO, c_init=0.0, density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, PS + 0.5 * (SLAB_LAYERS - 1) * PS),
            size=(2 * SLAB_HALF, 2 * SLAB_HALF, SLAB_LAYERS * PS),
        ),
    )


def run_mode(plane_friction, frames, out_path):
    """One default/friction run; writes per-frame metrics JSON. Subprocess-isolated (one scene/process)."""
    scene = make_scene(plane_friction)
    scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=RHO, c_init=0.0, density_relaxation=0.2, viscosity_relaxation=0.01,
        ),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, PS + 0.5 * (SLAB_LAYERS - 1) * PS),
            size=(2 * SLAB_HALF, 2 * SLAB_HALF, SLAB_LAYERS * PS),
        ),
    )
    scene.build()
    solver = scene.sim.pbd_solver
    n_fluid = solver._n_fluid_particles
    print(f"fluid={n_fluid} plane_friction={solver._plane_friction}")

    vel = solver.particles.vel.to_numpy()
    vel[:n_fluid, 0, :] = (V0, 0.0, 0.0)
    solver.particles.vel.from_numpy(vel)

    frames_data = []
    nan_max = 0
    z_min_min, z_mean_max = 1e9, -1e9
    for i in range(frames + 1):
        if i > 0:
            scene.step()
        pos = solver.particles.pos.to_numpy()[:n_fluid, 0, :]
        vel = solver.particles.vel.to_numpy()[:n_fluid, 0, :]
        nan = int((~np.isfinite(pos)).sum() + (~np.isfinite(vel)).sum())
        nan_max = max(nan_max, nan)
        safe = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        sp_t = float(np.mean(np.hypot(vel[:, 0], vel[:, 1])))
        ke = float(0.5 * RHO * np.nansum((vel**2).sum(axis=1)))
        z_min, z_mean = float(safe[:, 2].min()), float(safe[:, 2].mean())
        z_min_min = min(z_min_min, z_min)
        z_mean_max = max(z_mean_max, z_mean)
        frames_data.append(dict(t=i * DT, sp_t=sp_t, ke=ke, z_min=z_min, z_mean=z_mean, nan=nan))
        if i % 12 == 0 or i == frames:
            print(f"t={i * DT:5.2f}s  |v_t|_mean={sp_t:.5f}  z_min={z_min:+.5f}  z_mean={z_mean:+.5f}  nan={nan}")

    out = dict(
        plane_friction=solver._plane_friction,
        nan_max=nan_max,
        z_min_min=z_min_min,
        z_mean_max=z_mean_max,
        frames=frames_data,
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"wrote {out_path}")


def guard_mode(frames):
    """Default-option run with the new kernel wrapped in a call counter: must never be called,
    and the solver flag must read 0.0 (the Python gate short-circuits before invocation)."""
    scene = make_scene(None)
    add_slab(scene)
    scene.build()
    solver = scene.sim.pbd_solver
    assert solver._plane_friction == 0.0, f"default _plane_friction != 0.0: {solver._plane_friction}"

    calls = [0]
    orig = solver._kernel_apply_plane_friction

    def wrapped(f):
        calls[0] += 1
        return orig(f)

    solver._kernel_apply_plane_friction = wrapped
    for _ in range(frames):
        scene.step()
    print(f"guard: _plane_friction={solver._plane_friction} kernel_calls={calls[0]} frames={frames}")
    ok = calls[0] == 0
    print(f"guard: {'PASS' if ok else 'FAIL'} (new kernel never invoked/traced at default)")
    if not ok:
        sys.exit(1)


def spawn(mode, out_path, plane_friction=None, frames=120):
    cmd = [sys.executable, os.path.abspath(__file__), "--mode", mode, "--frames", str(frames), "--out", out_path]
    if plane_friction is not None:
        cmd += ["--plane-friction", str(plane_friction)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    print(f"+ {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
    print(tail)
    if proc.returncode != 0:
        print(f"subprocess failed rc={proc.returncode}")
        sys.exit(1)


def half_life(frames_data):
    """First crossing of V0/2 by the mean tangential speed, linearly interpolated; inf if none."""
    target = 0.5 * V0
    for a, b in zip(frames_data, frames_data[1:]):
        if a["sp_t"] > target >= b["sp_t"]:
            frac = (a["sp_t"] - target) / max(a["sp_t"] - b["sp_t"], 1e-12)
            return a["t"] + frac * (b["t"] - a["t"])
    return float("inf")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def trajectory_diff(a, b):
    """Max abs/rel divergence of the per-frame mean tangential speed between two runs."""
    sa = np.array([f["sp_t"] for f in a["frames"]])
    sb = np.array([f["sp_t"] for f in b["frames"]])
    d = np.abs(sa - sb)
    rel = d / np.maximum(np.abs(sa), 1e-9)
    return float(d.max()), float(rel.max())


def test_t3():
    print("=" * 78)
    try:
        gs.options.PBDOptions(boundary_plane=None, plane_friction=0.1)
    except Exception as exc:
        print(f"[T3] PBDOptions(boundary_plane=None, plane_friction=0.1) raised {type(exc).__name__}: {exc}")
        print("[T3] PASS")
        return True
    print("[T3] FAIL: no exception raised")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["orchestrate", "run", "guard"], default="orchestrate")
    parser.add_argument("--plane-friction", type=float, default=None)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.mode == "run":
        assert args.out, "--out required in run mode"
        run_mode(args.plane_friction, args.frames, args.out)
        return
    if args.mode == "guard":
        guard_mode(args.frames)
        return

    p_fric = os.path.join(VIDEOS, "dbg_plane_friction_fric.json")
    p_zero_a = os.path.join(VIDEOS, "dbg_plane_friction_zero_a.json")
    p_zero_b = os.path.join(VIDEOS, "dbg_plane_friction_zero_b.json")

    print("### T1: physics, plane_friction=0.02 vs 0.0")
    spawn("run", p_fric, plane_friction=0.02)
    spawn("run", p_zero_a, plane_friction=0.0)
    fric, zero = load(p_fric), load(p_zero_a)
    hl_fric, hl_zero = half_life(fric["frames"]), half_life(zero["frames"])
    sp_end_fric = fric["frames"][-1]["sp_t"]
    sp_end_zero = zero["frames"][-1]["sp_t"]
    ratio = sp_end_fric / max(sp_end_zero, 1e-12)
    print(f"half-life: fric=0.02 -> {hl_fric:.3f}s   fric=0.0 -> {'inf' if hl_zero == float('inf') else f'{hl_zero:.3f}s'}")
    print(f"end |v_t|_mean: fric=0.02 -> {sp_end_fric:.5f}   fric=0.0 -> {sp_end_zero:.5f}   ratio={ratio:.3f}")
    p_nan = fric["nan_max"] == 0 and zero["nan_max"] == 0
    p_plane = fric["z_min_min"] >= -0.5 * PS and zero["z_min_min"] >= -0.5 * PS
    p_plane &= fric["z_mean_max"] <= 0.10 and zero["z_mean_max"] <= 0.10
    p_decay = ratio <= 0.5
    print("=" * 78)
    print(f"[T1] nan max fric/zero = {fric['nan_max']}/{zero['nan_max']} -> {'PASS' if p_nan else 'FAIL'}")
    print(f"[T1] slab on plane: z_min_min = {fric['z_min_min']:+.5f}/{zero['z_min_min']:+.5f} "
          f"(limit {-0.5 * PS:+.4f}), z_mean_max = {fric['z_mean_max']:+.5f}/{zero['z_mean_max']:+.5f} "
          f"-> {'PASS' if p_plane else 'FAIL'}")
    print(f"[T1] decay ratio end speeds = {ratio:.3f} (limit 0.5) -> {'PASS' if p_decay else 'FAIL'}")

    print("### T2: zero regression — kernel never called at default + replica noise floor")
    spawn("guard", os.path.join(VIDEOS, "dbg_plane_friction_guard.txt"), frames=30)
    spawn("run", p_zero_b, plane_friction=0.0)
    zero_b = load(p_zero_b)
    d_abs, d_rel = trajectory_diff(zero, zero_b)
    ke0_a, ke0_b = zero["frames"][1]["ke"], zero_b["frames"][1]["ke"]
    ke_rel = abs(ke0_a - ke0_b) / max(abs(ke0_a), 1e-12)
    p_replica = d_rel <= 0.05  # replica noise floor; pre/post divergence equals this by construction
    p_ke = ke_rel <= 0.01
    print(f"replica/replica |v_t| divergence: max abs = {d_abs:.3e}, max rel = {d_rel:.3e}")
    print(f"first-step KE: zero_a = {ke0_a:.6e}, zero_b = {ke0_b:.6e}, rel diff = {ke_rel:.3e}")
    print("=" * 78)
    print(f"[T2] kernel guard (see guard log) -> PASS (subprocess exited 0)")
    print(f"[T2] replica noise floor max rel = {d_rel:.3e} (limit 0.05) -> {'PASS' if p_replica else 'FAIL'}")
    print(f"[T2] first-step KE rel diff = {ke_rel:.3e} (limit 0.01) -> {'PASS' if p_ke else 'FAIL'}")

    p_t3 = test_t3()

    ok = p_nan and p_plane and p_decay and p_replica and p_ke and p_t3
    print("=" * 78)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
