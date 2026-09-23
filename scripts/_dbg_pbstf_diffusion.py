"""C2a verification: concentration diffusion on the multiflow PBSTF path.

Three modes (run as separate processes, same stage-B cube recipe in all of them):

  --mode off   Zero regression. Diffusion left at the default (0 = off). 0.2 s run,
               aggregate metrics compared with the recorded stage-B run `area_s`
               (mf18_pbstf_stcube_area_s_metrics.csv @ t=0.2 s). Mass calibration
               must match the stage-A value bit-for-bit.

  --mode id    Identity consistency. c initialized by an x-split (x < median -> 1,
               else 0), diffusion OFF, 1 s run. The user-order c array must be
               bitwise identical at t=0 and t=1 s (reorder round-trips must not
               scramble particle identity).

  --mode on    Conservation + demo video. Same x-split, diffusion_coeff=0.01,
               zero gravity, 2 s run. Asserts sum(c) drift < 1%, nan=0, and
               physics aggregates consistent with stage-B `area_s` @ t=2 s (c is a
               passive scalar, so the dynamics must not change). Records the
               point-rendered video and a per-frame CSV, and prints the
               diffusion-rate calibration against the old PBD poly6 weights.

Usage:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe
  PYTHONIOENCODING=utf-8 "$PY" _dbg_pbstf_diffusion.py --mode off
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import genesis as gs  # noqa: E402
import quadrants as qd  # noqa: E402
from genesis.utils.misc import qd_to_numpy  # noqa: E402

assert "multiflow" in os.path.normpath(gs.__file__), f"Wrong genesis: {gs.__file__}"

DT = 1.0 / 240.0
SUBSTEPS = 4
PS = 0.002
CUBE_L = 0.030
CUBE_CENTER = (0.0, 0.0, 0.05)
RHO0 = 1000.0
DIFFUSION_EPS = 0.01

VIDEOS_DIR = os.path.join(WORKSPACE, "videos")
MP4 = os.path.join(VIDEOS_DIR, "_dbg_pbstf_diffusion.mp4")

# stage-B reference rows, mf18_pbstf_stcube_area_s_metrics.csv (same recipe as here:
# density_compliance=375000, area=0.0048, sdist=40, idist=180, visc=0, regular)
REF_02S = dict(nan=0, ke=8.523554242856335e-06, com_z=0.049999975, rho50=0.998965859413147,
               vmax=0.0861378163099289, n_surf=1178)
REF_2S = dict(nan=0, ke=1.5622744342635997e-07, com_z=0.04999694, rho50=0.9999244809150696,
              vmax=0.04398674890398979, n_surf=1178,
              sphericity=1.0011860366254726, vol_ratio=0.9423513673342746)
MASS_REF = 7.985668e-6  # stage-A build calibration (theory 1000/125224338.17)

FAILURES = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


@qd.kernel
def _kernel_split_c_x(solver: qd.template(), x_mid: qd.f32):
    # user-order write, before the manual reorder sync so both views agree at t=0
    for i, i_b in qd.ndrange(solver._n_particles, solver._B):
        if solver.particles[i, i_b].pos[0] < x_mid:
            solver.particles[i, i_b].c = 1.0
        else:
            solver.particles[i, i_b].c = 0.0


def build_scene(diffusion_coeff):
    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, 0.0)),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=PS,
            lower_bound=(-0.06, -0.06, -0.01),
            upper_bound=(0.06, 0.06, 0.11),
            max_solver_iterations=20,
            topology_rebuild_interval=2,
            diffusion_coeff=diffusion_coeff,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )
    cube = scene.add_entity(
        material=gs.materials.PBSTF.Liquid(
            rho=RHO0,
            density_compliance=375000.0,
            surface_tension_compliance=0.0048,
            surface_distance_compliance=40.0,
            interior_distance_compliance=180.0,
            surface_viscosity=0.0,
            interior_viscosity=0.0,
            sampler="regular",
        ),
        morph=gs.morphs.Box(size=(CUBE_L, CUBE_L, CUBE_L), pos=CUBE_CENTER),
        surface=gs.surfaces.Default(color=(0.35, 0.55, 0.95, 1.0), opacity=1.0),
    )
    cam = scene.add_camera(
        res=(960, 1280), pos=(0.075, -0.075, 0.05), lookat=(0.0, 0.0, 0.05), fov=40, GUI=False,
    )
    scene.build()
    return scene, cube, cam


def snapshot_reordered(solver):
    pos = qd_to_numpy(solver.particles_reordered.pos, transpose=True)[0]
    vel = qd_to_numpy(solver.particles_reordered.vel, transpose=True)[0]
    density = qd_to_numpy(solver.particles_reordered.density, transpose=True)[0]
    on_surface = qd_to_numpy(solver.on_surface, transpose=True)[0].astype(bool)
    return pos, vel, density, on_surface


def aggregates(pos, vel, density, on_surface, mass):
    nan_count = int((~np.isfinite(pos).all(axis=1) | ~np.isfinite(vel).all(axis=1)).sum())
    ke = float(0.5 * mass * np.sum(vel * vel))
    com_z = float(pos[:, 2].mean())
    vmax = float(np.linalg.norm(vel, axis=1).max())
    interior = density[~on_surface] / RHO0
    rho50 = float(np.percentile(interior, 50)) if len(interior) else float("nan")
    return dict(nan=nan_count, ke=ke, com_z=com_z, rho50=rho50, vmax=vmax, n_surf=int(on_surface.sum()))


def compare_with_ref(tag, m, ref):
    check(f"{tag}.nan", m["nan"] == ref["nan"], f"nan={m['nan']} (ref {ref['nan']})")
    check(f"{tag}.com_z", abs(m["com_z"] - ref["com_z"]) < 1e-4,
          f"com_z={m['com_z']:.8f} (ref {ref['com_z']:.8f})")
    check(f"{tag}.rho50", abs(m["rho50"] - ref["rho50"]) < 5e-3,
          f"rho50={m['rho50']:.6f} (ref {ref['rho50']:.6f})")
    check(f"{tag}.ke_order", 0.33 < m["ke"] / ref["ke"] < 3.0,
          f"ke={m['ke']:.4e} (ref {ref['ke']:.4e})")
    check(f"{tag}.vmax_order", 0.33 < m["vmax"] / ref["vmax"] < 3.0,
          f"vmax={m['vmax']:.4f} (ref {ref['vmax']:.4f})")
    check(f"{tag}.n_surf", abs(m["n_surf"] - ref["n_surf"]) <= 50,
          f"n_surf={m['n_surf']} (ref {ref['n_surf']})")


def rate_reference(pos, mass):
    """Diffusion-rate calibration on the t=0 lattice, numpy replica of both kernels."""
    i0 = int(np.argmin(np.linalg.norm(pos - np.array(CUBE_CENTER), axis=1)))
    d = np.linalg.norm(pos - pos[i0], axis=1).astype(np.float64)
    # PBSTF port: dc_i = eps * V * sum_j (c_j - c_i) * W_cubic(x_i - x_j), support 3*ps
    h = 3.0 * PS
    V = mass / RHO0
    coef = 8.0 / (np.pi * h**3)
    q = d / h
    W = np.zeros_like(d)
    m1 = q < 0.5
    W[m1] = coef * (6.0 * q[m1] ** 2 * (q[m1] - 1.0) + 1.0)
    m2 = (q >= 0.5) & (q < 1.0)
    W[m2] = 2.0 * coef * (1.0 - q[m2]) ** 3
    s_pbstf = V * W.sum()
    n_pbstf = int((q < 1.0).sum())
    # old PBD path: dc_i = eps * sum_{j != i, |x| < dist_scale} poly6 * (c_j - c_i),
    # poly6 dimensionless: d = |x| / dist_scale, dist_scale = particle_radius / 0.4 = 1.25*ps,
    # W = 315/(64 pi) * (1 - d^2)^3 for 0 < d < 1 (self excluded)
    dist_scale = (PS / 2.0) / 0.4
    dp = d / dist_scale
    mp = (dp > 0.0) & (dp < 1.0)
    s_pbd = float((315.0 / (64.0 * np.pi) * (1.0 - dp[mp] ** 2) ** 3).sum())
    n_pbd = int(mp.sum())
    print(
        f"[rate] interior particle {i0}: PBSTF V*sum(W)={s_pbstf:.6f} ({n_pbstf} nbrs incl self, "
        f"h={h}), PBD sum(poly6)={s_pbd:.6f} ({n_pbd} nbrs, dist_scale={dist_scale})"
    )
    print(
        f"[rate] per-substep relaxation at eps=0.01: PBSTF {0.01 * s_pbstf:.2e} vs PBD "
        f"{0.01 * s_pbd:.2e}; per-second (960 substeps): {9.6 * s_pbstf:.3f} vs {9.6 * s_pbd:.3f}"
    )
    print(
        f"[rate] same visible rate as the old PBD demo (eps=0.01) needs eps_pbstf ~= "
        f"{0.01 * s_pbd / s_pbstf:.5f}"
    )


def run_mode_off():
    scene, cube, cam = build_scene(0.0)
    solver = scene.sim.pbstf_solver
    solver._kernel_reorder_particles(0)
    solver._rebuild_topology(0)
    mass = float(solver._default_mass)
    print(f"[off] n={cube.n_particles} mass={mass:.10e} (stage-A ref {MASS_REF:.10e})")
    check("off.mass", abs(mass - MASS_REF) / MASS_REF < 1e-5, f"mass={mass:.10e}")
    n_frames = int(round(0.2 / DT))
    for _ in range(n_frames):
        scene.step()
    m = aggregates(*snapshot_reordered(solver), mass)
    print(f"[off] t=0.2s {m}")
    compare_with_ref("off", m, REF_02S)


def run_mode_id():
    scene, cube, cam = build_scene(0.0)
    solver = scene.sim.pbstf_solver
    pos0 = qd_to_numpy(solver.particles.pos, transpose=True)[0].copy()
    x_mid = float(np.median(pos0[:, 0]))
    _kernel_split_c_x(solver, x_mid)
    solver._kernel_reorder_particles(0)
    solver._rebuild_topology(0)
    c0 = qd_to_numpy(solver.particles.c, transpose=True)[0].copy()
    n_coffee = int((c0 > 0.5).sum())
    print(f"[id] x_mid={x_mid:.8f} c=1:{n_coffee} c=0:{len(c0) - n_coffee} (expect 1575/1800)")
    check("id.split_counts", n_coffee == 1575, f"c=1 count {n_coffee}")
    n_frames = int(round(1.0 / DT))
    for _ in range(n_frames):
        scene.step()
    c1 = qd_to_numpy(solver.particles.c, transpose=True)[0]
    bitwise = bool(np.array_equal(c0.view(np.int32), c1.view(np.int32)))
    check("id.bitwise", bitwise,
          f"bitwise equal: {bitwise}, max|dc|={float(np.abs(c1 - c0).max()):.3e}")
    pos, vel, density, on_surface = snapshot_reordered(solver)
    m = aggregates(pos, vel, density, on_surface, float(solver._default_mass))
    print(f"[id] t=1.0s physics (c passive, must look like stage B): {m}")
    check("id.nan", m["nan"] == 0, f"nan={m['nan']}")
    check("id.rho50", 0.99 < m["rho50"] < 1.01, f"rho50={m['rho50']:.6f}")
    np.savez(os.path.join(VIDEOS_DIR, "_dbg_pbstf_diffusion_id.npz"), c0=c0, c1=c1)


def run_mode_on():
    scene, cube, cam = build_scene(DIFFUSION_EPS)
    solver = scene.sim.pbstf_solver
    mass = float(solver._default_mass)
    pos0 = qd_to_numpy(solver.particles.pos, transpose=True)[0].copy()
    x_mid = float(np.median(pos0[:, 0]))
    _kernel_split_c_x(solver, x_mid)
    solver._kernel_reorder_particles(0)
    solver._rebuild_topology(0)
    rate_reference(pos0, mass)
    c0 = qd_to_numpy(solver.particles.c, transpose=True)[0].copy()
    sum_c0 = float(c0.sum(dtype=np.float64))
    sum_mc0 = mass * sum_c0
    print(f"[on] eps={DIFFUSION_EPS} sum(c)={sum_c0:.6f} sum(m*c)={sum_mc0:.6e}")

    def c_stats(c, pos):
        mixed = float(((c > 0.05) & (c < 0.95)).mean())
        left = float(c[pos[:, 0] < -0.008].mean())
        right = float(c[pos[:, 0] > 0.008].mean())
        return mixed, left, right

    mixed0, left0, right0 = c_stats(c0, pos0)
    print(f"[on] t=0: mixed_frac={mixed0:.4f} c_left={left0:.4f} c_right={right0:.4f}")

    cam.start_recording(save_to_filename=MP4, fps=30)
    n_frames = int(round(2.0 / DT))
    print_every = max(1, int(round(0.25 / DT)))
    rows = []
    t_start = time.perf_counter()
    for i in range(1, n_frames + 1):
        scene.step()
        c = qd_to_numpy(solver.particles.c, transpose=True)[0]
        drift = abs(float(c.sum(dtype=np.float64)) - sum_c0) / sum_c0
        nan_c = int((~np.isfinite(c)).sum())
        mixed, left, right = c_stats(c, pos0)
        rows.append((i * DT, drift, nan_c, mixed, left, right))
        if i % print_every == 0 or i == n_frames:
            pos, vel, density, on_surface = snapshot_reordered(solver)
            m = aggregates(pos, vel, density, on_surface, mass)
            print(
                f"t={i * DT:5.2f}s drift={drift:.3e} nan_c={nan_c} mixed={mixed:.4f} "
                f"c_l={left:.4f} c_r={right:.4f} | nan={m['nan']} KE={m['ke']:.3e} "
                f"rho50={m['rho50']:.4f}"
            )
    cam.stop_recording()
    wall = time.perf_counter() - t_start
    print(f"[perf] {n_frames} frames in {wall:.1f}s -> {1e3 * wall / n_frames:.2f} ms/frame")

    c1 = qd_to_numpy(solver.particles.c, transpose=True)[0]
    drift_c = abs(float(c1.sum(dtype=np.float64)) - sum_c0) / sum_c0
    drift_mc = abs(mass * float(c1.sum(dtype=np.float64)) - sum_mc0) / sum_mc0
    check("on.conserve_c", drift_c < 0.01, f"sum(c) drift {drift_c:.3e} (<1%)")
    check("on.conserve_mc", drift_mc < 0.01, f"sum(m*c) drift {drift_mc:.3e} (<1%)")
    check("on.nan_c", int((~np.isfinite(c1)).sum()) == 0, "c has no nan")
    pos, vel, density, on_surface = snapshot_reordered(solver)
    m = aggregates(pos, vel, density, on_surface, mass)
    print(f"[on] t=2.0s physics {m}")
    compare_with_ref("on", m, REF_2S)
    mixed1, left1, right1 = c_stats(c1, pos0)
    check("on.profile_softens", mixed1 > mixed0 + 0.05 and left1 < left0 - 0.01,
          f"mixed {mixed0:.4f}->{mixed1:.4f}, c_left {left0:.4f}->{left1:.4f}")

    with open(os.path.join(VIDEOS_DIR, "_dbg_pbstf_diffusion_on_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "sum_c_drift", "nan_c", "mixed_frac", "c_left", "c_right"])
        w.writerows(rows)
    np.savez(os.path.join(VIDEOS_DIR, "_dbg_pbstf_diffusion_on.npz"), c0=c0, c1=c1, pos=pos,
             on_surface=on_surface)

    # final-state sphericity vs the recorded stage-B area_s reconstruction
    import pysplashsurf
    import trimesh

    mesh_with_data, _ = pysplashsurf.reconstruction_pipeline(
        np.ascontiguousarray(pos, dtype=np.float64),
        particle_radius=0.5 * PS,
        smoothing_length=2.0,
        cube_size=0.8,
        iso_surface_threshold=0.6,
        mesh_smoothing_weights=True,
        mesh_smoothing_iters=25,
        normals_smoothing_iters=10,
        mesh_cleanup=True,
        compute_normals=True,
        multi_threading=True,
    )
    mesh = trimesh.Trimesh(vertices=mesh_with_data.mesh.vertices, faces=mesh_with_data.mesh.triangles,
                           process=False)
    vol = float(mesh.volume) if mesh.is_watertight else float("nan")
    sphericity = float(mesh.area**3 / (36.0 * np.pi * vol**2)) if np.isfinite(vol) and vol > 0 else float("nan")
    vol_ratio = vol / (mass * len(pos) / RHO0) if np.isfinite(vol) else float("nan")
    check("on.sphericity", abs(sphericity - REF_2S["sphericity"]) < 0.02,
          f"sphericity={sphericity:.4f} (ref {REF_2S['sphericity']:.4f})")
    check("on.vol_ratio", abs(vol_ratio - REF_2S["vol_ratio"]) < 0.05,
          f"vol_ratio={vol_ratio:.4f} (ref {REF_2S['vol_ratio']:.4f})")
    print(f"mp4: {MP4}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("off", "id", "on"), required=True)
    args = parser.parse_args()
    {"off": run_mode_off, "id": run_mode_id, "on": run_mode_on}[args.mode]()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} checks failed: {FAILURES})")
        sys.exit(1)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
