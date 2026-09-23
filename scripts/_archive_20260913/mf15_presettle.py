"""MF-15 helper: pre-settle the new ultimate demo scene (coffee in the handled mug + milk in the
jug) and save the particle positions to an .npz, for the MF-15 narrative demo (pour the milk to
75%, watch the coffee bulge and overflow onto the table) to start from an equilibrium state.

Both vessels are analytic SHELL boundaries (finite-thickness wall with an OUTER surface): the
mug is the static upright `boundary_cup_shell`, the jug is `boundary_pitcher_shell` kept upright
for the whole presettle. No boundary_group anywhere: shells are world solids for every particle,
so the only thing above the jug lip is the script mouth lid. Same anti-blast recipe as
mf12_presettle: dt=1/240 x substeps=4 (dt_sub=1/960 -> per-substep ballistic displacement smaller
than the clamp bands), velocity_damping=0.99, an axial pre-compression K_Z of the raw lattice
about each vessel floor, and a script lid on the milk column. The settled column height is
~0.67 x the sampled height (emergent bulk packing), so K_Z sits just ABOVE that ratio: the start
is slightly UNDER-dense and contracts downward, self-stabilizing; a start below the packing
ratio (e.g. the 0.62 inherited from mf12's wider vessels) is over-dense and blasts the surface
out of the vessel. The coffee column sits at 75% of the mug depth, so it needs no lid.

Run:
  PY=/c/Users/admin/.conda/envs/ipbf/python.exe; PYTHONIOENCODING=utf-8 \
    "$PY" multiflow/scripts/mf15_presettle.py --seconds 6
"""

import argparse
import os
import sys
import time

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multiflow/
sys.path.insert(0, os.path.join(WORKSPACE, "genesis-world"))

import numpy as np  # noqa: E402

import genesis as gs  # noqa: E402

print("genesis loaded from:", gs.__file__)
assert os.path.normpath(gs.__file__).startswith(os.path.normpath(WORKSPACE)), "Wrong genesis! Must be the multiflow copy."

VIDEOS_DIR = os.path.join(WORKSPACE, "videos")

PS = 0.008
DT = 1.0 / 240.0
SUBSTEPS = 4  # dt_sub = 1/960
Z_TABLE = 0.0

# mug (handled coffee cup): static upright cup shell sitting on the table
CUP_XY = (0.0, 0.0)
Z_FLOOR = 0.016  # inner cavity floor of BOTH vessels (each outer bottom sits at Z_FLOOR - t_bottom = table)
R_IN = 0.15  # mug cavity radius
L_CUP = 0.30  # mug cavity depth (inner floor -> rim)
T_WALL = 0.016
T_BOTTOM = 0.016
LIP_ROUND = 0.008
R_OUT = R_IN + T_WALL
Z_RIM = Z_FLOOR + L_CUP

# jug (milk pitcher): pitcher shell, kept upright on the table for the whole presettle
JUG_XY = (0.42, 0.0)
R_JUG = 0.125
L_JUG = 0.30

# fill targets as a fraction of the cavity depth, with the acceptance bands around them. The
# milk sits at ~80% (NOT the 85% the narrative would allow): the demo rebalances the settled
# state at a coarser dt_sub, and every layer of lip headroom removed comes back as milk sprayed
# over the lip at t=0.
FILL_C, FILL_M = 0.75, 0.80
BAND_C = (0.70, 0.80)
BAND_M = (0.78, 0.82)

R_C_SAMPLE = R_IN - 0.5 * PS
R_M_SAMPLE = R_JUG - 0.5 * PS

# uniform per-axis position jitter breaking the regular sampler's cubic lattice order (the raw
# lattice's chamfered-square silhouette survives into the poured stream and reads as an octagon)
POS_JITTER = 0.25 * PS
JITTER_SEED = 20260911


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--dens-iters", type=int, default=20)
    parser.add_argument("--leps", type=float, default=0.1)
    parser.add_argument("--st-comp", type=float, default=1.0)
    parser.add_argument("--vdamp", type=float, default=0.99)
    parser.add_argument("--h-c0", type=float, default=0.335, help="sampled coffee column height [m]")
    parser.add_argument("--h-m0", type=float, default=0.349, help="sampled milk column height [m]")
    parser.add_argument("--kz", type=float, default=0.68, help="axial pre-compression factor")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    h_c0, h_m0, k_z = args.h_c0, args.h_m0, args.kz
    out_path = os.path.join(VIDEOS_DIR, f"mf15{args.tag}_settled.npz")
    a_j = np.array([0.0, 0.0, 1.0])
    o_j = np.array([JUG_XY[0], JUG_XY[1], Z_FLOOR])

    gs.init(backend=gs.gpu, precision="32", seed=0)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0.0, 0.0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            particle_size=PS,
            lower_bound=(-0.35, -0.35, Z_TABLE),
            upper_bound=(0.80, 0.35, 0.80),
            boundary_cup_shell=(CUP_XY[0], CUP_XY[1], Z_FLOOR, R_IN, L_CUP, T_WALL, T_BOTTOM, LIP_ROUND),
            boundary_pitcher_shell=(*o_j, *a_j, R_JUG, L_JUG, T_WALL, T_BOTTOM, LIP_ROUND),
            boundary_plane=(Z_TABLE,),
            max_density_solver_iterations=args.dens_iters,
            max_viscosity_solver_iterations=1,
            density_lambda_epsilon=args.leps,
            surface_tension_enabled=True,
            st_compliance=args.st_comp,
            st_surface_density_factor=1.0,
            st_distance_enabled=True,
            st_distance_compliance=40.0,
            st_max_surface_neighbors=256,
            velocity_damping=args.vdamp,
            wall_adhesion_enabled=True,
            wall_adhesion_compliance=20.0,
        ),
        vis_options=gs.options.VisOptions(render_particle_as="points"),
        show_viewer=False,
    )
    coffee = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=1.0, density_relaxation=0.2, viscosity_relaxation=0.01
        ),
        morph=gs.morphs.Cylinder(
            radius=R_C_SAMPLE, height=h_c0, pos=(CUP_XY[0], CUP_XY[1], Z_FLOOR + 0.5 * PS + 0.5 * h_c0)
        ),
    )
    milk = scene.add_entity(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0, c_init=0.0, density_relaxation=0.2, viscosity_relaxation=0.01
        ),
        morph=gs.morphs.Cylinder(
            radius=R_M_SAMPLE, height=h_m0, pos=(JUG_XY[0], JUG_XY[1], Z_FLOOR + 0.5 * PS + 0.5 * h_m0)
        ),
    )
    scene.build()
    n_coffee, n_milk = coffee.n_particles, milk.n_particles
    print(f"coffee={n_coffee} milk={n_milk} total={n_coffee + n_milk}")
    print(
        f"mug shell: r_in={R_IN} L={L_CUP} rim z={Z_RIM:.3f} | jug shell: r={R_JUG} L={L_JUG} "
        f"rim z={Z_FLOOR + L_JUG:.3f} @xy={JUG_XY} | plane z={Z_TABLE}"
    )

    solver = scene.sim.pbd_solver
    assert type(solver.boundary).__name__ == "CubeBoundary", f"unexpected boundary: {type(solver.boundary)}"
    assert type(solver.boundary2).__name__ == "TiltedCylinderShellBoundary", f"unexpected boundary2: {type(solver.boundary2)}"
    assert type(solver.boundary_cup).__name__ == "TiltedCylinderShellBoundary", (
        f"unexpected boundary_cup: {type(solver.boundary_cup)}"
    )
    assert type(solver.boundary3).__name__ == "PlaneBoundary", f"unexpected boundary3: {type(solver.boundary3)}"
    assert solver._pitcher_is_shell and solver._has_boundary_cup_shell and solver._has_boundary_shell
    n_total = solver._n_fluid_particles
    assert n_total == n_coffee + n_milk

    # overwrite the sampled lattice before the first substep: a uniform position jitter first
    # (see POS_JITTER), then the axial pre-compression about each vessel floor, then zero
    # velocities (see module docstring for why K_Z)
    rng = np.random.default_rng(JITTER_SEED)
    print(f"lattice jitter: +/-{POS_JITTER:.4f} m per axis (seed {JITTER_SEED})")
    for name, ent, xy, r_sample in (("coffee", coffee, CUP_XY, R_C_SAMPLE), ("milk", milk, JUG_XY, R_M_SAMPLE)):
        pos = ent.get_particles_pos().cpu().numpy()
        pos += rng.uniform(-POS_JITTER, POS_JITTER, size=pos.shape)
        pos[:, 2] = Z_FLOOR + (pos[:, 2] - Z_FLOOR) * k_z
        ent.set_particles_pos(pos)
        ent.set_particles_vel(np.zeros_like(pos))
        r = np.linalg.norm(pos[:, :2] - np.asarray(xy), axis=1)
        print(
            f"preset {name}: n={pos.shape[0]} z in [{pos[:, 2].min():.4f}, {pos[:, 2].max():.4f}]"
            f"  r_max={r.max():.4f} (sample r={r_sample})"
        )

    # script lid on the milk column: just above the compressed fill line, never above the rim
    # minus a safety margin (particles pushed past s=L leave the keep region for good)
    s_cap = min(h_m0 * k_z + 2.0 * PS, L_JUG - 0.003)
    print(f"milk lid s_cap={s_cap:.4f} (compressed top {h_m0 * k_z:.4f}, rim margin {L_JUG - 0.003:.4f})")

    n_frames = int(round(args.seconds / DT))
    print_every = int(round(0.5 / DT))
    wall0 = time.time()
    nan_max = 0
    for i in range(n_frames):
        scene.step()

        pos_np = solver.particles.pos.to_numpy()
        m = pos_np[n_coffee:n_total, 0, :]
        s_m = (m - o_j) @ a_j
        over = s_m > s_cap
        if over.any():
            m[over] -= ((s_m[over] - s_cap)[:, None] * a_j)
            solver.particles.pos.from_numpy(pos_np)

        if (i + 1) % print_every == 0 or i == n_frames - 1:
            vel_np = solver.particles.vel.to_numpy()[:n_total, 0, :]
            ke = float(0.5 * np.nansum((vel_np**2).sum(axis=1)))
            nan = int((~np.isfinite(pos_np)).sum())
            nan_max = max(nan_max, nan)
            c = pos_np[:n_coffee, 0, :]
            r_c = np.linalg.norm(c[:, :2] - np.asarray(CUP_XY), axis=1)
            r_m = np.linalg.norm(m[:, :2] - np.asarray(JUG_XY), axis=1)
            n_in_c = int((r_c <= R_IN + PS).sum())
            n_in_m = int(((r_m <= R_JUG + PS) & (s_m > -PS) & (s_m < L_JUG + PS)).sum())
            print(
                f"t={(i + 1) * DT:5.2f}s / {args.seconds:.1f}s  KE={ke:.3e}  capped={int(over.sum())}"
                f"  n_cup={n_in_c}/{n_coffee}  n_jug={n_in_m}/{n_milk}"
                f"  z99 c={np.percentile(c[:, 2], 99):.4f} m={np.percentile(m[:, 2], 99):.4f}"
                f"  nan={nan}  ({(time.time() - wall0) / (i + 1) * 1000:.0f} ms/frame)"
            )

    # ---- containment verification + fill levels --------------------------------------------
    pos_np = solver.particles.pos.to_numpy()
    safe = np.nan_to_num(pos_np[:n_total, 0, :], nan=0.0)
    c, m = safe[:n_coffee], safe[n_coffee:n_total]
    r_c = np.linalg.norm(c[:, :2] - np.asarray(CUP_XY), axis=1)
    r_m = np.linalg.norm(m[:, :2] - np.asarray(JUG_XY), axis=1)
    s_m = (m - o_j) @ a_j
    nan = int((~np.isfinite(pos_np)).sum())

    z99_c, z99_m = float(np.percentile(c[:, 2], 99)), float(np.percentile(m[:, 2], 99))
    fill_c, fill_m = (z99_c - Z_FLOOR) / L_CUP, (z99_m - Z_FLOOR) / L_JUG
    print("=" * 70)
    print(
        f"coffee: n={n_coffee}  z in [{c[:, 2].min():.4f}, {c[:, 2].max():.4f}]  r_max={r_c.max():.4f}"
        f"  level z99={z99_c:.4f}  fill={fill_c:.3f}"
    )
    print(f"        bbox x [{c[:, 0].min():.4f}, {c[:, 0].max():.4f}]  y [{c[:, 1].min():.4f}, {c[:, 1].max():.4f}]")
    print(
        f"milk:   n={n_milk}  z in [{m[:, 2].min():.4f}, {m[:, 2].max():.4f}]  r_max={r_m.max():.4f}"
        f"  s in [{s_m.min():.4f}, {s_m.max():.4f}]  level z99={z99_m:.4f}  fill={fill_m:.3f}"
    )
    print(f"        bbox x [{m[:, 0].min():.4f}, {m[:, 0].max():.4f}]  y [{m[:, 1].min():.4f}, {m[:, 1].max():.4f}]")
    print(f"nan={nan} (max during run {nan_max})  particles {n_coffee}+{n_milk}={n_coffee + n_milk} (n_total {n_total})")

    assert nan == 0, "NaN in particle state"
    assert n_total == n_coffee + n_milk, "particle count not conserved"
    assert r_c.max() <= R_IN + PS + 1e-6, "coffee escaped the mug radially"
    assert c[:, 2].min() >= Z_FLOOR - PS, "coffee below the mug floor"
    assert c[:, 2].max() <= Z_FLOOR + FILL_C * L_CUP + 4.0 * PS, "coffee above the target fill line"
    assert r_m.max() <= R_JUG + PS + 1e-6, "milk escaped the jug radially"
    assert s_m.min() > -PS and s_m.max() <= L_JUG - 0.003 + 2.0 * PS, "milk outside the jug axially"

    ok_c = BAND_C[0] <= fill_c <= BAND_C[1]
    ok_m = BAND_M[0] <= fill_m <= BAND_M[1]
    sug_c = h_c0 * (FILL_C * L_CUP) / max(z99_c - Z_FLOOR, 1e-6)
    sug_m = h_m0 * (FILL_M * L_JUG) / max(z99_m - Z_FLOOR, 1e-6)
    print(
        f"fill bands: coffee {BAND_C} -> {'OK' if ok_c else 'OUT'}  milk {BAND_M} -> {'OK' if ok_m else 'OUT'}"
        f"  (suggested h_c0={sug_c:.4f} h_m0={sug_m:.4f})"
    )

    os.makedirs(VIDEOS_DIR, exist_ok=True)
    np.savez(
        out_path,
        pos=pos_np,
        n_coffee=n_coffee,
        n_milk=n_milk,
        ps=PS,
        coffee_r=R_C_SAMPLE,
        coffee_h0=h_c0,
        milk_r=R_M_SAMPLE,
        milk_h0=h_m0,
        k_z=k_z,
        dens_iters=args.dens_iters,
        leps=args.leps,
        st_comp=args.st_comp,
    )
    print(f"saved: {out_path}  pos shape {pos_np.shape}")
    if not (ok_c and ok_m):
        print("presettle FAILED calibration band - rerun with the suggested heights")
        sys.exit(3)
    print("presettle OK")


if __name__ == "__main__":
    main()
