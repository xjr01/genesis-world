"""IPBF example 04: artificial damping A/B test — settling / kinetic-energy decay.

Same dam-break scene as 02_dam_break.py, run twice (damping on / off) for 20 simulated
seconds; logs total kinetic energy sum_i ||v_i||^2 per frame to CSV and records the damped
run to mp4. Config: h = 1/480 s, ipbf_iterations=2, alpha=0, boundary particles on.
The damping (paper section 3.6, eqs. 16-18; Fig. 13) extracts kinetic energy once the motion
comes close to a stop. damping_alpha_star is dimensional: the paper's 1e-3 is tuned for its
own units (kernel radius 1); the default 3.2e6 is the equivalent for this meter-scale scene
(kernel radius 0.02 m) — see IPBFOptions docstring. It is currently considered too strong
(visibly thickens the flow), hence damping defaults to OFF.
Run: python examples/ipbf/04_settling_damping.py [--alpha-star 1.0] [--no-render]
"""

import argparse
import csv
import os

import torch

import genesis as gs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def build_scene(damping_enabled: bool, with_camera: bool, alpha_star: float | None = None):
    ipbf_kwargs = dict(
        lower_bound=(-0.5, -0.5, 0.0),
        upper_bound=(0.5, 0.5, 1.0),
        particle_size=0.01,
        ipbf_iterations=2,
        alpha=0.0,
        damping_enabled=damping_enabled,
    )
    if alpha_star is not None:
        ipbf_kwargs["damping_alpha_star"] = alpha_star
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60, substeps=8),
        ipbf_options=gs.options.IPBFOptions(**ipbf_kwargs),
        show_viewer=False,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    liquid = scene.add_entity(
        material=gs.materials.IPBF.Liquid(),
        morph=gs.morphs.Box(pos=(-0.25, 0.0, 0.25), size=(0.4, 0.8, 0.5)),
        surface=gs.surfaces.Default(color=(0.4, 0.8, 1.0), vis_mode="particle"),
    )
    cam = None
    if with_camera:
        cam = scene.add_camera(res=(1280, 960), pos=(1.8, -1.8, 1.2), lookat=(0, 0, 0.3), fov=35, GUI=False)
    scene.build()
    return scene, liquid, cam


def run(scene, liquid, cam, horizon, video_out=None):
    ke = []
    if cam is not None and video_out is not None:
        cam.start_recording(save_to_filename=video_out, fps=60)
    for frame in range(horizon):
        scene.step()
        vel = liquid.get_particles_vel()
        assert torch.isfinite(vel).all(), f"Non-finite velocities at frame {frame}!"
        ke.append(vel.norm(dim=-1).pow(2).sum().item())
    if cam is not None and video_out is not None:
        cam.stop_recording()
    return ke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, choices=["cpu", "gpu"])
    parser.add_argument("--horizon", type=int, default=1200)  # 20 s at dt=1/60
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "settling_kinetic_energy.csv"))
    parser.add_argument("--video", default=os.path.join(OUTPUT_DIR, "settling_damped.mp4"))
    parser.add_argument("--alpha-star", type=float, default=None,
                        help="override IPBFOptions.damping_alpha_star")
    parser.add_argument("--no-render", action="store_true", help="skip camera/recording (metrics only)")
    args = parser.parse_args()

    test_mode = "PYTEST_VERSION" in os.environ  # quick CPU smoke for tests/test_examples.py
    backend = args.backend or ("cpu" if test_mode else "gpu")
    if test_mode:
        args.horizon = min(args.horizon, 3)

    gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # damped run (also recorded unless --no-render)
    scene, liquid, cam = build_scene(damping_enabled=True, with_camera=not args.no_render, alpha_star=args.alpha_star)
    ke_damped = run(scene, liquid, cam, args.horizon, video_out=args.video if not args.no_render else None)
    pos = liquid.get_particles_pos()
    assert torch.isfinite(pos).all(), "Non-finite particle positions (damped run)!"
    print("damped run final pos stats: min", pos.min(axis=0).values, "max", pos.max(axis=0).values)
    del scene

    # undamped run
    scene, liquid, cam = build_scene(damping_enabled=False, with_camera=False, alpha_star=args.alpha_star)
    ke_undamped = run(scene, liquid, cam, args.horizon)
    pos = liquid.get_particles_pos()
    assert torch.isfinite(pos).all(), "Non-finite particle positions (undamped run)!"
    print("undamped run final pos stats: min", pos.min(axis=0).values, "max", pos.max(axis=0).values)

    with open(args.out, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["frame", "ke_damped", "ke_undamped"])
        for frame, (ke_d, ke_u) in enumerate(zip(ke_damped, ke_undamped)):
            writer.writerow([frame, ke_d, ke_u])

    print(f"kinetic energy: damped first/last = {ke_damped[0]:.4e} / {ke_damped[-1]:.4e}; "
          f"undamped first/last = {ke_undamped[0]:.4e} / {ke_undamped[-1]:.4e}")
    print("kinetic energy log saved to", args.out)
    if not args.no_render:
        print("video saved to", args.video)


if __name__ == "__main__":
    main()
