"""Drop or hang a shirt with vertex block descent (VBD) or FEM.Cloth + Incremental Potential Contact (IPC)."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import trimesh

import genesis as gs
from genesis.options.vis import DirectionalLight
from genesis.utils.geom import euler_to_R, trans_R_to_T
from genesis.utils.misc import tensor_to_array


@dataclass
class Benchmark:
    solver: str
    asset: str | None
    asset_sha256: str | None
    device: str
    device_name: str
    precision: str
    vertices: int
    triangles: int
    dt: float
    substeps: int
    scene_dt: float
    scene_substeps: int
    iterations: int | None
    is_line_search_enabled: bool | None
    contact_d_hat: float
    contact_search_radius: float | None
    shell_contact_stiffness: float | None
    is_contact_diagnostics_enabled: bool | None
    friction_epsilon: float
    convergence_tolerance: float | None
    thickness: float
    contact_thickness: float | None
    membrane_damping: float | None
    bending_damping: float | None
    bending_model: str | None
    shell_ground_stiffness: float | None
    shell_ground_friction: float | None
    shell_initialization: str | None
    shell_step_size: float | None
    contact_relaxation: float | None
    youngs_modulus: float
    poisson_ratio: float
    density: float
    shear_stiffness: float | None
    bending_rigidity: float | None
    ipc_bending_stiffness: float | None
    friction: float
    is_plastic: bool
    yield_angle_deg: float | None
    hardening: float | None
    initialization_seconds: float
    simulated_seconds: float
    repeats: int
    mean_step_ms: float
    p95_step_ms: float
    wall_seconds: float
    minimum_height: float


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-g", "--gpu", action="store_true", help="Run the simulation on the GPU.")
    parser.add_argument("-v", "--vis", action="store_true")
    parser.add_argument(
        "--solver", choices=("vbd", "ipc"), default="vbd", help="ipc uses FEM.Cloth and requires --gpu."
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=180, help="Number of frames; 180 covers 3 seconds at the default dt."
    )
    parser.add_argument("--dt", type=float, default=1 / 60, help="Frame duration in seconds.")
    parser.add_argument("--substeps", type=int, default=20, help="Integration substeps per frame.")
    parser.add_argument(
        "--scenario",
        choices=("shirt", "hang", "fold"),
        default="shirt",
        help="Drop a shirt, pin its shoulders, or fold a sheet.",
    )
    parser.add_argument(
        "-r",
        "--record",
        dest="is_recording_enabled",
        action="store_true",
        help="Save a video under the output directory.",
    )
    parser.add_argument("--precision", choices=("32", "64"), default="32")
    parser.add_argument("--iterations", type=int, help="VBD iterations per substep (default: 15).")
    parser.add_argument(
        "--line-search",
        dest="is_line_search_enabled",
        action="store_true",
        help="Enable local energy backtracking at extra cost.",
    )
    parser.add_argument(
        "--tolerance", type=float, help="VBD convergence tolerance; zero uses all iterations (default: 0)."
    )
    parser.add_argument("--plastic", action="store_true")
    parser.add_argument("--yield-angle", type=float, help="VBD plastic yield angle in degrees (default: 10).")
    parser.add_argument("--hardening", type=float, help="VBD plastic hardening (default: 0).")
    parser.add_argument("--youngs-modulus", type=float, default=320000.0 / 3, help="Young's modulus in Pa.")
    parser.add_argument("--poisson-ratio", type=float, default=1 / 3, help="Poisson's ratio.")
    parser.add_argument("--density", type=float, default=800.0, help="Volume density in kg/m^3.")
    parser.add_argument("--thickness", type=float, default=0.00025, help="Full shell thickness in metres.")
    parser.add_argument("--contact-thickness", type=float, help="VBD collision thickness in metres (default: 1e-6).")
    parser.add_argument("--membrane-damping", type=float, help="VBD membrane damping time in seconds (default: 2e-6).")
    parser.add_argument("--bending-damping", type=float, help="VBD bending damping time in seconds (default: 2e-6).")
    parser.add_argument(
        "--bending-model",
        choices=("QuadraticBending", "DiscreteShellBending"),
        help="VBD bending energy (default: quadratic).",
    )
    parser.add_argument(
        "--shear-stiffness", type=float, help="Additional VBD in-plane shear stiffness in N/m (default: 0)."
    )
    parser.add_argument("--bending-rigidity", type=float, help="VBD bending rigidity in N*m (default: 1e-6).")
    parser.add_argument(
        "--ipc-bending-stiffness",
        type=float,
        help="FEM.Cloth bending coefficient (default: 40); larger resists folding.",
    )
    parser.add_argument("--no-bending", action="store_true")
    parser.add_argument(
        "--friction", type=float, default=0.1, help="Self-contact friction; VBD ground friction is 0.3."
    )
    parser.add_argument("--contact-distance", type=float, default=0.002, help="Contact activation gap in metres.")
    parser.add_argument("--contact-stiffness", type=float, help="VBD self-contact stiffness in N/m (default: 100).")
    parser.add_argument(
        "--search-radius",
        type=float,
        help="VBD search distance in metres (default: 0.0045); smaller reduces queries and increases rebuilds.",
    )
    parser.add_argument(
        "--contact-diagnostics",
        dest="is_contact_diagnostics_enabled",
        action="store_true",
        help="Check for penetration and report warnings at extra cost.",
    )
    parser.add_argument(
        "--benchmark", dest="is_benchmark", action="store_true", help="Warm up 20 steps, then time five reset runs."
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("out/vbd"))
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.dt <= 0 or args.substeps <= 0:
        parser.error("--dt and --substeps must be positive")
    if args.is_recording_enabled and args.is_benchmark:
        parser.error("--record and --benchmark must run separately to keep rendering out of the timings")
    if args.solver == "ipc":
        if not args.gpu:
            parser.error("FEM.Cloth + IPC requires --gpu.")
        if args.scenario == "fold":
            parser.error(
                "--scenario fold requires VBD's moving and releasable vertex constraints; IPC supports shirt/hang."
            )
        if (
            args.plastic
            or args.is_line_search_enabled
            or args.is_contact_diagnostics_enabled
            or any(
                value is not None
                for value in (
                    args.iterations,
                    args.tolerance,
                    args.yield_angle,
                    args.hardening,
                    args.shear_stiffness,
                    args.bending_rigidity,
                    args.bending_model,
                    args.membrane_damping,
                    args.bending_damping,
                    args.contact_thickness,
                    args.contact_stiffness,
                    args.search_radius,
                )
            )
        ):
            parser.error("VBD solver/material controls require --solver vbd; IPC bending uses --ipc-bending-stiffness.")
        if args.ipc_bending_stiffness is None:
            args.ipc_bending_stiffness = 40.0
    else:
        if args.ipc_bending_stiffness is not None:
            parser.error("--ipc-bending-stiffness requires --solver ipc.")
        args.iterations = 15 if args.iterations is None else args.iterations
        args.tolerance = 0.0 if args.tolerance is None else args.tolerance
        args.yield_angle = 10.0 if args.yield_angle is None else args.yield_angle
        args.hardening = 0.0 if args.hardening is None else args.hardening
        args.shear_stiffness = 0.0 if args.shear_stiffness is None else args.shear_stiffness
        args.bending_rigidity = 1e-6 if args.bending_rigidity is None else args.bending_rigidity
        if args.bending_model is None:
            args.bending_model = "DiscreteShellBending" if args.plastic else "QuadraticBending"
        if args.plastic and args.bending_model != "DiscreteShellBending":
            parser.error("--plastic requires --bending-model DiscreteShellBending.")
        args.membrane_damping = 2e-6 if args.membrane_damping is None else args.membrane_damping
        args.bending_damping = 2e-6 if args.bending_damping is None else args.bending_damping
        args.contact_thickness = 1e-6 if args.contact_thickness is None else args.contact_thickness
        args.contact_stiffness = 100.0 if args.contact_stiffness is None else args.contact_stiffness
        args.search_radius = 0.0045 if args.search_radius is None else args.search_radius
    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision=args.precision)
    if args.is_benchmark:
        started = time.perf_counter()
    asset = None
    if args.scenario in ("shirt", "hang"):
        asset = Path(gs.__file__).parent / "assets" / "meshes" / "shirt.glb"
        mesh = trimesh.load_scene(asset, process=False).to_mesh()
        mesh.apply_transform(trans_R_to_T(R=euler_to_R((90.0, 0.0, 0.0))))
        mesh.apply_translation((0.0, 0.0, 0.35 - mesh.bounds[0, 2]))
        camera_pos = (1.2, -2.6, 1.5)
        camera_lookat = (0.0, 0.0, 0.65)
        surface = gs.surfaces.Default()
    else:
        x, y = np.meshgrid(np.linspace(-0.04, 0.04, 17), np.linspace(-0.03, 0.03, 13))
        vertices = np.column_stack((x.flat, y.flat, np.full(x.size, 0.012)))
        cells = np.arange(12)[:, None] * 17 + np.arange(16)[None, :]
        cells = cells.reshape(cells.size)
        faces = np.concatenate(
            (np.column_stack((cells, cells + 1, cells + 17)), np.column_stack((cells + 1, cells + 18, cells + 17)))
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        camera_pos = (0.22, -0.28, 0.2)
        camera_lookat = (0.0, 0.0, 0.035)
        surface = gs.surfaces.Default(color=(0.7, 0.15, 0.3, 1.0))
    coupler_options = None
    vbd_options = None
    if args.solver == "ipc":
        coupler_options = gs.options.IPCCouplerOptions(
            contact_d_hat=args.contact_distance,
            contact_eps_velocity=0.001,
        )
        material = gs.materials.FEM.Cloth(
            E=args.youngs_modulus,
            nu=args.poisson_ratio,
            rho=args.density,
            friction_mu=args.friction,
            # The shell radius integrates mass over 2*r, preserving the full specified thickness.
            thickness=0.5 * args.thickness,
            bending_stiffness=None if args.no_bending else args.ipc_bending_stiffness,
        )
    else:
        vbd_options = gs.options.VBDOptions(
            iterations=args.iterations,
            convergence_tolerance=args.tolerance,
            contact_d_hat=args.contact_distance,
            shell_contact_stiffness=args.contact_stiffness,
            contact_search_radius=args.search_radius,
            contact_relaxation=0.4,
            shell_step_size=1.0,
            shell_ground_stiffness=10.0,
            shell_ground_d_hat=0.0,
            shell_ground_friction=0.3,
            shell_ground_friction_epsilon=0.0001,
            is_contact_diagnostics_enabled=args.is_contact_diagnostics_enabled,
            is_line_search_enabled=args.is_line_search_enabled,
        )
        material = gs.materials.VBD.Cloth(
            E=args.youngs_modulus,
            nu=args.poisson_ratio,
            rho=args.density,
            friction_mu=args.friction,
            shear_stiffness=args.shear_stiffness,
            thickness=args.thickness,
            contact_thickness=args.contact_thickness,
            membrane_damping=args.membrane_damping,
            bending_damping=args.bending_damping,
            bending_model=None if args.no_bending else args.bending_model,
            bending_rigidity=args.bending_rigidity,
            is_bending_plasticity_enabled=args.plastic,
            bending_yield_angle_deg=args.yield_angle,
            bending_plastic_hardening=args.hardening,
        )
    frame_dt, frame_substeps = args.dt, args.substeps
    scene_steps = frame_substeps if args.solver == "ipc" else 1
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=frame_dt / scene_steps,
            substeps=frame_substeps // scene_steps,
            gravity=(0, 0, -10.0),
        ),
        coupler_options=coupler_options,
        vbd_options=vbd_options,
        vis_options=gs.options.VisOptions(
            ambient_light=(0.3, 0.3, 0.3),
            lights=(
                DirectionalLight(
                    dir=(-1, 1, -1),
                    color=(1.0, 1.0, 1.0),
                    intensity=5.0,
                ),
            ),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            realtime_factor=None,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    entity = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=(mesh,),
        ),
        material=material,
        surface=surface,
    )
    initial_positions = tensor_to_array(entity.init_positions)
    if args.scenario == "hang":
        shoulder_targets = np.array([[-0.2, 0.0, 1.09], [0.2, 0.0, 1.09]])
        distances = np.linalg.norm(initial_positions[:, None, :] - shoulder_targets[None, :, :], axis=-1)
        shoulder_positions = initial_positions[distances.argmin(axis=0)]
        distances = np.linalg.norm(initial_positions[:, None, :] - shoulder_positions[None, :, :], axis=-1)
        fixed_indices = np.flatnonzero((distances < 0.025).any(axis=1))
        entity.set_vertex_constraints(fixed_indices)
        print(f"Pinned {len(fixed_indices)} vertices in two shoulder patches (radius 0.025 m).", flush=True)
    elif args.scenario == "fold":
        fixed_indices = np.flatnonzero(initial_positions[:, 0] < -0.039)
        moving_indices = np.flatnonzero(initial_positions[:, 0] > 0.039)
        entity.set_vertex_constraints(fixed_indices)
        entity.set_vertex_constraints(moving_indices)
        moving_rest = initial_positions[moving_indices].copy()
    camera = None
    if args.is_recording_enabled:
        camera = scene.add_camera(res=(960, 720), pos=camera_pos, lookat=camera_lookat, debug=True)
    scene.build()
    if args.scenario == "hang" and (args.vis or args.is_recording_enabled):
        scene.draw_debug_spheres(shoulder_positions, radius=0.018, color=(1.0, 0.45, 0.05, 1.0))
        support_positions = shoulder_positions + (0.0, 0.0, 0.2)
        scene.draw_debug_line(support_positions[0], support_positions[1], radius=0.008, color=(0.3, 0.3, 0.3, 1.0))
        for shoulder_pos, support_pos in zip(shoulder_positions, support_positions):
            scene.draw_debug_line(shoulder_pos, support_pos, radius=0.003, color=(0.3, 0.3, 0.3, 1.0))
    if args.is_benchmark:
        if args.gpu:
            torch.cuda.synchronize()
        initialization_seconds = time.perf_counter() - started
        for _ in range(20):
            for _ in range(scene_steps):
                scene.step()
        scene.reset()
    if camera is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        camera.start_recording(
            save_to_filename=str(args.output_dir / f"{args.solver}_shell_{args.scenario}.mp4"), fps=1 / frame_dt
        )
    durations = [] if args.is_benchmark else None
    minimum_height = np.inf
    repeats = 5 if args.is_benchmark else 1
    if args.is_benchmark:
        started = time.perf_counter()
    for repeat in range(repeats):
        if repeat:
            scene.reset()
        for step in range(args.steps):
            if args.scenario == "fold" and step < args.steps // 2:
                angle = np.pi * min(1.0, 2 * step / max(1, args.steps - 1))
                targets = moving_rest.copy()
                targets[:, 0] = 0.04 * np.cos(angle)
                targets[:, 2] = 0.012 + 0.04 * np.sin(angle) + 0.001 * angle / np.pi
                entity.set_vertex_constraints(moving_indices, targets)
            elif args.scenario == "fold" and step == args.steps // 2:
                entity.remove_vertex_constraints(moving_indices)
            if args.is_benchmark:
                tick = time.perf_counter()
            for _ in range(scene_steps):
                scene.step()
            if args.is_benchmark:
                if args.gpu:
                    torch.cuda.synchronize()
                durations.append(time.perf_counter() - tick)
        if args.is_benchmark:
            minimum_height = min(minimum_height, tensor_to_array(entity.get_state().pos)[..., 2].min())
            print(f"Run {repeat + 1}/{repeats}: mean {np.mean(durations[-args.steps :]) * 1000:.3f} ms", flush=True)
    if camera is not None:
        camera.stop_recording()
    if not args.is_benchmark:
        return
    elapsed = time.perf_counter() - started
    result = Benchmark(
        args.solver,
        asset.relative_to(Path(gs.__file__).parent / "assets").as_posix() if asset is not None else None,
        hashlib.sha256(asset.read_bytes()).hexdigest() if asset is not None else None,
        str(gs.device),
        torch.cuda.get_device_name() if args.gpu else "CPU",
        args.precision,
        entity.n_vertices,
        entity.n_elements,
        frame_dt,
        frame_substeps,
        scene.dt,
        scene.sim.substeps,
        args.iterations,
        args.is_line_search_enabled if args.solver == "vbd" else None,
        args.contact_distance,
        args.search_radius,
        args.contact_stiffness,
        args.is_contact_diagnostics_enabled if args.solver == "vbd" else None,
        scene.vbd_options.friction_epsilon if args.solver == "vbd" else coupler_options.contact_eps_velocity,
        args.tolerance,
        args.thickness,
        args.contact_thickness,
        args.membrane_damping,
        args.bending_damping,
        (None if args.no_bending else args.bending_model) if args.solver == "vbd" else None,
        scene.vbd_options.shell_ground_stiffness if args.solver == "vbd" else None,
        scene.vbd_options.shell_ground_friction if args.solver == "vbd" else None,
        scene.vbd_options.shell_initialization if args.solver == "vbd" else None,
        scene.vbd_options.shell_step_size if args.solver == "vbd" else None,
        scene.vbd_options.contact_relaxation if args.solver == "vbd" else None,
        args.youngs_modulus,
        args.poisson_ratio,
        args.density,
        args.shear_stiffness,
        (0.0 if args.no_bending else args.bending_rigidity) if args.solver == "vbd" else None,
        (0.0 if args.no_bending else args.ipc_bending_stiffness) if args.solver == "ipc" else None,
        args.friction,
        args.plastic,
        args.yield_angle,
        args.hardening,
        initialization_seconds,
        args.steps * frame_dt,
        repeats,
        np.mean(durations) * 1000,
        np.percentile(durations, 95) * 1000,
        elapsed,
        float(minimum_height),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / f"{args.solver}_shell_{args.scenario}.json"
    report.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(report)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
