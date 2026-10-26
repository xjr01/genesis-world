"""Impulse-triggered shatter demo: a vase, bowl, or plate topples off a table and shatters on the floor.

A source mesh (vase from the shared Genesis assets, bowl / plate from ``assets/``) is partitioned by a
throwaway PBD pre-pass: volumetric variable-density seeds are drawn inside the solid wall, the PBD solid
registers the partition, and the true cut meshes are exported per fragment. Collision proxies come from a
VHACD convex decomposition of each fragment's closed cut mesh; the loader fuses the per-fragment blocks
into one watertight SDF collision geom, so resting contact follows the rendered surface. The delivered
scene is pure rigid: a compound parent body (one welded child per fragment, carrying surface-hull
collision proxies and the surface render pieces) owns the dynamics until
``PBDRigidFragmentBridge.arm_impact_trigger()`` fires on the calibrated floor impact, releasing the parked
fragment links (one per fragment, carrying the decomp-block collision geoms and the closed render pieces)
to the native rigid solver with the parent's exact state.

Seed provenance: each preset pins the md5 of the phase-12 calibration partition (seeds and per-seed
radii), so the demo reproduces the validated fragment layout bit-for-bit; ``--skip-seed-gate`` lifts the
pin for exploration. Impact thresholds are per-object calibrated impulse sums in N*s (table topple must
stay event-free, the floor slam must fire), carried over from the phase-12 calibration ladder.

Usage (from the repository root):

    python examples/fracture/impulse_shatter.py --object vase --video
    python examples/fracture/impulse_shatter.py --object bowl --video
    python examples/fracture/impulse_shatter.py --object plate --steps 300
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import trimesh
import trimesh.decomposition
from scipy.spatial import ConvexHull, QhullError

import genesis as gs
from genesis.engine.couplers.pbd_rigid_fragments import PBDRigidFragmentBridge
from genesis.utils.fracture_seeds import sample_volume_seeds
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array

PRESETS = {
    "vase": {
        "mesh": None,
        "kick_vx": -0.5,
        "kick_omega_y": -4.0,
        "threshold": 2.0,
        "count_range": (34, 44),
        "radius_range": (0.014, 0.05),
        "impact_band": (0.02, 0.10),
        "impact_scale": 0.5,
        "min_member": 40,
        "cv_min": 0.45,
        "particle_jitter": 0.0,
        "seeds_md5": "827497f9ef22e9f49c2215e4ca8da14b",
        "radii_md5": "2ee553a8d39d1d5dab203007d3c3f287",
    },
    "bowl": {
        "mesh": "bowl.obj",
        "kick_vx": -0.5,
        "kick_omega_y": -4.0,
        "threshold": 1.27,
        "count_range": (18, 26),
        "radius_range": (0.01, 0.03),
        "impact_band": (0.0, 0.05),
        "impact_scale": 0.5,
        "min_member": 65,
        "cv_min": 0.0,
        "particle_jitter": 0.0,
        "seeds_md5": "a1d09dacb379ee7edb63a563098f819d",
        "radii_md5": "dbcfd9c38ebb6dd21ad18ad1d97ec13d",
    },
    "plate": {
        "mesh": "plate.obj",
        "kick_vx": -0.5,
        "kick_omega_y": -2.0,
        "threshold": 1.03,
        "count_range": (22, 32),
        "radius_range": (0.014, 0.055),
        "impact_band": (0.0, 0.012),
        "impact_scale": 0.5,
        "min_member": 45,
        "cv_min": 0.0,
        "particle_jitter": 1.5e-4,
        "seeds_md5": "14cf60253e386a5182991b4259e1ffd7",
        "radii_md5": "3754452c7b844e0899332b75dbfe7ab7",
    },
}

DT = 2e-3
SPAWN = np.array([0.14, 0.0, 0.75])
TABLE_X0, TABLE_X1, TABLE_Y = 0.15, 0.45, 0.15
TABLE_TOP = 0.75
PARK_Z = -10.0
IMPACT_AXIS = (-1.0, 0.0)
MAX_ATTEMPTS = 8


def _source_mesh_path(object_name):
    mesh = PRESETS[object_name]["mesh"]
    if mesh is None:
        return Path(gs.utils.get_assets_dir(), "meshes", "vase.obj")
    return Path(__file__).resolve().parent / "assets" / mesh


def _hull_obj(points, path, tag_line):
    try:
        hull = ConvexHull(points)
    except QhullError:
        # nearly-coplanar member sets (a single particle layer slab): joggle instead of failing
        hull = ConvexHull(points, qhull_options="QJ")
    faces = hull.simplices.copy()
    cross = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    inward = np.einsum("ij,ij->i", cross, points[faces].mean(1) - points.mean(0)) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    lines = [tag_line]
    lines += [f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}" for v in points]
    lines += [f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}" for f in faces]
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
    return float(hull.volume)


def _tri_mesh(tris):
    verts = tris.reshape(-1, 3)
    uniq, inverse = np.unique(np.round(verts, 9), axis=0, return_inverse=True)
    return trimesh.Trimesh(vertices=uniq, faces=inverse.reshape(-1, 3), process=False)


def _tri_obj(tris, path, tag_line):
    mesh = _tri_mesh(tris)
    lines = [tag_line]
    lines += [f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}" for v in mesh.vertices]
    lines += [f"f {f[0]} {f[1]} {f[2]}" for f in mesh.faces + 1]
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def _asset_signature(mesh_path, preset, rng_seed):
    return {
        "mesh": str(Path(mesh_path).resolve()),
        "seed": rng_seed,
        "count_range": list(preset["count_range"]),
        "radius_range": list(preset["radius_range"]),
        "impact_band": list(preset["impact_band"]),
        "impact_scale": preset["impact_scale"],
        "min_member": preset["min_member"],
        "cv_min": preset["cv_min"],
        "particle_jitter": preset["particle_jitter"],
    }


def _generate_assets(mesh_path, object_name, work_dir, rng_seed, check_seeds=True):
    """PBD pre-pass: partition the source mesh and export every asset the delivered scene needs.

    Outputs into ``work_dir``:
      assets.json                     mass model + paths + provenance (the delivered scene reads this)
      proxies_surface/                parent-compound collision hulls (parent-COM frame, nearest-seed verts)
      render_surface/                 parent visual pieces (parent-COM frame, surface only)
      render_piece/                   fragment visual pieces (fragment-CM frame, surface + cut caps)
      decomp/                         VHACD convex blocks per fragment (fragment-CM frame) + manifest.json
    """
    preset = PRESETS[object_name]
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(mesh_path, force="mesh")
    render_verts_local = np.asarray(mesh.vertices, dtype=np.float64)
    tag_line = f"# impulse_shatter {object_name}"

    body = bridge = None
    seeds = info = None
    for attempt in range(MAX_ATTEMPTS):
        sigma = 0.55 + 0.15 * attempt
        info = {}
        seeds = sample_volume_seeds(
            mesh=mesh,
            count_range=preset["count_range"],
            radius_range=preset["radius_range"],
            radius_log_sigma=sigma,
            impact_axis=IMPACT_AXIS,
            impact_band=preset["impact_band"],
            impact_scale=preset["impact_scale"],
            rng_seed=rng_seed + attempt,
            info_out=info,
        )
        n_seeds = len(seeds)
        print(
            f"[attempt {attempt}] sigma={sigma:.2f} seeds={n_seeds} pool={info['pool_size']} (window {preset['count_range']})",
            flush=True,
        )
        if not (preset["count_range"][0] <= n_seeds <= preset["count_range"][1]):
            continue

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, substeps=10),
            pbd_options=gs.options.PBDOptions(
                lower_bound=(-2.0, -2.0, 0.0),
                upper_bound=(2.0, 2.0, 2.0),
                particle_size=0.005,
            ),
            show_viewer=False,
        )
        body = scene.add_entity(
            morph=gs.morphs.Mesh(file=str(Path(mesh_path).resolve()), pos=tuple(SPAWN)),
            material=gs.materials.PBD.Solid(
                sampler="regular",
                static_friction=0.8,
                kinetic_friction=0.7,
                stiffness=1.0,
                fracture_threshold=0.11,
                shear_threshold=0.2,
                rotation_threshold=0.25,
                seam_failure_threshold=1.0,
                bond_radius_factor=1.15,
                fragment_seeds=[tuple(s) for s in seeds],
            ),
        )
        if preset["particle_jitter"] > 0.0:
            # exactly-coplanar particle slabs (flat-plate centers) make the registration hulls degenerate
            jr = np.random.default_rng(rng_seed + 1)
            body._particles[:] = body._particles + jr.uniform(
                -preset["particle_jitter"], preset["particle_jitter"], body._particles.shape
            )
        bridge = PBDRigidFragmentBridge(scene.sim)
        try:
            bridge.register_solid_fragments(body, proxy_dir=str(work / "bridge_proxies"))
            scene.build()
        except QhullError as e:
            print(
                f"[attempt {attempt}] registration hit a coplanar-member hull ({e.__class__.__name__}); next seed",
                flush=True,
            )
            continue

        n = body.n_fragments
        members = [np.nonzero(body._particle_fragments == i_k)[0] for i_k in range(n)]
        sizes = np.array([len(m) for m in members])
        mass = bridge._fragment_mass.astype(np.float64)
        particle_total = float(body._particle_mass * len(body._particles))
        cv = float(mass.std() / mass.mean())
        gates = {
            "count": preset["count_range"][0] <= n <= preset["count_range"][1],
            "min_member": sizes.min() >= preset["min_member"],
            "owner_range": body._particle_fragments.max() == n - 1 and body._particle_fragments.min() == 0,
            "single_owner": bool((body._particle_member_counts == 1).all()),
            "mass_conservation": abs(mass.sum() - particle_total) <= 1e-6 * particle_total,
            "cv": (cv >= preset["cv_min"]) if preset["cv_min"] > 0 else True,
        }
        print(
            f"[attempt {attempt}] gates: min_member={sizes.min()} cv={cv:.3f} "
            + " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in gates.items()),
            flush=True,
        )
        if all(gates.values()):
            break
        body = bridge = None
    if body is None:
        raise SystemExit(f"impulse_shatter({object_name}): no attempt satisfied all partition gates")

    if check_seeds and preset["seeds_md5"] is not None:
        seeds_md5 = hashlib.md5(np.asarray(seeds, dtype=np.float64).tobytes()).hexdigest()
        radii_md5 = hashlib.md5(np.asarray(info["per_seed_radius"], dtype=np.float64).tobytes()).hexdigest()
        if (seeds_md5, radii_md5) != (preset["seeds_md5"], preset["radii_md5"]):
            raise SystemExit(
                f"impulse_shatter({object_name}): seed md5 mismatch (got {seeds_md5}/{radii_md5}, "
                f"expected {preset['seeds_md5']}/{preset['radii_md5']}); the calibration partition "
                "changed - pass --skip-seed-gate to explore a new layout"
            )
        print(f"[pre-pass] seeds md5 {seeds_md5} matches the calibration partition")

    n = body.n_fragments
    mass = bridge._fragment_mass.astype(np.float64)
    cm = bridge._fragment_rest_cm.astype(np.float64)
    inertia = bridge._fragment_rest_inertia.astype(np.float64)
    com = (mass[:, None] * cm).sum(0) / mass.sum()
    inertia_com = np.zeros((3, 3))
    for i_k in range(n):
        d = cm[i_k] - com
        inertia_com += inertia[i_k] + mass[i_k] * (np.eye(3) * d.dot(d) - np.outer(d, d))
    rest = body._particles.astype(np.float64)
    cv = float(mass.std() / mass.mean())

    # render-face ownership by nearest seed: the same Voronoi criterion that partitioned the particles
    seeds_arr = np.asarray(seeds, dtype=np.float64)
    d2 = ((render_verts_local[:, None, :] - seeds_arr[None, :, :]) ** 2).sum(-1)
    nearest_seed = np.argmin(d2, axis=1)

    surf_tris = np.asarray(body._surf_tris, dtype=np.float64)
    surf_owner = np.asarray(body._surf_fragments)
    cap_tris = np.asarray(body._wall_verts, dtype=np.float64).reshape(-1, 3, 3)
    cap_owner = np.asarray(body._wall_fragments)

    proxies_dir = work / "proxies_surface"
    surf_dir = work / "render_surface"
    piece_dir = work / "render_piece"
    decomp_dir = work / "decomp"
    for d in (proxies_dir, surf_dir, piece_dir, decomp_dir):
        d.mkdir(exist_ok=True)

    geoms_surface, warnings, vols = [], [], 0.0
    manifest = {}
    total_blocks = 0
    for i_k in range(n):
        sel = nearest_seed == i_k
        if sel.sum() < 8:
            warnings.append(
                {
                    "fragment": i_k,
                    "issue": "surface verts < 8 under nearest-seed assignment",
                    "verts": int(sel.sum()),
                    "handling": "parent proxy falls back to particle-member hull",
                }
            )
            pts_surface = rest[body._particle_fragments == i_k] - com
        else:
            pts_surface = render_verts_local[sel] + SPAWN - com
        vol = _hull_obj(pts_surface, proxies_dir / f"fragment_proxy_{i_k:02d}.obj", tag_line)
        vols += vol
        geoms_surface.append(
            {
                "fragment": i_k,
                "path": str(proxies_dir / f"fragment_proxy_{i_k:02d}.obj"),
                "volume": vol,
                "surface_verts": int(sel.sum()),
            }
        )

        # surface piece in the parent-COM frame (parent visual), closed piece in the fragment frame
        # (fragment visual + VHACD source: the caps close the cut so the decomposition is watertight)
        surf = surf_tris[surf_owner == i_k]
        caps = cap_tris[cap_owner == i_k]
        _tri_obj(surf - com, surf_dir / f"render_surface_{i_k:02d}.obj", tag_line)
        piece_tris = np.concatenate([surf - cm[i_k], caps - cm[i_k]])
        _tri_obj(piece_tris, piece_dir / f"render_piece_{i_k:02d}.obj", tag_line)

        piece = _tri_mesh(piece_tris)
        true_vol = float(piece.volume)
        parts = trimesh.decomposition.convex_decomposition(piece, maxConvexHulls=8, resolution=200000, minEdgeLength=1)
        blocks, block_total = [], 0.0
        for j, kwargs in enumerate(parts):
            blk = trimesh.Trimesh(**kwargs)
            blk.merge_vertices()
            blk.process(validate=True)
            if blk.volume <= 0:
                blk.invert()
            blk_vol = float(abs(blk.volume))
            block_total += blk_vol
            name = f"fragment_{i_k:02d}_block_{j}.obj"
            blk.export(str(decomp_dir / name))
            blocks.append(
                {
                    "block": j,
                    "path": name,
                    "volume_cm3": round(blk_vol * 1e6, 3),
                    "centroid_mm": [round(float(x) * 1e3, 2) for x in np.asarray(blk.center_mass)],
                }
            )
        manifest[str(i_k)] = {
            "blocks": len(blocks),
            "decomposed_cm3": round(block_total * 1e6, 2),
            "true_cm3": round(true_vol * 1e6, 2),
            "ratio": round(block_total / true_vol, 4) if true_vol > 0 else None,
            "blocks_detail": blocks,
        }
        total_blocks += len(blocks)
        print(f"[pre-pass] decomp {i_k:02d}: {len(blocks)} blocks, ratio {block_total / true_vol:.3f}", flush=True)
    (decomp_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    seeds_md5 = hashlib.md5(np.asarray(seeds, dtype=np.float64).tobytes()).hexdigest()
    radii_md5 = hashlib.md5(np.asarray(info["per_seed_radius"], dtype=np.float64).tobytes()).hexdigest()
    assets = {
        "object": object_name,
        "signature": _asset_signature(mesh_path, preset, rng_seed),
        "spawn_origin": SPAWN.tolist(),
        "mass": float(mass.sum()),
        "com": com.tolist(),
        "inertia_com_3x3": inertia_com.tolist(),
        "n_fragments": int(n),
        "fragment_mass": mass.tolist(),
        "fragment_cm": cm.tolist(),
        "fragment_inertia": inertia.tolist(),
        "geoms_surface": geoms_surface,
        "decomp_dir": str(decomp_dir),
        "total_blocks": total_blocks,
        "mass_cv": cv,
        "mass_max_min_ratio": float(mass.max() / mass.min()),
        "warnings": warnings,
        "seeds_md5": seeds_md5,
        "radii_md5": radii_md5,
    }
    (work / "assets.json").write_text(json.dumps(assets, indent=1))
    print(f"[pre-pass] {n} fragments, {total_blocks} VHACD blocks, assets -> {work}", flush=True)
    return assets


def _load_assets(mesh_path, object_name, work_dir, rng_seed, check_seeds):
    work = Path(work_dir)
    asset_file = work / "assets.json"
    signature = _asset_signature(mesh_path, PRESETS[object_name], rng_seed)
    if asset_file.exists():
        assets = json.loads(asset_file.read_text())
        if assets.get("signature") == signature:
            print(f"[assets] loaded from cache {work}")
            return assets
        print("[assets] signature changed, regenerating")
    return _generate_assets(mesh_path, object_name, work_dir, rng_seed, check_seeds=check_seeds)


def _fullinertia(inertia):
    return " ".join(
        format(v, ".17g")
        for v in (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
    )


def _rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", choices=list(PRESETS), default="vase")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None, help="impact impulse threshold in N*s")
    parser.add_argument("--min-closing", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--render-every", type=int, default=8)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--cam-pos", default="-1.45,-1.35,0.95")
    parser.add_argument("--cam-lookat", default="0.02,0.0,0.38")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-seed-gate", action="store_true", help="lift the phase-12 seed md5 pin")
    args = parser.parse_args()
    if args.threshold is not None and args.threshold <= 0:
        parser.error("--threshold must be positive")

    preset = PRESETS[args.object]
    mesh_path = _source_mesh_path(args.object)
    threshold = args.threshold if args.threshold is not None else preset["threshold"]
    work_dir = args.work_dir or str(Path(__file__).resolve().parent / ".cache" / f"{args.object}_seed{args.seed}")
    out_dir = Path(args.out_dir) if args.out_dir else Path(work_dir) / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    assets = _load_assets(mesh_path, args.object, work_dir, args.seed, check_seeds=not args.skip_seed_gate)

    n = assets["n_fragments"]
    mass = assets["mass"]
    com0 = np.asarray(assets["com"], dtype=np.float64)
    inertia0 = np.asarray(assets["inertia_com_3x3"], dtype=np.float64)
    frag_mass = np.asarray(assets["fragment_mass"], dtype=np.float64)
    frag_cm = np.asarray(assets["fragment_cm"], dtype=np.float64)
    frag_inertia = np.asarray(assets["fragment_inertia"], dtype=np.float64)
    d_frag = frag_cm - com0[None, :]
    decomp_dir = Path(assets["decomp_dir"])
    surf_dir = Path(work_dir) / "render_surface"
    piece_dir = Path(work_dir) / "render_piece"
    manifest = json.loads((decomp_dir / "manifest.json").read_text())

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=1),
        rigid_options=gs.options.RigidOptions(use_hibernation=False, iterations=args.iterations),
        show_viewer=False,
    )
    scene.add_entity(morph=gs.morphs.Plane(), material=gs.materials.Rigid())
    scene.add_entity(
        morph=gs.morphs.Box(lower=(TABLE_X0, -TABLE_Y, 0.0), upper=(TABLE_X1, TABLE_Y, TABLE_TOP), fixed=True),
        material=gs.materials.Rigid(),
    )

    # compound parent: one welded child per fragment carrying the surface-hull collision proxy and the
    # parent-COM-frame surface piece; the whole body spawns at the pre-pass pose (com0) with a freejoint
    assets_xml, children = [], []
    for i_k, geom in enumerate(assets["geoms_surface"]):
        hull_path = Path(geom["path"]).resolve().as_posix()
        vis_path = (surf_dir / f"render_surface_{i_k:02d}.obj").resolve().as_posix()
        assets_xml.append(f'  <mesh name="pfrag_{i_k}" file="{hull_path}"/>')
        assets_xml.append(f'  <mesh name="pvis_{i_k}" file="{vis_path}"/>')
        children.append(
            f'  <body name="pfrag_{i_k}" pos="0 0 0">'
            f'<geom name="obj_vis_{i_k}" type="mesh" mesh="pvis_{i_k}" contype="0" conaffinity="0"/>'
            f'<geom name="obj_geom_{i_k}" type="mesh" mesh="pfrag_{i_k}" group="3" friction="0.7 0 0" condim="3"/>'
            f"</body>"
        )
    parent_xml = "\n".join(
        [
            '<mujoco model="impulse_shatter_parent">',
            '  <compiler angle="radian" inertiafromgeom="false"/>',
            "  <asset>",
            *assets_xml,
            "  </asset>",
            "  <worldbody>",
            f'  <body name="object_parent" pos="{com0[0]:.17g} {com0[1]:.17g} {com0[2]:.17g}">',
            '    <freejoint name="object_free"/>',
            f'    <inertial pos="0 0 0" mass="{mass:.17g}" fullinertia="{_fullinertia(inertia0)}"/>',
            *children,
            "  </body>",
            "  </worldbody>",
            "</mujoco>",
        ]
    )
    parent = scene.add_entity(
        morph=gs.morphs.MJCF(file=parent_xml, convexify=False, decimate=False, recompute_inertia=False, align=False),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.2, 0.4, 0.6, 1.0), vis_mode="visual"),
    )

    # fragment links: parked far under the floor at build time, one free body per fragment carrying the
    # VHACD collision blocks (group 3) and the closed render piece (visual only)
    assets_xml, fr_bodies = [], []
    for i_k in range(n):
        vis_path = (piece_dir / f"render_piece_{i_k:02d}.obj").resolve().as_posix()
        assets_xml.append(f'  <mesh name="svis_{i_k}" file="{vis_path}"/>')
        for b in manifest[str(i_k)]["blocks_detail"]:
            bp = (decomp_dir / b["path"]).resolve().as_posix()
            assets_xml.append(f'  <mesh name="sblk_{i_k}_{b["block"]}" file="{bp}"/>')
        fi = _fullinertia(frag_inertia[i_k])
        blk_geoms = "".join(
            f'<geom name="sblk_geom_{i_k}_{b["block"]}" type="mesh" mesh="sblk_{i_k}_{b["block"]}" '
            'group="3" friction="0.7 0 0" condim="3"/>'
            for b in manifest[str(i_k)]["blocks_detail"]
        )
        fr_bodies.append(
            f'  <body name="sfrag_{i_k}" pos="{frag_cm[i_k, 0]:.17g} {frag_cm[i_k, 1]:.17g} {PARK_Z:.17g}">'
            f'    <joint name="sfree_{i_k}" type="free" damping="0"/>'
            f'    <inertial pos="0 0 0" mass="{frag_mass[i_k]:.17g}" fullinertia="{fi}"/>'
            f'    <geom name="svis_geom_{i_k}" type="mesh" mesh="svis_{i_k}" contype="0" conaffinity="0"/>'
            f"    {blk_geoms}"
            f"  </body>"
        )
    frag_xml = "\n".join(
        [
            '<mujoco model="impulse_shatter_fragments">',
            '  <compiler angle="radian" inertiafromgeom="false"/>',
            "  <asset>",
            *assets_xml,
            "  </asset>",
            "  <worldbody>",
            *fr_bodies,
            "  </worldbody>",
            "</mujoco>",
        ]
    )
    frag_entity = scene.add_entity(
        morph=gs.morphs.MJCF(file=frag_xml, convexify=False, decimate=False, recompute_inertia=False, align=False),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(color=(0.2, 0.4, 0.6, 1.0), vis_mode="visual"),
    )

    bridge = PBDRigidFragmentBridge(scene.sim)
    scene.sim._pbd_rigid_fragment_bridge = bridge
    bridge.register_full_parent(parent, [frag_entity.links[i] for i in range(n)], d_frag)
    cam_pos = [float(v) for v in args.cam_pos.split(",")]
    cam_lookat = [float(v) for v in args.cam_lookat.split(",")]
    cam = scene.add_camera(res=(1280, 720), pos=cam_pos, lookat=cam_lookat, fov=50)
    scene.build(n_envs=0)

    # The VHACD blocks fuse into one watertight SDF collision geom per fragment link (shared contact
    # parameters), so the entity ends up with exactly n collision geoms; total_blocks is asset provenance.
    frag_geoms = sum(len(link.geoms) for link in frag_entity.links)
    assert len(parent.geoms) == n
    assert frag_geoms == n, f"fragment collision geoms {frag_geoms} != {n} (per-link fusion broke)"
    print(
        f"REGISTRATION_OK parent_links={parent.n_links} parent_geoms={len(parent.geoms)} "
        f"frag_links={frag_entity.n_links} frag_geoms={frag_geoms} source_blocks={assets['total_blocks']}"
    )

    rigid = scene.rigid_solver
    rigid.set_global_sol_params([4e-4, 1.0, 0.9, 0.95, 1.0e-3, 0.5, 2.0])
    bridge.arm_impact_trigger(threshold, min_closing_speed=args.min_closing)
    parent.set_dofs_velocity(np.array([preset["kick_vx"], 0.0, 0.0, 0.0, preset["kick_omega_y"], 0.0]))

    render = trimesh.load(mesh_path, force="mesh")
    render_verts = np.asarray(render.vertices, dtype=np.float64) + SPAWN - com0
    piece_verts = []
    for i_k in range(n):
        piece = trimesh.load(piece_dir / f"render_piece_{i_k:02d}.obj", force="mesh")
        piece_verts.append(np.asarray(piece.vertices, dtype=np.float64))

    rows = {
        "t": np.zeros(args.steps + 1),
        "mode": np.zeros(args.steps + 1, dtype=np.int64),
        "parent_com": np.zeros((args.steps + 1, 3)),
        "parent_quat": np.zeros((args.steps + 1, 4)),
        "parent_vel": np.zeros((args.steps + 1, 3)),
        "parent_ang": np.zeros((args.steps + 1, 3)),
        "parent_KE": np.zeros(args.steps + 1),
        "frag_P": np.zeros((args.steps + 1, 3)),
        "frag_L": np.zeros((args.steps + 1, 3)),
        "frag_KE": np.zeros(args.steps + 1),
        "render_min_z": np.zeros(args.steps + 1),
    }

    def measure(step):
        pos = tensor_to_array(parent.get_pos())
        quat = tensor_to_array(parent.get_quat())
        vel = tensor_to_array(parent.get_vel())
        ang = tensor_to_array(parent.get_ang())
        rows["t"][step] = step * DT
        rows["parent_com"][step] = pos
        rows["parent_quat"][step] = quat
        rows["parent_vel"][step] = vel
        rows["parent_ang"][step] = ang
        rows["parent_KE"][step] = 0.5 * mass * vel.dot(vel) + 0.5 * ang.dot(inertia0 @ ang)
        rows["render_min_z"][step] = ((render_verts @ _rot(quat).T) + pos)[:, 2].min()

    def fragment_state():
        qrows = qd_to_numpy(rigid.rigid_info.qpos, transpose=True)[0]
        vrows = qd_to_numpy(rigid.dyn_state.dofs.vel, transpose=True)[0]
        pos, quat, vel, ang = [], [], [], []
        for k in range(n):
            qs, ds = bridge._fragment_q_start[k], bridge._fragment_dof_start[k]
            pos.append(qrows[qs : qs + 3])
            quat.append(qrows[qs + 3 : qs + 7])
            vel.append(vrows[ds : ds + 3])
            ang.append(vrows[ds + 3 : ds + 6])
        pos, quat, vel, ang = np.stack(pos), np.stack(quat), np.stack(vel), np.stack(ang)
        com_f = (frag_mass[:, None] * pos).sum(0) / frag_mass.sum()
        P = (frag_mass[:, None] * vel).sum(0)
        L = np.zeros(3)
        KE = 0.0
        for k in range(n):
            I_w = _rot(quat[k]) @ frag_inertia[k] @ _rot(quat[k]).T
            L += I_w @ ang[k] + frag_mass[k] * np.cross(pos[k] - com_f, vel[k])
            KE += 0.5 * frag_mass[k] * vel[k].dot(vel[k]) + 0.5 * ang[k].dot(I_w @ ang[k])
        return com_f, P, L, KE, pos, quat, vel, ang

    frames_dir = out_dir / "frames"
    if args.video:
        frames_dir.mkdir(exist_ok=True)
        frame = cam.render(rgb=True)[0]
        PIL.Image.fromarray(frame[..., :3] if frame.shape[-1] == 4 else frame).save(frames_dir / "f_0000.png")

    fire_step = None
    switch_state = None
    final_state = None
    rest_trace = []
    step_ms = []
    measure(0)
    loop_t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        scene.step()
        step_ms.append((time.perf_counter() - t0) * 1e3)
        measure(step)
        trig = bridge.read_impact_trigger()
        if fire_step is None and int(trig["event_count"][0]) > 0:
            fire_step = step
            print(f"TRIGGER fired at step {step} (t={step * DT:.3f}s) event sum={trig['last_sum'][0]:.4f} N.s")
        if fire_step is not None and step > fire_step:
            # the switch applied at this substep's head: the fragments own the dynamics from here on
            rows["mode"][step] = 1
            com_f, P_f, L_f, KE_f, pos_f, quat_f, _, _ = fragment_state()
            rows["frag_P"][step] = P_f
            rows["frag_L"][step] = L_f
            rows["frag_KE"][step] = KE_f
            rows["render_min_z"][step] = min(
                ((piece_verts[k] @ _rot(quat_f[k]).T) + pos_f[k])[:, 2].min() for k in range(n)
            )
            if step == fire_step + 1:
                switch_state = (com_f, P_f, L_f, KE_f)
            if step == args.steps:
                final_state = (pos_f, quat_f)
            if step > args.steps - 500 and step % 2 == 0:
                vrows = qd_to_numpy(rigid.dyn_state.dofs.vel, transpose=True)[0]
                vs = np.stack(
                    [vrows[bridge._fragment_dof_start[k] : bridge._fragment_dof_start[k] + 3] for k in range(n)]
                )
                ws = np.stack(
                    [vrows[bridge._fragment_dof_start[k] + 3 : bridge._fragment_dof_start[k] + 6] for k in range(n)]
                )
                rest_trace.append((step, vs, ws))
            if step == fire_step + 1 and args.video:
                # the frozen parent is a visual ghost now: park it under the floor so the fragment pieces
                # carry the look. Both qpos and qpos_next must be written - the parent is externally
                # driven, so the substep copyback replays qpos_next over qpos and would revert a
                # qpos-only write. Presentational only: the parent is contactless after the switch.
                pose = np.array([0.0, 0.0, -100.0, 1.0, 0.0, 0.0, 0.0])
                qp = qd_to_torch(rigid.rigid_info.qpos, transpose=True, copy=False)
                qn = qd_to_torch(rigid.rigid_info.qpos_next, transpose=True, copy=False)
                qp[0, 0:7] = torch.as_tensor(pose, dtype=gs.tc_float, device=gs.device)
                qn[0, 0:7] = torch.as_tensor(pose, dtype=gs.tc_float, device=gs.device)
                parent.set_pos(np.array([0.0, 0.0, -100.0]), zero_velocity=True)
        if args.video and step % args.render_every == 0:
            frame = cam.render(rgb=True)[0]
            PIL.Image.fromarray(frame[..., :3] if frame.shape[-1] == 4 else frame).save(
                frames_dir / f"f_{step // args.render_every:04d}.png"
            )

    np.savez(out_dir / "metrics.npz", **rows)
    wall_s = time.perf_counter() - loop_t0
    ms = np.asarray(step_ms)
    print(
        f"PERF ms/step mean={ms.mean():.2f} p50={np.median(ms):.2f} max={ms.max():.2f} steps={len(ms)} wall_s={wall_s:.2f}"
    )
    if fire_step is not None and fire_step < len(ms):
        print(
            f"PERF split: pre-fire(mean {ms[:fire_step].mean():.2f}, n={fire_step}) post-fire(mean {ms[fire_step:].mean():.2f}, n={len(ms) - fire_step})"
        )

    # continuity: the fragments' imported state (read one substep after the switch) vs the parent at fire
    continuity = None
    if switch_state is not None:
        com_f, P_f, L_f, KE_f = switch_state
        continuity = {
            "mass_err": abs(float(frag_mass.sum()) - mass),
            "com_err": float(np.linalg.norm(com_f - rows["parent_com"][fire_step])),
            "P_err": float(np.linalg.norm(P_f - mass * rows["parent_vel"][fire_step])),
            "L_err": float(np.linalg.norm(L_f - inertia0 @ rows["parent_ang"][fire_step])),
            "KE_err": abs(KE_f - rows["parent_KE"][fire_step]),
        }

    frag_min_z = None
    if final_state is not None:
        pos_f, quat_f = final_state
        frag_min_z = [float(((piece_verts[k] @ _rot(quat_f[k]).T) + pos_f[k])[:, 2].min() * 1e3) for k in range(n)]

    # visual rest gate over the last 0.5 s of the trace
    visual_rest = None
    vr = [(st, vs, ws) for st, vs, ws in rest_trace if st > args.steps - 250]
    if vr:
        vmax = np.zeros(n)
        wmax = np.zeros(n)
        for _, vs, ws in vr:
            vmax = np.maximum(vmax, np.linalg.norm(vs, axis=1))
            wmax = np.maximum(wmax, np.linalg.norm(ws, axis=1))
        visual_rest = {
            "max_v_mm_s": round(float(vmax.max()) * 1e3, 2),
            "max_w_rad_s": round(float(wmax.max()), 4),
            "pass": bool(vmax.max() < 0.003 and wmax.max() < 0.05),
            "per_piece_max_v_mm_s": [round(float(x) * 1e3, 2) for x in vmax],
            "per_piece_max_w_rad_s": [round(float(x), 4) for x in wmax],
        }

    com_trace = rows["parent_com"]
    summary = {
        "object": args.object,
        "seed": args.seed,
        "seeds_md5": assets["seeds_md5"],
        "n_fragments": n,
        "total_blocks": assets["total_blocks"],
        "mass_kg": mass,
        "threshold_Ns": threshold,
        "min_closing_mps": args.min_closing,
        "kick_vx": preset["kick_vx"],
        "kick_omega_y": preset["kick_omega_y"],
        "steps": args.steps,
        "dt": DT,
        "iterations": args.iterations,
        "fire_step": fire_step,
        "fire_time_s": fire_step * DT if fire_step else None,
        "continuity": continuity,
        "frag_min_z_final_mm": frag_min_z,
        "visual_rest": visual_rest,
        "trajectory": {
            # rows where the presentational park moved the parent under the floor stay out of the trace
            "com_min_z": float(com_trace[com_trace[:, 2] > -50.0, 2].min()),
            "com_x_min": float(com_trace[:, 0].min()),
            "left_table": bool(com_trace[:, 0].min() < TABLE_X0),
            "render_min_z_min": float(rows["render_min_z"].min()),
            "render_min_z_final": float(rows["render_min_z"][-1]),
        },
        "perf_ms_per_step": {"mean": float(ms.mean()), "p50": float(np.median(ms)), "max": float(ms.max())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if args.video and frames_dir.exists() and any(frames_dir.iterdir()) and shutil.which("ffmpeg"):
        mp4 = out_dir / "shatter_720p.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "30",
                "-i",
                str(frames_dir / "f_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(mp4),
            ],
            check=True,
            capture_output=True,
        )
        for tag, step in (
            ("intact", 0),
            ("trigger", fire_step or 0),
            ("fracturing", (fire_step or 0) + 10),
            ("settled", args.steps),
        ):
            src = frames_dir / f"f_{max(0, min(step, args.steps)) // args.render_every:04d}.png"
            if src.exists():
                shutil.copy(src, out_dir / f"keyframe_{tag}.png")
        print(f"VIDEO_OK {mp4}")
    print("SHATTER_OK")


if __name__ == "__main__":
    main()
