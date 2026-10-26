"""PBSTF water inside a PBD vase that topples off a table and shatters on the floor.

The vase is a fracturable ``PBD.Solid`` carrying a Voronoi wall partition and holds a low PBSTF
liquid fill; the vase asset is uniformly scaled to ``VASE_SCALE``. The partition comes from
``--seed-layout``: ``regular`` reuses the fixed grid of ``pbd_vase_fracture.py`` (24 fragments),
while ``random`` samples variable-density volumetric seeds through
``genesis.utils.fracture_seeds.sample_volume_seeds``. Every fragment's convex collision proxy
is registered twice: once as a native rigid-body proxy through ``PBDRigidFragmentBridge`` and once
inside a single hidden FEM tetrahedral boundary driven from the same fragment poses. That FEM proxy
(one merged tetra mesh, per-fragment independent vertices, each fragment the fan of its convex hull
over an interior center) is bound to one ``PBSTFAbsorbentBoxStaticColliderOptions`` deformable
collider whose absorption is effectively zero (a 1-voxel micro box with ``total_capacity == 0``), so
the liquid boundary inherits the fragment motion while the coupling stays strictly one-way: the
shards push the water, and the water receives no reaction path back into the vase or the shards.
Before the first step and before every step, the fragment links' world poses (public batched
getters, one host readback) rewrite the proxy vertices through ``FEMEntity.set_position``, and the
PBSTF solver synchronizes the collider surface from the proxy at the next substep.

The convex proxies approximate the curved wall by chord facets that bulge a few millimetres into the
cavity, so the liquid starts with a matching clearance to every hull and the broken wall shows small
gaps between liquid and shards. ``--prepare-proxies`` regenerates the bundled assets for the selected
fragment layout (each random layout seed keeps its own asset subdirectory); the demo reads them
directly otherwise, and verifies at build time that the runtime fragment meshes and spawn poses still
match the bundled ones.
"""

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import igl
import imageio
import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as geom_utils
import genesis.utils.mesh as mesh_utils
import genesis.utils.particle as particle_utils
from genesis.engine.couplers.pbd_rigid_fragments import PBDRigidFragmentBridge
from genesis.utils.fracture_seeds import sample_volume_seeds
from genesis.utils.misc import tensor_to_array

if __package__:
    from .pbd_vase_fracture import fragment_layout, sector_seeds
else:
    from pbd_vase_fracture import fragment_layout, sector_seeds

# Uniform scale of the vase asset; the demo vase simulates at half size.
VASE_SCALE = 0.5
N_FRAGMENTS = 24
# The random layout accepts any runtime fragment count within this span of N_FRAGMENTS.
RANDOM_LAYOUT_COUNT_SPAN = 6
PBSTF_PARTICLE_SIZE = 0.002
MESH_CAVITY_CLEARANCE = 0.004
HULL_CLEARANCE_FACTOR = 0.65
# Vase-local quantities below are already expressed in the scaled local frame.
WATER_SEED_LOCAL = (0.0, 0.0, 0.015)
WATER_MAX_HEIGHT_LOCAL = 0.05
PROXY_ENTITY_NAME = "vase_fragment_proxy"
# The collider box shrinks to a single voxel centered on a fragment-0 hull interior point, and the
# rate/capacity pair floors ``total_capacity`` at zero: the box binds the moving FEM boundary while
# capturing nothing.
COLLIDER_BOX_HALF = 1.0e-6
ABSORPTION_RATE = 1.0e-6
ABSORPTION_CAPACITY_FRACTION = 1.0e-6
# Minimum distance between a fragment's fan center and every hull face plane; keeps every fan tetra
# well conditioned under the explicit FEM integrator.
FAN_CENTER_MIN_CLEARANCE = 5.0e-4
# Hull triangulations can contain near-collinear sliver faces whose area sits below the runtime
# surface-degeneracy margin; they are dropped from the tetra fan (their footprint is far below one
# particle diameter, so the boundary stays watertight at liquid resolution).
DEGENERATE_FACE_AREA_TWICE = 1.0e-6
# Wall lattice spacing of the PBD vase particles; the unit for partition drift tolerances.
LATTICE_SPACING = 6.0e-3
# The proxy is rigidly driven from the fragment poses every step, so its Young's modulus only sets
# the explicit-integrator stiffness of the thinnest fan tets. Stability requires E < rho * (h/dt)^2
# with the tet height h tied to the fragment thickness, so the modulus scales with VASE_SCALE^2.
PROXY_YOUNGS_MODULUS = 1.0e4 * VASE_SCALE**2
# The overhang of the vase centre past the table edge scales with the vase so the half-size base is
# perched at the same ratio; the mesh local origin is the bottom centre, so z is the table top.
VASE_POS = (0.145, 0.0, 0.75)
SETTLE_VASE_POS = (-0.3, 0.0, 0.0)
TABLE_LOWER = (0.15, -0.15, 0.0)
TABLE_UPPER = (0.45, 0.15, 0.75)
PUSH_LIN_VEL = (-0.5, 0.0, 0.0)
PUSH_ANG_VEL = (0.0, -4.0, 0.0)
# The parsed contact time constants are far softer than the impact timescale of a m/s-scale slam at
# this substep dt, so a slammed fragment would sink visibly into the floor before the contact force
# ramps. 4e-4 is the solver's own stability floor (2 * substep dt); the remaining values keep the
# defaults of the native rigid solver.
SOL_PARAMS = (4e-4, 1.0, 0.9, 0.95, 1.0e-3, 0.5, 2.0)
RENDER_RES = (1280, 720)
# Camera offsets are roughly half the full-size ones (same look-at targets, halved stand-off), so
# the half-size vase still fills the frame.
CAMERA_POS_DROP = (-0.72, -0.93, 0.95)
CAMERA_LOOKAT_DROP = (-0.2, -0.2, 0.35)
CAMERA_POS_DROP_CLOSE = (-0.16, -0.4, 0.99)
CAMERA_LOOKAT_DROP_CLOSE = (0.12, 0.0, 0.78)
CAMERA_WIDE_TIME = 0.65
CAMERA_POS_SETTLE = (-0.58, -0.43, 0.33)
CAMERA_LOOKAT_SETTLE = (-0.3, 0.0, 0.1)
DROP_KEYFRAME_TIMES = (0.2, 0.45, 0.7, 1.2, 3.0)
SETTLE_KEYFRAME_TIMES = (0.3, 1.0, 3.0)
ASSET_DIR = Path(__file__).resolve().parent / "pbstf_vase_water_assets"
RUN_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "vase_pbstf_fracture"
DEFAULT_OUTPUT = RUN_OUTPUT_DIR / "pbstf_vase_water_fracture.mp4"


def layout_asset_dir(seed_layout, layout_seed):
    """Bundled-asset directory of a layout; each random layout seed gets its own subdirectory."""
    if seed_layout == "regular":
        return ASSET_DIR
    return ASSET_DIR / f"random_seed_{layout_seed}"


def layout_water_npz(seed_layout, layout_seed):
    """Water-asset NPZ path of a layout."""
    return layout_asset_dir(seed_layout, layout_seed) / "water_particles.npz"


def vase_mesh_file():
    return os.path.join(gs.utils.get_assets_dir(), "meshes", "vase.obj")


def make_vase_morph(pos=VASE_POS):
    """Vase mesh morph shared by the asset preparation scene and the simulation scene."""
    return gs.morphs.Mesh(
        file=vase_mesh_file(),
        scale=VASE_SCALE,
        pos=tuple(pos),
    )


def fragment_count_window(seed_layout):
    """Accepted runtime fragment-count window for a layout."""
    if seed_layout == "regular":
        return (N_FRAGMENTS, N_FRAGMENTS)
    return (N_FRAGMENTS - RANDOM_LAYOUT_COUNT_SPAN, N_FRAGMENTS + RANDOM_LAYOUT_COUNT_SPAN)


def build_fragment_seeds(seed_layout, layout_seed):
    """Voronoi fragment seeds in the scaled vase-local frame for the requested layout."""
    if seed_layout == "regular":
        return [
            (x * VASE_SCALE, y * VASE_SCALE, z * VASE_SCALE) for x, y, z in sector_seeds(*fragment_layout(N_FRAGMENTS))
        ]
    mesh = trimesh.load(vase_mesh_file(), force="mesh")
    mesh.apply_scale(VASE_SCALE)
    count_lo, count_hi = fragment_count_window(seed_layout)
    # The sampler's default radius band and impact-lobe height band target the full-size asset;
    # scaling both keeps the fragment-size mix and lobe position on the half-size wall.
    info = {}
    seeds = sample_volume_seeds(
        mesh=mesh,
        count_range=(count_lo, count_hi),
        rng_seed=layout_seed,
        radius_range=(0.018 * VASE_SCALE, 0.05 * VASE_SCALE),
        impact_band=(0.02 * VASE_SCALE, 0.10 * VASE_SCALE),
        info_out=info,
    )
    print(
        f"[random layout] accepted {len(seeds)} seeds (window {count_lo}-{count_hi}, seed {layout_seed}, "
        f"rescale {info.get('rescale_factor', 1.0)})"
    )
    return [tuple(seed) for seed in seeds]


def make_vase_surface():
    """Vase render surface shared by the preparation and simulation scenes.

    The surface changes the visual mesh's vertex merging, so both scenes must use the identical one
    to sample the same PBD particles.
    """
    return gs.surfaces.Default(
        color=(0.75, 0.75, 0.72, 0.55),
        opacity=0.08,
    )


def make_vase_material(seed_layout="regular", layout_seed=3):
    """Fracturable PBD solid material with the requested fragment Voronoi layout."""
    return gs.materials.PBD.Solid(
        sampler="regular",
        static_friction=0.8,
        kinetic_friction=0.7,
        stiffness=1.0,
        fracture_threshold=0.11,
        shear_threshold=0.2,
        rotation_threshold=0.25,
        seam_failure_threshold=1.0,
        fragment_seeds=build_fragment_seeds(seed_layout, layout_seed),
        bond_radius_factor=1.15,
    )


def world_vase_mesh():
    """Vase surface mesh moved to the spawn pose (identity rotation by construction)."""
    mesh = mesh_utils.load_mesh(vase_mesh_file()).copy()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.vertices = mesh.vertices * VASE_SCALE + np.array(VASE_POS)
    return mesh


def proxy_obj_name(i_k):
    return f"fragment_proxy_{i_k:02d}.obj"


def read_obj(path):
    """Vertices and triangle faces of a bridge-written proxy OBJ."""
    vertices = []
    faces = []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f":
            faces.append([int(part.split("/")[0]) - 1 for part in parts[1:4]])
    return np.array(vertices), np.array(faces, dtype=gs.np_int)


def fragment_link_idxs(scene, n_fragments):
    """Global rigid-solver link indices of the shadow links named ``fragment_i``."""
    links_by_name = {link.name: link for link in scene.rigid_solver.links}
    links_idx = []
    for i_k in range(n_fragments):
        name = f"fragment_{i_k}"
        if name not in links_by_name:
            gs.raise_exception(f"Missing fragment shadow link '{name}'.")
        links_idx.append(links_by_name[name].idx)
    return np.array(links_idx, dtype=gs.np_int)


def fragment_spawn_poses(scene, links_idx):
    """World poses of the fragment links through the public batched getters."""
    pos = tensor_to_array(scene.rigid_solver.get_links_pos(links_idx, relative=False))
    quat = tensor_to_array(scene.rigid_solver.get_links_quat(links_idx))
    return pos, quat


def water_candidate_positions():
    """Cavity lattice of the world-frame vase below the fill height."""
    seed = np.array(WATER_SEED_LOCAL) + np.array(VASE_POS)
    max_height = WATER_MAX_HEIGHT_LOCAL + VASE_POS[2]
    return particle_utils.mesh_cavity_to_particles(
        world_vase_mesh(),
        p_size=PBSTF_PARTICLE_SIZE,
        seed=seed,
        max_height=max_height,
        clearance=MESH_CAVITY_CLEARANCE,
        height_axis=2,
    )


def filter_water_by_proxies(candidates, spawn_pos, asset_dir):
    """Keep the candidates at least one hull clearance outside every fragment proxy hull."""
    clearance = HULL_CLEARANCE_FACTOR * PBSTF_PARTICLE_SIZE
    is_kept = np.ones(len(candidates), dtype=bool)
    for i_k in range(len(spawn_pos)):
        verts, faces = read_obj(asset_dir / proxy_obj_name(i_k))
        distance = igl.signed_distance(candidates, verts + spawn_pos[i_k], faces)[0]
        is_kept &= distance >= clearance
    return candidates[is_kept], int(len(candidates) - is_kept.sum()), clearance


def build_fragment_tetra_proxy(n_fragments, asset_dir):
    """Merge per-fragment convex-hull fans into one tetra mesh with per-fragment independent vertices.

    Each fragment contributes its faces-referenced hull vertices plus one interior center; every
    hull face fans into a tetra with that center, positively oriented to satisfy the tetra mesh
    morph validation. Returns the local-frame vertices, the global elements, the K+1 vertex offsets
    and fragment 0's interior center (the single collider box anchor).
    """
    vertex_blocks = []
    element_blocks = []
    vertex_offsets = [0]
    min_face_area_twice = math.inf
    min_tet_det = math.inf
    min_center_clearance = math.inf
    for i_k in range(n_fragments):
        verts, faces = read_obj(asset_dir / proxy_obj_name(i_k))
        used = np.unique(faces.reshape(-1))
        remap = np.full(len(verts), -1, dtype=gs.np_int)
        remap[used] = np.arange(len(used))
        block_vertices = verts[used]
        block_faces = remap[faces]

        face_edges_1 = block_vertices[block_faces[:, 1]] - block_vertices[block_faces[:, 0]]
        face_edges_2 = block_vertices[block_faces[:, 2]] - block_vertices[block_faces[:, 0]]
        face_normals = np.cross(face_edges_1, face_edges_2)
        face_area_twice = np.linalg.norm(face_normals, axis=1)
        is_degenerate_face = face_area_twice <= DEGENERATE_FACE_AREA_TWICE
        if is_degenerate_face.any():
            block_faces = block_faces[~is_degenerate_face]
            face_normals = face_normals[~is_degenerate_face]
            face_area_twice = face_area_twice[~is_degenerate_face]
            used = np.unique(block_faces.reshape(-1))
            remap = np.full(len(block_vertices), -1, dtype=gs.np_int)
            remap[used] = np.arange(len(used))
            block_vertices = block_vertices[used]
            block_faces = remap[block_faces]
        min_face_area_twice = min(min_face_area_twice, float(face_area_twice.min()))

        # Hull faces of these thin wall chunks can pass arbitrarily close to the vertex centroid,
        # which would leave a sliver fan tetra behind. Nudge the center along the worst face's
        # inward normal until every face plane clears the fan margin; the walk stays inside the
        # convex hull, so the center still stars the whole hull.
        center = block_vertices.mean(axis=0)
        face_origins = block_vertices[block_faces[:, 0]]
        face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]
        for _ in range(32):
            face_distances = -((center - face_origins) * face_normals).sum(axis=1)
            worst = int(face_distances.argmin())
            if face_distances[worst] >= FAN_CENTER_MIN_CLEARANCE:
                break
            center = center - face_normals[worst] * (FAN_CENTER_MIN_CLEARANCE - face_distances[worst])
        face_distances = -((center - face_origins) * face_normals).sum(axis=1)
        if face_distances.min() < 0.5 * FAN_CENTER_MIN_CLEARANCE:
            gs.raise_exception(
                f"Fragment {i_k} hull admits no fan center with the required face clearance "
                f"(worst {face_distances.min():.3e} m)."
            )
        min_center_clearance = min(min_center_clearance, float(face_distances.min()))

        block_elements = np.concatenate(
            (
                np.full((len(block_faces), 1), len(block_vertices), dtype=gs.np_int),
                block_faces,
            ),
            axis=1,
        )
        element_vertices = np.concatenate((block_vertices, center[None]))[block_elements]
        dets = np.linalg.det(element_vertices[:, 1:] - element_vertices[:, :1])
        # fan tets inherit the hull faces' outward orientation; swap the last two vertices to reach
        # the positive orientation the tetra mesh morph requires
        is_negative = dets <= 0.0
        if is_negative.any():
            flipped = block_elements.copy()
            flipped[is_negative, 2] = block_elements[is_negative, 3]
            flipped[is_negative, 3] = block_elements[is_negative, 2]
            block_elements = flipped
        min_tet_det = min(min_tet_det, float(np.abs(dets).min()))

        vertex_blocks.append(np.concatenate((block_vertices, center[None])))
        # element indices are block-local; shift them into the merged vertex array
        element_blocks.append(block_elements + vertex_offsets[-1])
        vertex_offsets.append(vertex_offsets[-1] + len(vertex_blocks[-1]))

    if min_face_area_twice <= 1.0e-6:
        gs.raise_exception(
            f"A fragment proxy hull face is within the runtime degeneracy margin (min 2*area {min_face_area_twice:.3e})."
        )
    return (
        np.concatenate(vertex_blocks),
        np.concatenate(element_blocks),
        np.array(vertex_offsets, dtype=gs.np_int),
        vertex_blocks[0][-1].copy(),
        min_face_area_twice,
        min_tet_det,
        min_center_clearance,
    )


def prepare_assets(args):
    """Export the fragment proxy OBJs and the filtered water spawn through the native bridge."""
    asset_dir = layout_asset_dir(args.seed_layout, args.layout_seed)
    water_npz = layout_water_npz(args.seed_layout, args.layout_seed)
    gs.init(backend=gs.cuda, precision="32", logging_level="info")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
        ),
        pbd_options=gs.options.PBDOptions(
            lower_bound=(-2.0, -2.0, 0.0),
            upper_bound=(2.0, 2.0, 2.0),
            particle_size=0.005,
        ),
        show_viewer=False,
    )
    vase = scene.add_entity(
        morph=make_vase_morph(),
        material=make_vase_material(args.seed_layout, args.layout_seed),
        surface=make_vase_surface(),
    )
    bridge = PBDRigidFragmentBridge(scene.sim)
    scene.sim._pbd_rigid_fragment_bridge = bridge
    asset_dir.mkdir(parents=True, exist_ok=True)
    for stale_obj in asset_dir.glob("fragment_proxy_*.obj"):
        stale_obj.unlink()
    bridge.register_solid_fragments(vase, proxy_dir=asset_dir)
    scene.build(n_envs=0)

    n_fragments = bridge.n_fragments
    count_lo, count_hi = fragment_count_window(args.seed_layout)
    if not count_lo <= n_fragments <= count_hi:
        gs.raise_exception(
            f"The '{args.seed_layout}' fragment layout produced {n_fragments} fragments, "
            f"expected {count_lo}-{count_hi}."
        )
    links_idx = fragment_link_idxs(scene, n_fragments)
    spawn_pos, spawn_quat = fragment_spawn_poses(scene, links_idx)
    identity_quats = np.tile(np.array((1.0, 0.0, 0.0, 0.0)), (n_fragments, 1))
    if not np.allclose(spawn_quat, identity_quats, atol=1.0e-9):
        gs.raise_exception("Fragment spawn quaternions must be identity for the bundled asset frame.")

    candidates = water_candidate_positions()
    positions, n_removed, clearance = filter_water_by_proxies(candidates, spawn_pos, asset_dir)
    if len(positions) == 0:
        gs.raise_exception("Water sampling left no particle outside the fragment hull boundary.")
    if not np.isfinite(positions).all():
        gs.raise_exception("Water sampling produced non-finite positions.")

    (
        proxy_vertices_local,
        proxy_elements,
        proxy_vertex_offsets,
        collider_center_local,
        min_face_area_twice,
        min_tet_det,
        min_center_clearance,
    ) = build_fragment_tetra_proxy(n_fragments, asset_dir)
    if not np.isfinite(proxy_vertices_local).all():
        gs.raise_exception("Fragment proxy tetra mesh contains non-finite vertices.")
    if proxy_elements.min() < 0 or proxy_elements.max() >= len(proxy_vertices_local):
        gs.raise_exception("Fragment proxy tetra elements reference vertices outside the merged mesh.")

    np.savez(
        water_npz,
        positions=positions,
        particle_size=np.array(PBSTF_PARTICLE_SIZE),
        vase_pos=np.array(VASE_POS),
        fragment_pos=spawn_pos,
        fragment_quat=spawn_quat,
        n_fragments=np.array(n_fragments),
        seed_layout=np.array(args.seed_layout),
        layout_seed=np.array(args.layout_seed),
        proxy_vertices_local=proxy_vertices_local,
        proxy_elements=proxy_elements,
        proxy_vertex_offsets=proxy_vertex_offsets,
        collider_center_local=collider_center_local,
    )

    n_verts = 0
    n_faces = 0
    for i_k in range(n_fragments):
        verts, faces = read_obj(asset_dir / proxy_obj_name(i_k))
        n_verts += len(verts)
        n_faces += len(faces)
    print(f"seed layout: {args.seed_layout} (layout seed {args.layout_seed})")
    print(f"fragments: {n_fragments}")
    print(f"proxy hulls: {n_verts} vertices / {n_faces} triangles")
    print(
        f"tetra proxy: {len(proxy_vertices_local)} vertices / {len(proxy_elements)} elements, "
        f"min 2*face area {min_face_area_twice:.3e} m^2, min |tet det| {min_tet_det:.3e} m^3, "
        f"min center clearance {min_center_clearance:.3e} m"
    )
    print(
        f"water candidates: {len(candidates)}, removed within {clearance * 1.0e3:.2f} mm of a hull: "
        f"{n_removed}, kept: {len(positions)}"
    )
    print(f"water z range: {positions[:, 2].min():.4f} .. {positions[:, 2].max():.4f}")
    print(f"assets written to {asset_dir}")


def load_water_asset(water_npz):
    if not water_npz.exists():
        gs.raise_exception(f"Missing {water_npz}; run with --prepare-proxies first.")
    with np.load(water_npz) as data:
        return {key: data[key] for key in data.files}


def hull_signed_volume(vertices, faces):
    return float(
        np.einsum("ij,ij->i", vertices[faces[:, 0]], np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum() / 6.0
    )


def verify_fragment_proxies(proxy_dir, n_fragments, asset_dir):
    """Check the runtime fragment partition against the bundled asset partition at layout level.

    The particle sampler carries cross-process float32 noise that flips exact Voronoi-tie lattice
    points between neighbouring fragments, so per-fragment membership and hull volume legitimately
    vary a few percent between processes; the comparison therefore checks the layout invariants
    (fragment count, total and per-fragment membership, hull bounds, total and per-fragment volume)
    and fails on any real drift such as a different layout, vase, or seed set. The liquid boundary
    itself is built from the bundled assets, so this check protects the rigid-side registration.
    """
    asset_volumes = np.empty(n_fragments)
    runtime_volumes = np.empty(n_fragments)
    for i_k in range(n_fragments):
        asset_verts, asset_faces = read_obj(asset_dir / proxy_obj_name(i_k))
        runtime_verts, runtime_faces = read_obj(Path(proxy_dir) / proxy_obj_name(i_k))
        if abs(len(asset_verts) - len(runtime_verts)) > max(3, int(0.15 * len(asset_verts))):
            gs.raise_exception(
                f"Fragment {i_k} proxy vertex count drifted between the bundled asset and the runtime mesh "
                f"({len(asset_verts)} vs {len(runtime_verts)}); regenerate the assets."
            )
        if not np.allclose(asset_verts.min(axis=0), runtime_verts.min(axis=0), atol=LATTICE_SPACING) or not np.allclose(
            asset_verts.max(axis=0), runtime_verts.max(axis=0), atol=LATTICE_SPACING
        ):
            gs.raise_exception(
                f"Fragment {i_k} proxy hull bounds drifted by more than one lattice spacing between the bundled "
                "asset and the runtime mesh; regenerate the assets."
            )
        asset_volumes[i_k] = hull_signed_volume(asset_verts, asset_faces)
        runtime_volumes[i_k] = hull_signed_volume(runtime_verts, runtime_faces)
        if abs(asset_volumes[i_k] - runtime_volumes[i_k]) > 0.35 * abs(asset_volumes[i_k]):
            gs.raise_exception(
                f"Fragment {i_k} proxy hull volume drifted beyond the Voronoi border swing between the bundled "
                f"asset and the runtime mesh ({asset_volumes[i_k]:.3e} vs {runtime_volumes[i_k]:.3e} m^3); "
                "regenerate the assets."
            )
    if abs(asset_volumes.sum() - runtime_volumes.sum()) > 0.01 * abs(asset_volumes.sum()):
        gs.raise_exception(
            f"The total fragment partition drifted between the bundled asset and the runtime mesh "
            f"({asset_volumes.sum():.3e} vs {runtime_volumes.sum():.3e} m^3); regenerate the assets."
        )


def drive_fragment_proxy(scene, links_idx, proxy, local_vertices, vertex_offsets):
    """Rewrite the proxy vertices from the fragment link poses (one batched read, one host copy)."""
    poses = tensor_to_array(
        torch.cat(
            (
                scene.rigid_solver.get_links_pos(links_idx, relative=False),
                scene.rigid_solver.get_links_quat(links_idx),
            ),
            dim=-1,
        )
    )
    world_vertices = np.empty_like(local_vertices)
    for i_k in range(len(vertex_offsets) - 1):
        block = slice(vertex_offsets[i_k], vertex_offsets[i_k + 1])
        world_vertices[block] = geom_utils.transform_by_quat(local_vertices[block], poses[i_k, 3:]) + poses[i_k, :3]
    proxy.set_position(world_vertices)
    proxy.set_velocity((0.0, 0.0, 0.0))
    return world_vertices


def save_keyframe(camera, keyframe_dir, index, sim_time):
    rgb = tensor_to_array(camera.render()[0])
    path = keyframe_dir / f"keyframe_{index:02d}_t{sim_time:05.2f}s.png"
    imageio.imwrite(path, rgb)
    gs.logger.info(f"Saved keyframe {path}")


def count_water_inside_table(positions):
    return int(np.all((positions > TABLE_LOWER) & (positions < TABLE_UPPER), axis=1).sum())


def measure_wall_clearance(scene, water, bridge, links_idx, proxy_dir, n_fragments):
    """Measure water clearance to every runtime fragment hull at the current physical state."""
    positions = tensor_to_array(water.get_particles_pos())
    poses = tensor_to_array(
        torch.cat(
            (
                scene.rigid_solver.get_links_pos(links_idx, relative=False),
                scene.rigid_solver.get_links_quat(links_idx),
            ),
            dim=-1,
        )
    )
    min_distances = []
    is_inside = np.zeros(len(positions), dtype=bool)
    for i_k in range(n_fragments):
        verts, faces = read_obj(Path(proxy_dir) / proxy_obj_name(i_k))
        world_vertices = geom_utils.transform_by_quat(verts, poses[i_k, 3:]) + poses[i_k, :3]
        distances = igl.signed_distance(positions, world_vertices, faces)[0]
        min_distances.append(float(distances.min()))
        is_inside |= distances < 0.0
    return {
        "sim_time_s": float(scene.cur_t),
        "fragment_handoffs_total": int(bridge.read_commit_counts().sum()),
        "water_min_wall_distance": float(min(min_distances)),
        "n_water_inside_fragment_hulls": int(is_inside.sum()),
        "n_water_inside_table": count_water_inside_table(positions),
        "water_min_distance_per_fragment": min_distances,
    }


def collect_metrics(
    scene, water, bridge, args, n_fragments, wall_time, proxy_deviations, collider, pbstf_options, water_material
):
    positions = tensor_to_array(water.get_particles_pos())
    commit_counts = bridge.read_commit_counts().reshape(-1)
    commit_rejected = bridge.read_commit_rejected().reshape(-1)
    return {
        "mode": "settle_only" if args.settle_only else "drop",
        "seed_layout": args.seed_layout,
        "layout_seed": args.layout_seed,
        "water_render": args.water_render,
        "duration_s": args.duration,
        "sim_time_s": float(scene.cur_t),
        "wall_time_s": wall_time,
        "n_fragments": n_fragments,
        "n_water_particles": int(water.n_particles),
        "water_particle_size": float(pbstf_options.particle_size),
        "pbf_max_solver_iterations": pbstf_options.max_solver_iterations,
        "pbf_iteration_option": "PBSTFOptions.max_solver_iterations",
        "pbstf_options": pbstf_options.model_dump(mode="json"),
        "water_material": {
            "sampler": water_material.sampler,
            "rho": water_material.rho,
            "density_compliance": water_material.density_compliance,
            "surface_tension_compliance": water_material.surface_tension_compliance,
            "surface_distance_compliance": water_material.surface_distance_compliance,
            "interior_distance_compliance": water_material.interior_distance_compliance,
            "surface_viscosity": water_material.surface_viscosity,
            "interior_viscosity": water_material.interior_viscosity,
            "c_init": water_material.c_init,
        },
        "wall_collider_options": [options.model_dump(mode="json") for options in pbstf_options.static_colliders],
        "all_positions_finite": bool(np.isfinite(positions).all()),
        "water_min_z": float(positions[:, 2].min()),
        "water_max_z": float(positions[:, 2].max()),
        "water_xy_min": [float(positions[:, 0].min()), float(positions[:, 1].min())],
        "water_xy_max": [float(positions[:, 0].max()), float(positions[:, 1].max())],
        "water_xy_spread": [float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1]))],
        "fragment_handoffs_total": int(commit_counts.sum()),
        "fragment_handoffs_per_fragment": [int(value) for value in commit_counts],
        "commit_rejected": [int(value) for value in commit_rejected],
        "n_static_colliders": len(scene.pbstf_solver._static_colliders),
        "n_deformable_static_colliders": sum(
            collider.is_deformable for collider in scene.pbstf_solver._static_colliders
        ),
        "n_water_inside_table": count_water_inside_table(positions),
        "collider_deformable_entity_name": collider.deformable_entity_name,
        "collider_total_capacity": collider.total_capacity,
        "collider_n_voxels": collider.n_voxels,
        "collider_voxel_capacity_max": int(collider.voxel_capacity.max()),
        "proxy_follow_max_deviation": float(max(proxy_deviations)),
        "proxy_follow_max_deviation_per_fragment": proxy_deviations,
    }


def run_sim(args):
    asset = load_water_asset(layout_water_npz(args.seed_layout, args.layout_seed))
    n_fragments = int(asset["n_fragments"])
    count_lo, count_hi = fragment_count_window(args.seed_layout)
    if not count_lo <= n_fragments <= count_hi:
        gs.raise_exception("Bundled water asset targets a different fragment count; regenerate with --prepare-proxies.")
    if "seed_layout" not in asset:
        gs.raise_exception("Bundled water asset predates the layout options; regenerate with --prepare-proxies.")
    if str(asset["seed_layout"]) != args.seed_layout or int(asset["layout_seed"]) != args.layout_seed:
        gs.raise_exception(
            "Bundled water asset was generated for a different layout; regenerate with --prepare-proxies."
        )
    if not np.isclose(float(asset["particle_size"]), PBSTF_PARTICLE_SIZE):
        gs.raise_exception("Bundled water asset uses a different particle size; regenerate with --prepare-proxies.")

    spawn_pos = asset["fragment_pos"]
    water_positions = asset["positions"]
    proxy_vertices_local = asset["proxy_vertices_local"]
    proxy_elements = asset["proxy_elements"]
    proxy_vertex_offsets = asset["proxy_vertex_offsets"]
    if len(proxy_vertex_offsets) != n_fragments + 1 or proxy_vertex_offsets[-1] != len(proxy_vertices_local):
        gs.raise_exception(
            "Bundled proxy vertex offsets do not match the merged mesh; regenerate with --prepare-proxies."
        )
    collider_center = asset["collider_center_local"] + spawn_pos[0]
    vase_pos = np.array(VASE_POS)
    if args.settle_only:
        shift = np.array(SETTLE_VASE_POS) - np.array(VASE_POS)
        vase_pos = vase_pos + shift
        water_positions = water_positions + shift
        spawn_pos = spawn_pos + shift
        collider_center = collider_center + shift
    proxy_spawn_vertices = proxy_vertices_local + np.repeat(spawn_pos, np.diff(proxy_vertex_offsets), axis=0)

    gs.init(backend=gs.cuda, precision="32", logging_level="info")
    static_colliders = [
        gs.options.PBSTFAbsorbentBoxStaticColliderOptions(
            pos=tuple(collider_center),
            quat=(1.0, 0.0, 0.0, 0.0),
            is_collider_adhesion_friction_enabled=True,
            collider_adhesion_compliance=50.0,
            collider_friction=0.1,
            absorption_rate=ABSORPTION_RATE,
            absorption_capacity_fraction=ABSORPTION_CAPACITY_FRACTION,
            lower=(-COLLIDER_BOX_HALF, -COLLIDER_BOX_HALF, -COLLIDER_BOX_HALF),
            upper=(COLLIDER_BOX_HALF, COLLIDER_BOX_HALF, COLLIDER_BOX_HALF),
            deformable_entity_name=PROXY_ENTITY_NAME,
        ),
        gs.options.PBSTFBoxStaticColliderOptions(
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            is_collider_adhesion_friction_enabled=True,
            collider_adhesion_compliance=50.0,
            collider_friction=0.5,
            lower=TABLE_LOWER,
            upper=TABLE_UPPER,
        ),
    ]
    pbstf_options = gs.options.PBSTFOptions(
        diffusion_coeff=0.001,
        particle_size=PBSTF_PARTICLE_SIZE,
        lower_bound=(-1.0, -1.0, 0.0),
        upper_bound=(1.0, 1.0, 1.5),
        max_solver_iterations=10,
        topology_rebuild_interval=10,
        max_surface_neighbors=128,
        max_localmesh_neighbors=64,
        enable_pca_normals=False,
        static_colliders=static_colliders,
    )
    water_material = gs.materials.PBSTF.Liquid(
        sampler="regular",
        rho=1000.0,
        density_compliance=375000.0,
        surface_tension_compliance=1.0 / 225.0,
        surface_distance_compliance=40.0,
        interior_distance_compliance=180.0,
        surface_viscosity=0.5,
        interior_viscosity=0.5,
        c_init=0.0,
    )

    camera_pos = CAMERA_POS_SETTLE if args.settle_only else CAMERA_POS_DROP_CLOSE
    camera_lookat = CAMERA_LOOKAT_SETTLE if args.settle_only else CAMERA_LOOKAT_DROP_CLOSE
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
        ),
        pbd_options=gs.options.PBDOptions(
            # the domain floor sits below the ground plane: the band above the domain floor would
            # otherwise push the resting vase's bottom particles and shatter its seams at spawn
            lower_bound=(-2.0, -2.0, -0.05),
            upper_bound=(2.0, 2.0, 2.0),
            particle_size=0.005,
        ),
        # the FEM proxy coincides with the fragment collision proxies, so rigid-FEM contact would
        # push the driven boundary non-rigidly and shove the shards off their own hulls
        coupler_options=gs.options.LegacyCouplerOptions(
            rigid_fem=False,
        ),
        pbstf_options=pbstf_options,
        viewer_options=gs.options.ViewerOptions(
            realtime_factor=1.0,
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=50,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            lower=TABLE_LOWER,
            upper=TABLE_UPPER,
            fixed=True,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(
            color=(0.36, 0.24, 0.14),
        ),
    )
    vase = scene.add_entity(
        morph=make_vase_morph(pos=vase_pos),
        material=make_vase_material(args.seed_layout, args.layout_seed),
        surface=make_vase_surface(),
    )
    # the hidden FEM proxy is the deformable geometry of the single PBSTF boundary; adding it before
    # the liquid orders the FEM solver ahead of the PBSTF solver, so the collider surface refreshes
    # right after the proxy pose write of the same frame
    fragment_proxy = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            vertices=proxy_spawn_vertices,
            elements=proxy_elements,
        ),
        material=gs.materials.FEM.Elastic(
            E=PROXY_YOUNGS_MODULUS,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Default(
            opacity=0.0,
        ),
        name=PROXY_ENTITY_NAME,
    )
    water = scene.add_entity(
        morph=gs.morphs.Particles(
            positions=water_positions,
        ),
        material=water_material,
        surface=gs.surfaces.Default(
            color=(0.08, 0.3, 0.8, 0.9),
            vis_mode="recon" if args.water_render == "recon" else "particle",
        ),
    )

    cam = None
    if args.record:
        cam = scene.add_camera(
            res=RENDER_RES,
            pos=camera_pos,
            lookat=camera_lookat,
            fov=50,
        )
    key_cam = None
    if args.keyframe_dir is not None:
        args.keyframe_dir.mkdir(parents=True, exist_ok=True)
        key_cam = scene.add_camera(
            res=RENDER_RES,
            pos=camera_pos,
            lookat=camera_lookat,
            fov=50,
        )

    bridge = PBDRigidFragmentBridge(scene.sim)
    scene.sim._pbd_rigid_fragment_bridge = bridge
    # the temporary registration directory must outlive build and run: the shadow links keep their
    # mesh geoms loaded from it for the whole simulation
    proxy_dir = Path(tempfile.mkdtemp(prefix="pbstf_vase_water_fracture_"))
    bridge.register_solid_fragments(vase, proxy_dir=proxy_dir)
    scene.build(n_envs=0)

    colliders = scene.pbstf_solver._static_colliders
    if len(colliders) != 2 or sum(collider.is_deformable for collider in colliders) != 1:
        gs.raise_exception("The scene requires one deformable fragment boundary and one analytic table collider.")
    collider = colliders[0]
    if collider.total_capacity != 0:
        gs.raise_exception(f"The fragment boundary collider must have zero capacity, got {collider.total_capacity}.")

    if bridge.n_fragments != n_fragments:
        gs.raise_exception(f"Runtime registration produced {bridge.n_fragments} fragments, expected {n_fragments}.")
    verify_fragment_proxies(proxy_dir, n_fragments, layout_asset_dir(args.seed_layout, args.layout_seed))

    links_idx = fragment_link_idxs(scene, n_fragments)
    spawn_pose_pos, spawn_pose_quat = fragment_spawn_poses(scene, links_idx)
    if not np.allclose(spawn_pose_pos, spawn_pos, atol=LATTICE_SPACING):
        gs.raise_exception("Fragment spawn poses drifted from the bundled asset; regenerate with --prepare-proxies.")
    if not np.allclose(spawn_pose_quat, asset["fragment_quat"], atol=1.0e-6):
        gs.raise_exception(
            "Fragment spawn quaternions drifted from the bundled asset; regenerate with --prepare-proxies."
        )

    scene.rigid_solver.set_global_sol_params(SOL_PARAMS)

    if not args.settle_only:
        vase_particles = tensor_to_array(vase.get_particles_pos())
        push_center = vase_particles.mean(axis=0)
        vase.set_particles_vel(np.cross(PUSH_ANG_VEL, vase_particles - push_center) + PUSH_LIN_VEL)
        water_particles = tensor_to_array(water.get_particles_pos())
        water.set_particles_vel(np.cross(PUSH_ANG_VEL, water_particles - push_center) + PUSH_LIN_VEL)

    key_times = args.keyframe_times
    if key_times is None:
        key_times = SETTLE_KEYFRAME_TIMES if args.settle_only else DROP_KEYFRAME_TIMES
    key_times = sorted(time_value for time_value in key_times if 0.0 < time_value < args.duration)
    next_key = 0

    if cam is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cam.start_recording(save_to_filename=str(args.output), fps=50)

    wall_start = time.perf_counter()
    n_steps = math.ceil(args.duration / scene.dt)
    proxy_target = None
    keyframe_diagnostics = []
    for _ in range(n_steps):
        if not args.settle_only:
            progress = min(scene.cur_t / CAMERA_WIDE_TIME, 1.0)
            blend = progress * progress * (3.0 - 2.0 * progress)
            camera_pos = (1.0 - blend) * np.array(CAMERA_POS_DROP_CLOSE) + blend * np.array(CAMERA_POS_DROP)
            camera_lookat = (1.0 - blend) * np.array(CAMERA_LOOKAT_DROP_CLOSE) + blend * np.array(CAMERA_LOOKAT_DROP)
            for camera in (cam, key_cam):
                if camera is not None:
                    camera.set_pose(pos=camera_pos, lookat=camera_lookat)
        if next_key < len(key_times) and scene.cur_t >= key_times[next_key]:
            save_keyframe(key_cam, args.keyframe_dir, next_key, scene.cur_t)
            keyframe_diagnostics.append(measure_wall_clearance(scene, water, bridge, links_idx, proxy_dir, n_fragments))
            next_key += 1
        proxy_target = drive_fragment_proxy(
            scene, links_idx, fragment_proxy, proxy_vertices_local, proxy_vertex_offsets
        )
        scene.step()
    wall_time = time.perf_counter() - wall_start

    if cam is not None:
        cam.stop_recording()
    if key_cam is not None:
        save_keyframe(key_cam, args.keyframe_dir, len(key_times), scene.cur_t)
        keyframe_diagnostics.append(measure_wall_clearance(scene, water, bridge, links_idx, proxy_dir, n_fragments))

    proxy_positions = tensor_to_array(fragment_proxy.get_state().pos)[0]
    proxy_deviations = []
    for i_k in range(n_fragments):
        block = slice(*proxy_vertex_offsets[i_k : i_k + 2])
        proxy_deviations.append(float(np.abs(proxy_positions[block] - proxy_target[block]).max()))
    metrics = collect_metrics(
        scene, water, bridge, args, n_fragments, wall_time, proxy_deviations, collider, pbstf_options, water_material
    )
    metrics["keyframe_wall_clearance"] = keyframe_diagnostics
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="ascii")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-r", "--record", action="store_true", default=False, help="record an mp4 of the shatter")
    parser.add_argument("--duration", type=float, default=8.0, help="simulation duration in seconds")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="recording output path")
    parser.add_argument(
        "--prepare-proxies",
        action="store_true",
        default=False,
        help="regenerate the bundled fragment proxy OBJs and water spawn, then exit",
    )
    parser.add_argument(
        "--settle-only",
        action="store_true",
        default=False,
        help="place the filled vase on the floor with no push, to inspect fragment release",
    )
    parser.add_argument(
        "--water-render",
        choices=("recon", "particle"),
        default="recon",
        help="render the liquid as a reconstructed surface or as particles",
    )
    parser.add_argument(
        "--seed-layout",
        choices=("regular", "random"),
        default="regular",
        help="fragment seed layout: the fixed wall grid or a volumetric random sample",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=3,
        help="RNG seed for the random fragment layout",
    )
    parser.add_argument("--metrics-output", type=Path, default=None, help="write the run metrics JSON here")
    parser.add_argument("--keyframe-dir", type=Path, default=None, help="save rendered keyframes here")
    parser.add_argument(
        "--keyframe-times",
        type=float,
        nargs="*",
        default=None,
        help="keyframe times in seconds (defaults follow the selected mode)",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.prepare_proxies:
        prepare_assets(args)
        return
    run_sim(args)


if __name__ == "__main__":
    main()
