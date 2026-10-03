from dataclasses import dataclass
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .config import GarmentFoldingScenarioConfig


@dataclass(frozen=True)
class GarmentLandmarks:
    hem_left: tuple[int, ...]
    hem_right: tuple[int, ...]
    hem_center: tuple[int, ...]
    sleeve_left: tuple[int, ...]
    sleeve_right: tuple[int, ...]

    def indices(self, name: str) -> tuple[int, ...]:
        if name == "hem_left":
            return self.hem_left
        if name == "hem_right":
            return self.hem_right
        if name == "hem_center":
            return self.hem_center
        if name == "sleeve_left":
            return self.sleeve_left
        if name == "sleeve_right":
            return self.sleeve_right
        raise ValueError(f"Unknown garment landmark: {name}")


@dataclass(frozen=True)
class GarmentFoldingRuntime:
    """Named scene handles used by task controllers and downstream adapters."""

    scene: gs.Scene
    garment: object
    table: object
    jaws: tuple[tuple[object, object], ...]
    camera: object | None
    initial_positions: np.ndarray
    landmarks: GarmentLandmarks
    garment_pos: tuple[float, float, float]
    garment_scale: float
    jaw_origins_local: tuple[tuple[float, float, float], ...]
    jaw_half_thickness: float

    def get_jaw_poses(self):
        """Return current jaw poses through the public rigid-entity state API."""
        return tuple(tuple((jaw.get_pos(), jaw.get_quat()) for jaw in pair) for pair in self.jaws)


def build_scene(
    config: GarmentFoldingScenarioConfig | None = None,
    *,
    show_viewer: bool = False,
    add_camera: bool = False,
) -> GarmentFoldingRuntime:
    """Build the garment-folding scene without starting task motion or recording."""
    config = config or GarmentFoldingScenarioConfig()
    solver = config.solver
    material = config.material
    assets = config.assets
    package_root = Path(__file__).resolve().parent
    garment_mesh = package_root / assets.garment_mesh
    garment_texture = package_root / assets.garment_texture
    if not garment_mesh.is_file():
        raise FileNotFoundError(f"Garment mesh does not exist: {garment_mesh}")
    if not garment_texture.is_file():
        raise FileNotFoundError(f"Garment texture does not exist: {garment_texture}")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=solver.dt, gravity=solver.gravity),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=solver.contact_d_hat,
            contact_resistance=solver.contact_resistance,
            newton_semi_implicit_enable=solver.newton_semi_implicit_enable,
            n_linesearch_iterations=solver.n_linesearch_iterations,
            contact_friction_enable=True,
            constraint_strength_translation=solver.constraint_strength_translation,
            constraint_strength_rotation=solver.constraint_strength_rotation,
            two_way_coupling=solver.two_way_coupling,
        ),
        viewer_options=gs.options.ViewerOptions(camera_pos=assets.camera_pos, camera_lookat=assets.camera_lookat),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(coup_friction=material.support_friction),
        surface=gs.surfaces.Default(color=(0.28, 0.31, 0.34, 1.0)),
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(size=assets.table_size, pos=assets.table_pos, fixed=True),
        material=gs.materials.Rigid(coup_friction=material.support_friction),
        surface=gs.surfaces.Default(color=(0.75, 0.72, 0.66, 1.0)),
    )
    garment = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(garment_mesh),
            pos=assets.garment_pos,
            scale=assets.garment_scale,
            decimate=False,
        ),
        material=gs.materials.FEM.Cloth(
            E=material.cloth_young_modulus,
            nu=material.cloth_poisson_ratio,
            rho=material.cloth_density,
            thickness=material.cloth_thickness,
            bending_stiffness=material.cloth_bending_stiffness,
            friction_mu=material.cloth_friction,
        ),
        surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(garment_texture))),
    )

    jaw_origins_local = assets.jaw_origins(config.task.task)
    jaw_half_thickness = 0.5 * assets.jaw_size[2]
    jaws = []
    for origin_local in jaw_origins_local:
        origin = tuple(
            world_origin + assets.garment_scale * value
            for world_origin, value in zip(assets.garment_pos, origin_local)
        )
        pair = []
        for sign in (-1.0, 1.0):
            pair.append(
                scene.add_entity(
                    morph=gs.morphs.Box(
                        size=assets.jaw_size,
                        pos=(
                            origin[0],
                            origin[1],
                            origin[2] + sign * (0.5 * config.task.open_gap + jaw_half_thickness),
                        ),
                    ),
                    material=gs.materials.Rigid(
                        coup_type="two_way_soft_constraint",
                        rho=material.jaw_density,
                        coup_friction=material.jaw_friction,
                        gravity_compensation=material.jaw_gravity_compensation,
                    ),
                    surface=gs.surfaces.Default(color=(0.7, 0.74, 0.8, 1.0)),
                )
            )
        jaws.append(tuple(pair))

    camera = None
    if add_camera:
        camera = scene.add_camera(
            res=assets.camera_res,
            pos=assets.camera_pos,
            lookat=assets.camera_lookat,
            fov=assets.camera_fov,
            GUI=False,
        )
    scene.build()

    initial_positions = tensor_to_array(garment.get_state().pos).reshape(-1, 3)
    landmark_specs = tuple(
        (name, assets.world_landmark(name))
        for name in ("hem_left", "hem_right", "hem_center", "sleeve_left", "sleeve_right")
    )
    landmark_indices = []
    for _, position in landmark_specs:
        distances = np.linalg.norm(initial_positions[:, :2] - position[:2], axis=1)
        landmark_indices.append(tuple(np.argsort(distances)[:6]))
    landmarks = GarmentLandmarks(*landmark_indices)
    return GarmentFoldingRuntime(
        scene=scene,
        garment=garment,
        table=table,
        jaws=tuple(jaws),
        camera=camera,
        initial_positions=initial_positions,
        landmarks=landmarks,
        garment_pos=assets.garment_pos,
        garment_scale=assets.garment_scale,
        jaw_origins_local=jaw_origins_local,
        jaw_half_thickness=jaw_half_thickness,
    )
