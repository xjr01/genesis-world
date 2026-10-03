from dataclasses import dataclass

import genesis as gs
from examples.multiflow.teapot import pbstf_surface_tension as legacy

from .config import TableWipingScenarioConfig


@dataclass(frozen=True)
class TableWipingRuntime:
    scene: gs.Scene
    settings: legacy.WipeSettings
    entities: tuple[object, ...]
    initial_qpos: gs.Tensor


def _create_settings(config: TableWipingScenarioConfig):
    solver = config.solver
    material = config.material
    assets = config.assets
    task = config.task
    manipulator = legacy.MopManipulatorSettings(
        entity_name="mop_manipulator",
        sponge_grid_resolution=assets.sponge_grid_resolution,
        sponge_density=assets.sponge_density,
        asset=assets.robot,
        is_visible=True,
        scale=1.0,
        base_pos=assets.robot_base_pos,
        base_quat=assets.robot_base_quat,
        hand_link_name="panda_link7",
        left_finger_link_name="panda_leftfinger",
        right_finger_link_name="panda_rightfinger",
        tool_center_point=task.tool_center_point,
        grasp_quat=task.grasp_quat,
        initial_qpos=task.initial_qpos,
        finger_open_qpos=task.finger_open_qpos,
        finger_closed_qpos=task.finger_closed_qpos,
    )
    wipe = legacy.WipeSettings(
        collider_idx=1,
        collider_entity_name="sponge",
        collider_lower=assets.sponge_lower,
        collider_upper=assets.sponge_upper,
        table_entity_name="wipe_table",
        table_pos=assets.table_pos,
        table_size=assets.table_size,
        liquid_lower=assets.liquid_lower,
        liquid_upper=assets.liquid_upper,
        start_pos=task.start_pos,
        end_pos=task.end_pos,
        quat=task.quat,
        settle_time=task.settle_time,
        wipe_time=task.wipe_time,
        mop_manipulator=manipulator,
    )
    table_collider = gs.options.PBSTFBoxStaticColliderOptions(
        pos=assets.table_pos,
        quat=task.quat,
        is_collider_adhesion_friction_enabled=True,
        collider_adhesion_compliance=material.collider_adhesion_compliance,
        collider_friction=material.collider_friction,
        lower=tuple(-0.5 * value for value in assets.table_size),
        upper=tuple(0.5 * value for value in assets.table_size),
    )
    sponge_collider = gs.options.PBSTFAbsorbentBoxStaticColliderOptions(
        pos=task.start_pos,
        quat=task.quat,
        is_collider_adhesion_friction_enabled=True,
        collider_adhesion_compliance=material.collider_adhesion_compliance,
        collider_friction=material.collider_friction,
        lower=assets.sponge_lower,
        upper=assets.sponge_upper,
        absorption_rate=material.absorption_rate,
        absorption_capacity_fraction=material.absorption_capacity_fraction,
        pbd_entity_name="sponge",
    )
    settings = legacy.CaseSettings(
        scale=300,
        dt=solver.dt,
        gravity=solver.gravity,
        lower_bound=solver.lower_bound,
        upper_bound=solver.upper_bound,
        camera_pos=(0.0, 0.3, 2.0 / 3.0),
        camera_lookat=(0.0, 1.0 / 30.0, -0.1),
        ground_height=assets.table_pos[1] - 0.5 * assets.table_size[1],
        static_colliders=(table_collider, sponge_collider),
        max_solver_iterations=solver.max_solver_iterations,
        max_surface_neighbors=solver.max_surface_neighbors,
        max_localmesh_neighbors=solver.max_localmesh_neighbors,
        enable_pca_normals=False,
        steps=task.steps,
        teapot=None,
        emitter=None,
        mop=wipe,
        sweep=None,
    )
    liquid_material = gs.materials.PBSTF.Liquid(
        sampler="regular",
        rho=material.density,
        density_compliance=material.density_compliance,
        surface_tension_compliance=material.surface_tension_compliance,
        surface_distance_compliance=material.surface_distance_compliance,
        interior_distance_compliance=material.interior_distance_compliance,
        surface_viscosity=material.surface_viscosity,
        interior_viscosity=material.interior_viscosity,
    )
    return settings, liquid_material


def build_scene(
    config: TableWipingScenarioConfig | None = None,
    *,
    show_viewer: bool = False,
) -> TableWipingRuntime:
    """Build the wiping scene from separate solver, material, asset and task inputs."""
    config = config or TableWipingScenarioConfig()
    settings, liquid_material = _create_settings(config)
    scene, entities = legacy.build_scene(
        case=legacy.CASE_MOP,
        show_viewer=show_viewer,
        settings=settings,
        particle_size=config.solver.particle_size,
        liquid_material=liquid_material,
        pbd_options=gs.options.PBDUnifiedOptions(
            particle_size=config.solver.pbd_particle_size,
            lower_bound=config.solver.lower_bound,
            upper_bound=config.solver.upper_bound,
            max_solver_iterations=config.solver.pbd_solver_iterations,
            constraint_acceleration=config.solver.pbd_constraint_acceleration,
        ),
        sponge_grid_resolution=config.assets.sponge_grid_resolution,
    )
    manipulator = settings.mop.mop_manipulator
    if manipulator is None:
        gs.raise_exception("The table-wiping task requires a manipulator.")
    initial_qpos = scene.get_entity(name=manipulator.entity_name).get_qpos()
    return TableWipingRuntime(scene=scene, settings=settings.mop, entities=entities, initial_qpos=initial_qpos)
