from dataclasses import dataclass

import genesis as gs
from genesis.utils import element

from .config import TableWipingScenarioConfig


@dataclass(frozen=True)
class TableWipingSettings:
    collider_idx: int
    collider_entity_name: str
    collider_lower: tuple[float, float, float]
    collider_upper: tuple[float, float, float]
    table_entity_name: str
    table_pos: tuple[float, float, float]
    table_size: tuple[float, float, float]
    liquid_lower: tuple[float, float, float]
    liquid_upper: tuple[float, float, float]
    start_pos: tuple[float, float, float]
    end_pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    settle_time: float
    wipe_time: float


@dataclass(frozen=True)
class TableWipingRuntime:
    scene: gs.Scene
    config: TableWipingScenarioConfig
    settings: TableWipingSettings
    entities: tuple[object, ...]
    liquid: object
    sponge: object
    table: object
    robot: object
    initial_qpos: gs.Tensor
    camera: object | None

    @property
    def control_dt(self) -> float:
        return self.config.solver.dt


def build_scene(
    config: TableWipingScenarioConfig | None = None,
    *,
    show_viewer: bool = False,
    add_camera: bool = False,
) -> TableWipingRuntime:
    """Build the self-contained PBSTF wiping task."""
    config = config or TableWipingScenarioConfig()
    solver = config.solver
    material = config.material
    assets = config.assets
    task = config.task
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
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=solver.dt,
            gravity=solver.gravity,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            disable_constraint=True,
        ),
        pbd_options=gs.options.PBDUnifiedOptions(
            particle_size=solver.pbd_particle_size,
            lower_bound=solver.lower_bound,
            upper_bound=solver.upper_bound,
            max_solver_iterations=solver.pbd_solver_iterations,
            constraint_acceleration=solver.pbd_constraint_acceleration,
        ),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=solver.particle_size,
            lower_bound=solver.lower_bound,
            upper_bound=solver.upper_bound,
            max_solver_iterations=solver.max_solver_iterations,
            topology_rebuild_interval=10,
            max_surface_neighbors=solver.max_surface_neighbors,
            max_localmesh_neighbors=solver.max_localmesh_neighbors,
            enable_pca_normals=False,
            static_colliders=[table_collider, sponge_collider],
        ),
        viewer_options=gs.options.ViewerOptions(
            refresh_rate=round(1.0 / solver.dt),
            camera_pos=(0.0, 0.3, 2.0 / 3.0),
            camera_lookat=(0.0, 1.0 / 30.0, -0.1),
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    liquid = scene.add_entity(
        morph=gs.morphs.Box(
            lower=assets.liquid_lower,
            upper=assets.liquid_upper,
        ),
        material=gs.materials.PBSTF.Liquid(
            sampler="regular",
            rho=material.density,
            density_compliance=material.density_compliance,
            surface_tension_compliance=material.surface_tension_compliance,
            surface_distance_compliance=material.surface_distance_compliance,
            interior_distance_compliance=material.interior_distance_compliance,
            surface_viscosity=material.surface_viscosity,
            interior_viscosity=material.interior_viscosity,
        ),
    )
    sponge_size = tuple(upper - lower for lower, upper in zip(assets.sponge_lower, assets.sponge_upper))
    sponge_pos = tuple(
        start + 0.5 * (lower + upper)
        for start, lower, upper in zip(task.start_pos, assets.sponge_lower, assets.sponge_upper)
    )
    sponge_vertices, sponge_elements = element.create_tetrahedral_grid(
        lower=tuple(-0.5 * size for size in sponge_size),
        upper=tuple(0.5 * size for size in sponge_size),
        resolution=assets.sponge_grid_resolution,
    )
    sponge = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=sponge_pos,
            vertices=sponge_vertices,
            elements=sponge_elements,
        ),
        material=gs.materials.PBD.Elastic(
            rho=assets.sponge_density,
            stretch_relaxation=0.25,
            volume_relaxation=0.15,
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.68, 0.12),
            vis_mode="tetrahedral",
        ),
        name="sponge",
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            pos=assets.table_pos,
            quat=task.quat,
            collision=True,
            fixed=True,
            size=assets.table_size,
        ),
        material=gs.materials.Rigid(
            coup_friction=0.0,
            is_coup_reaction_enabled=False,
        ),
        surface=gs.surfaces.Default(
            color=(0.36, 0.24, 0.14),
        ),
        name="wipe_table",
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=assets.robot,
            pos=assets.robot_base_pos,
            quat=assets.robot_base_quat,
            visualization=True,
            convexify=False,
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coup_friction=0.0,
            is_coup_reaction_enabled=False,
            gravity_compensation=1.0,
        ),
        name="mop_manipulator",
    )
    camera = None
    if add_camera:
        camera = scene.add_camera(
            res=assets.camera_res,
            pos=assets.camera_pos,
            lookat=assets.camera_lookat,
            up=assets.camera_up,
            fov=assets.camera_fov,
            GUI=False,
        )
    scene.build()
    robot.set_qpos(task.initial_qpos, zero_velocity=True)
    target_pos = (
        task.start_pos[0] + 0.5 * (assets.sponge_lower[0] + assets.sponge_upper[0]),
        task.start_pos[1] + assets.sponge_upper[1],
        task.start_pos[2] + 0.5 * (assets.sponge_lower[2] + assets.sponge_upper[2]),
    )
    initial_qpos = robot.inverse_kinematics(
        link=robot.get_link("panda_link7"),
        pos=target_pos,
        quat=task.grasp_quat,
        local_point=task.tool_center_point,
        init_qpos=robot.get_qpos(),
        pos_tol=1.0e-4,
        rot_tol=1.0e-4,
        dofs_idx_local=range(7),
    )
    initial_qpos[..., -2:] = task.finger_open_qpos
    robot.set_qpos(initial_qpos, zero_velocity=True)
    scene.reset(scene.get_state())
    settings = TableWipingSettings(
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
    )
    return TableWipingRuntime(
        scene=scene,
        config=config,
        settings=settings,
        entities=(liquid,),
        liquid=liquid,
        sponge=sponge,
        table=table,
        robot=robot,
        initial_qpos=initial_qpos.clone(),
        camera=camera,
    )
