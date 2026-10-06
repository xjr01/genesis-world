from dataclasses import dataclass

import torch

import genesis as gs

from .config import ButterSpreadingScenarioConfig
from .contact import ButterContact
from .task import knife_pose


@dataclass(frozen=True)
class ButterSpreadingRuntime:
    scene: gs.Scene
    config: ButterSpreadingScenarioConfig
    bread: object
    butter: object
    blade: object
    contact: ButterContact
    blade_xy_offset: tuple[float, float]
    task_origin: tuple[float, float, float]
    camera: object | None

    @property
    def control_dt(self) -> float:
        return self.config.solver.dt


def _rigid_material(config: ButterSpreadingScenarioConfig, *, is_blade: bool):
    material = config.material
    assets = config.assets
    return gs.materials.Rigid(
        rho=material.blade_density if is_blade else material.support_density,
        friction=material.blade_friction if is_blade else material.support_friction,
        coup_friction=material.blade_coupling_friction if is_blade else 1.0,
        coup_restitution=0.0,
        coup_softness=material.coupling_softness,
        needs_coup=True,
        sdf_cell_size=assets.blade_sdf_cell_size if is_blade else assets.support_sdf_cell_size,
        sdf_min_res=32,
        sdf_max_res=128,
    )


def build_scene(
    config: ButterSpreadingScenarioConfig | None = None,
    *,
    show_viewer: bool = False,
    add_camera: bool = False,
) -> ButterSpreadingRuntime:
    """Build the calibrated, self-contained butter-spreading MPM task."""
    config = config or ButterSpreadingScenarioConfig()
    solver = config.solver
    material = config.material
    assets = config.assets
    initial_local_blade = knife_pose(config.task, 0.0)
    initial_blade = tuple(origin + value for origin, value in zip(assets.task_origin, initial_local_blade))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=solver.dt, substeps=1, gravity=solver.gravity),
        mpm_options=gs.options.MPMOptions(
            dt=solver.dt,
            gravity=solver.gravity,
            lower_bound=solver.lower_bound,
            upper_bound=solver.upper_bound,
            particle_size=solver.bread_particle_size,
            grid_density=solver.grid_density,
            enable_CPIC=solver.is_cpic_enabled,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_mpm=True),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            lower=(
                -0.5 * assets.support_size_xy[0],
                -0.5 * assets.support_size_xy[1],
                assets.plate_height - assets.support_depth,
            ),
            upper=(0.5 * assets.support_size_xy[0], 0.5 * assets.support_size_xy[1], assets.plate_height),
            fixed=True,
        ),
        material=_rigid_material(config, is_blade=False),
    )
    bread = scene.add_entity(
        morph=gs.morphs.Box(pos=assets.bread_center, size=assets.bread_size),
        material=gs.materials.MPM.PorousBread(
            E=material.bread_youngs_modulus,
            nu=material.bread_poissons_ratio,
            rho=material.bread_density,
            compaction_yield_pressure=material.bread_compaction_yield_pressure,
            compaction_hardening=material.bread_compaction_hardening,
            densification_strain=material.bread_densification_strain,
            densification_hardening=material.bread_densification_hardening,
            min_plastic_volume_ratio=material.bread_min_plastic_volume_ratio,
            shear_yield_stress=material.bread_shear_yield_stress,
            shear_hardening=material.bread_shear_hardening,
            sampler="regular",
            particle_size=solver.bread_particle_size,
        ),
        surface=gs.surfaces.Plastic(color=(0.72, 0.55, 0.35), roughness=0.9),
        vis_mode="particle",
    )
    blade = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(
                initial_blade[0] + assets.blade_xy_offset[0],
                initial_blade[1] + assets.blade_xy_offset[1],
                initial_blade[2],
            ),
            size=assets.blade_size,
        ),
        material=_rigid_material(config, is_blade=True),
    )
    butter = scene.add_entity(
        morph=gs.morphs.Box(pos=assets.butter_center, size=assets.butter_size),
        material=gs.materials.MPM.HerschelBulkleyButter(
            shear_modulus=material.butter_shear_modulus,
            bulk_modulus=material.butter_bulk_modulus,
            yield_stress=material.butter_yield_stress,
            consistency=material.butter_consistency,
            flow_exponent=material.butter_flow_exponent,
            rho=material.butter_density,
            dt=solver.dt,
            sampler="regular",
            particle_size=solver.butter_particle_size,
        ),
        surface=gs.surfaces.Plastic(color=(0.97, 0.78, 0.30), roughness=0.3),
        vis_mode="particle",
    )
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

    bread_pos = bread.get_particles_pos()
    bread_pos[:, 2] += assets.plate_height + 0.5 * solver.bread_particle_size - bread_pos[:, 2].min()
    bread.set_particles_pos(bread_pos)
    bread.set_particles_vel(gs.zeros_like(bread_pos))
    butter_pos = butter.get_particles_pos()
    material_gap = 0.5 * (solver.bread_particle_size + solver.butter_particle_size)
    butter_pos[:, 2] += bread.get_particles_pos()[:, 2].max() + material_gap - butter_pos[:, 2].min()
    if solver.butter_sampling == "stratified":
        generator = torch.Generator(device=butter_pos.device)
        generator.manual_seed(solver.butter_sampling_seed)
        offsets = torch.empty_like(butter_pos).uniform_(generator=generator)
        butter_pos += (offsets - 0.5) * 0.9 * solver.butter_particle_size
    butter.set_particles_pos(butter_pos)
    butter.set_particles_vel(gs.zeros_like(butter_pos))

    contact_config = config.contact
    contact = ButterContact(
        bread,
        butter,
        blade,
        spacing=solver.butter_particle_size,
        dt=solver.dt,
        blade_size=assets.blade_size,
        lower_bound=solver.lower_bound,
        upper_bound=solver.upper_bound,
        bread_stress=contact_config.bread_normal_stress,
        blade_stress=contact_config.blade_normal_stress,
        bread_contact_range=contact_config.bread_contact_range,
        bread_slip_time=contact_config.bread_slip_time,
        bread_shear_stress=contact_config.bread_shear_stress,
        blade_contact_range=contact_config.butter_contact_range,
        blade_slip_time=contact_config.blade_slip_time,
        blade_shear_stress=contact_config.blade_shear_stress,
        blade_normal_relaxation=solver.dt / contact_config.blade_normal_relaxation_time,
        blade_max_separation_speed=contact_config.blade_max_separation_speed,
        blade_contact_margin=contact_config.blade_contact_margin,
        is_equilibrium_adhesion=contact_config.is_equilibrium_adhesion,
    )
    scene.register_pre_step_callback(contact)
    scene.reset(scene.get_state())

    return ButterSpreadingRuntime(
        scene=scene,
        config=config,
        bread=bread,
        butter=butter,
        blade=blade,
        contact=contact,
        blade_xy_offset=assets.blade_xy_offset,
        task_origin=assets.task_origin,
        camera=camera,
    )
