# PBSTF Surface Tension Demo Configuration

This document describes the `teapot`, `sweep`, and `mop` cases in
[`pbstf_surface_tension.py`](pbstf_surface_tension.py). Position-based surface tension flow (PBSTF) uses the Y axis as
the vertical axis in these three cases, so gravity is `(0.0, -9.8, 0.0)`. Positions are in world space unless a field
is explicitly described as collider-local. Quaternions use W-X-Y-Z order.

## Running the cases

Run from the repository root:

```shell
python examples/teapot/pbstf_surface_tension.py --case teapot --vis
python examples/teapot/pbstf_surface_tension.py --case sweep --vis
python examples/teapot/pbstf_surface_tension.py --case mop --vis
```

The command-line options are:

| Option | Meaning |
| --- | --- |
| `--case` | Selects a case. This document covers `teapot`, `sweep`, and `mop`. |
| `--scale` | Sets particle radius to `1 / scale` and particle diameter to `2 / scale`. |
| `--dt` | Overrides the simulation step duration in seconds. |
| `--steps` | Overrides the number of simulation steps. |
| `-v`, `--vis` | Opens the viewer. |
| `--record` | Opens the viewer, records it, and prompts for an output path when the run ends. |

Higher `scale` gives smaller particles and more spatial detail, with particle count, memory, and runtime generally
growing approximately cubically. The `teapot` case requires `scale >= 20` so its initial liquid sampling remains dense
enough. The motion schedules use simulated seconds, so changing `dt` changes how many steps each motion phase takes.
Changing `steps` changes the total simulated duration without changing the schedule itself.

`--scale`, `--dt`, and `--steps` are the only command-line parameter overrides. Geometry, material, trajectory, and
absorption values are configured in `case_settings()` and `_case_liquid_material()`. Teapot geometry and manipulator
values are configured in [`fluid_helper.py`](fluid_helper.py).

## Shared solver defaults

`build_scene()` converts `scale` to `particle_size = 2 / scale` and uses a topology rebuild interval of 10 steps.

| Setting | `teapot` | `sweep` and `mop` |
| --- | ---: | ---: |
| `scale` | `20` | `20` |
| `particle_size` | `0.1` | `0.1` |
| `dt` | `0.01` | `0.01` |
| gravity | `(0.0, -9.8, 0.0)` | `(0.0, -9.8, 0.0)` |
| lower bound | `(-20.0, -6.04186, -20.0)` | `(-6.0, -1.0, -4.0)` |
| upper bound | `(20.0, 15.0, 20.0)` | `(6.0, 4.0, 4.0)` |
| solver iterations | `5` | `10` |
| surface-neighbor capacity | `128` | `128` |
| local-mesh-neighbor capacity | `64` | `64` |
| principal component analysis normals | disabled | disabled |
| default steps | `10000` | `1000` |
| default simulated duration | `100 s` | `10 s` |

The liquid material parameters are:

| Setting | `teapot` | `sweep` and `mop` |
| --- | ---: | ---: |
| sampler | `regular` | `regular` |
| rest density | `1000.0` | `1000.0` |
| density compliance | `150.0` | `150.0` |
| surface-tension compliance | `3.0` | `1.0` |
| surface-distance compliance | `40.0` | `40.0` |
| interior-distance compliance | `180.0` | `180.0` |
| surface viscosity | `0.2` | `0.5` |
| interior viscosity | `0.05` | `0.5` |
| collider adhesion and friction | enabled | enabled |
| collider-adhesion compliance | `20.0` | `20.0` |
| collider friction | `0.01` | `0.5` |

Compliance values trade enforcement strength for softness: lower values enforce the corresponding condition more
strongly. Higher viscosity damps relative particle motion more strongly. Collider friction only affects unabsorbed
particles because absorbed particles follow the absorbent collider directly.

## Teapot case

The `teapot` case fills a transformed Utah teapot mesh with liquid, represents the teapot wall with a moving mesh
static collider, and updates a KUKA arm and Shadow Hand to follow the authored grasp pose.

### Teapot geometry and liquid seed

These values come from `create_teapot_settings()` in `fluid_helper.py`:

| Field | Default | Effect |
| --- | --- | --- |
| `asset` | `meshes/utah_teapot_modified.obj` | Visual mesh, cavity sampling mesh, and collider source. |
| `mesh_scale` | `2.25` | Uniformly scales the teapot and its local grasp position. |
| `offset` | `(0.0, -3.79, 0.0)` | Initial world position. |
| `quat` | `(sqrt(0.5), 0.0, -sqrt(0.5), 0.0)` | Initial world orientation. |
| collider `sdf_res` | `150` | Signed distance field resolution for the teapot wall. |
| `particles_seed` | `(0.0, -3.15, 0.0)` | Seed point used to find the interior liquid cavity. |
| `particles_max_height` | `0.7` | Highest world-Y level filled with liquid. |
| `particles_vel` | `(0.0, 0.0, 0.0)` | Initial liquid velocity. |

The cavity sampler keeps half a particle diameter of clearance from the wall. Raising `particles_max_height` adds
liquid if the seed remains connected to the desired cavity. Moving `particles_seed` outside that connected cavity can
select the wrong region or produce no useful sample.

The mesh collider is collider index 0. `update_teapot_case()` moves this collider and the transparent visual mesh to
the pose from `teapot_pose()`. The arm pose is recomputed with inverse kinematics from the same teapot-local grasp.

### Teapot motion schedule

The teapot rotates around the world-X line through `turning_axis_pos`. That point is derived from the local grasp
position, `mesh_scale`, `offset`, and `quat`.

| Simulated time | Motion |
| --- | --- |
| `0.0` to `18.0 s` | Rotate at `1.5 deg/s` to `27 deg`. |
| `18.0` to `28.0 s` | Hold at `27 deg`. |
| `28.0` to `29.6 s` | Rotate back at `5 deg/s` to `19 deg`. |
| after `29.6 s` | Hold at `19 deg`. |

Edit `turning_rate_initial`, `stop_angle_initial`, `hold_end`, `turning_rate_final`, and `stop_angle_final` in
`teapot_pose()` to change this schedule. Keep the returned linear and angular velocities consistent with the pose so
the moving collider supplies the intended wall velocity to the fluid.

### Manipulator parameters

`TeapotManipulatorSettings` in `fluid_helper.py` groups the following controls:

- KUKA asset, scale, base pose, end-effector link, and initial joint positions.
- Shadow Hand asset, scale, mount transform, and initial joint positions.
- Teapot-local `grasp_pos` and `grasp_quat`.
- End-effector `tool_center_point` used by inverse kinematics.
- Viewer `camera_pos` and `camera_lookat`.

Use [`pbstf_teapot_grasp_editor.py`](pbstf_teapot_grasp_editor.py) when authoring or checking the grasp transform. The
arm and hand have collision disabled in this demo; the PBSTF mesh static collider is the fluid boundary.

## Sweep and mop cases

`sweep` and `mop` use the same table, liquid region, rest dimensions, and wiping trajectory:

- `sweep` uses an analytic `PBSTFBoxStaticColliderOptions` box that pushes the water.
- `mop` renders a soft finite element method (FEM) sponge and binds its absorbent PBSTF collider to the deformed FEM
  surface. The liquid sees the sponge as a one-way boundary and contributes no force to its FEM dynamics.

Both cases use these `WipeSettings` values:

| Field | Default | Meaning |
| --- | --- | --- |
| `collider_idx` | `1` | Moving sweep-box or sponge index; the table is collider index 0. |
| `collider_entity_name` | `"sponge"` / `"sweep_collider"` | Ordinary visual entity corresponding to the moving collider. |
| `collider_lower` | `(-0.6, 0.02, -1.2)` | Collider rest-space lower corner. |
| `collider_upper` | `(0.6, 0.85, 1.2)` | Collider rest-space upper corner. |
| `table_entity_name` | `"wipe_table"` | Rendered rigid table entity, also used by the FEM sponge collision projection. |
| `table_pos` | `(0.0, -0.25, 0.0)` | Table world position. |
| `table_size` | `(12.0, 0.5, 8.0)` | Table dimensions. Its top surface is at world Y = 0. |
| `liquid_lower` | `(-2.5, 0.05, -0.7)` | Initial liquid lower corner. |
| `liquid_upper` | `(-0.5, 0.35, 0.7)` | Initial liquid upper corner. |
| `start_pos` | `(-3.5, 0.0, 0.0)` | Moving box position before the stroke. |
| `end_pos` | `(3.5, 0.0, 0.0)` | Moving box position after the stroke. |
| `quat` | `(1.0, 0.0, 0.0, 0.0)` | Moving box and table orientation. |
| `settle_time` | `1.0 s` | Time allowed for the initial liquid to settle. |
| `wipe_time` | `5.0 s` | Duration of the linear wiping stroke. |

`wipe_pose()` keeps the active collider at `start_pos` through `settle_time`, linearly interpolates to `end_pos`
during `wipe_time`, and keeps it at `end_pos` afterward. With the default `dt`, settling lasts 100 steps and the
stroke ends at step 600. `update_wipe_case()` moves the analytic sweep box and its ordinary visual entity together.
`update_mop_case()` coordinates the Franka, sponge simulation, deformable collider geometry, and the same trajectory.

### Mop gripper and sponge phases

The mop uses `urdf/panda_bullet/panda.urdf` at scale `15`. The seven arm joints come directly from inverse kinematics
(IK), and the two finger joints are authored positions. The tool center targets the sponge top center with the finger
opening aligned to world X, keeping the gripper above the liquid region.

| Simulated time | Sponge and gripper behavior |
| --- | --- |
| `0.0` to `1.0 s` | Fingers close from `0.60` to `0.40` while gravity and rigid point collisions drive the sponge deformation. |
| `1.0` to `6.0 s` | IK moves the hand along the wiping stroke while the sponge remains fully simulated. |
| after `6.0 s` | The hand holds the end pose while the sponge continues responding to elasticity, gravity, and contact. |

The FEM sponge uses a regular tetrahedral grid and a linear-corotated elastic material with `E=1e4`, `nu=0.4`, and
`rho=30`. Earth gravity remains `(0.0, -9.8, 0.0)`; the low foam density gives the sponge a light physical weight.
No sponge vertex is attached to a rigid link. Every vertex is instead projected outside every coupled collider geom.
The projection removes only inward normal motion, so the contacts are frictionless and transfer no reaction force to
the kinematically driven robot or fixed table.

The Panda loads collision geometry from every authored link. The rendered fixed rigid table also serves FEM collision
queries. Both use one-way rigid-to-FEM coupling. They are separate from the two PBSTF static colliders, so the liquid
still sees only the analytic table and the absorbent sponge surface.

The sponge is authored from the refined tetrahedral boundary, so its rendered triangles are the same triangles as the
FEM volume boundary. `update_mop_case()` synchronizes those boundary vertices and the embedded absorption voxels on
every step. Runtime fluid queries use the current triangles directly, which keeps the collider current while the FEM
shape continues changing. `collider_lower` and `collider_upper` define the rest-space material voxel lattice; the
deformed tetrahedral boundary defines the contact and absorption surface.

## Configuring the absorbent box

The default mop collider is configured in `case_settings()` as follows:

```python
wipe_collider = gs.options.PBSTFAbsorbentBoxStaticColliderOptions(
    pos=wipe.start_pos,
    quat=wipe.quat,
    lower=wipe.collider_lower,
    upper=wipe.collider_upper,
    absorption_rate=2000.0,
    absorption_capacity_fraction=1.0,
    fem_entity_name="sponge",
)
```

`lower` and `upper` define the rest material grid in collider-local coordinates. `pos` and `quat` place that frame in
world space. `fem_entity_name` binds the collision surface and material targets to the named volumetric FEM entity.
Leaving `sdf_res` unset keeps exact triangle queries active for the continuously deforming shape. The two absorption
fields control different parts of the behavior.

### `absorption_rate`

`absorption_rate` controls both admission throughput and inward motion. Each absorbent collider in each environment
earns `absorption_rate * dt` capture credit per simulation substep, and each new particle binding consumes one credit.
With persistent contact and free capacity, the sustained upper bound is therefore approximately `absorption_rate`
newly captured particles per simulated second. Idle credit is bounded to about one additional particle so a dry box
cannot accumulate a large burst before it reaches water.

After admission, a target at Manhattan voxel distance `d` uses:

```text
beta = 1 - exp(-absorption_rate * dt / (d + 1))
progress_new = progress + beta * (1 - progress)
local_pos_new = local_pos + beta * (target_local_pos - local_pos)
```

The nearest-voxel time constant is `1 / absorption_rate`; a target at distance `d` has time constant
`(d + 1) / absorption_rate`. At the default `2000.0 s^-1`, persistent contacts admit about 2000 particles per
simulated second. The corresponding motion time constants are `0.0005 s` at distance zero, `0.001 s` at distance one,
and `0.002 s` at distance three.

- Raise the rate for more captures per second and a faster, more abrupt inward trajectory.
- Lower the rate for fewer captures per second and a slower, smoother inward trajectory.
- The value must be greater than zero.

### `absorption_capacity_fraction`

`absorption_capacity_fraction` is the fraction of the box volume available for liquid at rest. It is greater than zero
and at most one. Total capacity is converted to an integer particle count:

```text
particle_rest_volume = particle_mass / liquid_rest_density
total_capacity = floor(
    absorption_capacity_fraction * box_volume / particle_rest_volume
)
```

The default box has volume `1.2 * 0.83 * 2.4 = 2.3904`. Its fraction of `1.0` makes that full volume available before
conversion to particle slots. The exact particle count depends on PBSTF mass calibration and particle resolution.

- Raise the fraction to absorb more liquid and delay saturation.
- Lower the fraction to saturate sooner and make the box push additional water sooner.
- Use a normal `PBSTFBoxStaticColliderOptions` when no absorption is desired, as the `sweep` case does.

### Voxel layout and saturation

The box is divided automatically using the PBSTF support radius:

```text
grid_res = ceil((upper - lower) / support_radius)
```

At the default `scale=20`, `particle_size=0.1` and `support_radius=0.3`. The default mop box therefore uses a
`(4, 3, 8)` grid. Integer slot counts are distributed uniformly over these 96 voxels while preserving the exact total
capacity.

Each rest-grid voxel center is embedded in one FEM tetrahedron with barycentric coordinates. Synchronization evaluates
those coordinates from the current tetrahedron vertices, so targets follow the grip deformation. Capacity remains
derived from the rest box volume and retains the same per-voxel and total slot counts throughout deformation.

When a particle contacts the sponge, its collider-local position selects the nearest deformed voxel center. The
solver then searches the unchanged six-neighbor material graph in breadth-first order: the contact voxel at graph
distance zero, face-adjacent neighbors at distance one, and successive layers. Within one graph-distance layer,
deformed physical center distance and stable voxel index determine the order. The rest-grid six-neighbor graph remains
fixed throughout deformation. The first candidate with an available slot captures the particle.

Captured particles remain active and visible. They follow sponge translation and rotation while converging to their
current embedded voxel targets, and they leave density, surface, distance, viscosity, adhesion, and friction
processing. Each capture remains assigned to its reserved material voxel. Explicitly setting a captured particle's
position, velocity, or active state releases its absorption binding. Scene state save and restore includes all
bindings, capture credit, voxel distances, progress, local targets, dynamic collider poses, deformed surfaces,
embedded voxel positions, search orders, and SDF activation; the SDF is restored from the saved surface geometry and
wetness is rebuilt from the restored binding progress.

Changing `scale` also changes `particle_size`, support radius, voxel count, graph distances, calibrated particle
volume, and therefore the integer capacity distribution. Since the admission limit counts particles, its volumetric
rate is `absorption_rate * particle_rest_volume`; retune `absorption_rate` inversely with particle rest volume to keep
the same liquid volume per second after a resolution change. The rate keeps the same continuous-time meaning when `dt`
changes, although a smaller `dt` resolves contacts and motion more finely.

### Reading wetness

Wetness is available from the solver after the scene is built:

```python
from examples.teapot.pbstf_surface_tension import CASE_MOP, build_scene, case_settings, get_wipe_settings

settings = case_settings(CASE_MOP)
wipe = get_wipe_settings(settings)
scene, _ = build_scene(case=CASE_MOP)

wetness = scene.pbstf_solver.get_static_collider_wetness(
    collider_idx=wipe.collider_idx,
)
```

Every value lies in `[0, 1]` and equals the summed capture progress divided by that voxel's slot capacity, clamped to
the valid range. Axes follow local X, Y, and Z from `lower` to `upper`.

- A single-environment scene returns `[nx, ny, nz]`.
- A batched scene returns `[B, nx, ny, nz]`.
- Pass `envs_idx` to request selected batched environments.
- Calling the getter for the table or a normal sweep box raises an error because those colliders have no wetness.

For live inspection, call `get_static_collider_wetness()` after `scene.step()`. The returned tensor is a fresh value and
can be used for visualization or logging without modifying the solver's internal wetness field.
