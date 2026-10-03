# Minimal particle-fluid integration baseline

## Goal

This phase keeps the existing Genesis public API stable while giving an external adapter a small, explicit contract
for the fork's particle-fluid solvers. Existing calls to `Scene`, `scene.step()`, `scene.reset()`, solver properties and
entity state methods remain valid.

## Configuration boundaries

The integration uses three distinct parameter groups:

1. **Solver options** configure a global numerical algorithm. Examples are particle size, iteration counts, hash-grid
   bounds, surface-topology capacity and static colliders. They remain in `IPBFOptions`, `IPBSTFOptions` and
   `PBSTFOptions`.
2. **Material options** configure one simulated entity. Examples are density, compliance, surface tension,
   concentration and viscosity coefficients. They remain in `gs.materials.<solver>.Liquid`.
3. **Asset and task options** describe a concrete example: mesh paths, URDF paths, poses, dimensions, controller
   timing and camera settings. They belong to the example and never enter the solver integration contract.

PBSTF allocates collider data while building the solver, so its `static_colliders` field receives object-bound collider
descriptions. The descriptions still originate from the example's asset/boundary configuration. This is a construction
dependency; mesh paths, object poses, absorption behavior and other task choices are not reusable solver parameters.

`ParticleFluidProperties` contains only the common values with equivalent meanings across the covered solvers:
density and particle size. `create_particle_fluid_setup` maps them to a typed solver-options/material pair. All other
parameters stay explicit in their native Genesis type. In particular, the helper does not translate a generic
viscosity value because PBSTF viscosity filters and IPBF XSPH coefficients are different numerical quantities.

Particle size is a solver option expressed as a diameter in meters. Examples and adapters pass this quantity directly.
Object mesh scale remains an entity/asset property and has no shared configuration field with particle size.

## Adapter flow

An adapter selects a backend once and keeps the existing Scene construction path:

```python
from genesis.integrations import ParticleFluidProperties, ParticleFluidSolver, create_particle_fluid_setup

fluid = create_particle_fluid_setup(
    ParticleFluidProperties(density=1000.0, particle_size=0.008),
    ParticleFluidSolver.PBSTF,
)
scene = gs.Scene(pbstf_options=fluid.solver_options)
liquid = scene.add_entity(morph=particle_morph, material=fluid.material)
```

Advanced tasks pass an explicit `PBSTFOptions` and `PBSTF.Liquid` to the same helper. The helper validates that their
common density and particle size agree with the generic specification.

Rigid links used as one-way fluid boundaries are synchronized through `StaticColliderLinkSynchronizer`. The adapter
registers it once after `scene.build()`, after which ordinary `scene.step()` calls keep the boundary poses current:

```python
sync = StaticColliderLinkSynchronizer(scene, scene.pbstf_solver, links_idx=(cup_link_idx,), colliders_idx=(0,))
scene.register_pre_step_callback(sync)
```

## UniRoboSim-Genesis adapter requirements

An adapter that already supports an official Genesis particle-fluid solver reuses its provider/session/world lifecycle,
WorldSpec compilation, asset resolution, entity-path mapping, units, quaternion conversion and particle state transport.
Supporting this fork's PBSTF, IPBSTF or IPBF implementation adds the following backend work:

1. Pin and probe an explicit fork engine profile. Capability discovery must distinguish the available solver and
   boundary features from an official Genesis installation.
2. Map the portable density and particle diameter to `ParticleFluidProperties`, select `ParticleFluidSolver`, and keep
   algorithm-specific solver and material values in typed Genesis backend options.
3. Compile a fluid-boundary binding from one resolved entity description. The rigid entity, fluid collider and cavity
   sampler must share the asset revision, pose and scale required by their respective representations.
4. Maintain internal entity-path-to-link and entity-path-to-collider mappings. Collider indices are build products and
   never form part of the portable task interface.
5. Register `StaticColliderLinkSynchronizer` after `scene.build()` for every moving rigid boundary. Static boundaries
   require the collider description but no runtime pose callback.
6. Include the selected solver's particle, boundary and shared analytic-boundary fields in state, checkpoint and full
   or partial environment reset behavior.
7. Rebuild the scene when boundary geometry, scale, particle size, solver type or build-time interaction parameters
   change. Runtime rigid pose changes flow through the registered synchronizer.

Cup meshes, cavity meshes, fill fractions, concentration labels, robot motion and absorbent tools remain task or asset
data. A portable fluid-boundary relation belongs in the shared specification when multiple backends implement it;
until then it can be a typed Genesis backend extension.

## State and reset contract

PBSTF snapshots include particles, concentrations, static-collider poses, deformable-collider state and absorption
state. IPBSTF snapshots include particles and static-collider poses, and restore only the selected environments.
IPBF snapshots include particles, concentrations, boundary membership and the optional movable-pitcher pose.

The IPBF pitcher is one global analytic boundary. A full reset restores its pose. A partial parallel-environment reset
restores the selected particles and keeps the current shared pitcher pose because one global pose cannot represent a
different value per environment. An adapter that needs independent pitcher poses must use one scene per environment
until the boundary storage becomes batched.

## Construction phases

1. Preserve current public Scene, material, entity and solver access paths.
2. Add typed common-property mapping and explicit solver-specific escape hatches.
3. Add a reusable rigid-link/static-collider pre-step synchronizer.
4. Complete IPBSTF and IPBF snapshot state needed by reset and checkpoint users.
5. Split one multi-physics example so solver, material and asset/task parameters are visibly separate.
6. Validate configuration mapping, synchronization, state restoration and the existing particle-fluid tests.

Future solver registration can replace hard-coded Simulator construction internally. It is intentionally outside this
phase because the adapter contract above does not depend on how Simulator stores or constructs its solvers.

## Completion status

The six construction phases above are implemented in this baseline. The adapter-facing helpers live in
`genesis/integrations/particle_fluid.py`. The coffee-water example keeps its entry point while its solver/material
configuration and asset/boundary configuration live under `examples/multiphysics/coffee_water`.

The acceptance suite consists of the common-property mapping tests, the IPBF pitcher snapshot test, the IPBSTF
partial-environment reset test and the existing particle-fluid tests. IPBSTF execution requires a CUDA test host.
