# Symmetric blade adhesion and particle-driven films

The authored pipeline uses the same contact rules on the two local blade faces.
Butter uses one fixed constitutive preset throughout the trajectory. Tool poses,
particle spacing, the physics grid and integration step retain the standalone
firm-smooth settings.

## Contact and residue

`contact.py:AdhesionParameters` defines finite cohesive traction, static anchoring,
sliding and rebonding. Contact state follows separation, relative tangential motion
and contact history. Simulation time and the sign of the blade face select no
adhesion coefficients.

The actual deforming bread surface supplies a competing-contact event. A particle or the downward material column supporting its blade interface
that reaches this surface uses the bounded transferred-contact law, permitting
sliding and deposition. Its transfer state resets after separation from both
interfaces. This is an interface history rule; the butter's elastic moduli, yield
stress, consistency and flow exponent stay fixed. The same event applies to both
blade faces, including material that flows around an edge.

The static tangential traction has fixed blade-frame directional limits: the
local X direction supports pickup, while the local Y draw direction remains
slippery. Both faces share these limits. The dish retains its native contact and
friction settings.

A calibrated bulk-traction limiter reduces the normal pull on thick columns.
A conservative bilinear volume scatter and a neighboring-column maximum measure the local
column on each side in the rotating blade frame. This geometry-dependent limiter
keeps a thick stock block from being pulled or rotated as strongly as a thin
slice. It is a demo-preserving contact approximation, evaluated identically on
both faces, rather than a measured material law. Its depth thresholds and force
limits remain fixed throughout the run.

The explicit kinematic residue fallback selects existing particle IDs, capped at
128 particles per face (55.3 cubic millimeters total at the default spacing).
Capture requires sustained near-face contact, bounded slip speed and a thin local
material column, including neighboring columns. The local thickness grid is
refreshed every ten integration steps. A thick slice or stock surface therefore
cannot become a set of kinematic handles. Captured particles remain active in
the material point method and retain their identity and mass. After integration,
their local anchors follow the measured rigid blade pose. Bread contact, excessive
separation, loss of the face footprint or incoming thick material releases them.
A cooldown prevents immediate recapture.

This fallback is an explicit external position/velocity constraint. Its forces
and work are not an independently measured physical adhesion model. Recorded
`residue_faces` and the peak constrained count over saved frames make its
contribution auditable; the recorded peak does not sample every integration step.
The native collision projection remains responsible for steel nonpenetration.

## Hybrid rendering

`blade_film.py` projects each side's particle volume into the actual blade's local
plane. A column containing thick material remains in the bulk reconstruction.
Thin nearby particles contribute complementary weights to the bulk density and
the film, with a smooth optical partition between 1.5 and 3 local particle layers.
The partition changes rendering weights; it changes no physics or material values.

Both films reuse the earlier standalone cutting demo's pale transparent film
appearance: color (0.96, 0.84, 0.56), roughness 0.38 and optical depth 0.28 per
particle layer. Bulk butter retains the approved yellow appearance.
Per-vertex opacity derives from normalized particle-volume splats. A transparent
surface lies 8 micrometers outside each broad steel face, follows the actual blade
transform and respects the authored outline. This is an approximate thin-coating
bidirectional reflectance distribution function (BRDF): metal remains visible
through sparse coverage. Thick remnants retain reconstructed three-dimensional
geometry. The renderer reads prepared film arrays and requires no simulation or
surface-reconstruction dependencies in its Isaac environment.

The bread coating retains its normalized anisotropic reconstruction and bounded
sampling-gap repair. Released blade particles continue into that reconstruction
through their original identities. No permanent painted residue is left behind
when particles leave the blade.

## Inspection and comparison

The normal standalone `asset_pipeline` command prepares both films automatically.
`asset_render --frame-indices ... --camera-eye X Y Z --camera-target X Y Z
--focal-length MM --output-name NAME` produces separate close-up artifacts.
Keep the inspection camera outside the bread, board and counter meshes.

`residue_audit --run RUN` reports both faces' thin-particle counts, volume and
coverage at saved checkpoints. `adhesion_compare --run RUN --reference BASELINE`
checks complete-run pickup, deposition, spread extent, slice deformation and
stock displacement against explicit limits. It also checks that the fixed butter
preset and prescribed trajectory match. A successful comparison still requires
inspection of the complete animation and both blade faces.
