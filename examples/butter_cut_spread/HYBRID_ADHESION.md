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

The explicit deposition fallback selects original particle IDs after sustained
near-face contact and outward relative separation above 0.01 m/s. Static contact
alone does not deposit material. Each face has a fixed capacity of 512 particles,
with at most one particle per square cell of width two particle spacings. Both faces use the
same contact duration, outline clearance, capacity and release conditions.
Capture does not require the neighboring bulk column to have become thin.

Deposited film keeps its original particle ID and represented volume, but is
excluded from material point method (MPM) transfers while attached. Its local
anchor follows the actual blade pose. During downward insertion into a thick
column, the tangential component of downward motion slips past the coating.
This permits a continuous wipe without using fixed particles to drag the stock.
No clock time or named demo phase selects a different rule.

Bread contact, loss of the face footprint or excessive separation releases the
particle back into MPM. Released film has relaxed elastic deformation and affine
velocity; a cooldown prevents immediate recapture. This is an explicit dissipative
deposition approximation, not a measured adhesion or energy-conserving model.
Recorded `particle_active` and `residue_faces` must agree exactly: each original
butter particle is either active bulk or attached film. No particle is deleted,
duplicated or painted onto the knife without represented material volume.
The peak film count is measured over saved frames, not every integration step.
The native collision projection remains responsible for steel nonpenetration.
The primitive diagnostic `demo` keeps deposition disabled because it has no
authored-film replay or mesh-anchor projection; the authored cutting and spreading
stages share the complete deposition law.

## Hybrid rendering

`blade_film.py` projects each side's particle volume into the actual blade's local
plane. A column containing thick material remains in the bulk reconstruction.
Thin nearby particles contribute complementary weights to the bulk density and
the film, with a smooth optical partition between 1.5 and 3 local particle layers.
The partition changes rendering weights; it changes no physics or material values.

Both films reuse the earlier standalone cutting demo's pale transparent film
appearance: color (0.96, 0.84, 0.56), roughness 0.38 and optical depth 0.28 per
0.4 mm reference layer. Optical extinction uses physical thickness rather than
the current simulation spacing, so a sampling change does not change the color
of the same amount of coating. Bulk butter retains the approved yellow appearance.
Per-vertex opacity derives from normalized particle-volume splats. A transparent
surface lies 8 micrometers outside each broad steel face, follows the actual blade
transform and respects the authored outline. This is an approximate thin-coating
bidirectional reflectance distribution function (BRDF): metal remains visible
through sparse coverage. Thick remnants retain reconstructed three-dimensional
geometry. The renderer reads prepared film arrays and requires no simulation or
surface-reconstruction dependencies in its Isaac environment.

The sheet uses a 1.5 mm quadratic kernel to connect the coarse deposited samples
into a thin wipe. The 0.9 mm kernel for detecting thick columns is unchanged.
This optical smoothing preserves the sheet volume and complementary bulk weights;
it creates neither additional butter nor a fixed rectangular coverage mask.

`materials/blade_film.mdl` applies particle coverage to the entire coating BSDF,
including its specular component. Uncovered steel therefore receives no extra
reflection or refraction. The coating has a small fixed specular contribution
and diffuse tint, with stochastic geometric coverage exposing the original
steel underneath. This deliberately avoids `UsdPreviewSurface` opacity, which
can retain a reflective/refractive interface even at zero opacity in Isaac RTX.
Triangles with no coverage are omitted, and each uploaded triangle receives the
same outward blade-face normal. No rectangular transparent card follows the tool.
See [the artifact investigation](FILM_ARTIFACT_VALIDATION.md) for the zero-coverage
ablation and alignment measurements.

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
