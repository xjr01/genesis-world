# Deposited blade film: complete cutting and spreading comparison

The reference is the symmetric force-plus-residue trajectory used by commit
`8fcefc5c` (state SHA-256
`5e5a426977e3e175e61d2d67761969e4a05fa02e4836d7c541e02f6e94073b0e`).
The selected deposited-film trajectory is `coherent-film-b`, SHA-256
`4f9d4705d7130631d9ba738e79e1b87f008ab67412d4a9c8bf092252763e947a`.
Both complete 168,000 steps, 8.4 seconds and 203 recorded frames, with 549,158
butter particles and 272,258 bread particles. Material parameters and prescribed
knife poses match exactly. The fixed butter preset has not been softened by phase.

## Why the previous residue looked sparse

The earlier active-particle fallback only captured already-thin neighborhoods.
Increasing its count would make fixed particles act as handles on the bulk.
The standalone cutting reference instead removed captured IDs from bulk transfers
and displayed them as a deposited film. The authored pipeline now uses that
explicit deposition approximation on both faces, with spatial capacity, separation
and bread-contact release. It does not restore the old one-sided contact rule.

Immediate deposition during compression was rejected: trial A increased stock
motion and slice deformation, and failed the complete comparison. The selected
rule requires established contact and outward relative separation. It preserves
the fixed force law while leaving a bounded amount of coating behind.

## Macroscopic comparison

| Metric | Current symmetric reference | Deposited film |
| --- | ---: | ---: |
| Slice pickup | 99.9632% | 99.9609% |
| Final deposition | 83.9407% | 83.4490% |
| Spread span | 78.726 mm | 78.986 mm |
| Cut slice deformation RMS | 1.082 mm | 1.085 mm |
| Lift slice deformation RMS | 1.554 mm | 1.606 mm |
| Stock displacement median | 1.074 mm | 1.424 mm |
| Stock displacement P99 | 2.625 mm | 3.455 mm |
| Bread maximum displacement | 20.186 mm | 17.347 mm |

All existing `adhesion_compare` limits pass against the current symmetric
reference. Bread same-ID positional difference P99 is 0.916 mm; its centroid
difference is 0.132 mm. No sampled interior blade-face crossings were observed.
These are saved-frame measurements, not a continuous-time guarantee.

The older pre-symmetric `standalone-firm-smooth` trajectory is also retained as a
historical comparison. The cumulative change from that older version fails its
stock-median, deposition-centroid and bread-P99 difference limits. This update
is accepted against the user's current symmetric version, not claimed identical
to every earlier calibration.

## Film accounting and appearance

At the saved lift frame, 78 original IDs are deposited on the negative face and
162 on the positive face. The saved peak over the full run is 700, within the
512-per-face cap. All IDs are accounted for as active bulk or deposited film;
1,134 distinct IDs are observed returning to active bulk between saved frames.
The maximum recorded normal-anchor error is 0.0000179 mm, and no deposited point
is outside the actual blade-face outline.

At the 3.7-second checkpoint, positive-face thin volume rises from 8.110 to
45.865 cubic millimeters. With the new normalized 1.5 mm display kernel, the
region above 2% opacity covers 402.16 square millimeters. The 0.9 mm thickness
classifier is unchanged. Smoothing preserves represented volume and the
complementary bulk weights. Tint, roughness and optical depth retain the pale
standalone-cutting settings, using its 0.4 mm physical reference layer. This
corrects the previous dependence of extinction on simulation particle spacing.
The zero-coverage-safe native shader is retained.

The physical model is deliberately approximate: deposited particles temporarily
leave MPM and bread traction, follow the knife kinematically, and re-enter with
relaxed elastic/affine state. This is dissipative and is not an energy-conserving
adhesion solver. No new particle IDs or permanent painted patches are introduced.

Physics stepping took 685.7 seconds on this machine, compared with 562.8 seconds
for the preceding reference. Additional deposition bookkeeping has a runtime
cost; these wall times are not an isolated benchmark, as CPU validation also ran
during the candidate simulation.

## Verification

All 21 cases in the cutting/contact, spreading, blade-film and sampling-gap test
files pass. The two bread-contact cases also pass on the CPU in double precision.
The film tests were repeated after optical calibration; identical physical sheet
thickness now produces identical opacity at different particle spacings.

The independent source export prepares the bundled assets and completes a fresh
200-step smoke run without importing another checkout. Existing native SDF caches
are reused; this is not an empty-cache installation timing claim.

All 203 reconstructed frames were rendered from the independent source export
at 128 verified path-tracing samples. Final state, reconstruction and replay
hashes match. The film partition's maximum relative volume error is 2.22e-16,
and its sheet-volume error is 2.65e-23 cubic meters. The display/collision blade
alignment error remains 3.66e-8 meters. Surface preparation including the optical
refresh took 355.1 seconds; complete replay took 583.4 seconds.

The delivery includes a complete 203-frame before/after video, a 59-frame moving
blade detail, matched-camera back-face images after cutting, and both-face
inspection images. Isaac emitted one topology/normal fallback warning on the
planar coating mesh; this is not a warning-free renderer claim. The inspected
cutting, spreading and withdrawal frames show coverage within the knife outline
without the earlier rectangular card silhouette.

The inactive-film bread-contact regression was also run with its exclusion guard
removed: it failed because an inactive particle acquired a 0.0463 m/s tangential
velocity change. Restoring the guard passes both unbatched and batched CPU checks.
The distance calculation now uses the exact norm where division is already
guarded; this removes a small regularization bias exposed by the double-precision
analytical blade-contact check. The authored demo has this separate blade force
disabled, so that norm correction does not change its recorded physical trajectory.
