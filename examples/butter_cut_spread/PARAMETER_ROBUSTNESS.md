# Reference hardness with bounded blade adhesion

The selected local profile `fixed-reference-hardness-20261027` restores all six
butter material parameters from `51c83edc8411d0629294b86e4227a93e4a4e7f52`:
G=80000 Pa, K=150000 Pa, yield=600 Pa, consistency=25 Pa s^0.5,
flow exponent=0.5 and density=900 kg/m3. The earlier consistency=15 calibration
is superseded. One fixed material is used throughout cutting and spreading.

Relative to `51c83edc`, only two blade adhesion values differ: normal traction
2000 -> 1500 Pa and thick-column traction 200 -> 60 Pa. Static/sliding/cross shear
remain 1500/50/50 Pa. The contact law, both-face symmetry, film capture/release,
pale film optics, bread interaction, original demo trajectory and solver are unchanged.
There is no phase-dependent or speed-dependent material adjustment.

## Complete original demo

The full 8.4-second, 203-frame run `firm-v6-original` passes every existing
`adhesion_compare` check against `coherent-film-b`, including exact material equality
and identical prescribed poses. Acceptance limits were not changed.

| Measurement | 51c83edc reference | Selected |
| --- | ---: | ---: |
| Slice pickup (%) | 99.961 | 99.972 |
| Slice deposition (%) | 83.449 | 83.210 |
| Deposited span (mm) | 78.986 | 79.464 |
| Cut shape RMS (mm) | 1.085 | 0.991 |
| Lift shape RMS (mm) | 1.606 | 1.405 |
| Stock displacement P99 (mm) | 3.455 | 2.399 |

Bread same-particle position difference P99 is 0.178 mm.
The selected complete state SHA256 is `a7bffa72e40243bd1b7e77d1968babd476f00bc0f1beac1e6d6d36b6c794140b`.

## Paired trajectory sensitivity

All cases use 549158 butter particles at 0.6 mm, 272258 bread particles at 0.8 mm,
a 50 microsecond timestep and grid density 768. Each before/after pair uses the
same prescribed poses and the same reference material. Only the two normal
adhesion limits differ. The stress trajectories are diagnostic inputs; production
demo and robot trajectories were not edited.

| Condition | Stock P99 mm, before / after | Peak stock IDs raised >5 mm, before / after | Raised-slice RMS mm, before / after |
| --- | ---: | ---: | ---: |
| Robot command replay | 1.355 / 1.212 | 1758 / 1737 | 4.168 / 4.056 |
| 2x lift, no outward peel | 2.938 / 2.311 | 685 / 731 | 1.086 / 0.963 |
| 2x cut and lift, no outward peel | 6.428 / 4.892 | 358 / 452 | 1.292 / 1.057 |

Stock selection includes particles immediately next to the cut. It differs from
the full-demo audit, which excludes a margin. Raised-subset RMS uses the lifted
IDs of each run; `firm-v6-validation.json` also reports fits on identical lifted
IDs across each pair to separate shape deformation from pickup differences.

These results bound the tested cases, not all possible knife speeds. Unwanted
stock pickup is not zero and improvements in displacement do not necessarily
mean fewer lifted particles. Thin film on the blade is intentional. Higher yield
and modulus trials were rejected because slice-edge curling worsened despite
some improvements in central shape retention.

The robot result replays command-derived knife poses in the reference scene,
through cutting/lifting/hold at 3.2 seconds. It excludes robot dynamics, measured
tracking and grasp slip. Actual remote-run recordings were unavailable. The
commanded cut leaves about 2.15 mm clearance over the initially configured butter
bottom. Fixed material parameters cannot make an incomplete cut equivalent to
a fully severed slice; robot path correction remains a separate task.

## Rendering and integration

Both selected trajectories have complete Isaac PathTracing videos at 128 samples:
`original-cut-spread-isaac.mp4` (203 frames) and `robot-command-cut-lift-isaac.mp4`
(40 frames). These are continuous recorded-state renders, not five-state previews.
Film shader, optical settings and bulk appearance match the reference. Render
and reconstruction state hashes match the simulations; every decoded frame and
mesh upload was checked. Local deliverables are in `out/firmer-isaac-20261027`.

Use `configs/asset_firm_smooth.json` with the current `contact.py:AdhesionParameters`.
Start a fresh simulation output directory; do not resume old-material checkpoints.
For the AGSim package, material is configured in `task_library/bread_0694/task.yaml`
and active adhesion in `robot_film_contact.py:AdhesionParameters`. The legacy YAML
`contact.normal_stress` is not the installed adapter's traction. The installer
records `fixed-reference-hardness-20261027`; both profiles must stay synchronized.

The default preset combines the exact reference material with the reduced blade
adhesion limits above. The standalone pipeline includes the solver, contact law,
surface reconstruction, film rendering and bundled input assets.
