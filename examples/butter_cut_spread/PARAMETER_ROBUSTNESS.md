# Fixed butter parameters across prescribed knife trajectories

The firm-smooth asset preset uses one material and one symmetric contact law
throughout cutting, lifting and spreading. The 2026-10-27 calibration changes only:

| Parameter | Reference | Selected |
| --- | ---: | ---: |
| Material consistency (Pa s^0.5) | 25 | 15 |
| Blade normal traction (Pa) | 2000 | 1500 |
| Thick-column normal traction limit (Pa) | 200 | 60 |

Shear modulus remains 80 kPa, bulk modulus 150 kPa, yield stress 600 Pa,
flow exponent 0.5 and density 900 kg/m3. Static/sliding/cross-direction shear
traction remains 1500/50/50 Pa. Film capture, release, pale optical appearance,
the two face rules, bread contact and trajectory generation are unchanged.
The material file and `contact.py:AdhesionParameters` are both required parts
of this calibration. Changing only the material file does not reproduce it.

The lower consistency reduces the rate-dependent resistance after yield without
lowering the elastic stiffness or the yield threshold. The lower normal limits
reduce pulling on attached material, especially thick stock columns. These are
empirical demo parameters, not measurements of a particular butter product.

## Validation scope

All runs use 549158 butter particles at 0.6 mm, 272258 bread particles at 0.8 mm,
a 50 microsecond step and grid density 768. There is no reduction in spatial
resolution or change to the solver in this calibration. The reference is commit
`51c83edc8411d0629294b86e4227a93e4a4e7f52`, run `coherent-film-b`.

Three trajectory conditions were evaluated:

1. The unchanged original 8.4-second cutting/spreading sequence, 203 saved frames.
2. A prescribed replay derived from the supplied robot joint commands and a fixed
   grasp transform, mapped into the reference scene, through lift and hold at 3.2 s.
3. The original cut followed by a 0.4-second lift instead of 0.8 seconds, with no
   outward peel, then a hold through 3.7 s. This is a sensitivity experiment only.

The robot's production trajectory was not edited. These experiments do not include
robot dynamics, measured grasp slip or actual tracking error. The remote run's
particle and measured-pose recordings were unavailable. The robot command replay
retains about 2.15 mm clearance above the initially configured butter bottom;
parameter tuning cannot establish that this trajectory cuts through the stock.

## Original complete sequence

| Measurement | Reference | Selected |
| --- | ---: | ---: |
| Slice pickup | 99.961% | 99.975% |
| Slice deposition | 83.449% | 83.217% |
| Deposited span | 78.986 mm | 80.359 mm |
| Cut slice rigid-fit RMS residual | 1.085 mm | 0.989 mm |
| Lift slice rigid-fit RMS residual | 1.606 mm | 1.373 mm |
| Stock displacement P99 | 3.455 mm | 1.975 mm |
| Bread same-particle position difference P99 | -- | 0.428 mm |

All existing macroscopic gates in `adhesion_compare` pass, including pickup,
deposition, span, deformation, stock motion, bread motion and identical prescribed
poses. `aligned_audit` also passes: finite particles, complete identity accounting,
bounded film capacity, no film outside the authored face and no sampled interior
side changes. The old comparator's overall result is intentionally **false**:
its additional same-material equality test detects consistency 25 versus 15.
That test and all effect thresholds were left intact. The parameter-calibration
report separately requires that consistency is the sole material change.

## Lift sensitivity and remaining limitations

The following stock selection includes particles immediately next to the cut plane,
unlike the original full-demo stock audit, which excludes a two-spacing margin.
Slice identities are selected using the blade plane at the cut checkpoint. RMS
fits use all initially selected slice IDs, including material left behind.

| Measurement | Robot reference | Robot selected | Fast reference | Fast selected |
| --- | ---: | ---: | ---: | ---: |
| Slice raised over 20 mm | 94.485% | 95.960% | 99.940% | 99.970% |
| All-slice rigid-fit RMS | 15.362 mm | 13.402 mm | 1.718 mm | 1.383 mm |
| Raised-subset rigid-fit RMS | 4.168 mm | 3.874 mm | 1.086 mm | 0.956 mm |
| Stock displacement P99 | 1.355 mm | 1.122 mm | 2.938 mm | 1.871 mm |
| Peak stock particles raised over 5 mm | 1758 | 1749 | 685 | 747 |

The large all-slice robot residual includes the separation between the lifted
slice and material left near the stock. The raised-subset residual is included
so that this cannot be mistaken for the deformation of only the lifted piece.

The selected parameters improve shape retention and reduce bulk displacement in
both lift tests. They **do not eliminate extra stock pickup**: the robot count is
almost unchanged and the fast-lift count increases by 62 particles, equivalent
to 13.392 mm3 of sampled material. There is no claim of lower unwanted pickup for
every trajectory, or of fidelity for arbitrary speed, cut depth or knife shape.
Actual robot motion still requires its own validation.

Other evaluated candidates were not selected: halving consistency alone reduced
raised-slice deformation but did not improve stock pickup; reducing sliding and
cross shear worsened pickup and lift deformation; reducing static anchoring to
600 Pa reduced stock movement but worsened deformation of the raised slice.

## Running and integrating

Use the normal standalone `asset_pipeline` with its default
`configs/asset_firm_smooth.json` and a fresh output directory. Re-prepare inputs
when source fingerprints change; do not resume a checkpoint made with the old
material/contact parameters. No phase-specific material override is required.

For the supplied AGSim package, the active material is in
`task_library/bread_0694/task.yaml`, while the active adhesion defaults are in
`src/agsim_genesis/tasks/bread_0694/robot_film_contact.py:AdhesionParameters`.
The older `contact.normal_stress` entry in the YAML is not the normal traction
used by the installed `RobotFilmContact` adapter. Its integration receipt records
the actual parameters and profile `fixed-butter-robust-20261027`.

The local calibration artifacts are in `out/robust-original-balanced`,
`out/robust-robot-{base,balanced}` and `out/robust-fast-{base,balanced}`. Each run
records material, effective adhesion, prescribed motion, experiment scope and
particle states. The complete original candidate state SHA256 is
`1f5e20ea749c3b8712cf8e4638c6472d2cca9de83c0266d414ee0d65668b3369`.
The deliverable `parameter-validation.json` contains the individual checks and
measurements; rendered previews cover five states, not a full new Isaac animation.

The selected defaults pass 21 focused GPU tests covering cutting contact, material
integration, spreading, blade-film partition and supported-gap reconstruction.
Five Isaac preview states were reviewed at 128 samples; the final spread remains
visually close to the reference. These checks do not constitute robot-dynamics
validation or a guarantee that all possible spreading holes are eliminated.
