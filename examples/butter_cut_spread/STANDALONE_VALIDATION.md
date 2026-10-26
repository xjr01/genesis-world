# Standalone integration validation — 2026-10-26

Validated on Windows with Python 3.11, RTX 5090 D v2, the Genesis checkout based
on `1d3b9344`, and a separate Isaac Sim 5.1 environment. This record concerns the
bundled fixed firm-smooth configuration and the current rendering code.

## Repository independence

- Exported only staged repository files into a new directory, without local
  simulation output, ignored assets or another checkout's source files.
- Verified that all `examples.*` and `genesis.*` imports in this example resolve
  within that exported repository.
- Verified the 49-file bundled scene and all relative USD texture dependencies.
  Archive SHA-256: `ac04e8aa16eab45e8dd35e2585104bdf6e538b503edbfa73d72395d3cf88a3a9`.
- Prepared from the bundle with empty, isolated Genesis and Quadrants caches;
  generated all three rigid mesh SDF records locally, then ran 200 physics steps.
  Initial butter and bread particles exactly match the separately prepared full run.
- No original Blender files, machine cache directory, old simulation or separate
  renderer checkout was needed for the exported copy's preparation and smoke run.

## Complete physical run

| Check | Result |
| --- | --- |
| Duration / steps / saved frames | 8.4 s / 168,000 / 203 |
| Butter / bread particles | 549,158 / 272,258 |
| Sampling / time step | 0.6 mm butter, 0.8 mm bread / 50 microseconds |
| Constant material | G=80 kPa, K=150 kPa, yield=600 Pa, consistency=25, n=0.5 |
| Pickup / deposition | 99.9517% / 80.9605% |
| Final spread span | 76.5366 mm |
| Sampled interior blade-side changes | 0 |
| Particle activity / finiteness / aligned audit | Passed |
| Decoded particle video | 203 frames |

Trajectory SHA-256: `a1c2be7b1154715f7e9f439a28869a675b10d261c2aedb3c07b2eb8aae9e8fbb`.
No material interpolation or phase softening is used. The physical state is
newly simulated, not copied from an earlier result.

## Isaac and regression checks

The exported repository reconstructed and rendered indices 0, 50, 101, 152 and
202 of the new trajectory. All five captures report 128 effective path-tracing
samples and zero mesh upload error. The 1280 x 720 preview decodes to five frames;
simulation, reconstruction and rendering hashes match. The final image was
visually inspected. This integration check does **not** regenerate a complete
203-frame Isaac video.

33 focused tests passed: 28 asset, trajectory, surface, gap-repair and mesh-upload
checks, plus five GPU cases covering CPIC tool selection, finite-range contact,
authored tool pose and consolidated tool state updates. Ruff checks, formatting
and staged whitespace checks passed.

The first preparation uses empty caches and can take several minutes. Cold
preparation and GPU tests ran concurrently with parts of the full simulation;
these elapsed times are not used as a performance benchmark. Historical isolated
performance measurements and visual limitations remain in `PERFORMANCE.md`.
