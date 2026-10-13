# Teapot Comparison

Compare PBSTF, PBD, DFSPH, and WCSPH using the same teapot motion and initial particle state.
Run the commands below from the repository root in an environment using this checkout of Genesis and a CUDA GPU.
Ray-traced rendering additionally requires the LuisaRender backend and `pysplashsurf`.

| Method | Script | Method selection |
| --- | --- | --- |
| PBSTF | `examples/teapot/pbstf_teapot.py` | - |
| PBD | `examples/teapot/pbd_liquid_teapot.py` | - |
| DFSPH | `examples/teapot/sph_teapot.py` | `--pressure-solver DFSPH` |
| WCSPH | `examples/teapot/sph_teapot.py` | `--pressure-solver WCSPH` |

`examples/teapot/render_teapot_comparison.py` runs the selected methods and assembles their comparison video.
The workflow is to save simulation checkpoints once, then reuse them while adjusting rendering.

## Simulation

Run a single method with an interactive particle preview:

```powershell
python examples/teapot/pbstf_teapot.py --seconds 50 --vis
python examples/teapot/pbd_liquid_teapot.py --seconds 50 --vis
python examples/teapot/sph_teapot.py --pressure-solver DFSPH --seconds 50 --vis
python examples/teapot/sph_teapot.py --pressure-solver WCSPH --seconds 50 --vis
```

Omit `--vis` to run headlessly. Add `--record -o out/teapot_render/particles` to save a particle video.

To simulate all four methods and save checkpoints without rendering:

```powershell
python examples/teapot/render_teapot_comparison.py --simulate-only --seconds 50 --fps 25 --playback-speed 1 --save-checkpoints out/teapot_checkpoints -o out/teapot_render
```

Each method writes `metadata.json` and `frame_XXXXXXXX.npz` under `out/teapot_checkpoints/METHOD/`.
The frame number is the simulation step. With the default `dt = 0.01`, this command saves every four steps.
Each method's checkpoint directory must be empty when saving; choose a new directory for a new simulation.
If checkpoints already exist, proceed directly to rendering.

To save just one method, select it with `--methods PBSTF`, or use its script directly:

```powershell
python examples/teapot/pbstf_teapot.py --seconds 50 --fps 25 --playback-speed 1 --save-checkpoints out/teapot_checkpoints -o out/teapot_render
```

DFSPH can become unstable and end early with an error. Previously saved frames remain available for rendering.

## Rendering from checkpoints

Replay restores saved particle states and teapot poses without advancing the simulation.
Duration, frame rate, and playback speed come from the checkpoints unless overridden.
The current camera, materials, and lights are applied during rendering.

Render water surfaces with Genesis RayTracer and assemble the four-panel comparison:

```powershell
python examples/teapot/render_teapot_comparison.py --replay-checkpoints out/teapot_checkpoints --preview render --spp 128 -o out/teapot_render
```

Render the faster particle version:

```powershell
python examples/teapot/render_teapot_comparison.py --replay-checkpoints out/teapot_checkpoints --preview particles -o out/teapot_render
```

| Output | Ray-traced water | Particles |
| --- | --- | --- |
| Comparison video | `out/teapot_render/teapot_comparison.mp4` | `out/teapot_render/teapot_comparison_particles.mp4` |
| Method videos | `out/teapot_render/METHOD.mp4` | `out/teapot_render/particles/METHOD.mp4` |
| Logs and run metadata | `out/teapot_render/` | `out/teapot_render/particles/` |

Existing outputs with the same names are overwritten. The two rendering modes use separate output paths.
The default panel order is PBSTF, PBD, DFSPH, WCSPH, read left to right and top to bottom.
A shorter method recording holds its last frame, with its stopping time labeled, until the comparison ends.

Ray-traced rendering uses the clay teapot, tea table, white walls, and water materials in
`examples/teapot/teapot_render.py`. Environment and texture assets are downloaded on first use into `out/teapot_assets/`.

To render only PBSTF as a video:

```powershell
python examples/teapot/pbstf_teapot.py --replay-checkpoints out/teapot_checkpoints --preview render --spp 128 -o out/teapot_render
```

To inspect one saved state while tuning materials and lighting:

```powershell
python examples/teapot/pbstf_teapot.py --replay-checkpoints out/teapot_checkpoints --replay-time 3 --preview render --spp 32 -o out/teapot_stills
```

`--replay-time` is measured in simulation seconds. This example renders step 300 at the default time step and saves
`out/teapot_stills/PBSTF_step_00000300.png`. If the requested time falls between checkpoints, the earlier frame is used.
Use the PBD or SPH entry point from the table to render another method.

## Compose existing videos

To rebuild the comparison from existing method videos and their initial-state files:

```powershell
python examples/teapot/render_teapot_comparison.py --compose-only --seconds 50 --fps 25 --playback-speed 1 -o out/teapot_render
python examples/teapot/render_teapot_comparison.py --compose-only --preview particles --seconds 50 --fps 25 --playback-speed 1 -o out/teapot_render
```

Match duration, frame rate, and playback speed to the recordings. These commands only assemble the video.

## Useful options

| Option | Usage |
| --- | --- |
| `--seconds N` | Simulation time, or the replay interval from zero to N seconds. Single-method scripts also accept `--steps N`. |
| `--fps N --playback-speed S` | Output frame rate and simulated seconds per video second. `--playback-speed 1` gives real-time playback. |
| `--checkpoint-every N` | Save every N simulation steps. Smaller values preserve finer motion and use more disk space. |
| `--spp N` | Ray-tracing samples per pixel. Higher values reduce noise and take longer; use 32 for drafts and 128 for final rendering. |
| `--methods PBSTF PBD WCSPH` | Select methods and panel order in the comparison runner. The default includes all four. |
| `--jobs N` | Concurrent methods in the comparison runner. Defaults to 1; higher values require more GPU memory. |
| `--timings` | Write per-stage timing CSV files; GPU synchronization reduces execution overlap. |

Replay uses the saved states; increasing the output frame rate beyond the checkpoint cadence repeats states.
For fresh runs, single-method scripts default to 100 simulated seconds at 1x playback; the comparison runner defaults
to 32 seconds at 2x playback. Both default to 25 fps. The commands above set 50 seconds at 1x explicitly when saving.
