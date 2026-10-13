![Genesis World teaser](https://raw.githubusercontent.com/YilingQiao/Genesis/readme-assets/videos/HeroShot_Final.png)

# Genesis World

[![PyPI - Version](https://img.shields.io/pypi/v/genesis-world)](https://pypi.org/project/genesis-world/)
[![PyPI Downloads](https://static.pepy.tech/badge/genesis-world)](https://pepy.tech/projects/genesis-world)
[![Documentation](https://app.readthedocs.org/projects/genesis-world/badge/?version=latest)](https://genesis-world.readthedocs.io/en/latest/)
[![GitHub Issues](https://img.shields.io/github/issues/Genesis-Embodied-AI/genesis-world)](https://github.com/Genesis-Embodied-AI/genesis-world/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/Genesis-Embodied-AI/genesis-world)](https://github.com/Genesis-Embodied-AI/genesis-world/discussions)



**Genesis World** is a simulation platform for physical AI developments. It combines a unified multi-physics engine, a photo-realistic renderer ([Nyx](https://github.com/Genesis-Embodied-AI/genesis-nyx)), and a cross-platform compiler ([Quadrants](https://github.com/Genesis-Embodied-AI/quadrants)) behind a Pythonic simulation interface. Genesis World is designed to scale from a single laptop kernel to datacenter-grade GPUs, while remaining easy to read, extend, and embed in research code.

It was previously named **Genesis** and started as an academic project since Dec 2024, and its development is now officially supported by [Genesis AI](https://www.genesis.ai/).

For more technical details, refer to our [blog post](https://genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward).

## Table of Contents

1. [What is Genesis World?](#what-is-genesis-world)
2. [Catalogue](#catalogue)
3. [Quick Installation](#quick-installation)
4. [Docker](#docker)
5. [Contribution](#contributing-to-genesis)
6. [Support](#support)
7. [License and Acknowledgments](#license-and-acknowledgments)
8. [Citation](#citation)

## What is Genesis World?

![Genesis World stack](https://raw.githubusercontent.com/YilingQiao/Genesis/readme-assets/videos/diagram_white_lum.png)

Genesis World occupies the four layers inside the dashed box. Above sits whatever you build (robotics environments, ML pipelines, data generation, agentic simulation); below sits whatever compute backend you have.

- **Simulation Interface** — the user-facing API: asset parsing (URDF, MJCF, OBJ, GLB, USD, …), entity accessors, controllers, sensors, parallel and heterogeneous environments, and a built-in GUI.
- **Physics** — a unified multi-physics engine integrating Rigid, FEM, MPM, Particle (PBD / SPH), [uipc](https://github.com/spiriMirror/libuipc), an explicit coupler, and SAP, all sharing one scene and one state.
- **Render** — three rendering paths plug in as camera sensors: **[Nyx](https://github.com/Genesis-Embodied-AI/genesis-nyx)** (our in-house renderer designed for robotics), Luisa (DSL ray tracer), and Pyrender (rasterizer).
- **Compiler** — **[Quadrants](https://github.com/Genesis-Embodied-AI/quadrants)** lowers Python kernel code to CUDA, AMD ROCm, Apple Metal, Vulkan, x86, and ARM64. It carries Genesis's autodiff, GPU graphs, and fastcache machinery.

### Documentation
- [Genesis World](https://genesis-world.readthedocs.io/en/latest/)
- [Quadrants](https://genesis-embodied-ai.github.io/quadrants/index.html)
- [Nyx](https://genesis-embodied-ai.github.io/genesis-nyx/latest/)

## Catalogue

This branch contains the shared teapot comparison scene in `examples/teapot/`.

| Method | Entry point |
|---|---|
| PBSTF | [pbstf_teapot.py](./examples/teapot/pbstf_teapot.py) |
| PBD | [pbd_liquid_teapot.py](./examples/teapot/pbd_liquid_teapot.py) |
| DFSPH / WCSPH | [sph_teapot.py](./examples/teapot/sph_teapot.py) |

All methods share the teapot geometry, prescribed motion, initial particle positions and velocities, camera, gravity,
and floor at Y = -20. Solver-specific material parameters and substeps remain configurable in their entry points.
The defaults use 188,716 particles, a particle diameter of 0.1, and a simulation time step of 0.01 seconds on CUDA.

```bash
python examples/teapot/pbstf_teapot.py --vis --preview particles
python examples/teapot/pbd_liquid_teapot.py --vis --preview render
python examples/teapot/sph_teapot.py --pressure-solver WCSPH --vis --preview render
python examples/teapot/render_teapot_comparison.py --jobs 2
```

The rendered preview uses Genesis RayTracer with reconstructed water surfaces and requires the optional LuisaRender
backend and `pysplashsurf`. Lighting and surface assets are downloaded into `out/teapot_assets/`, with their CC0 sources
recorded in `sources.csv`. The comparison runner saves four method videos, initial and final particle states, run
settings, preview images, and `out/teapot_comparison/teapot_comparison.mp4`. Its defaults cover 32 simulated seconds at
2x playback, 25 frames per second, and 32 samples per pixel. Use `--jobs 1` to reduce peak GPU memory, or
`--compose-only` to combine existing recordings after verifying their initial states match exactly.

Save simulation checkpoints once, then render them with the current camera, lighting, materials, and sample count:

```bash
python examples/teapot/pbstf_teapot.py --seconds 32 --playback-speed 2 --save-checkpoints out/teapot_checkpoints
python examples/teapot/pbstf_teapot.py --replay-checkpoints out/teapot_checkpoints --preview render --spp 64 -o out/teapot_rerender
python examples/teapot/pbstf_teapot.py --replay-checkpoints out/teapot_checkpoints --replay-time 24 --preview render --spp 64 -o out/teapot_stills
```

Each method writes compressed particle positions, velocities, active flags, and teapot poses under `DIR/METHOD`,
plus JSON metadata. The cache is for rendering replay. Saving uses an empty method directory and defaults to the video
cadence (every eight steps with 2x playback at 25 fps); `--checkpoint-every 1` preserves finer motion at a higher disk
cost. Replay skips physics and selects the closest saved frame at or before each requested time. Increasing the
output frame rate repeats cached states. `--replay-time` saves one PNG. `--timings` writes stage timings to CSV and
synchronizes GPU work for accurate measurement, reducing execution overlap.

The comparison runner accepts the same save/replay flags and a method selection:

```bash
python examples/teapot/render_teapot_comparison.py --methods PBSTF PBD WCSPH --simulate-only --save-checkpoints out/teapot_comparison_cache
python examples/teapot/render_teapot_comparison.py --methods PBSTF PBD WCSPH --replay-checkpoints out/teapot_comparison_cache --spp 64 -o out/teapot_comparison_rerender
```

`--simulate-only` saves the selected trajectories with rendering disabled. Non-finite particle states raise an error;
the video encoder finalizes the available frames. A recording marked `unstable` is labeled accordingly during composition.

## Quick Installation

### Using pip

Install **PyTorch** first following the [official instructions](https://pytorch.org/get-started/locally/).

Then, install Genesis via PyPI:
```bash
pip install genesis-world  # Requires Python>=3.10,<3.14;
```

For the latest version to date, make sure that `pip` is up-to-date via `pip install --upgrade pip`, then run command:
```bash
pip install git+https://github.com/Genesis-Embodied-AI/genesis-world.git
```
Note that the package must still be updated manually to sync with main branch.

Users seeking to contribute are encouraged to install Genesis in editable mode. First, make sure that `genesis-world` has been uninstalled, then clone the repository and install locally:
```bash
git clone https://github.com/Genesis-Embodied-AI/genesis-world.git
cd genesis-world
pip install -e ".[dev]"
```
It is recommended to systematically execute `pip install -e ".[dev]"` after moving HEAD to make sure that all dependencies and entrypoints are up-to-date.

### Optional extras

| | |
|---|---|
| IPC solver (uipc backend) | `pip install pyuipc` *(Linux / Windows x86, NVIDIA GPU)* |
| Nyx renderer | `pip install gs-nyx` — see [genesis-nyx](https://github.com/Genesis-Embodied-AI/genesis-nyx) |

Quadrants is bundled with Genesis automatically; no extra install. The standalone wheel (`pip install quadrants`) is documented at [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants) for users who want the compiler outside Genesis.

### Using uv

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager.

**Install uv:**
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Quick start with uv:**
```bash
git clone https://github.com/Genesis-Embodied-AI/genesis-world.git
cd genesis-world
uv sync
```

Then install PyTorch for your platform:

```bash
# NVIDIA GPU (CUDA 12.6 as an example)
uv pip install torch --index-url https://download.pytorch.org/whl/cu126

# CPU only (Linux/Windows)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (Metal/MPS)
uv pip install torch
```

Run an example:
```bash
uv run examples/teapot/pbstf_teapot.py --vis
```

## Docker

If you want to use Genesis from Docker, you can first build the Docker image as:

```bash
docker build -t genesis -f docker/Dockerfile docker
```

Then you can run the examples inside the docker image (mounted to `/workspace/examples`):

```bash
xhost +local:root # Allow the container to access the display

docker run --gpus all --rm -it \
-e DISPLAY=$DISPLAY \
-e LOCAL_USER_ID="$(id -u)" \
-v /dev/dri:/dev/dri \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $(pwd):/workspace \
--name genesis genesis:latest
```

### AMD users
AMD users can use Genesis using the `docker/Dockerfile.amdgpu` file, which is built by running:
```
docker build -t genesis-amd -f docker/Dockerfile.amdgpu docker
```

and can then be used by running:

```xhost +local:docker \
docker run -it --network=host \
 --device=/dev/kfd \
 --device=/dev/dri \
 --group-add=video \
 --ipc=host \
 --cap-add=SYS_PTRACE \
 --security-opt seccomp=unconfined \
 --shm-size 8G \
 -v $PWD:/workspace \
 -e DISPLAY=$DISPLAY \
 genesis-amd
 ```

The examples will be accessible from `/workspace/examples`. Note: AMD users should use the ROCm (HIP) backend. This means you will need to call `gs.init(backend=gs.amdgpu)` to initialise Genesis.

## Contributing to Genesis

The Genesis project is an open and collaborative effort. We welcome all forms of contributions from the community, including:

- **Pull requests** for new features or bug fixes.
- **Bug reports** through GitHub Issues.
- **Suggestions** to improve Genesis's usability.

Refer to our [contribution guide](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/.github/contributing/PULL_REQUESTS.md) for more details.

## Support

- Report bugs or request features via GitHub [Issues](https://github.com/Genesis-Embodied-AI/genesis-world/issues).
- Join discussions or ask questions on GitHub [Discussions](https://github.com/Genesis-Embodied-AI/genesis-world/discussions).

## License and Acknowledgments

The Genesis source code is licensed under Apache 2.0.

Genesis's development has been made possible thanks to these open-source projects:

- [Taichi](https://github.com/taichi-dev/taichi): the original compiler that [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants) forked from in June 2025. Kudos to the Taichi team for their technical support over the years.
- [libuipc](https://github.com/spiriMirror/libuipc): IPC solver backend.
- [FluidLab](https://github.com/zhouxian/FluidLab): Reference MPM solver implementation.
- [SPH_Taichi](https://github.com/erizmr/SPH_Taichi): Reference SPH solver implementation.
- [Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) and [PBF3D](https://github.com/WASD4959/PBF3D): Reference PBD solver implementations.
- [MuJoCo](https://github.com/google-deepmind/mujoco): Reference for rigid body dynamics.
- [libccd](https://github.com/danfis/libccd): Reference for collision detection.
- [PyRender](https://github.com/mmatl/pyrender): Rasterization-based renderer.
- [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) and [LuisaRender](https://github.com/LuisaGroup/LuisaRender): Ray-tracing DSL.
- [Madrona](https://github.com/shacklettbp/madrona) and [Madrona-mjx](https://github.com/shacklettbp/madrona_mjx): Batch renderer backend

## Citation

If you use Genesis in your research, please consider citing:

```bibtex
@article{
   genesis2026genesisworld,
   author = {Genesis AI Team},
   title = {The Role of Simulation in Scalable Robotics, Genesis World 1.0, and the Path Forward},
   journal = {Genesis AI Blog},
   month = {May},
   year = {2026},
   url = {https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward},
}
```
```bibtex
@misc{
  Genesis,
  author = {Genesis Authors},
  title = {Genesis: A Generative and Universal Physics Engine for Robotics and Beyond},
  month = {December},
  year = {2024},
  url = {https://github.com/Genesis-Embodied-AI/genesis-world}
}
```
