"""Portable numerical checkpoints for visual replay of rigid, deformable, and liquid entities.

Scene descriptions contain pickled Genesis options and use the assets of the recording checkout. Load descriptions
created by a trusted recording process. Frames contain only numerical arrays and can be read independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import pickle
from typing import Literal
from zipfile import BadZipFile

import numpy as np

import genesis as gs
from genesis.engine.entities.pbd_entity import PBDBaseEntity
from genesis.engine.entities.pbstf_entity import PBSTFEntity
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.materials.base import Material
from genesis.options.morphs import Morph
from genesis.options.surfaces import Surface
from genesis.utils.misc import tensor_to_array


@dataclass(frozen=True)
class ReplayEntity:
    name: str
    morph: Morph
    material: Material
    surface: Surface
    kind: Literal["rigid", "pbd", "pbstf"]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ReplayScene:
    version: int
    dt: float
    precision: str
    n_envs: int
    env_spacing: tuple[float, float]
    sim_options: gs.options.SimOptions
    rigid_options: gs.options.RigidOptions
    pbd_options: gs.options.PBDOptions | gs.options.PBDUnifiedOptions
    pbstf_options: gs.options.PBSTFOptions
    vis_options: gs.options.VisOptions
    viewer_options: gs.options.ViewerOptions
    entities: tuple[ReplayEntity, ...]


@dataclass(frozen=True)
class ReplayFrame:
    index: int
    time: float
    # Entity order follows ReplayScene: rigid qpos; particle pos, active, then liquid concentration.
    arrays: tuple[np.ndarray, ...]


def describe_scene(scene: gs.Scene, dt: float) -> ReplayScene:
    """Capture construction options and array shapes for the supported visual entity types."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Checkpoint dt must be finite and positive.")
    entities = []
    batch_shape = (scene.n_envs,) if scene.n_envs else ()
    for entity in scene.entities:
        if isinstance(entity, RigidEntity):
            kind = "rigid"
            shape = (*batch_shape, entity.n_qs)
        elif isinstance(entity, PBSTFEntity):
            kind = "pbstf"
            shape = (*batch_shape, entity.n_particles, 3)
        elif isinstance(entity, PBDBaseEntity) and isinstance(entity.material, gs.materials.PBD.Elastic):
            kind = "pbd"
            shape = (*batch_shape, entity.n_particles, 3)
        else:
            raise ValueError(f"Checkpoint replay does not support {type(entity).__name__}: {entity.name}.")
        entities.append(ReplayEntity(entity.name, entity.morph, entity.material, entity.surface, kind, shape))
    return ReplayScene(
        version=1,
        dt=dt,
        precision=str(np.dtype(gs.np_float).itemsize * 8),
        n_envs=scene.n_envs,
        env_spacing=scene.env_spacing,
        sim_options=scene.sim_options,
        rigid_options=scene.rigid_options,
        pbd_options=scene.pbd_options,
        pbstf_options=scene.pbstf_options,
        vis_options=scene.vis_options,
        viewer_options=scene.viewer_options,
        entities=tuple(entities),
    )


class CheckpointWriter:
    """Write frame zero followed by one checkpoint per caller-supplied time interval.

    Each completed frame is atomically published, so an interrupted recording retains its completed prefix.
    """

    def __init__(self, scene: gs.Scene, directory: Path, dt: float):
        self.scene = scene
        self.directory = Path(directory)
        self.description = describe_scene(scene, dt)
        self.next_frame = 0
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(f"Checkpoint directory must be empty: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / "scene.pkl.part"
        with temporary.open("wb") as stream:
            pickle.dump(self.description, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(self.directory / "scene.pkl")

    def write_frame(self) -> Path:
        """Capture the current visual state and commit the next sequential frame."""
        arrays = []
        for entity in self.scene.entities:
            if isinstance(entity, RigidEntity):
                arrays.append(tensor_to_array(entity.get_qpos()))
            elif isinstance(entity, (PBDBaseEntity, PBSTFEntity)):
                arrays.extend(
                    (tensor_to_array(entity.get_particles_pos()), tensor_to_array(entity.get_particles_active()))
                )
                if isinstance(entity, PBSTFEntity):
                    arrays.append(tensor_to_array(entity.get_particles_concentration()))
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError(f"Checkpoint {self.next_frame} contains non-finite state.")
        path = self.directory / f"frame-{self.next_frame:08d}.npz"
        temporary = path.with_suffix(".npz.part")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, self.next_frame, self.next_frame * self.description.dt, *arrays)
        temporary.replace(path)
        self.next_frame += 1
        return path


class CheckpointReader:
    """Index a completed checkpoint prefix and load one numerical frame at a time."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        path = self.directory / "scene.pkl"
        try:
            with path.open("rb") as stream:
                description = pickle.load(stream)
        except (OSError, EOFError, pickle.UnpicklingError) as error:
            raise ValueError(f"Cannot read checkpoint scene {path}: {error}") from error
        if not isinstance(description, ReplayScene) or description.version != 1:
            raise ValueError("Unsupported checkpoint scene format; record a new sequence with this checkout.")
        if not math.isfinite(description.dt) or description.dt <= 0 or description.precision not in ("32", "64"):
            raise ValueError("Invalid checkpoint time interval or precision.")
        self.description = description
        self.frames = tuple(sorted(self.directory.glob("frame-*.npz")))
        if not self.frames:
            raise ValueError(f"No completed checkpoint frames in {self.directory}.")
        for index, path in enumerate(self.frames):
            if path.name != f"frame-{index:08d}.npz":
                raise ValueError(f"Missing or misnumbered checkpoint frame {index}: {path.name}")
        self.shapes = []
        self.is_boolean = []
        for entity in description.entities:
            if entity.kind not in ("rigid", "pbd", "pbstf"):
                raise ValueError(f"Unsupported checkpoint entity kind: {entity.kind}")
            self.shapes.append(entity.shape)
            self.is_boolean.append(False)
            if entity.kind != "rigid":
                self.shapes.append(entity.shape[:-1])
                self.is_boolean.append(True)
                if entity.kind == "pbstf":
                    self.shapes.append(entity.shape[:-1])
                    self.is_boolean.append(False)

    def read_frame(self, index: int) -> ReplayFrame:
        """Read and validate the index, timestamp, shapes, and numerical values of one frame."""
        if not 0 <= index < len(self.frames):
            raise IndexError(f"Checkpoint index {index} is outside [0, {len(self.frames) - 1}].")
        path = self.frames[index]
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {f"arr_{slot}" for slot in range(len(self.shapes) + 2)}:
                    raise ValueError("Unexpected checkpoint array layout.")
                saved_index, saved_time = data["arr_0"], data["arr_1"]
                if saved_index.shape != () or saved_index.dtype.kind not in "iu" or saved_index != index:
                    raise ValueError("Checkpoint index disagrees with its filename.")
                if saved_time.shape != () or saved_time != index * self.description.dt:
                    raise ValueError("Checkpoint timestamp disagrees with its time interval.")
                arrays = tuple(data[f"arr_{slot + 2}"] for slot in range(len(self.shapes)))
                for array, shape, is_boolean in zip(arrays, self.shapes, self.is_boolean):
                    if array.shape != shape or array.dtype.kind != ("b" if is_boolean else "f"):
                        raise ValueError("Checkpoint array shape or type disagrees with the scene.")
                    if not np.isfinite(array).all():
                        raise ValueError("Checkpoint contains non-finite state.")
        except (OSError, ValueError, EOFError, BadZipFile) as error:
            raise ValueError(f"Cannot read checkpoint {path}: {error}") from error
        return ReplayFrame(index, saved_time.item(), arrays)

    def build_scene(self, *, is_viewer_shown: bool = True) -> gs.Scene:
        """Rebuild the recorded entities and initialize their rendering resources."""
        description = self.description
        viewer_options = description.viewer_options.model_copy(deep=True)
        viewer_options.res = (960, 720)
        viewer_options.realtime_factor = None
        # The replay loop pumps window events while paused and owns every OpenGL operation.
        viewer_options.run_in_thread = False
        scene = gs.Scene(
            sim_options=description.sim_options,
            rigid_options=description.rigid_options,
            pbd_options=description.pbd_options,
            pbstf_options=description.pbstf_options,
            vis_options=description.vis_options,
            viewer_options=viewer_options,
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=False,
            ),
            show_viewer=is_viewer_shown,
        )
        for entity in description.entities:
            scene.add_entity(
                morph=entity.morph,
                material=entity.material,
                surface=entity.surface,
                name=entity.name,
            )
        scene.build(n_envs=description.n_envs, env_spacing=description.env_spacing)
        for entity, recorded in zip(scene.entities, description.entities):
            count = entity.n_qs if isinstance(entity, RigidEntity) else entity.n_particles
            expected = recorded.shape[-1] if recorded.kind == "rigid" else recorded.shape[-2]
            if count != expected:
                raise ValueError(f"Rebuilt entity {entity.name} has {count} values; checkpoint requires {expected}.")
        return scene


def apply_frame(scene: gs.Scene, frame: ReplayFrame) -> None:
    """Restore an entire visual frame through entity setters. The caller owns the viewer lock."""
    arrays = iter(frame.arrays)
    for entity in scene.entities:
        if isinstance(entity, RigidEntity):
            qpos = next(arrays)
            if entity.n_qs:
                entity.set_qpos(qpos, zero_velocity=False)
        elif isinstance(entity, (PBDBaseEntity, PBSTFEntity)):
            entity.set_particles_pos(next(arrays))
            entity.set_particles_active(next(arrays))
            if isinstance(entity, PBSTFEntity):
                entity.set_particles_concentration(next(arrays))
        else:
            raise ValueError(f"Unsupported replay entity: {entity.name}")
    scene.visualizer.update(force=True)
