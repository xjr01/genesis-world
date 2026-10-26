"""Validate offline artifacts and record native SDF caches without modifying the engine."""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.utils.mesh import get_gsd_path


@dataclass
class StartupTimings:
    previous: float = field(default_factory=time.perf_counter)
    seconds: dict[str, float] = field(default_factory=dict)

    def mark(self, name):
        now = time.perf_counter()
        self.seconds[name] = now - self.previous
        self.previous = now


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def source_signature():
    folder = Path(__file__).parent
    return {
        name: file_digest(folder / name)
        for name in (
            "asset_demo.py",
            "aligned_motion.py",
            "asset_runtime.py",
            "geometry.py",
            "config.py",
            "contact.py",
            "asset_contact.py",
            "aligned_contact.py",
            "cutting_scene.py",
            "face_footprint.py",
            "asset_material.json",
        )
    }


def particle_morph(folder, name, spacing, prepare):
    path = folder / f"{name}-particles.npy"
    if prepare:
        (mesh,) = gs.Mesh.from_morph_surface(gs.morphs.Mesh(file=str(folder / f"{name}.obj")))
        points = mesh.particlize(spacing, "regular").astype(gs.np_float)
        np.save(path, points)
    else:
        points = np.load(path)
    return gs.morphs.Particles(positions=points)


def sdf_records(scene):
    records = []
    for entity in scene.rigid_solver.entities:
        for link in entity.links:
            for geom in link.geoms:
                if geom.type.name != "MESH":
                    continue
                path = Path(get_gsd_path(geom.init_verts, geom.init_faces, geom.sdf_res, geom.sdf_cell_size))
                records.append(
                    {
                        "geom_index": geom.idx,
                        "triangles": len(geom.init_faces),
                        "resolution": geom.sdf_res.tolist(),
                        "cells": int(np.prod(geom.sdf_res)),
                        "cell_size_m": geom.sdf_cell_size.tolist(),
                        "cache_path": str(path),
                        "cache_exists": path.is_file(),
                    }
                )
    return records


def verify_prepared(folder):
    path = folder / "runtime-preparation.json"
    if not path.is_file():
        raise ValueError("Assets are not prepared. Run prepare_runtime before simulation.")
    report = json.loads(path.read_text())
    if report["trajectory_sources"] != source_signature():
        raise ValueError("Simulation setup changed; prepare the assets again before simulation.")
    for name, digest in report["files"].items():
        if file_digest(folder / name) != digest:
            raise ValueError(f"Prepared asset changed: {name}. Re-run offline preparation.")
    if any(not Path(record["cache_path"]).is_file() for record in report["sdf"]):
        raise ValueError("Native SDF cache is missing on this machine. Re-run offline preparation.")
    return report


def write_preparation(folder, scene, seconds):
    names = (
        "config.json",
        "assets.json",
        "butter.obj",
        "dish.obj",
        "bread.obj",
        "dish-collision.obj",
        "board-collision.obj",
        "blade.obj",
        "blade-face.npz",
        "butter-particles.npy",
        "bread-particles.npy",
        "motion.npz",
        "support.npz",
    )
    report = {
        "files": {name: file_digest(folder / name) for name in names if (folder / name).is_file()},
        "trajectory_sources": source_signature(),
        "sdf": sdf_records(scene),
        "preparation_seconds": seconds,
        "scope": "Offline initial particles, trajectory and native SDF cache; warm-up state is discarded.",
    }
    (folder / "runtime-preparation.json").write_text(json.dumps(report, indent=2))
    return report
