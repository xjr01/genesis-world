import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from examples.butter_spreading import audit, pipeline, ripples
from examples.butter_spreading.config import RunConfig
from examples.butter_spreading.motion import knife_pose
from examples.butter_spreading.surface import splat


@pytest.mark.parametrize("backend", [None])
def test_butter_audit_rejects_incomplete_and_modified_samples(tmp_path, monkeypatch):
    config = RunConfig(butter_particle_size=0.0006)
    config_path = tmp_path / "run-config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    (tmp_path / "completion.json").write_text(
        json.dumps(
            {
                "is_complete": True,
                "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    positions = np.stack(np.meshgrid(np.arange(8), np.arange(4), np.arange(4), indexing="ij"), axis=-1)
    positions = positions.reshape(-1, 3) * config.butter_particle_size + (-0.05, 0.0, 0.028)
    bread = positions - (0.0, 0.0, 0.01)
    times = np.array([0.0, config.dt])
    state_path = tmp_path / "mpm-state.npz"
    np.savez(
        state_path,
        particles=np.stack((positions, positions)),
        particle_velocities=np.zeros((2, len(positions), 3)),
        bread_particles=np.stack((bread, bread)),
        knife_positions=np.array([knife_pose(time_s) for time_s in times]),
        time_s=times,
        fps=24,
    )
    state_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "particle_spacing_m": config.butter_particle_size,
                "bread_particle_spacing_m": config.bread_particle_size,
                "butter_particle_count": len(positions),
                "bread_particle_count": len(bread),
                "knife_collider_size_m": [0.022, 0.054, 0.005],
                "knife_collider_xy_offset_m": [0.0021, 0.0035],
                "particle_constraints": False,
                "dt_s": config.dt,
                "frames": 2,
                "total_steps": 1,
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "audit.json"
    arguments = ["audit", "--state", str(state_path), "--output", str(report_path)]
    monkeypatch.setattr(sys, "argv", [*arguments, "--preview"])
    audit.main()
    report = json.loads(report_path.read_text())
    assert report["is_passed"] and not report["is_full_trajectory"]

    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit, match="1"):
        audit.main()
    report = json.loads(report_path.read_text())
    assert not report["is_passed"]
    assert not report["checks"]["full_time_coverage"]

    config_path.write_text(config.model_copy(update={"seed": 18}).model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [*arguments, "--preview"])
    with pytest.raises(SystemExit, match="1"):
        audit.main()
    assert not json.loads(report_path.read_text())["checks"]["configuration_hash_matches"]


@pytest.mark.parametrize("backend", [None])
def test_butter_surface_conserves_rest_volume_and_translation():
    points = np.array([[0.003, 0.003, 0.003], [0.004, 0.004, 0.004]])
    origin = np.zeros(3)
    shape = (20, 20, 20)
    voxel = 0.0005
    volume = 0.001**3
    density = splat(points, origin, shape, voxel, 0.002, volume)
    translated = splat(points + (voxel, 0.0, 0.0), origin, shape, voxel, 0.002, volume)
    assert density.sum() * voxel**3 == pytest.approx(len(points) * volume, rel=1e-6)
    np.testing.assert_allclose(translated[1:], density[:-1], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("backend", [None])
def test_butter_pipeline_preserves_failure_and_existing_outputs(tmp_path, monkeypatch):
    output = tmp_path / "episode"
    monkeypatch.setattr(sys, "argv", ["pipeline", "--output-dir", str(output)])
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 7))
    with pytest.raises(subprocess.CalledProcessError) as error:
        pipeline.main()
    assert error.value.returncode == 7
    assert not (output / "pipeline-completion.json").exists()
    timing_bytes = (output / "full-pipeline-timing.json").read_bytes()
    assert json.loads(timing_bytes)["stages"][0]["exit_code"] == 7
    with pytest.raises(SystemExit, match="2"):
        pipeline.main()
    assert (output / "full-pipeline-timing.json").read_bytes() == timing_bytes


@pytest.mark.parametrize("backend", [None])
def test_butter_ripples_rejects_unrelated_surface(tmp_path, monkeypatch):
    state = tmp_path / "state.npz"
    surface = tmp_path / "surface.npz"
    np.savez(state, particles=np.zeros((2, 8, 3)))
    state.with_suffix(".json").write_text('{"particle_spacing_m": 0.0006}', encoding="utf-8")
    np.savez(surface, kernel_width_m=0.001, voxel_size_m=0.0003, state_sha256="unrelated")
    monkeypatch.setattr(
        sys,
        "argv",
        ["ripples", "--state", str(state), "--surface", str(surface), "--output", str(tmp_path / "ripples.json")],
    )
    with pytest.raises(ValueError, match="hashes differ"):
        ripples.main()
