"""Check that complete particle and Isaac videos replay the same aligned simulation."""

import argparse
import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    (run / "final-validation.json").unlink(missing_ok=True)
    config = json.loads((run / "config.json").read_text())
    simulation = json.loads((run / "simulation.json").read_text())
    physics = json.loads((run / "aligned-audit.json").read_text())
    contact = json.loads((run / "asset-audit.json").read_text())
    surface = json.loads((run / "surface-report.json").read_text())
    rendering = json.loads((run / "aligned-cut-spread-isaac.json").read_text())
    display = json.loads((run / "asset-display-check.json").read_text())
    with (run / "state.npz").open("rb") as source:
        state_hash = hashlib.file_digest(source, "sha256").hexdigest()
    video_reports = {}
    poster = Image.new("RGB", (1280, 1080), (20, 20, 20))
    poster_indices = tuple(min(simulation["frames"] - 1, round(t * config["fps"])) for t in (0, 3.7, 5.2, 6, 7.5, 8.4))
    for name in ("aligned-cut-spread-particles", "aligned-cut-spread-isaac"):
        count = 0
        with av.open(str(run / f"{name}.mp4")) as container:
            stream = container.streams.video[0]
            frame_rate = float(stream.average_rate)
            width, height = stream.width, stream.height
            for index, frame in enumerate(container.decode(video=0)):
                count += 1
                if name.endswith("isaac") and index in poster_indices:
                    slot = poster_indices.index(index)
                    poster.paste(
                        frame.to_image().resize((640, 360), Image.Resampling.LANCZOS),
                        ((slot % 2) * 640, (slot // 2) * 360),
                    )
        video_reports[name] = {"frames": count, "fps": frame_rate, "width": width, "height": height}
    matches = all(report["state_sha256"] == state_hash for report in (simulation, physics, contact, surface, rendering))
    is_sampling_verified = bool(
        rendering.get("render_mode") == "PathTracing"
        and rendering.get("effective_samples") == rendering["samples"]
        and all(frame.get("effective_samples") == rendering["samples"] for frame in rendering["frames"])
    )
    is_passed = bool(
        simulation["is_complete"]
        and physics["is_passed"]
        and rendering["is_complete"]
        and matches
        and is_sampling_verified
        and surface["frames"] == simulation["frames"]
        and [frame["index"] for frame in rendering["frames"]] == list(range(simulation["frames"]))
        and all(report["frames"] == simulation["frames"] for report in video_reports.values())
        and all(report["fps"] == config["fps"] for report in video_reports.values())
        and contact["domain_boundary_unique_particles"] == 0
        and max(frame["blade_penetration_max_m"] for frame in contact["frames"]) < config["particle_size"] / 4
        and len(rendering["active_lights"]) == 4
        and all(path.startswith("/World/Lights/") for path in rendering["active_lights"])
    )
    report = {
        "is_passed": is_passed,
        "state_sha256": state_hash,
        "all_state_hashes_match": matches,
        "is_sampling_verified": is_sampling_verified,
        "effective_samples": rendering.get("effective_samples"),
        "videos": video_reports,
        "pickup_fraction": physics["pickup_fraction"],
        "deposition_fraction": physics["deposition_fraction"],
        "spread_span_gain_m": physics["spread_span_gain_m"],
        "advanced_deposition_fraction": physics["advanced_deposition_fraction"],
        "sampled_interior_side_changes": physics["sampled_interior_side_changes"],
        "maximum_saved_frame_blade_penetration_m": max(frame["blade_penetration_max_m"] for frame in contact["frames"]),
        "maximum_sampled_dish_penetration_m": max(frame["dish_penetration_max_m"] for frame in contact["frames"]),
        "bread_board_sampled_penetration_m": physics["bread_board_max_penetration_m"],
        "asset_display_check": display,
        "active_lights": rendering["active_lights"],
        "coating_voxel_m": surface["coating_voxel_m"],
        "simulation_seconds": simulation["elapsed_seconds"],
        "physical_run_reused": simulation.get("repack", {}).get("physical_run_reused", False),
        "surface_seconds": surface["elapsed_seconds"],
        "render_seconds": rendering["elapsed_seconds"],
        "scope": "Saved-frame contact checks and video/state correspondence; not a continuous-time penetration proof.",
    }
    (run / "final-validation.json").write_text(json.dumps(report, indent=2))
    ImageDraw.Draw(poster).text(
        (8, 8), "Recorded simulation: initial / pickup / transfer / approach / spread / final", fill="white"
    )
    poster.save(run / "validation-contact-sheet.jpg", quality=95)
    print(json.dumps(report, indent=2))
    if not is_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
