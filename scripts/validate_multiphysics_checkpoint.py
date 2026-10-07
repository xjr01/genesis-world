"""Exercise paired Scene and task-controller reset/checkpoint on one normalized scenario."""

import argparse
import time
from pathlib import Path

import torch

import genesis as gs
from examples.multiphysics import butter_spreading, coffee_water, garment_folding, litter_scoop, table_wiping
from genesis.utils.video_encoder import VideoEncoder


def _observable_state(scenario, runtime):
    if scenario == "coffee_water":
        entities = (runtime.coffee, runtime.water, runtime.sponge)
        tensors = tuple(
            tensor for entity in entities for tensor in (entity.get_particles_pos(), entity.get_particles_vel())
        )
        tensors += (runtime.robot.get_qpos(), runtime.water_cup.get_pos(), runtime.rod.get_pos())
    elif scenario == "table_wiping":
        entities = (runtime.liquid, runtime.sponge)
        tensors = tuple(
            tensor for entity in entities for tensor in (entity.get_particles_pos(), entity.get_particles_vel())
        )
        tensors += (runtime.robot.get_qpos(),)
    elif scenario == "litter_scoop":
        entities = (runtime.sand, runtime.water)
        tensors = tuple(
            tensor for entity in entities for tensor in (entity.get_particles_pos(), entity.get_particles_vel())
        )
        tensors += (runtime.scene.dem_solver.get_sdf_obstacle_pos(), runtime.scene.dem_solver.get_sdf_obstacle_quat())
    elif scenario == "garment_folding":
        state = runtime.garment.get_state()
        tensors = (state.pos, state.vel)
    else:
        entities = (runtime.bread, runtime.butter)
        tensors = tuple(
            tensor for entity in entities for tensor in (entity.get_particles_pos(), entity.get_particles_vel())
        )
        tensors += (runtime.blade.get_pos(), runtime.blade.get_quat())
    tensors = tuple(torch.as_tensor(tensor) for tensor in tensors)
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise RuntimeError(f"{scenario} contains non-finite state after checkpoint validation")
    return tuple(tensor.clone() for tensor in tensors)


def _build(scenario, *, add_camera):
    if scenario == "coffee_water":
        config = coffee_water.CoffeeWaterScenarioConfig()
        runtime = coffee_water.build_scene(config, is_recording=add_camera)
        controller = coffee_water.CoffeeWaterController(config.task)
    elif scenario == "table_wiping":
        config = table_wiping.TableWipingScenarioConfig()
        runtime = table_wiping.build_scene(config, add_camera=add_camera)
        controller = table_wiping.TableWipingController(config.task)
    elif scenario == "litter_scoop":
        config = litter_scoop.LitterScoopScenarioConfig()
        runtime = litter_scoop.build_scene(config, add_camera=add_camera)
        controller = litter_scoop.LitterScoopController(config.task)
    elif scenario == "garment_folding":
        config = garment_folding.GarmentFoldingScenarioConfig()
        runtime = garment_folding.build_scene(config, add_camera=add_camera)
        controller = garment_folding.GarmentFoldingController(config.task)
    else:
        config = butter_spreading.ButterSpreadingScenarioConfig()
        runtime = butter_spreading.build_scene(config, add_camera=add_camera)
        controller = butter_spreading.ButterSpreadingController(config.task, config.solver.dt)
    return runtime, controller


def _controller_state(scenario, controller, runtime):
    return controller.get_state(runtime) if scenario == "coffee_water" else controller.get_state()


def _assert_restored(scenario, runtime, expected):
    for actual_tensor, expected_tensor in zip(_observable_state(scenario, runtime), expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=1e-6, rtol=1e-6)


def _record_frame(runtime, video, *, phase):
    print(f"checkpoint phase: {phase}", flush=True)
    if video is not None:
        rgb, *_ = runtime.camera.render()
        video.write(rgb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        choices=("coffee_water", "table_wiping", "litter_scoop", "garment_folding", "butter_spreading"),
    )
    parser.add_argument("--record", action="store_true", help="Record diagnostic phase frames at 5 FPS")
    parser.add_argument(
        "--output", type=Path, default=Path("out/multiphysics_checkpoint"), help="Diagnostic output root"
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1)
    video = None
    try:
        build_started = time.perf_counter()
        runtime, controller = _build(args.scenario, add_camera=args.record)
        print(f"checkpoint build_seconds={time.perf_counter() - build_started:.3f}", flush=True)
        if args.record:
            args.output.mkdir(parents=True, exist_ok=True)
            video = VideoEncoder(str(args.output / f"{args.scenario}-checkpoint.mp4"), fps=5, codec="libx264")

        controller.reset(runtime)
        initial_observables = _observable_state(args.scenario, runtime)
        _record_frame(runtime, video, phase="initial")
        controller.step(runtime)
        checkpoint_scene_state = runtime.scene.get_state()
        checkpoint_controller_state = _controller_state(args.scenario, controller, runtime)
        checkpoint_observables = _observable_state(args.scenario, runtime)
        _record_frame(runtime, video, phase="checkpoint")
        controller.step(runtime)
        _observable_state(args.scenario, runtime)
        _record_frame(runtime, video, phase="advance")

        runtime.scene.restore(checkpoint_scene_state)
        controller.set_state(runtime, checkpoint_controller_state)
        _assert_restored(args.scenario, runtime, checkpoint_observables)
        if controller.step_index != checkpoint_controller_state.step_index:
            raise RuntimeError("Controller task phase was not restored.")
        _record_frame(runtime, video, phase="restore")
        controller.step(runtime)
        _observable_state(args.scenario, runtime)
        _record_frame(runtime, video, phase="continuation")

        runtime.scene.reset()
        controller.reset(runtime)
        _assert_restored(args.scenario, runtime, initial_observables)
        if controller.step_index != 0:
            raise RuntimeError("Controller task phase was not reset.")
        _record_frame(runtime, video, phase="reset")
        controller.step(runtime)
        _observable_state(args.scenario, runtime)
        _record_frame(runtime, video, phase="restart")
        print(f"{args.scenario} checkpoint/reset validation passed", flush=True)
    finally:
        if video is not None:
            video.close()
        gs.destroy()


if __name__ == "__main__":
    main()
