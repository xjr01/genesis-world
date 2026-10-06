"""Exercise paired Scene and task-controller reset/checkpoint on one normalized scenario."""

import argparse

import torch

import genesis as gs


def _assert_finite(scenario, runtime):
    if scenario == "coffee_water":
        entities = (runtime.coffee, runtime.water)
        tensors = tuple(tensor for entity in entities for tensor in (entity.get_particles_pos(), entity.get_particles_vel()))
    elif scenario == "table_wiping":
        tensors = (runtime.liquid.get_particles_pos(), runtime.sponge.get_particles_pos())
    elif scenario == "litter_scoop":
        entities = (runtime.sand, runtime.water)
        tensors = tuple(tensor for entity in entities for tensor in (entity.get_particles_pos(), entity.get_particles_vel()))
    elif scenario == "garment_folding":
        state = runtime.garment.get_state()
        tensors = (state.pos, state.vel)
    else:
        entities = (runtime.bread, runtime.butter)
        tensors = tuple(tensor for entity in entities for tensor in (entity.get_particles_pos(), entity.get_particles_vel()))
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise RuntimeError(f"{scenario} contains non-finite state after checkpoint validation")


def _build(scenario):
    if scenario == "coffee_water":
        from examples.multiphysics.coffee_water import CoffeeWaterController, CoffeeWaterScenarioConfig, build_scene

        config = CoffeeWaterScenarioConfig()
        runtime = build_scene(config)
        controller = CoffeeWaterController(config.task)
    elif scenario == "table_wiping":
        from examples.multiphysics.table_wiping import TableWipingController, TableWipingScenarioConfig, build_scene

        config = TableWipingScenarioConfig()
        runtime = build_scene(config)
        controller = TableWipingController(config.task)
    elif scenario == "litter_scoop":
        from examples.multiphysics.litter_scoop import LitterScoopController, LitterScoopScenarioConfig, build_scene

        config = LitterScoopScenarioConfig()
        runtime = build_scene(config)
        controller = LitterScoopController(config.task)
    elif scenario == "garment_folding":
        from examples.multiphysics.garment_folding import (
            GarmentFoldingController,
            GarmentFoldingScenarioConfig,
            build_scene,
        )

        config = GarmentFoldingScenarioConfig()
        runtime = build_scene(config)
        controller = GarmentFoldingController(config.task)
    else:
        from examples.multiphysics.butter_spreading import (
            ButterSpreadingController,
            ButterSpreadingScenarioConfig,
            build_scene,
        )

        config = ButterSpreadingScenarioConfig()
        runtime = build_scene(config)
        controller = ButterSpreadingController(config.task, config.solver.dt)
    return runtime, controller


def _controller_state(scenario, controller, runtime):
    return controller.get_state(runtime) if scenario == "coffee_water" else controller.get_state()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        choices=("coffee_water", "table_wiping", "litter_scoop", "garment_folding", "butter_spreading"),
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1)
    runtime, controller = _build(args.scenario)

    controller.reset(runtime)
    initial_scene_state = runtime.scene.get_state()
    controller.step(runtime)
    checkpoint_scene_state = runtime.scene.get_state()
    checkpoint_controller_state = _controller_state(args.scenario, controller, runtime)
    controller.step(runtime)
    _assert_finite(args.scenario, runtime)

    runtime.scene.restore(checkpoint_scene_state)
    controller.set_state(runtime, checkpoint_controller_state)
    controller.step(runtime)
    _assert_finite(args.scenario, runtime)

    runtime.scene.reset(initial_scene_state)
    controller.reset(runtime)
    controller.step(runtime)
    _assert_finite(args.scenario, runtime)
    print(f"{args.scenario} checkpoint/reset validation passed")


if __name__ == "__main__":
    main()
