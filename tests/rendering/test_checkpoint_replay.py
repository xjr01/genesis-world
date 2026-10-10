from dataclasses import replace
import math
import pickle
import time

import numpy as np

import av
import pytest

from examples.genesis_origin_example.rendering.replay_checkpoints import Command, Playback, ReplayPlayer

import genesis as gs
from genesis.utils.checkpoint_replay import CheckpointReader, CheckpointWriter, apply_frame
from genesis.utils.element import create_tetrahedral_grid
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key

from ..conftest import IS_INTERACTIVE_VIEWER_AVAILABLE, SKIP_NO_VIEWER
from ..utils import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cuda])
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("vis_mode", [None, "particle", "recon"])
def test_checkpoint_roundtrip(tmp_path, n_envs, vis_mode, show_viewer):
    scene = gs.Scene(
        pbd_options=gs.options.PBDUnifiedOptions(
            particle_size=0.02,
        ),
        pbstf_options=gs.options.PBSTFOptions(
            particle_size=0.02,
            lower_bound=(-0.4, -0.4, -0.1),
            upper_bound=(0.4, 0.4, 0.4),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -0.65, 0.45),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    rigid = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.15, 0.0, 0.1),
            size=(0.08, 0.08, 0.08),
        ),
    )
    liquid = None
    if vis_mode is not None:
        liquid = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.1),
                size=(0.08, 0.08, 0.08),
            ),
            material=gs.materials.PBSTF.Liquid(
                sampler="regular",
                c_init=0.0,
            ),
            surface=gs.surfaces.Default(
                vis_mode=vis_mode,
            ),
        )
    vertices, elements = create_tetrahedral_grid(
        lower=(-0.04, -0.04, -0.04), upper=(0.04, 0.04, 0.04), resolution=(2, 2, 2)
    )
    sponge = scene.add_entity(
        morph=gs.morphs.TetrahedralMesh(
            pos=(0.15, 0.0, 0.1),
            vertices=vertices,
            elements=elements,
        ),
        material=gs.materials.PBD.Elastic(),
    )
    camera = scene.add_camera(res=(128, 128), pos=(0.0, -0.65, 0.45), lookat=(0.0, 0.0, 0.1))
    scene.build(n_envs=n_envs)
    writer = CheckpointWriter(scene, tmp_path / "checkpoints", dt=0.002)
    writer.write_frame()
    image_initial, *_ = camera.render()
    rigid_initial = tensor_to_array(rigid.get_pos())
    if liquid is not None:
        liquid_initial = tensor_to_array(liquid.get_particles_pos())
    sponge_initial = tensor_to_array(sponge.get_particles_pos())

    rigid.set_pos(rigid.get_pos() + 0.03)
    if liquid is not None:
        liquid.set_particles_pos(liquid.get_particles_pos() + 0.04)
        liquid.set_particles_concentration(0.75)
        is_active = liquid.get_particles_active()
        is_active[..., : liquid.n_particles // 2] = False
        liquid.set_particles_active(is_active)
    sponge.set_particles_pos(sponge.get_particles_pos() * 1.2)
    writer.write_frame()
    scene.visualizer.update()
    image_changed, *_ = camera.render()
    assert np.abs(image_initial / 255.0 - image_changed / 255.0).max() > 0.1

    reader = CheckpointReader(writer.directory)
    assert_equal(len(reader.frames), 2)
    assert_equal(reader.read_frame(1).time, 0.002)
    for index in (0, 1, 0):
        apply_frame(scene, reader.read_frame(index))
        assert_allclose(rigid.get_pos(), rigid_initial + index * 0.03, atol=1e-7)
        assert_allclose(sponge.get_particles_pos(), sponge_initial * (1.0 + index * 0.2), atol=1e-7)
        if liquid is not None:
            assert_allclose(liquid.get_particles_pos(), liquid_initial + index * 0.04, atol=1e-7)
            assert_equal(liquid.get_particles_concentration(), index * 0.75)
            assert_equal(liquid.get_particles_active(), is_active if index else True)
        image, *_ = camera.render()
        assert_allclose(image, image_changed if index else image_initial, atol=1.0)

    with pytest.raises(FileExistsError, match="must be empty"):
        CheckpointWriter(scene, writer.directory, dt=0.002)
    with pytest.raises(IndexError):
        reader.read_frame(-1)
    with pytest.raises(IndexError):
        reader.read_frame(2)
    (writer.directory / "frame-00000002.npz.part").write_bytes(b"interrupted write")
    assert_equal(len(CheckpointReader(writer.directory).frames), 2)
    reader.frames[1].rename(writer.directory / "frame-00000002.npz")
    with pytest.raises(ValueError, match="Missing or misnumbered"):
        CheckpointReader(writer.directory)
    (writer.directory / "frame-00000002.npz").rename(reader.frames[1])
    reader.frames[1].write_bytes(b"broken frame")
    with pytest.raises(ValueError, match="Cannot read checkpoint"):
        reader.read_frame(1)
    with (writer.directory / "scene.pkl").open("wb") as stream:
        pickle.dump(replace(reader.description, version=2), stream)
    with pytest.raises(ValueError, match="Unsupported checkpoint scene format"):
        CheckpointReader(writer.directory)


@pytest.mark.required
def test_playback_timeline():
    playback = Playback(dt=0.002, frame_count=101)
    assert_equal(playback.is_playing, False)
    playback.toggle()
    assert_equal(playback.is_playing, True)
    playback.frame_index = 25
    playback.toggle()
    assert_equal(playback.is_playing, False)
    playback.seek(offset=-1)
    assert_equal(playback.frame_index, 24)
    playback.seek(offset=1)
    assert_equal(playback.time, 0.05)
    playback.toggle()
    assert_equal(playback.frame_index, 25)
    playback.seek(offset=1000)
    assert_equal(playback.frame_index, 100)
    playback.toggle()
    assert_equal(playback.frame_index, 0)
    playback.seek(offset=-1)
    assert_equal(playback.frame_index, 0)
    playback.hold(offset=1, now=60.0)
    assert_equal(playback.frame_index, 1)
    assert_equal(playback.repeat_seek(now=60.1), False)
    assert_equal(playback.repeat_seek(now=60.21), True)
    assert_equal(playback.frame_index, 2)
    playback.repeat_seek(now=61.0)
    assert_equal(playback.frame_index, 3)
    playback.hold(offset=-1, now=61.0)
    playback.release(offset=1)
    playback.repeat_seek(now=61.21)
    assert_equal(playback.frame_index, 1)
    playback.release(offset=-1)
    assert_equal(playback.repeat_seek(now=62.0), False)
    assert_equal(playback.frame_index, 1)
    playback.hold(offset=-1, now=62.0)
    playback.repeat_seek(now=63.0)
    assert_equal(playback.frame_index, 0)
    playback.toggle()
    assert_equal(playback.repeat_seek(now=64.0), False)


@pytest.mark.required
@pytest.mark.skipif(not IS_INTERACTIVE_VIEWER_AVAILABLE, reason=SKIP_NO_VIEWER)
@pytest.mark.parametrize(
    "dt, sample_indices",
    [(0.002, (0, 10, 20)), (0.003, (0, 6, 13, 20)), (0.04, np.repeat(np.arange(21), 2))],
)
def test_replay_recording(tmp_path, dt, sample_indices):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=(960, 720),
            run_in_thread=False,
            realtime_factor=None,
            camera_pos=(0.3, -0.8, 0.5),
            camera_lookat=(0.1, 0.0, 0.1),
        ),
        show_viewer=True,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    scene.build()
    writer = CheckpointWriter(scene, tmp_path / "checkpoints", dt=dt)
    for index in range(21):
        entity.set_pos((index * 0.01, 0.0, 0.1))
        writer.write_frame()
    player = ReplayPlayer(CheckpointReader(writer.directory), scene, tmp_path / "videos")
    try:
        player.controls.viewer.on_key_press(Key.RIGHT, modifiers=0)
        player.handle_command(player.controls.commands.get_nowait(), now=0.0)
        time.sleep(1.0 / scene.viewer.refresh_rate)
        scene.viewer.update()
        player.tick(now=0.21)
        assert_allclose(entity.get_pos(), (0.02, 0.0, 0.1), atol=1e-7)
        player.controls.viewer.on_key_release(Key.RIGHT, modifiers=0)
        player.handle_command(player.controls.commands.get_nowait(), now=0.21)
        player.tick(now=0.5)
        assert_equal(player.playback.frame_index, 2)
        player.controls.viewer.on_key_press(Key.LEFT, modifiers=0)
        player.handle_command(player.controls.commands.get_nowait(), now=0.5)
        time.sleep(1.0 / scene.viewer.refresh_rate)
        scene.viewer.update()
        player.tick(now=0.71)
        assert_allclose(entity.get_pos(), (0.0, 0.0, 0.1), atol=1e-7)
        player.controls.on_deactivate()
        player.handle_command(player.controls.commands.get_nowait(), now=0.71)
        player.show_frame(index=5)
        player.tick(now=1.0)
        assert_equal(player.playback.frame_index, 5)
        scene.viewer.set_camera_pose(pos=(-0.4, -0.5, 0.25), lookat=(0.1, 0.0, 0.1))
        camera_pose = scene.viewer.camera_pose
        for is_recording in (False, True):
            player.show_frame(index=0)
            if is_recording:
                player.controls.viewer.set_size(width=800, height=600)
                player.controls.viewer.dispatch_events()
                player.handle_command(Command.RECORD, now=0.0)
            player.handle_command(Command.PLAY_PAUSE, now=0.0)
            visited_frames = [0]
            is_pause_checked = False
            start = time.perf_counter()
            while player.playback.is_playing and time.perf_counter() - start < 15.0:
                # Large clock jumps exercise playback under arbitrarily slow loading or rendering.
                player.tick(now=100.0 * (time.perf_counter() - start))
                index = player.playback.frame_index
                assert_allclose(entity.get_pos(), (index * 0.01, 0.0, 0.1), atol=1e-7)
                if index != visited_frames[-1]:
                    visited_frames.append(index)
                if index == 5 and not is_pause_checked:
                    player.handle_command(Command.PLAY_PAUSE, now=100.0)
                    if is_recording:
                        player.controls.viewer.set_size(width=800, height=480)
                        player.controls.viewer.dispatch_events()
                    player.tick(now=1000.0)
                    assert_equal(player.playback.frame_index, 5)
                    assert_allclose(scene.viewer.camera_pose, camera_pose, atol=1e-7)
                    player.handle_command(Command.PLAY_PAUSE, now=1000.0)
                    is_pause_checked = True
                scene.viewer.update()
                time.sleep(0.001)
            assert_equal(visited_frames, np.arange(21))
            assert_equal(is_pause_checked, True)
            assert_equal(player.playback.is_playing, False)
            assert_equal(player.is_recording, False)
        with av.open(str(player.video_path)) as video:
            stream = video.streams.video[0]
            frames = []
            timestamps = []
            for frame in video.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
                timestamps.append(frame.pts * frame.time_base.numerator / frame.time_base.denominator)
            assert_equal(len(frames), len(sample_indices))
            frame_interval = stream.average_rate.denominator / stream.average_rate.numerator
            assert_allclose(frame_interval, 0.02, atol=1e-9)
            assert_allclose(timestamps, np.arange(len(sample_indices)) * 0.02, atol=1e-6)
            assert_equal((stream.width, stream.height), (800, 600))
            duration = stream.duration * stream.time_base.numerator / stream.time_base.denominator
            assert_allclose(duration, len(sample_indices) * 0.02, atol=1e-6)
            assert 0.0 <= duration - 21 * dt < 0.02
            for frame, index in zip(frames, sample_indices):
                player.show_frame(index)
                reference, *_ = scene.viewer.render_offscreen(
                    player.controls.camera, player.controls.render_target, skip_markers=True
                )
                assert np.abs(frame / 255.0 - reference / 255.0).mean() < 1.0 / 255.0
        first_path = player.video_path
        player.show_frame(index=11)
        player.handle_command(Command.RECORD, now=200.0)
        player.handle_command(Command.NEXT, now=200.0)
        player.handle_command(Command.PREVIOUS, now=200.0)
        player.handle_command(Command.RELEASE_PREVIOUS, now=200.0)
        player.tick(now=1000.0)
        player.handle_command(Command.RECORD, now=1000.0)
        assert player.video_path != first_path
        with av.open(str(player.video_path)) as video:
            stream = video.streams.video[0]
            assert_equal((stream.width, stream.height), (800, 480))
            assert_equal(len(list(video.decode(video=0))), math.ceil(3 * dt * 50))
        player.handle_command(Command.RECORD, now=1000.0)
        reference, *_ = scene.viewer.render_offscreen(
            player.controls.camera, player.controls.render_target, skip_markers=True
        )
        scene.viewer.stop()
        player.run()
        with av.open(str(player.video_path)) as video:
            frames = [frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)]
            assert_equal(len(frames), math.ceil(dt * 50))
            for frame in frames:
                assert np.abs(frame / 255.0 - reference / 255.0).mean() < 1.0 / 255.0
    finally:
        player.stop_recording()
        player.loader.shutdown(wait=True, cancel_futures=True)
        scene.destroy()
