"""Preview checkpoint sequences and record the current view as an MP4 video.

Run ``python -m examples.genesis_origin_example.rendering.replay_checkpoints PATH`` where PATH contains scene.pkl and
frame-*.npz.
--surface reconstructs liquid surfaces with concentration colors; --particles displays liquid particles at lower cost.
These flags are mutually exclusive; omitting both uses the recorded display modes. Existing checkpoints support either.
Space plays/pauses, Left/Right step or hold to scrub, R starts/stops a video, I expands the viewer help, and Esc exits.
Holding an arrow repeats up to 30 checkpoints per second after a short delay. Mouse orbit, pan, and zoom stay available
while paused. Playback samples checkpoints at 50 frames per second; --fps selects another rate. --sequential displays
every checkpoint for inspection at the cost of slower playback. Recording retains simulation duration when playback
skips checkpoints; manual seeks each contribute one checkpoint interval. Pauses add no frames. Video duration rounds
up to one output frame; slower checkpoint rates repeat video frames.
--cache-surfaces DIRECTORY reuses reconstructed surfaces across runs, trading disk space for faster replay.
--precompute fills that cache at the selected playback rate without opening a viewer; use it before repeated playback.
Resize the window before pressing R: each video keeps the window resolution from the start of its recording.
Constant-quality encoding preserves detail at the cost of longer encoding time and potentially larger files.
Scene descriptions use pickle and must come from a trusted recording process with the same Genesis checkout/assets.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from fractions import Fraction
import math
from pathlib import Path
from queue import SimpleQueue
import time

import pyglet

import genesis as gs
from genesis.ext.pyrender import Renderer
from genesis.utils.checkpoint_replay import CheckpointReader, ReplayFrame, apply_frame
from genesis.utils.video_encoder import VideoEncoder
from genesis.vis.keybindings import Key, KeyAction, Keybind
from genesis.vis.viewer_plugins import ViewerPlugin


class Command(Enum):
    PLAY_PAUSE = auto()
    PREVIOUS = auto()
    NEXT = auto()
    RELEASE_PREVIOUS = auto()
    RELEASE_NEXT = auto()
    STOP_SEEK = auto()
    RECORD = auto()
    QUIT = auto()


@dataclass
class Playback:
    """A sampled checkpoint cursor with exact single-checkpoint manual seeks."""

    dt: float
    frame_count: int
    fps: int = 50
    is_sequential: bool = False
    frame_index: int = 0
    sample_origin: int = 0
    is_playing: bool = False
    seek_offset: int = 0
    next_seek_time: float = 0.0

    @property
    def time(self) -> float:
        return self.frame_index * self.dt

    def next_index(self, index: int) -> int:
        """Return the next output sample's checkpoint, including the sequence's final checkpoint."""
        if self.is_sequential:
            return min(index + 1, self.frame_count - 1)
        ratio = Fraction(str(self.dt)) * self.fps
        sample = math.ceil((index - self.sample_origin + 1) * ratio)
        return min(self.sample_origin + sample // ratio, self.frame_count - 1)

    def toggle(self) -> None:
        self.seek_offset = 0
        if self.is_playing:
            self.is_playing = False
        else:
            if self.frame_index == self.frame_count - 1:
                self.frame_index = 0
                self.sample_origin = 0
            self.is_playing = True

    def seek(self, offset: int) -> None:
        self.is_playing = False
        self.frame_index = min(max(self.frame_index + offset, 0), self.frame_count - 1)

    def hold(self, offset: int, now: float) -> None:
        self.seek(offset)
        self.seek_offset = offset
        self.next_seek_time = now + 0.2

    def release(self, offset: int) -> None:
        if self.seek_offset == offset:
            self.seek_offset = 0

    def repeat_seek(self, now: float) -> bool:
        if self.seek_offset and now >= self.next_seek_time:
            self.seek(self.seek_offset)
            # Advance one checkpoint per display opportunity, including after a slow frame.
            self.next_seek_time = now + 1.0 / 30.0
            return True
        return False


class ReplayControls(ViewerPlugin):
    """Queue viewer input and render status on the window's OpenGL thread."""

    def __init__(self):
        super().__init__()
        self.commands: SimpleQueue[Command] = SimpleQueue()
        self.status = "Loading checkpoints"
        self.label: pyglet.text.Label | None = None
        self.render_target: Renderer | None = None
        self.is_frame_drawn = False

    def build(self, viewer, camera, scene):
        super().build(viewer, camera, scene)
        self.render_target = Renderer(
            viewport_width=viewer.viewport_size[0],
            viewport_height=viewer.viewport_size[1],
            jit=viewer.gs_context.jit,
            point_size=viewer.render_flags["point_size"],
        )
        viewer.remove_keybind("record_video")
        viewer.register_keybinds(
            Keybind("play_pause", Key.SPACE, callback=self.commands.put, args=(Command.PLAY_PAUSE,)),
            Keybind(
                "previous_checkpoint_hold_to_rewind",
                Key.LEFT,
                key_action=KeyAction.PRESS,
                callback=self.commands.put,
                args=(Command.PREVIOUS,),
            ),
            Keybind(
                "next_checkpoint_hold_to_advance",
                Key.RIGHT,
                key_action=KeyAction.PRESS,
                callback=self.commands.put,
                args=(Command.NEXT,),
            ),
            Keybind("stop_rewind", Key.LEFT, callback=self.commands.put, args=(Command.RELEASE_PREVIOUS,)),
            Keybind("stop_advance", Key.RIGHT, callback=self.commands.put, args=(Command.RELEASE_NEXT,)),
            Keybind("start_stop_recording", Key.R, callback=self.commands.put, args=(Command.RECORD,)),
            Keybind("quit_and_save_recording", Key.ESCAPE, callback=self.commands.put, args=(Command.QUIT,)),
        )

    def on_draw(self):
        if self.label is None:
            self.label = pyglet.text.Label(font_size=12, anchor_x="right", anchor_y="top", color=(255, 255, 255, 255))
        self.label.text = self.status
        self.label.x = self.viewer.viewport_size[0] - 12
        self.label.y = self.viewer.viewport_size[1] - 12
        self.label.draw()
        self.is_frame_drawn = True

    def on_close(self):
        self.commands.put(Command.QUIT)
        if self.render_target is not None:
            self.render_target.delete()
            self.render_target = None
        if self.label is not None:
            self.label.delete()
            self.label = None

    def on_deactivate(self):
        self.commands.put(Command.STOP_SEEK)


class ReplayPlayer:
    """Display sampled checkpoints and stream video with one frame of read-ahead in host memory."""

    def __init__(self, reader: CheckpointReader, scene: gs.Scene, output: Path, *, fps=50, is_sequential=False):
        self.reader = reader
        self.scene = scene
        self.output = Path(output)
        self.fps = fps
        # Rational intervals keep video samples aligned with checkpoint boundaries over long recordings.
        self.recording_frame_ratio = Fraction(str(reader.description.dt)) * self.fps
        self.recorded_checkpoint_count = 0
        self.recorded_frame_count = 0
        self.playback = Playback(reader.description.dt, len(reader.frames), fps=fps, is_sequential=is_sequential)
        self.next_play_time = 0.0
        self.controls = ReplayControls()
        self.encoder: VideoEncoder | None = None
        self.video_path: Path | None = None
        self.displayed_frame: int | None = None
        self.loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-reader")
        self.pending_frame: Future[ReplayFrame] | None = None
        self.is_viewer_updated = False
        self.scene.viewer.add_plugin(self.controls)
        self.show_frame(index=0)

    @property
    def is_recording(self) -> bool:
        return self.encoder is not None

    def show_frame(self, index: int, frame: ReplayFrame | None = None, *, checkpoint_count: int = 1) -> None:
        self.playback.frame_index = index
        self.update_status()
        if index != self.displayed_frame:
            if frame is None:
                frame = self.reader.read_frame(index)
            self.controls.is_frame_drawn = False
            with self.scene.viewer.lock:
                apply_frame(self.scene, frame)
            self.displayed_frame = index
            self.is_viewer_updated = True
            self.capture_frame(checkpoint_count)

    def capture_frame(self, checkpoint_count: int = 1) -> None:
        if self.encoder is not None and self.scene.viewer.is_alive():
            self.recorded_checkpoint_count += checkpoint_count
            self.playback.sample_origin = self.playback.frame_index + 1 - self.recorded_checkpoint_count
            frame_count = math.ceil(self.recorded_checkpoint_count * self.recording_frame_ratio)
            if self.recorded_frame_count < frame_count:
                rgb, *_ = self.scene.viewer.render_offscreen(
                    self.controls.camera, self.controls.render_target, skip_markers=True
                )
                while self.recorded_frame_count < frame_count:
                    self.encoder.write(rgb)
                    self.recorded_frame_count += 1

    def stop_recording(self) -> None:
        if self.encoder is not None:
            try:
                self.encoder.close()
            finally:
                self.encoder = None
                self.playback.sample_origin = 0
            if self.video_path.is_file():
                gs.logger.info(f"Saved video: {self.video_path}")
            else:
                gs.logger.info("Recording ended before playback produced a video frame.")

    def handle_command(self, command: Command, now: float) -> None:
        if (
            command in (Command.PLAY_PAUSE, Command.PREVIOUS, Command.NEXT, Command.RECORD)
            and self.pending_frame is not None
        ):
            self.pending_frame.cancel()
            self.pending_frame = None
        if command == Command.PLAY_PAUSE:
            self.playback.toggle()
            self.next_play_time = now + max(self.playback.dt, 1.0 / self.fps)
            self.show_frame(self.playback.frame_index)
        elif command in (Command.PREVIOUS, Command.NEXT):
            self.playback.hold(offset=-1 if command == Command.PREVIOUS else 1, now=now)
            self.show_frame(self.playback.frame_index)
        elif command in (Command.RELEASE_PREVIOUS, Command.RELEASE_NEXT):
            self.playback.release(offset=-1 if command == Command.RELEASE_PREVIOUS else 1)
        elif command == Command.STOP_SEEK:
            self.playback.seek_offset = 0
        elif command == Command.RECORD:
            if self.is_recording:
                self.stop_recording()
            else:
                self.output.mkdir(parents=True, exist_ok=True)
                index = 1
                while (self.output / f"recording-{index:04d}.mp4").exists():
                    index += 1
                self.video_path = self.output / f"recording-{index:04d}.mp4"
                width, height = self.controls.viewer.viewport_size
                self.controls.render_target.viewport_width = width
                self.controls.render_target.viewport_height = height
                self.recorded_checkpoint_count = 0
                self.recorded_frame_count = 0
                self.encoder = VideoEncoder(
                    str(self.video_path),
                    self.fps,
                    codec="libx264",
                    bitrate=0,
                    codec_options={"crf": "16", "preset": "medium", "tune": "zerolatency"},
                )
                self.capture_frame()
                gs.logger.info(
                    f"Recording playback to {self.video_path} at {width}x{height}, {self.fps:g} frames per second"
                )

    def tick(self, now: float) -> None:
        """Display the next sampled checkpoint once the current frame has been drawn and its interval elapsed."""
        if self.controls.is_frame_drawn and self.playback.repeat_seek(now):
            self.show_frame(self.playback.frame_index)
        if self.playback.is_playing:
            frame = None
            if (
                self.controls.is_frame_drawn
                and now >= self.next_play_time
                and self.pending_frame is not None
                and self.pending_frame.done()
            ):
                frame = self.pending_frame.result()
                self.pending_frame = None
            if self.pending_frame is None:
                current = frame.index if frame is not None else self.playback.frame_index
                index = self.playback.next_index(current)
                if index > current:
                    self.pending_frame = self.loader.submit(self.reader.read_frame, index)
            if frame is not None:
                self.show_frame(frame.index, frame, checkpoint_count=frame.index - self.playback.frame_index)
                self.next_play_time = now + max(self.playback.dt, 1.0 / self.fps)
            if self.playback.frame_index == self.playback.frame_count - 1 and self.controls.is_frame_drawn:
                self.playback.is_playing = False
                self.stop_recording()
        self.update_status()

    def update_status(self) -> None:
        state = "PLAYING" if self.playback.is_playing else "PAUSED"
        recording = " | REC" if self.is_recording else ""
        self.controls.status = (
            f"Frame {self.playback.frame_index}/{self.playback.frame_count - 1} | "
            f"{self.playback.time:.3f}s | {state}{recording}"
        )

    def run(self) -> None:
        try:
            while self.scene.viewer.is_alive():
                self.is_viewer_updated = False
                while not self.controls.commands.empty():
                    command = self.controls.commands.get_nowait()
                    if command == Command.QUIT:
                        return
                    self.handle_command(command, time.perf_counter())
                self.tick(time.perf_counter())
                if not self.is_viewer_updated and self.scene.viewer.is_alive():
                    self.scene.viewer.update()
                time.sleep(0.001)
        finally:
            try:
                self.stop_recording()
            finally:
                self.loader.shutdown(wait=True, cancel_futures=True)
                self.scene.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing scene.pkl and frame-*.npz.")
    parser.add_argument("--output", type=Path, help="Video output directory (default: DIRECTORY/recordings).")
    liquid_display = parser.add_mutually_exclusive_group()
    liquid_display.add_argument(
        "--surface",
        dest="liquid_vis_mode",
        action="store_const",
        const="recon",
        help="Override recorded liquid display modes with reconstructed surfaces, adding reconstruction cost.",
    )
    liquid_display.add_argument(
        "--particles",
        dest="liquid_vis_mode",
        action="store_const",
        const="particle",
        help="Override recorded liquid display modes with particles for faster inspection of individual samples.",
    )
    parser.add_argument(
        "--fps", type=int, default=50, help="Preview and video rate; higher rates retain more motion detail."
    )
    parser.add_argument(
        "--sequential",
        dest="is_sequential",
        action="store_true",
        help="Display every checkpoint for inspection, including checkpoints between video samples.",
    )
    parser.add_argument(
        "--cache-surfaces",
        type=Path,
        help="Store liquid surface meshes here for faster repeated replay at extra disk cost.",
    )
    parser.add_argument(
        "--precompute",
        dest="is_precomputing",
        action="store_true",
        help="Prepare sampled surface meshes without a viewer; requires --cache-surfaces.",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive.")
    if args.is_precomputing and args.cache_surfaces is None:
        parser.error("--precompute requires --cache-surfaces DIRECTORY.")
    reader = CheckpointReader(args.directory, surface_cache=args.cache_surfaces, liquid_vis_mode=args.liquid_vis_mode)
    if args.is_precomputing and not any(
        issubclass(entity.material_type, gs.materials.PBSTF.Liquid) and entity.surface.vis_mode == "recon"
        for entity in reader.description.entities
    ):
        parser.error("--precompute requires liquid surfaces; select --surface or use a surface-mode recording.")
    gs.init(backend=gs.cuda, precision=reader.description.precision, logging_level="info")
    scene = reader.build_scene(is_viewer_shown=not args.is_precomputing)
    if args.is_precomputing:
        playback = Playback(reader.description.dt, len(reader.frames), fps=args.fps, is_sequential=args.is_sequential)
        try:
            while True:
                apply_frame(scene, reader.read_frame(playback.frame_index))
                gs.logger.info(f"Cached checkpoint {playback.frame_index}/{playback.frame_count - 1}")
                if playback.frame_index == playback.frame_count - 1:
                    break
                playback.frame_index = playback.next_index(playback.frame_index)
        finally:
            scene.destroy()
        return
    player = ReplayPlayer(
        reader, scene, args.output or args.directory / "recordings", fps=args.fps, is_sequential=args.is_sequential
    )
    player.run()


if __name__ == "__main__":
    main()
