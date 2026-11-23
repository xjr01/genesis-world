"""Serve a local cloth-state picker and action-plan editor for Genesis outputs."""

import datetime
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlparse

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from .action_plan import ActionPlan, PathLimits, compile_action_plan
from .drafts import default_plan
from .ik_options import IKOptions
from .material import MaterialAtlas, material_atlas
from .plan_io import load_compiled, save_compiled, write_json
from .points import Selections
from .preview import Camera, pose_preview, robot_preview
from .trajectory import file_hash
from .trajectory_edit import PathEdits, duration_scale, edit_path
from .workspace import Preparation, read_workspace


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    compiled: str
    edits: PathEdits


class IKRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    compiled: str
    options: IKOptions = Field(default_factory=IKOptions)


class PoseRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    source_frame: StrictInt
    hand: Literal["left", "right"]
    role: Literal["grasp", "placement"]
    quat_wxyz: tuple[float, float, float, float]


class DraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan: ActionPlan
    pull_axis: Literal["x", "y"] = "x"
    pull_sign: Literal[-1, 1] = -1


def compiled_folder(workspace: Path, relative: str):
    """Resolve existing action artifacts within their owning workspace."""
    path = (workspace / relative).resolve(strict=True)
    if not path.is_relative_to(workspace.resolve()) or not (path / "tcp-path.npz").is_file():
        raise ValueError("Choose a compiled directory within this workspace.")
    return path


@dataclass(frozen=True)
class WorkbenchData:
    workspace: Path
    metadata: Preparation
    positions: np.ndarray
    faces: np.ndarray
    rest_vertices: np.ndarray
    frames: np.ndarray
    robot_q: np.ndarray
    atlas: MaterialAtlas
    camera: Camera
    ipc_names: tuple[str, ...]
    ipc_transforms: np.ndarray | None

    @classmethod
    def load(cls, workspace: Path, replay: Path | None):
        metadata = read_workspace(workspace)
        with np.load(workspace / "cloth-state.npz", allow_pickle=False) as state:
            positions = state["cloth_pos"][None].copy()
            faces = state["faces"].copy()
            rest = state["rest_raw_xyz"].copy()
            frames = np.array([metadata.start_frame])
        with np.load(workspace / "seed.npz", allow_pickle=False) as state:
            robot_q = state["robot_q"][None].copy()
        replay = Path(metadata.replay) if replay is None and metadata.replay is not None else replay
        ipc_names, ipc_transforms = (), None
        if replay is not None:
            with np.load(replay, allow_pickle=False) as state:
                if state["cloth_sha256"].item() != metadata.cloth_sha256:
                    raise ValueError("Replay cloth hash differs from the workspace mesh.")
                positions = state["cloth_pos"].copy()
                frames = state["source_frames"].copy()
                order = [tuple(state["joint_names"].tolist()).index(name) for name in metadata.joint_names]
                robot_q = state["robot_q"][:, order].copy()
                if "ipc_link_names" in state and "ipc_link_transforms" in state:
                    ipc_names = tuple(state["ipc_link_names"].tolist())
                    ipc_transforms = state["ipc_link_transforms"].copy()
        if positions.shape != (len(frames), len(rest), 3) or not np.isfinite(positions).all():
            raise ValueError("Replay must contain finite positions matching the workspace topology.")
        if robot_q.shape != (len(frames), len(metadata.joint_names)) or not np.isfinite(robot_q).all():
            raise ValueError("Replay robot samples must match the frame count and joint names.")
        camera_file = workspace / "camera.json"
        camera = (
            Camera.model_validate_json(camera_file.read_text(encoding="utf-8")) if camera_file.exists() else Camera()
        )
        return cls(
            workspace,
            metadata,
            positions,
            faces,
            rest,
            frames,
            robot_q,
            material_atlas(rest),
            camera,
            ipc_names,
            ipc_transforms,
        )


class WorkbenchServer(HTTPServer):
    def __init__(self, address, data: WorkbenchData):
        self.data = data
        super().__init__(address, WorkbenchHandler)


class WorkbenchHandler(BaseHTTPRequestHandler):
    server: WorkbenchServer

    def send_payload(self, payload: bytes, content_type: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send_payload(
                (Path(__file__).resolve().parents[1] / "ui" / "workbench.html").read_bytes(), "text/html; charset=utf-8"
            )
            return
        if url.path in ("/render.js", "/workbench.js", "/workbench.css"):
            content_type = "text/css" if url.path.endswith(".css") else "text/javascript"
            self.send_payload((Path(__file__).resolve().parents[1] / "ui" / url.path[1:]).read_bytes(), content_type)
            return
        if url.path == "/api/state":
            data = self.server.data
            try:
                index = int(parse_qs(url.query).get("index", [0])[0])
                if not 0 <= index < len(data.frames):
                    raise ValueError("Frame index outside replay.")
                plan = json.loads((data.workspace / "plan.json").read_text(encoding="utf-8"))
                payload = {
                    "vertices": data.positions[index].tolist(),
                    "faces": data.faces.tolist(),
                    "rest_vertices": data.rest_vertices.tolist(),
                    "source_frame": data.frames[index].item(),
                    "frames": data.frames.tolist(),
                    "plan": plan,
                    "cloth_sha256": data.metadata.cloth_sha256,
                    "start_frame": data.metadata.start_frame,
                    "atlas": data.atlas.payload(),
                    "camera": data.camera.model_dump(),
                    "is_physics_ready": data.metadata.is_physics_ready,
                    "reference_path": "reference-path" if (data.workspace / "reference-path").exists() else None,
                }
                self.send_payload(json.dumps(payload, allow_nan=False).encode(), "application/json")
            except (ValueError, OSError) as error:
                self.send_payload(json.dumps({"error": str(error)}).encode(), "application/json", status=400)
            return
        if url.path in ("/api/pose", "/api/robot"):
            data = self.server.data
            try:
                query = parse_qs(url.query)
                index = int(query.get("index", [0])[0])
                if not 0 <= index < len(data.frames):
                    raise ValueError("Frame index outside replay.")
                urdf = (
                    Path(data.metadata.robot_path)
                    if data.metadata.robot_path is not None
                    else Path(data.metadata.asset_root) / "assets/robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf"
                )
                if url.path == "/api/robot":
                    payload = {
                        "meshes": robot_preview(
                            urdf, data.metadata.joint_names, data.robot_q[index], data.metadata.robot_base_pos_m
                        )
                    }
                else:
                    measured = (
                        None if data.ipc_transforms is None else dict(zip(data.ipc_names, data.ipc_transforms[index]))
                    )
                    payload = pose_preview(
                        urdf,
                        data.metadata.joint_names,
                        data.robot_q[index],
                        query.get("hand", ["right"])[0],
                        data.metadata.robot_base_pos_m,
                        measured,
                    )
                payload["source_frame"] = data.frames[index].item()
                self.send_payload(json.dumps(payload, allow_nan=False).encode(), "application/json")
            except (ValueError, OSError) as error:
                self.send_payload(json.dumps({"error": str(error)}).encode(), "application/json", status=400)
            return
        self.send_error(404)

    def do_POST(self):
        # Browser writes require a same-origin request to this localhost service.
        if self.headers.get("Origin") != "http://" + self.headers.get("Host", ""):
            self.send_error(403, "Same-origin browser request required")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024 * 1024:
                raise ValueError("Expected a JSON request of at most 1 MiB.")
            text = self.rfile.read(length).decode("utf-8")
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            data = self.server.data
            if self.path == "/api/points":
                selections = Selections.model_validate_json(text)
                if selections.source_frame != data.metadata.start_frame:
                    raise ValueError("Select points at the prepared plan start frame before saving.")
                folder = data.workspace / "edits"
                folder.mkdir(exist_ok=True)
                path = folder / f"points-{timestamp}.json"
                write_json(path, selections.model_dump())
                result = {"saved": str(path.resolve())}
            elif self.path == "/api/compile":
                plan = ActionPlan.model_validate_json(text)
                if plan.start_frame != data.metadata.start_frame or plan.fps != 60:
                    raise ValueError("Plan must match the prepared frame and 60 Hz action rate.")
                compiled = compile_action_plan(plan)
                folder = data.workspace / "actions" / timestamp
                save_compiled(folder, plan, compiled)
                result = {
                    "saved": str(folder.resolve()),
                    "accepted": compiled.is_accepted,
                    "left_pos": compiled.left.pos.tolist(),
                    "right_pos": compiled.right.pos.tolist(),
                    "source_frames": compiled.source_frames.tolist(),
                    "compiled": str(folder.relative_to(data.workspace)),
                    "scope": "TCP path only; use ik_preflight before export and physics",
                }
            elif self.path == "/api/path":
                request = IKRequest.model_validate_json(text)
                folder = compiled_folder(data.workspace, request.compiled)
                compiled = load_compiled(folder / "tcp-path.npz")
                result = {
                    "compiled": request.compiled,
                    "left_pos": compiled.left.pos.tolist(),
                    "right_pos": compiled.right.pos.tolist(),
                    "source_frames": compiled.source_frames.tolist(),
                    "accepted": compiled.is_accepted,
                }
            elif self.path == "/api/edit":
                request = EditRequest.model_validate_json(text)
                source_folder = compiled_folder(data.workspace, request.compiled)
                source = load_compiled(source_folder / "tcp-path.npz")
                compiled = edit_path(source, request.edits)
                plan = ActionPlan.model_validate_json((source_folder / "plan.json").read_text(encoding="utf-8"))
                plan.limits = PathLimits(
                    max_speed_mps=request.edits.limits.max_speed_mps, max_accel_mps2=request.edits.limits.max_accel_mps2
                )
                folder = data.workspace / "actions" / timestamp
                save_compiled(folder, plan, compiled)
                write_json(
                    folder / "edits.json",
                    {
                        "source_compiled": str(source_folder),
                        "source_tcp_path_sha256": file_hash(source_folder / "tcp-path.npz"),
                        "edits": request.edits.model_dump(),
                    },
                )
                report = {
                    "accepted": compiled.is_accepted,
                    "required_duration_scale": duration_scale(compiled, request.edits.limits),
                    "limits": request.edits.limits.model_dump(),
                    "left_max_speed_mps": compiled.left.max_speed_mps,
                    "right_max_speed_mps": compiled.right.max_speed_mps,
                    "left_max_accel_mps2": compiled.left.max_accel_mps2,
                    "right_max_accel_mps2": compiled.right.max_accel_mps2,
                }
                write_json(folder / "path-report.json", report)
                result = {
                    **report,
                    "compiled": str(folder.relative_to(data.workspace)),
                    "saved": str(folder.resolve()),
                    "left_pos": compiled.left.pos.tolist(),
                    "right_pos": compiled.right.pos.tolist(),
                    "source_frames": compiled.source_frames.tolist(),
                }
            elif self.path == "/api/ik":
                request = IKRequest.model_validate_json(text)
                folder = compiled_folder(data.workspace, request.compiled)
                options_file = folder / "ik-options.json"
                if (folder / "ik-report.json").exists():
                    raise ValueError("IK already checked; save a new path to change settings.")
                write_json(options_file, request.options.model_dump())
                environment = os.environ.copy()
                environment.setdefault("GS_CACHE_FILE_PATH", str(data.workspace / ".cache" / "genesis"))
                environment.setdefault("QD_OFFLINE_CACHE_FILE_PATH", str(data.workspace / ".cache" / "quadrants"))
                environment["PYTHONUTF8"] = "1"
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "examples.garment_folding.tools.ik_preflight",
                        "--workspace",
                        str(data.workspace),
                        "--compiled",
                        str(folder),
                        "--options",
                        str(options_file),
                    ],
                    cwd=Path(__file__).resolve().parents[4],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                    env=environment,
                )
                report_file = folder / "ik-report.json"
                if not report_file.exists():
                    raise ValueError((process.stderr or process.stdout)[-3000:])
                result = json.loads(report_file.read_text(encoding="utf-8"))
                result["saved"] = str(report_file)
            elif self.path == "/api/pose":
                request = PoseRequest.model_validate_json(text)
                if (
                    request.hand not in ("left", "right")
                    or request.role not in ("grasp", "placement")
                    or request.source_frame not in data.frames
                ):
                    raise ValueError("Expected a replay frame, left/right hand and grasp/placement role.")
                quat = np.array(request.quat_wxyz)
                if abs(np.linalg.norm(quat) - 1) > 1e-5:
                    raise ValueError("Pose quaternion must have unit norm.")
                folder = data.workspace / "orientations"
                folder.mkdir(exist_ok=True)
                path = folder / f"pose-{timestamp}.json"
                index = np.flatnonzero(data.frames == request.source_frame)[0]
                urdf = (
                    Path(data.metadata.robot_path)
                    if data.metadata.robot_path is not None
                    else Path(data.metadata.asset_root) / "assets/robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf"
                )
                measured = (
                    None if data.ipc_transforms is None else dict(zip(data.ipc_names, data.ipc_transforms[index]))
                )
                source_pose = pose_preview(
                    urdf,
                    data.metadata.joint_names,
                    data.robot_q[index],
                    request.hand,
                    data.metadata.robot_base_pos_m,
                    measured,
                )
                write_json(
                    path,
                    {
                        **request.model_dump(),
                        "schema": "workbench-gripper-orientation-v1",
                        "quaternion_frame": "world",
                        "quaternion_order": "wxyz",
                        "source_quat_wxyz": source_pose["quat_wxyz"],
                        "position_m": source_pose["position_m"],
                        "pose_source": source_pose["pose_source"],
                        "preview_only": True,
                        "ik_checked": False,
                        "physics_checked": False,
                    },
                )
                result = {"saved": str(path)}
            elif self.path == "/api/default-plan":
                request = DraftRequest.model_validate_json(text)
                source = request.plan
                if source.start_frame != data.metadata.start_frame:
                    raise ValueError("Draft start frame must match the workspace.")
                result = default_plan(
                    source.start, source.start_frame, request.pull_axis, request.pull_sign
                ).model_dump()
            else:
                self.send_error(404)
                return
            self.send_payload(json.dumps(result, allow_nan=False).encode(), "application/json")
        except (ValueError, ValidationError, OSError, subprocess.TimeoutExpired) as error:
            self.send_payload(json.dumps({"error": str(error)}).encode(), "application/json", status=400)
