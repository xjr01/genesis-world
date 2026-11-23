"""Measured robot poses, finger collision meshes and shirt material labels."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial.transform import Rotation

from .action_plan import Endpoint, Hands
from .kinematics import TCP_LOCAL


class Camera(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="allow")
    pos_m: tuple[float, float, float] = (0.68, 0, 2.45)
    lookat_m: tuple[float, float, float] = (0.68, 0, 0.8)
    up: tuple[float, float, float] = (0, 1, 0)
    vertical_fov_deg: float = Field(default=42, gt=0, lt=180)

    @model_validator(mode="after")
    def validate_axes(self):
        direction = np.array(self.lookat_m) - self.pos_m
        if np.linalg.norm(np.cross(direction, self.up)) < 1e-12:
            raise ValueError("Camera direction and up must define distinct axes.")
        return self


def origin_transform(element):
    """Read a robot description origin as a homogeneous transform."""
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return transform


def link_transforms(urdf: Path, joint_names: tuple[str, ...], qpos: np.ndarray, base_pos=(0, 0, 0.17)):
    """Evaluate the robot description tree with name-addressed measured joints."""
    root = ET.parse(urdf).getroot()
    pending = list(root.findall("joint"))
    children = {joint.find("child").get("link") for joint in pending}
    bases = [link.get("name") for link in root.findall("link") if link.get("name") not in children]
    if len(bases) != 1 or qpos.shape != (len(joint_names),) or not np.isfinite(qpos).all():
        raise ValueError("Expected one robot root and finite name-addressed joints.")
    transforms = {bases[0]: np.eye(4)}
    transforms[bases[0]][:3, 3] = base_pos
    while pending:
        ready = [joint for joint in pending if joint.find("parent").get("link") in transforms]
        if not ready:
            raise ValueError("Robot joint tree is unresolved.")
        for joint in ready:
            motion = np.eye(4)
            kind = joint.get("type")
            if kind != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                value = qpos[joint_names.index(joint.get("name"))]
                if kind == "prismatic":
                    motion[:3, 3] = axis * value
                elif kind in ("revolute", "continuous"):
                    motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
                else:
                    raise ValueError(f"Unsupported joint type: {kind}")
            transforms[joint.find("child").get("link")] = (
                transforms[joint.find("parent").get("link")] @ origin_transform(joint.find("origin")) @ motion
            )
            pending.remove(joint)
    return transforms


def measured_hands(transforms, qpos, names):
    """Measure both tool center points (TCPs) from robot link transforms."""
    endpoints = []
    for hand, prefix in (("left", 1), ("right", 2)):
        transform = transforms[f"{hand}_link{prefix}6"]
        position = transform[:3, 3] + transform[:3, :3] @ TCP_LOCAL
        quat = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
        opening = np.mean(qpos[[names.index(f"{hand}_joint{prefix}{number}") for number in (7, 8)]])
        endpoints.append(Endpoint(pos=tuple(position), quat=tuple(quat), opening=np.clip(opening, 0, 0.044)))
    return Hands(left=endpoints[0], right=endpoints[1])


@dataclass(frozen=True)
class CollisionGeometry:
    vertices: np.ndarray
    faces: np.ndarray


@lru_cache(maxsize=32)
def collision_geometry(urdf: Path, link_name: str) -> CollisionGeometry:
    """Load collision geometry in link-local coordinates for pose inspection."""
    link = ET.parse(urdf).getroot().find(f"link[@name='{link_name}']")
    meshes = []
    for collision in link.findall("collision"):
        geometry = collision.find("geometry/mesh")
        if geometry is not None:
            mesh = trimesh.load(urdf.parent / geometry.get("filename"), force="mesh", process=False)
            mesh.vertices *= np.fromstring(geometry.get("scale", "1 1 1"), sep=" ")
        elif collision.find("geometry/box") is not None:
            mesh = trimesh.creation.box(extents=np.fromstring(collision.find("geometry/box").get("size"), sep=" "))
        elif collision.find("geometry/cylinder") is not None:
            shape = collision.find("geometry/cylinder")
            mesh = trimesh.creation.cylinder(radius=float(shape.get("radius")), height=float(shape.get("length")))
        elif collision.find("geometry/sphere") is not None:
            mesh = trimesh.creation.icosphere(radius=float(collision.find("geometry/sphere").get("radius")))
        else:
            raise ValueError("Unsupported collision geometry for robot preview.")
        mesh.apply_transform(origin_transform(collision.find("origin")))
        meshes.append(mesh)
    combined = trimesh.util.concatenate(meshes).convex_hull
    return CollisionGeometry(combined.vertices.copy(), combined.faces.copy())


def pose_preview(
    urdf: Path, names: tuple[str, ...], qpos: np.ndarray, hand: str, base_pos=(0, 0, 0.17), measured_transforms=None
):
    """Return measured tool axes and finger meshes relative to the tool center."""
    if hand not in ("left", "right"):
        raise ValueError("Expected left or right hand.")
    transforms = link_transforms(urdf, names, qpos, base_pos)
    if measured_transforms is not None:
        transforms.update(measured_transforms)
    prefix = 1 if hand == "left" else 2
    tool = transforms[f"{hand}_link{prefix}6"]
    rotation = Rotation.from_matrix(tool[:3, :3]).as_matrix()
    position = tool[:3, 3] + tool[:3, :3] @ TCP_LOCAL
    fingers = []
    for number in (7, 8):
        name = f"{hand}_link{prefix}{number}"
        joint = ET.parse(urdf).getroot().find(f"joint[@name='{hand}_joint{prefix}{number}']")
        axis = origin_transform(joint.find("origin"))[:3, :3] @ np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        if not np.allclose(axis, (0, 1 if number == 7 else -1, 0)):
            raise ValueError("Finger opening axis differs from tool Y.")
        geometry = collision_geometry(urdf, name)
        transform = transforms[name]
        world = geometry.vertices @ transform[:3, :3].T + transform[:3, 3]
        fingers.append(
            {
                "name": name,
                "vertices_local_m": ((world - position) @ rotation).tolist(),
                "faces": geometry.faces.tolist(),
            }
        )
    return {
        "hand": hand,
        "position_m": position.tolist(),
        "quat_wxyz": Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]].tolist(),
        "fingers": fingers,
        "axis_semantics": {"x": "tool length", "y": "finger opening/closing", "z": "third axis"},
        "geometry_source": "URDF collision mesh convex hull",
        "pose_source": "measured IPC transforms"
        if measured_transforms is not None
        else "measured robot_q URDF forward kinematics",
        "preview_only": True,
        "ik_checked": False,
        "physics_checked": False,
    }


def robot_preview(urdf: Path, names: tuple[str, ...], qpos: np.ndarray, base_pos=(0, 0, 0.17)):
    """Return robot collision meshes transformed by measured joint positions."""
    transforms = link_transforms(urdf, names, qpos, base_pos)
    meshes = []
    for link in ET.parse(urdf).getroot().findall("link"):
        if not link.findall("collision"):
            continue
        name = link.get("name")
        geometry = collision_geometry(urdf, name)
        transform = transforms[name]
        meshes.append(
            {
                "name": name,
                "vertices_world_m": (geometry.vertices @ transform[:3, :3].T + transform[:3, 3]).tolist(),
                "faces": geometry.faces.tolist(),
            }
        )
    return meshes
