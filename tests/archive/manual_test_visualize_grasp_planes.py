from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pybullet as p
import pybullet_data
import yaml

from src.sim.assets import benchmark_part_urdf


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    if n <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [float(x) / n for x in v]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a: list[float], b: list[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _orthonormal_tangent(normal: list[float], tangent_hint: list[float]) -> list[float]:
    # Remove normal component from tangent hint.
    d = _dot(tangent_hint, normal)
    t = [
        tangent_hint[0] - d * normal[0],
        tangent_hint[1] - d * normal[1],
        tangent_hint[2] - d * normal[2],
    ]
    t = _normalize(t)
    if _dot(t, t) < 1e-12:
        # Fallback if tangent_hint was near parallel to normal.
        c = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
        t = _normalize(_cross(c, normal))
    return t


def _rotate_object_to_world(vec_obj: list[float], obj_quat: list[float]) -> list[float]:
    m = p.getMatrixFromQuaternion(obj_quat)
    r00, r01, r02 = float(m[0]), float(m[1]), float(m[2])
    r10, r11, r12 = float(m[3]), float(m[4]), float(m[5])
    r20, r21, r22 = float(m[6]), float(m[7]), float(m[8])
    x = float(vec_obj[0])
    y = float(vec_obj[1])
    z = float(vec_obj[2])
    return [
        r00 * x + r01 * y + r02 * z,
        r10 * x + r11 * y + r12 * z,
        r20 * x + r21 * y + r22 * z,
    ]


def _transform_object_to_world(point_obj: list[float], obj_pos: list[float], obj_quat: list[float]) -> list[float]:
    p_world, _ = p.multiplyTransforms(obj_pos, obj_quat, point_obj, [0.0, 0.0, 0.0, 1.0])
    return [float(p_world[0]), float(p_world[1]), float(p_world[2])]


def _create_sphere_marker(pos: list[float], radius: float, rgba: list[float]) -> int:
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=[float(c) for c in rgba])
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[float(v) for v in pos],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )


def _create_plane_patch(center: list[float], quat: list[float], size: float, thickness: float, rgba: list[float]) -> int:
    hx = float(size) * 0.5
    hy = float(size) * 0.5
    hz = float(thickness) * 0.5
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[hx, hy, hz],
        rgbaColor=[float(c) for c in rgba],
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[float(v) for v in center],
        baseOrientation=[float(v) for v in quat],
    )


def _quat_from_basis(tangent: list[float], bitangent: list[float], normal: list[float]) -> list[float]:
    # Build 3x3 matrix with basis vectors as columns.
    m = [
        tangent[0],
        bitangent[0],
        normal[0],
        tangent[1],
        bitangent[1],
        normal[1],
        tangent[2],
        bitangent[2],
        normal[2],
    ]
    # Convert matrix -> quaternion.
    tr = m[0] + m[4] + m[8]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[7] - m[5]) / s
        qy = (m[2] - m[6]) / s
        qz = (m[3] - m[1]) / s
    elif (m[0] > m[4]) and (m[0] > m[8]):
        s = math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0
        qw = (m[7] - m[5]) / s
        qx = 0.25 * s
        qy = (m[1] + m[3]) / s
        qz = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0
        qw = (m[2] - m[6]) / s
        qx = (m[1] + m[3]) / s
        qy = 0.25 * s
        qz = (m[5] + m[7]) / s
    else:
        s = math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0
        qw = (m[3] - m[1]) / s
        qx = (m[2] + m[6]) / s
        qy = (m[5] + m[7]) / s
        qz = 0.25 * s
    return [float(qx), float(qy), float(qz), float(qw)]


def run(yaml_path: str, distance_mm: float, patch_size_mm: float, patch_thickness_mm: float) -> None:
    with Path(yaml_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    part_id = int(data["part_id"])
    pose = data["object_pose_world"]
    obj_pos = [float(v) for v in pose["position_xyz"]]
    obj_quat = [float(v) for v in pose["orientation_xyzw"]]
    grasp_points = list(data.get("grasp_points", []))
    if not grasp_points:
        raise RuntimeError(f"No grasp_points in {yaml_path}")

    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=50.0,
        cameraPitch=-30.0,
        cameraTargetPosition=[obj_pos[0], obj_pos[1], obj_pos[2]],
    )

    p.loadURDF(
        str(benchmark_part_urdf(part_id)),
        basePosition=obj_pos,
        baseOrientation=obj_quat,
        useFixedBase=True,
    )

    d = float(distance_mm) / 1000.0
    patch_size = float(patch_size_mm) / 1000.0
    patch_thickness = float(patch_thickness_mm) / 1000.0

    for gp in grasp_points:
        p_obj = [float(v) for v in gp["position_obj_xyz"]]
        n_obj = _normalize([float(v) for v in gp["normal_obj_xyz"]])
        t_obj = _normalize([float(v) for v in gp["tangent_obj_xyz"]])

        p_world = _transform_object_to_world(p_obj, obj_pos, obj_quat)
        n_world = _normalize(_rotate_object_to_world(n_obj, obj_quat))
        t_world_hint = _normalize(_rotate_object_to_world(t_obj, obj_quat))
        t_world = _orthonormal_tangent(n_world, t_world_hint)
        b_world = _normalize(_cross(n_world, t_world))

        plane_center = [
            p_world[0] + d * n_world[0],
            p_world[1] + d * n_world[1],
            p_world[2] + d * n_world[2],
        ]
        plane_quat = _quat_from_basis(t_world, b_world, n_world)

        # Grasp point (green).
        _create_sphere_marker(p_world, radius=0.004, rgba=[0.1, 0.9, 0.1, 1.0])
        # Offset plane patch (blue, semi-transparent).
        _create_plane_patch(
            center=plane_center,
            quat=plane_quat,
            size=patch_size,
            thickness=patch_thickness,
            rgba=[0.1, 0.4, 1.0, 0.45],
        )
        # Small normal line for orientation debugging.
        p.addUserDebugLine(
            p_world,
            [p_world[0] + 0.02 * n_world[0], p_world[1] + 0.02 * n_world[1], p_world[2] + 0.02 * n_world[2]],
            [1.0, 1.0, 0.0],
            lineWidth=2.0,
            lifeTime=0.0,
        )

    print(f"Loaded {len(grasp_points)} grasp points from {yaml_path}")
    print("Press Ctrl+C in terminal to quit.")
    try:
        while p.isConnected(client_id):
            p.stepSimulation()
            time.sleep(1.0 / 120.0)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp-yaml", required=True, help="Path to saved grasp points yaml")
    parser.add_argument("--distance-mm", type=float, default=20.0, help="Offset distance d from grasp point along normal")
    parser.add_argument("--patch-size-mm", type=float, default=35.0, help="Visual patch size for each plane")
    parser.add_argument("--patch-thickness-mm", type=float, default=1.0, help="Visual patch thickness")
    args = parser.parse_args()
    run(
        yaml_path=args.grasp_yaml,
        distance_mm=args.distance_mm,
        patch_size_mm=args.patch_size_mm,
        patch_thickness_mm=args.patch_thickness_mm,
    )


if __name__ == "__main__":
    main()
