from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pybullet as p
import pybullet_data
import yaml

from src.sim.assets import AR10_URDF, benchmark_part_urdf
from src.sim.hand_model import HandModel


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    if n <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [float(x) / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _scale(v: list[float], s: float) -> list[float]:
    return [float(v[0]) * s, float(v[1]) * s, float(v[2]) * s]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [float(a[0] + b[0]), float(a[1] + b[1]), float(a[2] + b[2])]


def _orthonormal_tangent(normal: list[float], tangent_hint: list[float]) -> list[float]:
    d = _dot(tangent_hint, normal)
    t = [
        tangent_hint[0] - d * normal[0],
        tangent_hint[1] - d * normal[1],
        tangent_hint[2] - d * normal[2],
    ]
    t = _normalize(t)
    if _dot(t, t) < 1e-12:
        c = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
        t = _normalize(_cross(c, normal))
    return t


def _rotate_vector_around_axis(v: list[float], axis: list[float], angle_rad: float) -> list[float]:
    # Rodrigues rotation formula.
    a = _normalize(axis)
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    term1 = _scale(v, c)
    term2 = _scale(_cross(a, v), s)
    term3 = _scale(a, _dot(a, v) * (1.0 - c))
    return _add(_add(term1, term2), term3)


def _transform_local_to_world(point_local: list[float], pos_world: list[float], quat_world: list[float]) -> list[float]:
    p_world, _ = p.multiplyTransforms(pos_world, quat_world, point_local, [0.0, 0.0, 0.0, 1.0])
    return [float(p_world[0]), float(p_world[1]), float(p_world[2])]


def _rotate_local_to_world(vec_local: list[float], quat_world: list[float]) -> list[float]:
    m = p.getMatrixFromQuaternion(quat_world)
    x = float(vec_local[0])
    y = float(vec_local[1])
    z = float(vec_local[2])
    return [
        float(m[0]) * x + float(m[1]) * y + float(m[2]) * z,
        float(m[3]) * x + float(m[4]) * y + float(m[5]) * z,
        float(m[6]) * x + float(m[7]) * y + float(m[8]) * z,
    ]


def _mat_from_basis(x_axis: list[float], y_axis: list[float], z_axis: list[float]) -> list[list[float]]:
    return [
        [float(x_axis[0]), float(y_axis[0]), float(z_axis[0])],
        [float(x_axis[1]), float(y_axis[1]), float(z_axis[1])],
        [float(x_axis[2]), float(y_axis[2]), float(z_axis[2])],
    ]


def _mat_transpose(a: list[list[float]]) -> list[list[float]]:
    return [
        [float(a[0][0]), float(a[1][0]), float(a[2][0])],
        [float(a[0][1]), float(a[1][1]), float(a[2][1])],
        [float(a[0][2]), float(a[1][2]), float(a[2][2])],
    ]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = float(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
    return out


def _mat_vec_mul(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [
        float(mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2]),
        float(mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2]),
        float(mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2]),
    ]


def _mat_from_quat(quat: list[float]) -> list[list[float]]:
    m = p.getMatrixFromQuaternion(quat)
    return [
        [float(m[0]), float(m[1]), float(m[2])],
        [float(m[3]), float(m[4]), float(m[5])],
        [float(m[6]), float(m[7]), float(m[8])],
    ]


def _quat_from_mat(m: list[list[float]]) -> list[float]:
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    return [float(qx), float(qy), float(qz), float(qw)]


def _load_object_grasp_point(yaml_path: str, grasp_point_id: str | None) -> tuple[int, list[float], list[float], dict]:
    with Path(yaml_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    points = list(data.get("grasp_points", []))
    if not points:
        raise RuntimeError(f"No grasp_points in {yaml_path}")
    if grasp_point_id is None:
        gp = points[0]
    else:
        gp = next((x for x in points if str(x.get("id")) == str(grasp_point_id)), None)
        if gp is None:
            raise RuntimeError(f"grasp_point_id={grasp_point_id} not found")
    return (
        int(data["part_id"]),
        [float(v) for v in data["object_pose_world"]["position_xyz"]],
        [float(v) for v in data["object_pose_world"]["orientation_xyzw"]],
        gp,
    )


def _load_hand_reference(yaml_path: str, grasp_pose: str) -> tuple[dict, list[float]]:
    with Path(yaml_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    refs = dict(data.get("reference_points", {}))
    if grasp_pose not in refs:
        raise RuntimeError(f"grasp_pose={grasp_pose} not found in {yaml_path}")
    hand_pose_world = data.get("hand_pose_world", {})
    preferred_quat = [float(v) for v in hand_pose_world.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])]
    return refs[grasp_pose], preferred_quat


def run(
    object_yaml: str,
    hand_yaml: str,
    grasp_pose: str,
    grasp_point_id: str | None,
    distance_mm: float,
    align_mode: str,
    twist_offset_deg: float,
) -> None:
    part_id, obj_pos, obj_quat, gp = _load_object_grasp_point(object_yaml, grasp_point_id)
    hand_ref, preferred_hand_quat = _load_hand_reference(hand_yaml, grasp_pose)

    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf")

    p.resetDebugVisualizerCamera(
        cameraDistance=0.85,
        cameraYaw=55.0,
        cameraPitch=-25.0,
        cameraTargetPosition=[obj_pos[0], obj_pos[1], obj_pos[2] + 0.05],
    )

    part_id_sim = p.loadURDF(
        str(benchmark_part_urdf(part_id)),
        basePosition=obj_pos,
        baseOrientation=obj_quat,
        useFixedBase=True,
    )
    _ = part_id_sim

    p_obj = [float(v) for v in gp["position_obj_xyz"]]
    n_obj = _normalize([float(v) for v in gp["normal_obj_xyz"]])
    t_obj = _normalize([float(v) for v in gp["tangent_obj_xyz"]])
    gp_world = _transform_local_to_world(p_obj, obj_pos, obj_quat)
    n_obj_world = _normalize(_rotate_local_to_world(n_obj, obj_quat))
    t_obj_world = _orthonormal_tangent(n_obj_world, _rotate_local_to_world(t_obj, obj_quat))
    b_obj_world = _normalize(_cross(n_obj_world, t_obj_world))

    p_hand = [float(v) for v in hand_ref["position_hand_xyz"]]
    n_hand = _normalize([float(v) for v in hand_ref["normal_hand_xyz"]])
    t_hand = _normalize([float(v) for v in hand_ref["tangent_hand_xyz"]])
    t_hand = _orthonormal_tangent(n_hand, t_hand)
    b_hand = _normalize(_cross(n_hand, t_hand))

    n_hand_world_target = [-n_obj_world[0], -n_obj_world[1], -n_obj_world[2]]
    if align_mode == "full_pose":
        t_hand_world_target = t_obj_world
    else:
        # Parallel-plane alignment only: keep a stable hand twist from preferred hand orientation.
        pref_rot = _mat_from_quat(preferred_hand_quat)
        pref_t_world = _normalize(_mat_vec_mul(pref_rot, t_hand))
        t_hand_world_target = _orthonormal_tangent(n_hand_world_target, pref_t_world)
        if abs(float(twist_offset_deg)) > 1e-12:
            t_hand_world_target = _normalize(
                _rotate_vector_around_axis(t_hand_world_target, n_hand_world_target, math.radians(float(twist_offset_deg)))
            )
    b_hand_world_target = _normalize(_cross(n_hand_world_target, t_hand_world_target))
    t_hand_world_target = _normalize(_cross(b_hand_world_target, n_hand_world_target))

    r_world_target = _mat_from_basis(t_hand_world_target, b_hand_world_target, n_hand_world_target)
    r_hand_local = _mat_from_basis(t_hand, b_hand, n_hand)
    r_hand_base_world = _mat_mul(r_world_target, _mat_transpose(r_hand_local))
    hand_quat_world = _quat_from_mat(r_hand_base_world)

    target_ref_world = [
        gp_world[0] + float(distance_mm) / 1000.0 * n_obj_world[0],
        gp_world[1] + float(distance_mm) / 1000.0 * n_obj_world[1],
        gp_world[2] + float(distance_mm) / 1000.0 * n_obj_world[2],
    ]
    hand_ref_offset_world = _mat_vec_mul(r_hand_base_world, p_hand)
    hand_base_world = [
        target_ref_world[0] - hand_ref_offset_world[0],
        target_ref_world[1] - hand_ref_offset_world[1],
        target_ref_world[2] - hand_ref_offset_world[2],
    ]

    hand_id = p.loadURDF(
        str(AR10_URDF),
        basePosition=hand_base_world,
        baseOrientation=hand_quat_world,
        useFixedBase=True,
    )
    HandModel(hand_id).reset_open_pose()

    print(f"part_id={part_id}")
    print(f"grasp_point_id={gp.get('id')}")
    print(f"grasp_pose={grasp_pose}")
    print(f"align_mode={align_mode}")
    print(f"twist_offset_deg={float(twist_offset_deg)}")
    print(f"target_ref_world={[round(v, 5) for v in target_ref_world]}")
    print(f"hand_base_world={[round(v, 5) for v in hand_base_world]}")

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
    parser.add_argument("--object-yaml", required=True, help="Saved object grasp-point yaml")
    parser.add_argument("--hand-yaml", default="artifacts/hand_reference_points.yaml", help="Saved hand ref yaml")
    parser.add_argument("--grasp-pose", default="tripod")
    parser.add_argument("--grasp-point-id", default=None)
    parser.add_argument("--distance-mm", type=float, default=20.0)
    parser.add_argument(
        "--align-mode",
        choices=["parallel_only", "full_pose"],
        default="parallel_only",
        help="parallel_only aligns only plane normals and keeps stable twist; full_pose also matches tangent.",
    )
    parser.add_argument(
        "--twist-offset-deg",
        type=float,
        default=0.0,
        help="Additional twist around plane normal (used in parallel_only), e.g. 180.",
    )
    args = parser.parse_args()
    run(
        object_yaml=args.object_yaml,
        hand_yaml=args.hand_yaml,
        grasp_pose=args.grasp_pose,
        grasp_point_id=args.grasp_point_id,
        distance_mm=args.distance_mm,
        align_mode=args.align_mode,
        twist_offset_deg=args.twist_offset_deg,
    )


if __name__ == "__main__":
    main()
