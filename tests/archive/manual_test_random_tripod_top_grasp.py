from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import pybullet as p
import pybullet_data
import yaml

from src.sim.assets import AR10_URDF
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


def _quat_from_basis(tangent: list[float], bitangent: list[float], normal: list[float]) -> list[float]:
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
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[float(size) * 0.5, float(size) * 0.5, float(thickness) * 0.5],
        rgbaColor=[float(c) for c in rgba],
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[float(v) for v in center],
        baseOrientation=[float(v) for v in quat],
    )


def _set_body_visible(body_id: int, visible: bool) -> None:
    rgba = [1.0, 1.0, 1.0, 1.0 if visible else 0.0]
    p.changeVisualShape(int(body_id), -1, rgbaColor=rgba)
    for j in range(p.getNumJoints(int(body_id))):
        p.changeVisualShape(int(body_id), int(j), rgbaColor=rgba)


def _load_tripod_ref(hand_yaml: str) -> dict:
    with Path(hand_yaml).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    refs = dict(data.get("reference_points", {}))
    if "tripod" not in refs:
        raise RuntimeError(f"'tripod' missing in {hand_yaml}")
    return refs["tripod"]


def run(hand_yaml: str, object_x: float, object_y: float, distance_mm: float, hz: float) -> None:
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError("pygame is required. Install with: pip install pygame") from exc

    tripod_ref = _load_tripod_ref(hand_yaml)

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        raise RuntimeError("No gamepad found. Connect controller and retry.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Controller: {js.get_name()}")

    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(
        cameraDistance=0.9,
        cameraYaw=45.0,
        cameraPitch=-26.0,
        cameraTargetPosition=[float(object_x), float(object_y), 0.06],
    )

    hand_id = p.loadURDF(
        str(AR10_URDF),
        basePosition=[float(object_x) - 0.30, float(object_y), 0.10],
        baseOrientation=p.getQuaternionFromEuler([math.radians(90.0), 0.0, 0.0]),
        useFixedBase=True,
    )
    HandModel(hand_id).send_q_target([0.0] * 10)
    _set_body_visible(hand_id, False)
    hand_visible = False
    hand_rest_pos, hand_rest_quat = p.getBasePositionAndOrientation(hand_id)

    object_id = -1
    gp_marker_id = -1
    gp_plane_id = -1
    gp_line_id = -1
    info_text = -1
    pregrasp_pos = [0.0, 0.0, 0.0]
    pregrasp_quat = [0.0, 0.0, 0.0, 1.0]
    rng = random.Random()

    def spawn_random_object() -> None:
        nonlocal object_id, gp_marker_id, gp_plane_id, gp_line_id, info_text, pregrasp_pos, pregrasp_quat, hand_visible
        if object_id >= 0:
            p.removeBody(object_id)
        if gp_marker_id >= 0:
            p.removeBody(gp_marker_id)
        if gp_plane_id >= 0:
            p.removeBody(gp_plane_id)
        if gp_line_id >= 0:
            p.removeUserDebugItem(gp_line_id)
        if info_text >= 0:
            p.removeUserDebugItem(info_text)

        obj_type = "sphere" if rng.random() < 0.5 else "cube"
        if obj_type == "sphere":
            radius = rng.uniform(0.018, 0.045)
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0.95, 0.95, 0.95, 1.0])
            z = radius
            object_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[float(object_x), float(object_y), float(z)],
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
            )
            gp_world = [float(object_x), float(object_y), float(z + radius)]
            obj_size_txt = f"r={radius*1000.0:.1f}mm"
        else:
            side = rng.uniform(0.035, 0.090)
            half = side * 0.5
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=[0.95, 0.95, 0.95, 1.0])
            z = half
            object_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[float(object_x), float(object_y), float(z)],
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
            )
            gp_world = [float(object_x), float(object_y), float(z + half)]
            obj_size_txt = f"a={side*1000.0:.1f}mm"

        n_obj = [0.0, 0.0, 1.0]
        t_obj = [1.0, 0.0, 0.0]
        d = float(distance_mm) / 1000.0
        plane_center = [gp_world[0] + d * n_obj[0], gp_world[1] + d * n_obj[1], gp_world[2] + d * n_obj[2]]
        plane_q = _quat_from_basis(t_obj, _cross(n_obj, t_obj), n_obj)

        gp_marker_id = _create_sphere_marker(gp_world, radius=0.005, rgba=[0.05, 0.9, 0.2, 1.0])
        gp_plane_id = _create_plane_patch(
            center=plane_center,
            quat=plane_q,
            size=0.030,
            thickness=0.001,
            rgba=[0.30, 0.60, 1.00, 0.45],
        )
        gp_line_id = p.addUserDebugLine(gp_world, plane_center, [0.2, 0.8, 1.0], lineWidth=2.0, lifeTime=0.0)

        # Compute hand pregrasp for tripod.
        p_hand = [float(v) for v in tripod_ref["position_hand_xyz"]]
        n_hand = _normalize([float(v) for v in tripod_ref["normal_hand_xyz"]])
        t_hand = _orthonormal_tangent(n_hand, [float(v) for v in tripod_ref["tangent_hand_xyz"]])
        b_hand = _normalize(_cross(n_hand, t_hand))
        t_hand = _normalize(_cross(b_hand, n_hand))

        n_target = [-n_obj[0], -n_obj[1], -n_obj[2]]
        t_target = t_obj
        b_target = _normalize(_cross(n_target, t_target))
        t_target = _normalize(_cross(b_target, n_target))
        r_world_target = _mat_from_basis(t_target, b_target, n_target)
        r_hand_local = _mat_from_basis(t_hand, b_hand, n_hand)
        r_hand_world = _mat_mul(r_world_target, _mat_transpose(r_hand_local))
        pregrasp_quat = _quat_from_basis(
            [r_hand_world[0][0], r_hand_world[1][0], r_hand_world[2][0]],
            [r_hand_world[0][1], r_hand_world[1][1], r_hand_world[2][1]],
            [r_hand_world[0][2], r_hand_world[1][2], r_hand_world[2][2]],
        )
        hand_offset = _mat_vec_mul(r_hand_world, p_hand)
        pregrasp_pos = [
            plane_center[0] - hand_offset[0],
            plane_center[1] - hand_offset[1],
            plane_center[2] - hand_offset[2],
        ]

        info_text = p.addUserDebugText(
            text=f"type={obj_type} {obj_size_txt}  grip=tripod dist_mm={int(distance_mm)}",
            textPosition=[float(object_x), float(object_y), 0.20],
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.3,
            lifeTime=0.0,
        )

        if hand_visible:
            p.resetBasePositionAndOrientation(hand_id, pregrasp_pos, pregrasp_quat)

    spawn_random_object()
    print("Controls:")
    print("  Y: toggle hand at tripod pregrasp")
    print("  A: spawn new random object (sphere/cube)")
    print("  Menu: quit")

    y_latch = False
    a_latch = False
    dt = 1.0 / float(hz)
    try:
        while p.isConnected(cid):
            pygame.event.pump()
            y_btn = bool(js.get_numbuttons() > 3 and js.get_button(3))
            a_btn = bool(js.get_numbuttons() > 0 and js.get_button(0))
            menu = bool(js.get_numbuttons() > 7 and js.get_button(7))

            if menu:
                break
            if y_btn and not y_latch:
                hand_visible = not hand_visible
                _set_body_visible(hand_id, hand_visible)
                if hand_visible:
                    p.resetBasePositionAndOrientation(hand_id, pregrasp_pos, pregrasp_quat)
                else:
                    p.resetBasePositionAndOrientation(hand_id, hand_rest_pos, hand_rest_quat)
            if a_btn and not a_latch:
                spawn_random_object()

            y_latch = y_btn
            a_latch = a_btn
            p.stepSimulation()
            time.sleep(dt)
    finally:
        try:
            pygame.quit()
        except Exception:
            pass
        if p.isConnected(cid):
            p.disconnect(cid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-yaml", default="artifacts/hand_reference_points.yaml")
    parser.add_argument("--object-x", type=float, default=0.80)
    parser.add_argument("--object-y", type=float, default=0.00)
    parser.add_argument("--distance-mm", type=float, default=35.0, help="Tripod pregrasp distance")
    parser.add_argument("--hz", type=float, default=120.0)
    args = parser.parse_args()
    run(
        hand_yaml=args.hand_yaml,
        object_x=args.object_x,
        object_y=args.object_y,
        distance_mm=args.distance_mm,
        hz=args.hz,
    )


if __name__ == "__main__":
    main()
