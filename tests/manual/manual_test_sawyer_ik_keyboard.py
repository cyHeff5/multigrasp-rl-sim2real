from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import pybullet as p
import pybullet_data
import yaml

from src.sim.assets import AR10_URDF, SAWYER_URDF, benchmark_part_urdf
from src.sim.hand_model import HandModel
from src.sim.mounting import apply_hand_friction, mount_hand_to_arm
from src.sim.sawyer_arm import SawyerHelper

BASE_YAW = -math.pi * 0.5
ARM_HOME_GAIN = 0.35
ARM_HOME_FORCE = 2000.0
ARM_CTRL_GAIN = 0.40
ARM_CTRL_FORCE = 2200.0
ROTATION_STEP_SCALE = 0.35
AUTO_ALIGN_MAX_POS_STEP_M = 0.003
AUTO_ALIGN_ROT_ALPHA = 0.12
AUTO_ALIGN_POS_TOL_M = 0.0025
AUTO_ALIGN_NORMAL_TOL_DEG = 6.0


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


def _rotate_local_to_world(vec_local: list[float], quat_world: list[float]) -> list[float]:
    m = p.getMatrixFromQuaternion(quat_world)
    r00, r01, r02 = float(m[0]), float(m[1]), float(m[2])
    r10, r11, r12 = float(m[3]), float(m[4]), float(m[5])
    r20, r21, r22 = float(m[6]), float(m[7]), float(m[8])
    x = float(vec_local[0])
    y = float(vec_local[1])
    z = float(vec_local[2])
    return [
        r00 * x + r01 * y + r02 * z,
        r10 * x + r11 * y + r12 * z,
        r20 * x + r21 * y + r22 * z,
    ]


def _transform_local_to_world(point_local: list[float], pos_world: list[float], quat_world: list[float]) -> list[float]:
    p_world, _ = p.multiplyTransforms(pos_world, quat_world, point_local, [0.0, 0.0, 0.0, 1.0])
    return [float(p_world[0]), float(p_world[1]), float(p_world[2])]


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


def _quat_to_rpy(quat: list[float]) -> list[float]:
    return [float(v) for v in p.getEulerFromQuaternion(quat)]


def _quat_normalize(quat: list[float]) -> list[float]:
    n = math.sqrt(sum(float(v) * float(v) for v in quat))
    if n <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [float(v) / n for v in quat]


def _quat_slerp(q0: list[float], q1: list[float], t: float) -> list[float]:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3])
    if dot < 0.0:
        q1 = [-q1[0], -q1[1], -q1[2], -q1[3]]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        out = [
            (1.0 - t) * q0[0] + t * q1[0],
            (1.0 - t) * q0[1] + t * q1[1],
            (1.0 - t) * q0[2] + t * q1[2],
            (1.0 - t) * q0[3] + t * q1[3],
        ]
        return _quat_normalize(out)
    theta0 = math.acos(dot)
    sin_theta0 = math.sin(theta0)
    theta = theta0 * float(t)
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    return _quat_normalize(
        [
            s0 * q0[0] + s1 * q1[0],
            s0 * q0[1] + s1 * q1[1],
            s0 * q0[2] + s1 * q1[2],
            s0 * q0[3] + s1 * q1[3],
        ]
    )


def _mat_from_quat(quat: list[float]) -> list[list[float]]:
    m = p.getMatrixFromQuaternion(quat)
    return [
        [float(m[0]), float(m[1]), float(m[2])],
        [float(m[3]), float(m[4]), float(m[5])],
        [float(m[6]), float(m[7]), float(m[8])],
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


def _quat_from_mat(m: list[list[float]]) -> list[float]:
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif (m[0][0] > m[1][1]) and (m[0][0] > m[2][2]):
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


def _create_sphere_marker(pos: list[float], radius: float, rgba: list[float]) -> int:
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=[float(c) for c in rgba])
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[float(v) for v in pos],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )


def _reset_body_pose(body_id: int, pos: list[float], quat: list[float]) -> None:
    p.resetBasePositionAndOrientation(
        int(body_id),
        [float(v) for v in pos],
        [float(v) for v in quat],
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


def _draw_grasp_points_from_yaml(
    yaml_path: str,
    selected_part: int,
    object_id: int,
    distance_mm: float,
    patch_size_mm: float,
    patch_thickness_mm: float,
) -> list[dict]:
    path = Path(yaml_path)
    if not path.exists():
        print(f"grasp_points_yaml_missing={path}")
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if int(data.get("part_id", -1)) != int(selected_part):
        print(f"grasp_points_part_mismatch=yaml:{data.get('part_id')} selected:{selected_part}")
    grasp_points = list(data.get("grasp_points", []))
    if not grasp_points:
        print(f"grasp_points_empty={path}")
        return []

    d = float(distance_mm) / 1000.0
    patch_size = float(patch_size_mm) / 1000.0
    patch_thickness = float(patch_thickness_mm) / 1000.0

    visuals: list[dict] = []
    for gp in grasp_points:
        marker_id = _create_sphere_marker([0.0, 0.0, 0.0], radius=0.004, rgba=[0.1, 0.9, 0.1, 1.0])
        plane_id = _create_plane_patch(
            center=[0.0, 0.0, 0.0],
            quat=[0.0, 0.0, 0.0, 1.0],
            size=patch_size,
            thickness=patch_thickness,
            rgba=[0.1, 0.4, 1.0, 0.45],
        )
        visuals.append(
            {
                "gp": gp,
                "marker_id": marker_id,
                "plane_id": plane_id,
                "line_id": -1,
                "distance_m": d,
            }
        )
    _update_grasp_visuals(object_id=object_id, visuals=visuals)
    return visuals


def _update_grasp_visuals(object_id: int, visuals: list[dict]) -> None:
    obj_pos_raw, obj_quat_raw = p.getBasePositionAndOrientation(int(object_id))
    obj_pos = [float(v) for v in obj_pos_raw]
    obj_quat = [float(v) for v in obj_quat_raw]
    for item in visuals:
        gp = item["gp"]
        d = float(item["distance_m"])
        p_obj = [float(v) for v in gp["position_obj_xyz"]]
        n_obj = _normalize([float(v) for v in gp["normal_obj_xyz"]])
        t_obj_hint = _normalize([float(v) for v in gp["tangent_obj_xyz"]])
        p_world = _transform_local_to_world(p_obj, obj_pos, obj_quat)
        n_world = _normalize(_rotate_local_to_world(n_obj, obj_quat))
        t_world = _orthonormal_tangent(n_world, _rotate_local_to_world(t_obj_hint, obj_quat))
        b_world = _normalize(_cross(n_world, t_world))
        plane_center = [
            p_world[0] + d * n_world[0],
            p_world[1] + d * n_world[1],
            p_world[2] + d * n_world[2],
        ]
        plane_quat = _quat_from_basis(t_world, b_world, n_world)
        _reset_body_pose(int(item["marker_id"]), p_world, [0.0, 0.0, 0.0, 1.0])
        _reset_body_pose(int(item["plane_id"]), plane_center, plane_quat)
        item["line_id"] = p.addUserDebugLine(
            p_world,
            [p_world[0] + 0.02 * n_world[0], p_world[1] + 0.02 * n_world[1], p_world[2] + 0.02 * n_world[2]],
            [1.0, 1.0, 0.0],
            lineWidth=2.0,
            lifeTime=0.0,
            replaceItemUniqueId=int(item["line_id"]),
        )


def _load_grasp_points(yaml_path: str) -> list[dict]:
    path = Path(yaml_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("grasp_points", []))


def _load_hand_reference_point(yaml_path: str, grasp_pose: str) -> dict | None:
    path = Path(yaml_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    refs = dict(data.get("reference_points", {}))
    return refs.get(grasp_pose)


def _set_default_camera() -> None:
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=50.0,
        cameraPitch=-35.0,
        cameraTargetPosition=[-0.01, 0.37, -0.24],
    )


def _home_sawyer(robot_id: int, helper: SawyerHelper) -> None:
    home_q = [
        1.664604,
        -1.449988765591175,
        -0.03997614956493058,
        1.3400699248337562,
        -1.520061695852127,
        1.1699913835803561,
        3.2498732514288604,
    ]
    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=helper.joints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=home_q,
        positionGains=[ARM_HOME_GAIN] * len(helper.joints),
        forces=[ARM_HOME_FORCE] * len(helper.joints),
    )
    for _ in range(240):
        p.stepSimulation()


def _spawn_pedestal_with_benchmark_part(part_id: int | None = None) -> tuple[int, int, int]:
    # Fixed pedestal and one benchmark part for manual checks.
    pedestal_height = 0.05
    pedestal_half_extents = [0.08, 0.08, pedestal_height * 0.5]
    pedestal_xy = [0.80, 0.00]
    pedestal_z = pedestal_height * 0.5

    ped_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=pedestal_half_extents)
    ped_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=pedestal_half_extents,
        rgbaColor=[0.75, 0.75, 0.75, 1.0],
    )
    pedestal_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=ped_col,
        baseVisualShapeIndex=ped_vis,
        basePosition=[pedestal_xy[0], pedestal_xy[1], pedestal_z],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )

    selected_part = int(part_id) if part_id is not None else random.randint(1, 14)
    object_id = p.loadURDF(
        str(benchmark_part_urdf(selected_part)),
        basePosition=[pedestal_xy[0], pedestal_xy[1], pedestal_height + 0.12],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        useFixedBase=False,
    )

    # Align bottom of the benchmark part to the pedestal top.
    aabb_min, _ = p.getAABB(object_id)
    delta_z = float(pedestal_height - float(aabb_min[2]))
    obj_pos, obj_quat = p.getBasePositionAndOrientation(object_id)
    p.resetBasePositionAndOrientation(
        object_id,
        [float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2] + delta_z)],
        obj_quat,
    )
    return pedestal_id, object_id, selected_part


def _apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    return value


def _print_controls(enable_keyboard: bool, gamepad_enabled: bool) -> None:
    print("Sawyer IK Test (Gamepad first)")
    if gamepad_enabled:
        print("Gamepad mapping:")
        print("  Left stick: XY translation (forward/backward/left/right)")
        print("  A/B: Z up/down")
        print("  RT/LT: rotate EE around its own axis (wrist joint)")
        print("  Right stick: EE roll/pitch orientation (stabilized, in-place)")
        print("  RB/LB: rotate object around z-axis")
        print("  X: auto-align selected hand reference to nearest grasp point")
        print("  Menu: quit")
    if enable_keyboard:
        print("Keyboard fallback enabled:")
        print("  Arrow keys: XY, PgUp/PgDn: Z")
        print("  Q/E yaw, W/S pitch, A/D roll")
        print("  R reset, ESC quit")


def _make_gamepad(enable_keyboard: bool):
    try:
        import pygame
    except Exception:
        if enable_keyboard:
            return None, None
        raise RuntimeError("pygame is required for gamepad control. Install it with: pip install pygame")

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        if enable_keyboard:
            pygame.quit()
            return None, None
        pygame.quit()
        raise RuntimeError("No gamepad detected. Connect an Xbox controller or run with --enable-keyboard.")

    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Using gamepad: {js.get_name()}")
    return pygame, js


def _read_gamepad_delta(js, pygame_mod, pos_step: float, rot_step: float, deadzone: float, axis_bias: dict):
    pygame_mod.event.pump()

    def axis(i: int) -> float:
        if i < js.get_numaxes():
            return float(js.get_axis(i))
        return 0.0

    left_x = _apply_deadzone(axis(0) - axis_bias.get(0, 0.0), deadzone)
    left_y = _apply_deadzone(axis(1) - axis_bias.get(1, 0.0), deadzone)
    right_x = _apply_deadzone(axis(2) - axis_bias.get(2, 0.0), deadzone)
    right_y = _apply_deadzone(axis(3) - axis_bias.get(3, 0.0), deadzone)

    def normalize_trigger(raw: float) -> float:
        # SDL mappings vary: triggers can be [-1,1] (idle -1) or [0,1] (idle 0).
        if raw < 0.0:
            v = (raw + 1.0) * 0.5
        else:
            v = raw
        return max(0.0, min(1.0, float(v)))

    lb = bool(js.get_numbuttons() > 4 and js.get_button(4))
    rb = bool(js.get_numbuttons() > 5 and js.get_button(5))
    btn_a = bool(js.get_numbuttons() > 0 and js.get_button(0))
    btn_b = bool(js.get_numbuttons() > 1 and js.get_button(1))
    btn_x = bool(js.get_numbuttons() > 2 and js.get_button(2))
    btn_y = bool(js.get_numbuttons() > 3 and js.get_button(3))
    btn_menu = bool(js.get_numbuttons() > 7 and js.get_button(7))

    lt = normalize_trigger(axis(4))
    rt = normalize_trigger(axis(5))

    d_xyz = [
        left_y * pos_step,
        left_x * pos_step,
        (1.0 if btn_a else 0.0) * pos_step - (1.0 if btn_b else 0.0) * pos_step,
    ]
    # Use right stick for orientation tilt (roll/pitch).
    d_rpy = [
        -right_x * rot_step,
        -right_y * rot_step,
        0.0,
    ]
    # Rotate around EE axis via wrist joint using RT/LT analog triggers.
    ee_roll_delta = (rt - lt) * rot_step
    # RB rotates object +z, LB rotates object -z.
    object_rot_cmd = (1.0 if rb else 0.0) - (1.0 if lb else 0.0)
    move_xyz = any(abs(v) > 1e-12 for v in d_xyz)
    rotate_pose = any(abs(v) > 1e-12 for v in d_rpy)
    rotate_wrist = abs(ee_roll_delta) > 1e-12
    return d_xyz, d_rpy, ee_roll_delta, object_rot_cmd, move_xyz, rotate_pose, rotate_wrist, btn_x, btn_y, btn_menu


def _calibrate_gamepad_bias(js, pygame_mod, samples: int = 120, sleep_s: float = 0.002) -> dict:
    # Sample neutral stick values once at startup to cancel constant drift.
    sums = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    n = 0
    for _ in range(max(1, samples)):
        pygame_mod.event.pump()
        for idx in sums:
            if idx < js.get_numaxes():
                sums[idx] += float(js.get_axis(idx))
        n += 1
        time.sleep(sleep_s)
    return {k: (v / float(n)) for k, v in sums.items()}


def _read_keyboard_delta(pos_step: float, rot_step: float):
    keys = p.getKeyboardEvents()
    esc_key = 27
    d_xyz = [0.0, 0.0, 0.0]
    d_rpy = [0.0, 0.0, 0.0]

    if p.B3G_LEFT_ARROW in keys and (keys[p.B3G_LEFT_ARROW] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
        d_xyz[1] += pos_step
    if p.B3G_RIGHT_ARROW in keys and (keys[p.B3G_RIGHT_ARROW] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
        d_xyz[1] -= pos_step
    if p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
        d_xyz[0] += pos_step
    if p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
        d_xyz[0] -= pos_step
    if p.B3G_PAGE_UP in keys and (keys[p.B3G_PAGE_UP] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
        d_xyz[2] += pos_step
    if p.B3G_PAGE_DOWN in keys and (keys[p.B3G_PAGE_DOWN] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
        d_xyz[2] -= pos_step

    if ord("Q") in keys and (keys[ord("Q")] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
        d_rpy[2] += rot_step
    if ord("E") in keys and (keys[ord("E")] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
        d_rpy[2] -= rot_step
    if ord("W") in keys and (keys[ord("W")] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
        d_rpy[1] += rot_step
    if ord("S") in keys and (keys[ord("S")] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
        d_rpy[1] -= rot_step
    if ord("A") in keys and (keys[ord("A")] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
        d_rpy[0] += rot_step
    if ord("D") in keys and (keys[ord("D")] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
        d_rpy[0] -= rot_step

    reset = bool(ord("R") in keys and (keys[ord("R")] & p.KEY_WAS_TRIGGERED))
    quit_req = bool(esc_key in keys and (keys[esc_key] & p.KEY_WAS_TRIGGERED))
    changed = any(abs(v) > 1e-12 for v in (d_xyz + d_rpy))
    return d_xyz, d_rpy, changed, reset, quit_req


def _rotate_xy(vec_xyz: list[float], yaw_rad: float) -> list[float]:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    x = float(vec_xyz[0])
    y = float(vec_xyz[1])
    return [
        c * x - s * y,
        s * x + c * y,
        float(vec_xyz[2]),
    ]


def run(
    pos_step: float,
    rot_step_deg: float,
    hz: float,
    deadzone: float,
    enable_keyboard: bool,
    part_id: int | None,
    grasp_yaml: str | None,
    plane_distance_mm: float,
    plane_size_mm: float,
    plane_thickness_mm: float,
    hand_ref_yaml: str,
    grasp_pose: str,
    auto_align_distance_mm: float,
    auto_align_seconds: float,
    object_rot_speed_deg: float,
) -> None:
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    _set_default_camera()
    p.loadURDF("plane.urdf")
    _, object_id, selected_part = _spawn_pedestal_with_benchmark_part(part_id=part_id)
    print(f"benchmark_part_id={selected_part}")
    if grasp_yaml is None:
        grasp_yaml = f"artifacts/grasp_points_part_{selected_part}.yaml"
    grasp_points_local = _load_grasp_points(grasp_yaml)
    hand_ref = _load_hand_reference_point(hand_ref_yaml, grasp_pose)
    p_hand_local = None
    n_hand_local = None
    t_hand_local = None
    b_hand_local = None
    if hand_ref is None:
        print(f"hand_reference_missing pose={grasp_pose} yaml={hand_ref_yaml}")
    else:
        p_hand_local = [float(v) for v in hand_ref["position_hand_xyz"]]
        n_hand_local = _normalize([float(v) for v in hand_ref["normal_hand_xyz"]])
        t_hand_local = _orthonormal_tangent(
            n_hand_local,
            _normalize([float(v) for v in hand_ref["tangent_hand_xyz"]]),
        )
        b_hand_local = _normalize(_cross(n_hand_local, t_hand_local))
        print(f"hand_reference_loaded pose={grasp_pose}")
    grasp_visuals = _draw_grasp_points_from_yaml(
        yaml_path=grasp_yaml,
        selected_part=selected_part,
        object_id=object_id,
        distance_mm=plane_distance_mm,
        patch_size_mm=plane_size_mm,
        patch_thickness_mm=plane_thickness_mm,
    )
    print(f"visualized_grasp_points={len(grasp_visuals)}")

    robot_id = p.loadURDF(
        str(SAWYER_URDF),
        basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, BASE_YAW]),
        useFixedBase=True,
    )
    hand_id = p.loadURDF(str(AR10_URDF))
    mount_hand_to_arm(robot_id, hand_id, ee_link="right_l6")
    apply_hand_friction(hand_id)
    hand = HandModel(hand_id)
    hand.send_q_target([0.0] * 10)

    helper = SawyerHelper(robot_id, ee_link_name="right_l6")
    _home_sawyer(robot_id, helper)

    target_xyz, target_rpy = helper.ee_pose()
    target_xyz = target_xyz.copy()
    target_rpy = target_rpy.copy()
    target_q = [p.getJointState(robot_id, j)[0] for j in helper.joints]
    wrist_low, wrist_high = helper.limits[-1]
    wrist_target = float(target_q[-1])
    x_latch = False
    auto_align_active = False
    auto_align_gp: dict | None = None

    pygame_mod, gamepad = _make_gamepad(enable_keyboard=enable_keyboard)
    axis_bias = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    if gamepad is not None:
        axis_bias = _calibrate_gamepad_bias(gamepad, pygame_mod)
    _print_controls(enable_keyboard=enable_keyboard, gamepad_enabled=gamepad is not None)

    dt = 1.0 / float(hz)
    rot_step = rot_step_deg * 3.141592653589793 / 180.0

    try:
        y_latch = False
        while p.isConnected(client_id):
            pose_changed = False
            req_move_xyz = False
            req_rotate_pose = False
            reset_req = False
            quit_req = False

            if gamepad is not None:
                (
                    d_xyz,
                    d_rpy,
                    ee_roll_delta,
                    object_rot_cmd,
                    move_xyz,
                    rotate_pose,
                    rotate_wrist,
                    btn_x,
                    btn_y,
                    btn_menu,
                ) = _read_gamepad_delta(
                    gamepad,
                    pygame_mod,
                    pos_step=pos_step,
                    rot_step=rot_step,
                    deadzone=deadzone,
                    axis_bias=axis_bias,
                )
                d_xyz_world = _rotate_xy(d_xyz, BASE_YAW)
                target_xyz[0] += d_xyz_world[0]
                target_xyz[1] += d_xyz_world[1]
                target_xyz[2] += d_xyz_world[2]
                # Keep orientation updates smaller and separated from translation for IK stability.
                d_rpy_scaled = [float(v) * ROTATION_STEP_SCALE for v in d_rpy]
                target_rpy[0] += d_rpy_scaled[0]
                target_rpy[1] += d_rpy_scaled[1]
                target_rpy[2] += d_rpy_scaled[2]
                pose_changed = pose_changed or move_xyz or rotate_pose
                req_move_xyz = req_move_xyz or move_xyz
                req_rotate_pose = req_rotate_pose or rotate_pose
                wrist_target = max(wrist_low, min(wrist_high, wrist_target + ee_roll_delta))
                if abs(object_rot_cmd) > 1e-12:
                    obj_pos_now, obj_quat_now = p.getBasePositionAndOrientation(object_id)
                    obj_rpy_now = list(p.getEulerFromQuaternion(obj_quat_now))
                    obj_rpy_now[2] += float(object_rot_cmd) * math.radians(float(object_rot_speed_deg)) * dt
                    new_quat = p.getQuaternionFromEuler(obj_rpy_now)
                    p.resetBasePositionAndOrientation(object_id, obj_pos_now, new_quat)
                if (move_xyz or rotate_pose or abs(ee_roll_delta) > 1e-12 or abs(object_rot_cmd) > 1e-12) and auto_align_active:
                    auto_align_active = False
                    auto_align_gp = None
                    print("auto_align_cancelled=manual_input")
                if btn_y and not y_latch:
                    sawyer_q = [p.getJointState(robot_id, j)[0] for j in helper.joints]
                    hand_q = hand.get_q_measured()
                    print("sawyer_joints =", [round(float(v), 6) for v in sawyer_q])
                    print("hand_q_measured =", [round(float(v), 6) for v in hand_q])
                    y_latch = True
                if not btn_y:
                    y_latch = False
                quit_req = quit_req or btn_menu

                if btn_x and not x_latch:
                    if not grasp_points_local:
                        print("auto_align_skipped: no grasp points loaded")
                    elif p_hand_local is None or n_hand_local is None or t_hand_local is None or b_hand_local is None:
                        print("auto_align_skipped: hand reference not loaded")
                    else:
                        obj_pos_now, obj_quat_now = p.getBasePositionAndOrientation(object_id)
                        hand_pos_now, hand_quat_now = p.getBasePositionAndOrientation(hand_id)
                        current_ref_world = _transform_local_to_world(p_hand_local, list(hand_pos_now), list(hand_quat_now))

                        best = None
                        for gp in grasp_points_local:
                            p_obj = [float(v) for v in gp["position_obj_xyz"]]
                            p_world = _transform_local_to_world(p_obj, list(obj_pos_now), list(obj_quat_now))
                            dx = p_world[0] - current_ref_world[0]
                            dy = p_world[1] - current_ref_world[1]
                            dz = p_world[2] - current_ref_world[2]
                            d2 = dx * dx + dy * dy + dz * dz
                            if best is None or d2 < best[0]:
                                best = (d2, gp, p_world)

                        _, gp_best, gp_world = best
                        auto_align_gp = gp_best
                        auto_align_active = True
                        print(
                            "auto_align_start target_gp="
                            f"{gp_best.get('id')} current_ref_world={[round(v, 4) for v in current_ref_world]}"
                        )
                x_latch = btn_x

            if enable_keyboard:
                d_xyz, d_rpy, changed, reset_kb, quit_kb = _read_keyboard_delta(pos_step, rot_step)
                d_xyz_world = _rotate_xy(d_xyz, BASE_YAW)
                target_xyz[0] += d_xyz_world[0]
                target_xyz[1] += d_xyz_world[1]
                target_xyz[2] += d_xyz_world[2]
                target_rpy[0] += d_rpy[0]
                target_rpy[1] += d_rpy[1]
                target_rpy[2] += d_rpy[2]
                pose_changed = pose_changed or changed
                req_move_xyz = req_move_xyz or any(abs(v) > 1e-12 for v in d_xyz)
                req_rotate_pose = req_rotate_pose or any(abs(v) > 1e-12 for v in d_rpy)
                reset_req = reset_req or reset_kb
                quit_req = quit_req or quit_kb

            if reset_req:
                target_xyz, target_rpy = helper.ee_pose()
                target_xyz = target_xyz.copy()
                target_rpy = target_rpy.copy()
                target_q = [p.getJointState(robot_id, j)[0] for j in helper.joints]
                wrist_target = float(target_q[-1])
                pose_changed = False

            if quit_req:
                break

            if (
                auto_align_active
                and auto_align_gp is not None
                and p_hand_local is not None
                and n_hand_local is not None
                and t_hand_local is not None
                and b_hand_local is not None
            ):
                obj_pos_now, obj_quat_now = p.getBasePositionAndOrientation(object_id)
                hand_pos_now, hand_quat_now = p.getBasePositionAndOrientation(hand_id)
                ee_xyz_now, ee_rpy_now = helper.ee_pose()
                ee_quat_now = p.getQuaternionFromEuler(ee_rpy_now)

                p_obj = [float(v) for v in auto_align_gp["position_obj_xyz"]]
                n_obj_world = _normalize(
                    _rotate_local_to_world(
                        [float(v) for v in auto_align_gp["normal_obj_xyz"]],
                        list(obj_quat_now),
                    )
                )
                gp_world = _transform_local_to_world(p_obj, list(obj_pos_now), list(obj_quat_now))
                target_ref_world = [
                    gp_world[0] + float(auto_align_distance_mm) / 1000.0 * n_obj_world[0],
                    gp_world[1] + float(auto_align_distance_mm) / 1000.0 * n_obj_world[1],
                    gp_world[2] + float(auto_align_distance_mm) / 1000.0 * n_obj_world[2],
                ]
                current_ref_world = _transform_local_to_world(p_hand_local, list(hand_pos_now), list(hand_quat_now))
                err = [
                    target_ref_world[0] - current_ref_world[0],
                    target_ref_world[1] - current_ref_world[1],
                    target_ref_world[2] - current_ref_world[2],
                ]
                err_norm = math.sqrt(err[0] * err[0] + err[1] * err[1] + err[2] * err[2])
                if err_norm > 1e-12:
                    step = min(float(AUTO_ALIGN_MAX_POS_STEP_M), err_norm)
                    pos_step_vec = [err[0] * step / err_norm, err[1] * step / err_norm, err[2] * step / err_norm]
                else:
                    pos_step_vec = [0.0, 0.0, 0.0]

                current_hand_rot = _mat_from_quat(list(hand_quat_now))
                current_t_world = _normalize(_mat_vec_mul(current_hand_rot, t_hand_local))
                current_n_world = _normalize(_mat_vec_mul(current_hand_rot, n_hand_local))
                n_hand_world_target = [-n_obj_world[0], -n_obj_world[1], -n_obj_world[2]]
                t_hand_world_target = _orthonormal_tangent(n_hand_world_target, current_t_world)
                b_hand_world_target = _normalize(_cross(n_hand_world_target, t_hand_world_target))
                t_hand_world_target = _normalize(_cross(b_hand_world_target, n_hand_world_target))
                r_world_target = _mat_from_basis(t_hand_world_target, b_hand_world_target, n_hand_world_target)
                r_hand_local = _mat_from_basis(t_hand_local, b_hand_local, n_hand_local)
                r_hand_base_world = _mat_mul(r_world_target, _mat_transpose(r_hand_local))
                hand_quat_target = _quat_from_mat(r_hand_base_world)
                blended_hand_quat = _quat_slerp(list(hand_quat_now), hand_quat_target, float(AUTO_ALIGN_ROT_ALPHA))
                blended_hand_pos = [
                    float(hand_pos_now[0]) + pos_step_vec[0],
                    float(hand_pos_now[1]) + pos_step_vec[1],
                    float(hand_pos_now[2]) + pos_step_vec[2],
                ]

                ee_to_hand_pos, ee_to_hand_quat = p.multiplyTransforms(
                    *p.invertTransform(ee_xyz_now.tolist(), ee_quat_now),
                    hand_pos_now,
                    hand_quat_now,
                )
                hand_to_ee_pos, hand_to_ee_quat = p.invertTransform(ee_to_hand_pos, ee_to_hand_quat)
                ee_pos_target, ee_quat_target = p.multiplyTransforms(
                    blended_hand_pos,
                    blended_hand_quat,
                    hand_to_ee_pos,
                    hand_to_ee_quat,
                )
                target_xyz = [float(v) for v in ee_pos_target]
                target_rpy = _quat_to_rpy(list(ee_quat_target))
                pose_changed = True

                normal_dot = max(-1.0, min(1.0, _dot(current_n_world, n_hand_world_target)))
                normal_err_deg = math.degrees(math.acos(normal_dot))
                if err_norm <= float(AUTO_ALIGN_POS_TOL_M) and normal_err_deg <= float(AUTO_ALIGN_NORMAL_TOL_DEG):
                    auto_align_active = False
                    auto_align_gp = None
                    print(
                        "auto_align_done "
                        f"ref_error_mm={round(1000.0 * err_norm, 2)} "
                        f"normal_err_deg={round(normal_err_deg, 2)}"
                    )

            if grasp_visuals:
                _update_grasp_visuals(object_id=object_id, visuals=grasp_visuals)

            # Re-run IK only when Cartesian EE target changed.
            if pose_changed:
                ee_xyz_now, ee_rpy_now = helper.ee_pose()
                if req_move_xyz:
                    # Position has priority: hold current EE orientation while translating.
                    q = helper._ik(target_xyz, ee_rpy_now)
                elif req_rotate_pose:
                    # Rotate in-place: keep current EE position while adjusting orientation.
                    q = helper._ik(ee_xyz_now, target_rpy)
                    target_xyz = ee_xyz_now.copy()
                else:
                    q = helper._ik(target_xyz, target_rpy)
                target_q = q.tolist()
            target_q[-1] = wrist_target

            p.setJointMotorControlArray(
                bodyUniqueId=robot_id,
                jointIndices=helper.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_q,
                positionGains=[ARM_CTRL_GAIN] * len(helper.joints),
                forces=[ARM_CTRL_FORCE] * len(helper.joints),
            )
            p.stepSimulation()
            time.sleep(dt)
    finally:
        try:
            if pygame_mod is not None:
                pygame_mod.quit()
        except Exception:
            pass
        if p.isConnected(client_id):
            p.disconnect(client_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos-step", type=float, default=0.003, help="Cartesian step in meters per control tick")
    parser.add_argument("--rot-step-deg", type=float, default=2.0, help="Euler step in degrees per control tick")
    parser.add_argument("--hz", type=float, default=120.0, help="Control loop frequency")
    parser.add_argument("--deadzone", type=float, default=0.30, help="Analog deadzone for gamepad sticks")
    parser.add_argument("--enable-keyboard", action="store_true", help="Enable keyboard input in addition to gamepad")
    parser.add_argument("--part-id", type=int, default=None, help="Benchmark part id [1..14]. If omitted, random.")
    parser.add_argument(
        "--grasp-yaml",
        type=str,
        default=None,
        help="Path to grasp points yaml. Default: artifacts/grasp_points_part_<part-id>.yaml",
    )
    parser.add_argument("--plane-distance-mm", type=float, default=20.0, help="Projected plane offset distance")
    parser.add_argument("--plane-size-mm", type=float, default=35.0, help="Projected plane size")
    parser.add_argument("--plane-thickness-mm", type=float, default=1.0, help="Projected plane thickness")
    parser.add_argument("--hand-ref-yaml", type=str, default="artifacts/hand_reference_points.yaml")
    parser.add_argument("--grasp-pose", type=str, default="tripod")
    parser.add_argument("--auto-align-distance-mm", type=float, default=20.0)
    parser.add_argument("--auto-align-seconds", type=float, default=1.2)
    parser.add_argument("--object-rot-speed-deg", type=float, default=55.0)
    args = parser.parse_args()
    run(
        pos_step=args.pos_step,
        rot_step_deg=args.rot_step_deg,
        hz=args.hz,
        deadzone=args.deadzone,
        enable_keyboard=args.enable_keyboard,
        part_id=args.part_id,
        grasp_yaml=args.grasp_yaml,
        plane_distance_mm=args.plane_distance_mm,
        plane_size_mm=args.plane_size_mm,
        plane_thickness_mm=args.plane_thickness_mm,
        hand_ref_yaml=args.hand_ref_yaml,
        grasp_pose=args.grasp_pose,
        auto_align_distance_mm=args.auto_align_distance_mm,
        auto_align_seconds=args.auto_align_seconds,
        object_rot_speed_deg=args.object_rot_speed_deg,
    )


if __name__ == "__main__":
    main()
