from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pybullet as p
import pybullet_data
import yaml

from src.sim.assets import AR10_URDF
from src.sim.hand_model import HandModel


GRASP_POSES = [
    "medium_wrap",
    "tripod",
    "power_sphere",
    "thumb_1_finger",
]


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    if n <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [float(x) / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _project_on_plane(v: list[float], normal: list[float]) -> list[float]:
    d = _dot(v, normal)
    return [
        float(v[0] - d * normal[0]),
        float(v[1] - d * normal[1]),
        float(v[2] - d * normal[2]),
    ]


def _rotate_world_to_hand(vec_world: list[float], hand_quat: list[float]) -> list[float]:
    m = p.getMatrixFromQuaternion(hand_quat)
    r00, r01, r02 = float(m[0]), float(m[1]), float(m[2])
    r10, r11, r12 = float(m[3]), float(m[4]), float(m[5])
    r20, r21, r22 = float(m[6]), float(m[7]), float(m[8])
    x = float(vec_world[0])
    y = float(vec_world[1])
    z = float(vec_world[2])
    return [
        r00 * x + r10 * y + r20 * z,
        r01 * x + r11 * y + r21 * z,
        r02 * x + r12 * y + r22 * z,
    ]


def _world_point_to_hand(point_world: list[float], hand_pos: list[float], hand_quat: list[float]) -> list[float]:
    inv_pos, inv_quat = p.invertTransform(hand_pos, hand_quat)
    p_hand, _ = p.multiplyTransforms(inv_pos, inv_quat, point_world, [0.0, 0.0, 0.0, 1.0])
    return [float(p_hand[0]), float(p_hand[1]), float(p_hand[2])]


def _make_marker(radius: float, rgba: list[float]) -> int:
    vis = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=float(radius),
        rgbaColor=[float(c) for c in rgba],
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _choose_tangent_world(normal_world: list[float]) -> list[float]:
    candidate = _project_on_plane([0.0, 0.0, 1.0], normal_world)
    if math.sqrt(sum(v * v for v in candidate)) < 1e-6:
        candidate = _project_on_plane([0.0, 1.0, 0.0], normal_world)
    return _normalize(candidate)


def run(
    output_yaml: str,
    hz: float,
    move_speed_deg: float,
    deadzone: float,
    marker_radius: float,
    hand_xyz: list[float],
    hand_rpy_deg: list[float],
) -> None:
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError("pygame is required. Install with: pip install pygame") from exc

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        raise RuntimeError("No gamepad found. Connect an Xbox controller and retry.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Controller: {js.get_name()}")

    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf")

    hand_quat = p.getQuaternionFromEuler([math.radians(v) for v in hand_rpy_deg])
    hand_id = p.loadURDF(
        str(AR10_URDF),
        basePosition=[float(v) for v in hand_xyz],
        baseOrientation=hand_quat,
        useFixedBase=True,
    )
    HandModel(hand_id).reset_open_pose()

    p.resetDebugVisualizerCamera(
        cameraDistance=0.35,
        cameraYaw=35.0,
        cameraPitch=-25.0,
        cameraTargetPosition=[hand_xyz[0], hand_xyz[1], hand_xyz[2] + 0.03],
    )

    red_id = _make_marker(marker_radius, [1.0, 0.1, 0.1, 1.0])
    green_markers: dict[str, int] = {}
    saved_points: dict[str, dict] = {}

    print("Controls:")
    print("  Left stick X/Y: move red point over hand surface")
    print("  LB/RB: select grasp pose")
    print("  A: save reference point for active grasp pose")
    print("  Menu: quit")

    dt = 1.0 / max(1e-6, float(hz))
    move_speed = math.radians(float(move_speed_deg))
    ray_len = 0.25

    az = 0.0
    el = 0.0
    active_pose_idx = 0
    a_latch = False
    lb_latch = False
    rb_latch = False
    out_path = Path(output_yaml)
    debug_text_id = -1

    try:
        while p.isConnected(client_id):
            pygame.event.pump()

            ax0 = float(js.get_axis(0)) if js.get_numaxes() > 0 else 0.0
            ax1 = float(js.get_axis(1)) if js.get_numaxes() > 1 else 0.0
            if abs(ax0) < deadzone:
                ax0 = 0.0
            if abs(ax1) < deadzone:
                ax1 = 0.0

            az += ax0 * move_speed * dt
            el += -ax1 * move_speed * dt
            el = max(math.radians(-85.0), min(math.radians(85.0), el))

            lb_pressed = bool(js.get_numbuttons() > 4 and js.get_button(4))
            rb_pressed = bool(js.get_numbuttons() > 5 and js.get_button(5))
            a_pressed = bool(js.get_numbuttons() > 0 and js.get_button(0))
            menu_pressed = bool(js.get_numbuttons() > 7 and js.get_button(7))

            if lb_pressed and not lb_latch:
                active_pose_idx = (active_pose_idx - 1) % len(GRASP_POSES)
                print(f"active_pose={GRASP_POSES[active_pose_idx]}")
            if rb_pressed and not rb_latch:
                active_pose_idx = (active_pose_idx + 1) % len(GRASP_POSES)
                print(f"active_pose={GRASP_POSES[active_pose_idx]}")
            lb_latch = lb_pressed
            rb_latch = rb_pressed

            if menu_pressed:
                break

            dir_world = [
                math.cos(el) * math.cos(az),
                math.cos(el) * math.sin(az),
                math.sin(el),
            ]
            hand_pos, hand_quat_now = p.getBasePositionAndOrientation(hand_id)
            ray_from = [
                float(hand_pos[0] + ray_len * dir_world[0]),
                float(hand_pos[1] + ray_len * dir_world[1]),
                float(hand_pos[2] + ray_len * dir_world[2]),
            ]
            ray_to = [
                float(hand_pos[0] - ray_len * dir_world[0]),
                float(hand_pos[1] - ray_len * dir_world[1]),
                float(hand_pos[2] - ray_len * dir_world[2]),
            ]
            hit = p.rayTest(ray_from, ray_to)[0]
            hit_id = int(hit[0])
            if hit_id != hand_id:
                p.stepSimulation()
                time.sleep(dt)
                continue

            hit_pos = [float(hit[3][0]), float(hit[3][1]), float(hit[3][2])]
            hit_normal_world = _normalize([float(hit[4][0]), float(hit[4][1]), float(hit[4][2])])
            p.resetBasePositionAndOrientation(red_id, hit_pos, [0.0, 0.0, 0.0, 1.0])

            if a_pressed and not a_latch:
                tangent_world = _choose_tangent_world(hit_normal_world)
                p_hand = _world_point_to_hand(hit_pos, list(hand_pos), list(hand_quat_now))
                n_hand = _normalize(_rotate_world_to_hand(hit_normal_world, list(hand_quat_now)))
                t_hand = _normalize(_rotate_world_to_hand(tangent_world, list(hand_quat_now)))
                pose_name = GRASP_POSES[active_pose_idx]
                entry = {
                    "position_hand_xyz": [round(v, 6) for v in p_hand],
                    "normal_hand_xyz": [round(v, 6) for v in n_hand],
                    "tangent_hand_xyz": [round(v, 6) for v in t_hand],
                }
                saved_points[pose_name] = entry

                if pose_name not in green_markers:
                    green_markers[pose_name] = _make_marker(marker_radius, [0.1, 0.9, 0.1, 1.0])
                p.resetBasePositionAndOrientation(green_markers[pose_name], hit_pos, [0.0, 0.0, 0.0, 1.0])

                payload = {
                    "hand_urdf": str(AR10_URDF),
                    "hand_pose_world": {
                        "position_xyz": [float(v) for v in hand_pos],
                        "orientation_xyzw": [float(v) for v in hand_quat_now],
                    },
                    "reference_points": saved_points,
                }
                _save_yaml(out_path, payload)
                print(f"saved {pose_name} -> {out_path}")
                print(entry)
            a_latch = a_pressed

            # Show active pose above hand.
            debug_text_id = p.addUserDebugText(
                text=f"active: {GRASP_POSES[active_pose_idx]}",
                textPosition=[hand_pos[0], hand_pos[1], hand_pos[2] + 0.12],
                textColorRGB=[1.0, 1.0, 1.0],
                textSize=1.2,
                lifeTime=0.0,
                replaceItemUniqueId=debug_text_id,
            )

            p.stepSimulation()
            time.sleep(dt)
    finally:
        try:
            pygame.quit()
        except Exception:
            pass
        if p.isConnected(client_id):
            p.disconnect(client_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-yaml", default="artifacts/hand_reference_points.yaml")
    parser.add_argument("--hz", type=float, default=120.0)
    parser.add_argument("--move-speed-deg", type=float, default=140.0, help="Surface cursor angular speed")
    parser.add_argument("--deadzone", type=float, default=0.20)
    parser.add_argument("--marker-radius", type=float, default=0.004)
    parser.add_argument("--hand-x", type=float, default=0.0)
    parser.add_argument("--hand-y", type=float, default=0.0)
    parser.add_argument("--hand-z", type=float, default=0.18)
    parser.add_argument("--hand-roll-deg", type=float, default=90.0)
    parser.add_argument("--hand-pitch-deg", type=float, default=0.0)
    parser.add_argument("--hand-yaw-deg", type=float, default=0.0)
    args = parser.parse_args()

    run(
        output_yaml=args.output_yaml,
        hz=args.hz,
        move_speed_deg=args.move_speed_deg,
        deadzone=args.deadzone,
        marker_radius=args.marker_radius,
        hand_xyz=[args.hand_x, args.hand_y, args.hand_z],
        hand_rpy_deg=[args.hand_roll_deg, args.hand_pitch_deg, args.hand_yaw_deg],
    )


if __name__ == "__main__":
    main()
