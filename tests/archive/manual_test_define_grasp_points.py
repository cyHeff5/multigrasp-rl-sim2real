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


def _dot(a: list[float], b: list[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _project_on_plane(v: list[float], normal: list[float]) -> list[float]:
    d = _dot(v, normal)
    return [v[0] - d * normal[0], v[1] - d * normal[1], v[2] - d * normal[2]]


def _rotate_world_to_object(vec_world: list[float], obj_quat: list[float]) -> list[float]:
    # getMatrixFromQuaternion gives local->world; transpose maps world->local.
    m = p.getMatrixFromQuaternion(obj_quat)
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


def _world_point_to_object(point_world: list[float], obj_pos: list[float], obj_quat: list[float]) -> list[float]:
    inv_pos, inv_quat = p.invertTransform(obj_pos, obj_quat)
    p_obj, _ = p.multiplyTransforms(inv_pos, inv_quat, point_world, [0.0, 0.0, 0.0, 1.0])
    return [float(p_obj[0]), float(p_obj[1]), float(p_obj[2])]


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
    # Prefer global +Z projected to surface; fallback to +X when near singular.
    candidate = _project_on_plane([0.0, 0.0, 1.0], normal_world)
    if math.sqrt(sum(v * v for v in candidate)) < 1e-6:
        candidate = _project_on_plane([1.0, 0.0, 0.0], normal_world)
    return _normalize(candidate)


def run(
    part_id: int,
    output_yaml: str,
    hz: float,
    move_speed_deg: float,
    deadzone: float,
    marker_radius: float,
    object_xyz: list[float],
    object_rpy_deg: list[float],
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

    p.resetDebugVisualizerCamera(
        cameraDistance=0.7,
        cameraYaw=50.0,
        cameraPitch=-30.0,
        cameraTargetPosition=[object_xyz[0], object_xyz[1], object_xyz[2]],
    )

    obj_quat = p.getQuaternionFromEuler([math.radians(v) for v in object_rpy_deg])
    obj_id = p.loadURDF(
        str(benchmark_part_urdf(part_id)),
        basePosition=[float(v) for v in object_xyz],
        baseOrientation=obj_quat,
        useFixedBase=True,
    )

    red_id = _make_marker(marker_radius, [1.0, 0.1, 0.1, 1.0])
    saved_points: list[dict] = []
    green_marker_ids: list[int] = []

    print("Controls:")
    print("  Left stick X/Y: move red point over surface")
    print("  A: save grasp point (adds green marker + writes yaml)")
    print("  Menu: quit")

    dt = 1.0 / max(1e-6, float(hz))
    move_speed = math.radians(float(move_speed_deg))
    ray_len = 0.6

    # Start from +X direction.
    az = 0.0
    el = 0.0
    a_latch = False

    out_path = Path(output_yaml)

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

            dir_world = [
                math.cos(el) * math.cos(az),
                math.cos(el) * math.sin(az),
                math.sin(el),
            ]
            obj_pos, obj_quat_now = p.getBasePositionAndOrientation(obj_id)
            ray_from = [
                float(obj_pos[0] + ray_len * dir_world[0]),
                float(obj_pos[1] + ray_len * dir_world[1]),
                float(obj_pos[2] + ray_len * dir_world[2]),
            ]
            ray_to = [
                float(obj_pos[0] - ray_len * dir_world[0]),
                float(obj_pos[1] - ray_len * dir_world[1]),
                float(obj_pos[2] - ray_len * dir_world[2]),
            ]
            hit = p.rayTest(ray_from, ray_to)[0]
            hit_id = int(hit[0])
            if hit_id == obj_id:
                hit_pos = [float(hit[3][0]), float(hit[3][1]), float(hit[3][2])]
                hit_normal_world = _normalize([float(hit[4][0]), float(hit[4][1]), float(hit[4][2])])
                p.resetBasePositionAndOrientation(red_id, hit_pos, [0.0, 0.0, 0.0, 1.0])
            else:
                p.stepSimulation()
                time.sleep(dt)
                continue

            btn_a = bool(js.get_numbuttons() > 0 and js.get_button(0))
            btn_menu = bool(js.get_numbuttons() > 7 and js.get_button(7))
            if btn_menu:
                break

            if btn_a and not a_latch:
                tangent_world = _choose_tangent_world(hit_normal_world)
                p_obj = _world_point_to_object(hit_pos, list(obj_pos), list(obj_quat_now))
                n_obj = _normalize(_rotate_world_to_object(hit_normal_world, list(obj_quat_now)))
                t_obj = _normalize(_rotate_world_to_object(tangent_world, list(obj_quat_now)))

                gp_id = f"gp_{len(saved_points) + 1:03d}"
                gp = {
                    "id": gp_id,
                    "position_obj_xyz": [round(v, 6) for v in p_obj],
                    "normal_obj_xyz": [round(v, 6) for v in n_obj],
                    "tangent_obj_xyz": [round(v, 6) for v in t_obj],
                }
                saved_points.append(gp)

                green_id = _make_marker(marker_radius, [0.1, 0.9, 0.1, 1.0])
                p.resetBasePositionAndOrientation(green_id, hit_pos, [0.0, 0.0, 0.0, 1.0])
                green_marker_ids.append(green_id)

                payload = {
                    "part_id": int(part_id),
                    "part_urdf": str(benchmark_part_urdf(part_id)),
                    "object_pose_world": {
                        "position_xyz": [float(v) for v in obj_pos],
                        "orientation_xyzw": [float(v) for v in obj_quat_now],
                    },
                    "grasp_points": saved_points,
                }
                _save_yaml(out_path, payload)
                print(f"saved {gp_id} -> {out_path}")
                print(gp)
            a_latch = btn_a

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
    parser.add_argument("--part-id", type=int, required=True, help="Benchmark part id in [1, 14]")
    parser.add_argument("--output-yaml", default="artifacts/grasp_points.yaml")
    parser.add_argument("--hz", type=float, default=120.0)
    parser.add_argument("--move-speed-deg", type=float, default=140.0, help="Surface cursor angular speed")
    parser.add_argument("--deadzone", type=float, default=0.20)
    parser.add_argument("--marker-radius", type=float, default=0.004)
    parser.add_argument("--obj-x", type=float, default=0.40)
    parser.add_argument("--obj-y", type=float, default=0.00)
    parser.add_argument("--obj-z", type=float, default=0.05)
    parser.add_argument("--obj-roll-deg", type=float, default=0.0)
    parser.add_argument("--obj-pitch-deg", type=float, default=0.0)
    parser.add_argument("--obj-yaw-deg", type=float, default=0.0)
    args = parser.parse_args()

    run(
        part_id=args.part_id,
        output_yaml=args.output_yaml,
        hz=args.hz,
        move_speed_deg=args.move_speed_deg,
        deadzone=args.deadzone,
        marker_radius=args.marker_radius,
        object_xyz=[args.obj_x, args.obj_y, args.obj_z],
        object_rpy_deg=[args.obj_roll_deg, args.obj_pitch_deg, args.obj_yaw_deg],
    )


if __name__ == "__main__":
    main()
