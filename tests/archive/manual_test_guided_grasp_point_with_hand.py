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

HAND_CLOSE_SPEED_PER_SEC = 1.0


def _make_body_non_colliding(body_id: int) -> None:
    # Disable collisions for base (-1) and all links.
    for link in range(-1, p.getNumJoints(int(body_id))):
        p.setCollisionFilterGroupMask(int(body_id), int(link), 0, 0)


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
    a = _normalize(axis)
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    term1 = _scale(v, c)
    term2 = _scale(_cross(a, v), s)
    term3 = _scale(a, _dot(a, v) * (1.0 - c))
    return _add(_add(term1, term2), term3)


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


def _world_to_local_point(point_world: list[float], pos_world: list[float], quat_world: list[float]) -> list[float]:
    inv_pos, inv_quat = p.invertTransform(pos_world, quat_world)
    p_local, _ = p.multiplyTransforms(inv_pos, inv_quat, point_world, [0.0, 0.0, 0.0, 1.0])
    return [float(p_local[0]), float(p_local[1]), float(p_local[2])]


def _rotate_world_to_local(vec_world: list[float], quat_world: list[float]) -> list[float]:
    m = p.getMatrixFromQuaternion(quat_world)
    x = float(vec_world[0])
    y = float(vec_world[1])
    z = float(vec_world[2])
    # transpose(R) * v
    return [
        float(m[0]) * x + float(m[3]) * y + float(m[6]) * z,
        float(m[1]) * x + float(m[4]) * y + float(m[7]) * z,
        float(m[2]) * x + float(m[5]) * y + float(m[8]) * z,
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


def _load_hand_refs(path: str) -> dict[str, dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    refs = dict(data.get("reference_points", {}))
    if not refs:
        raise RuntimeError(f"No reference_points in {path}")
    return refs


def _spawn_pedestal_and_part(part_id: int, x: float, y: float, pedestal_h: float) -> tuple[int, int]:
    # Round platform with 4 cm diameter.
    ped_radius = 0.02
    ped_half_h = pedestal_h * 0.5
    ped_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=ped_radius, height=pedestal_h)
    ped_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=ped_radius, length=pedestal_h, rgbaColor=[0.85, 0.85, 0.85, 1.0])
    ped_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=ped_col,
        baseVisualShapeIndex=ped_vis,
        basePosition=[float(x), float(y), ped_half_h],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )
    obj_id = p.loadURDF(
        str(benchmark_part_urdf(part_id)),
        basePosition=[float(x), float(y), pedestal_h + 0.12],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        useFixedBase=True,
    )
    # Align bottom to top of pedestal.
    aabb_min, _ = p.getAABB(obj_id)
    pos, quat = p.getBasePositionAndOrientation(obj_id)
    dz = float(pedestal_h - float(aabb_min[2]))
    p.resetBasePositionAndOrientation(obj_id, [float(pos[0]), float(pos[1]), float(pos[2] + dz)], quat)
    return ped_id, obj_id


def _save_pregrasp_record(path: str, entry: dict) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    records = list(data.get("records", []))
    records.append(entry)
    data["records"] = records
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return len(records)


def _append_grasp_point(
    path: str,
    part_id: int,
    grasp_point: dict,
    object_pose_world: dict,
    part_urdf: str,
) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    data["part_id"] = int(part_id)
    data["part_urdf"] = str(part_urdf)
    data["object_pose_world"] = object_pose_world

    points = list(data.get("grasp_points", []))
    next_idx = len(points) + 1
    point = dict(grasp_point)
    point["id"] = f"gp_{next_idx:03d}"
    points.append(point)
    data["grasp_points"] = points

    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return next_idx


def run(
    part_id: int,
    hand_ref_yaml: str,
    initial_ref: str | None,
    initial_distance_mm: float,
    point_speed_deg: float,
    deadzone: float,
    obj_x: float,
    obj_y: float,
    pedestal_h: float,
) -> None:
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError("pygame is required. Install with: pip install pygame") from exc

    refs = _load_hand_refs(hand_ref_yaml)
    ref_names = sorted(refs.keys())
    if initial_ref is not None and initial_ref in refs:
        ref_idx = ref_names.index(initial_ref)
    else:
        ref_idx = 0

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        raise RuntimeError("No gamepad found.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Controller: {js.get_name()}")

    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    plane_id = p.loadURDF("plane.urdf")
    _make_body_non_colliding(plane_id)
    ped_id, obj_id = _spawn_pedestal_and_part(part_id, obj_x, obj_y, pedestal_h)
    _make_body_non_colliding(ped_id)

    p.resetDebugVisualizerCamera(
        cameraDistance=0.9,
        cameraYaw=45.0,
        cameraPitch=-28.0,
        cameraTargetPosition=[float(obj_x), float(obj_y), float(pedestal_h + 0.06)],
    )

    hand_id = p.loadURDF(str(AR10_URDF), basePosition=[obj_x - 0.25, obj_y, pedestal_h + 0.2], useFixedBase=True)
    _make_body_non_colliding(hand_id)
    hand = HandModel(hand_id)
    hand_scalar = 0.0
    hand.send_q_target([hand_scalar] * 10)

    red_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.005, rgbaColor=[1.0, 0.1, 0.1, 1.0])
    point_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=red_vis,
        basePosition=[obj_x, obj_y, pedestal_h + 0.1],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )

    print("Controls:")
    print("  Left stick: move grasp point slowly over object surface")
    print("  Right stick Y: open/close hand")
    print("  RB/LB: distance +/- 5 mm")
    print("  RT/LT: next/prev hand reference point")
    print("  Y: hand twist +90 deg")
    print("  A: save pre-grasp record")
    print("  Menu: quit")

    dt = 1.0 / 120.0
    ray_len = 0.6
    az = 0.0
    el = 0.0
    distance_mm = float(initial_distance_mm)
    twist_steps = 0
    lb_latch = False
    rb_latch = False
    a_latch = False
    y_latch = False
    lt_latch = False
    rt_latch = False
    text_id = -1

    def norm_trigger(raw: float) -> float:
        if raw < 0.0:
            return max(0.0, min(1.0, (raw + 1.0) * 0.5))
        return max(0.0, min(1.0, raw))

    try:
        while p.isConnected(client_id):
            pygame.event.pump()
            ax0 = float(js.get_axis(0)) if js.get_numaxes() > 0 else 0.0
            ax1 = float(js.get_axis(1)) if js.get_numaxes() > 1 else 0.0
            right_y = float(js.get_axis(3)) if js.get_numaxes() > 3 else 0.0
            if abs(ax0) < float(deadzone):
                ax0 = 0.0
            if abs(ax1) < float(deadzone):
                ax1 = 0.0
            if abs(right_y) < float(deadzone):
                right_y = 0.0

            az += ax0 * math.radians(float(point_speed_deg)) * dt
            el += -ax1 * math.radians(float(point_speed_deg)) * dt
            el = max(math.radians(-85.0), min(math.radians(85.0), el))
            hand_scalar = max(
                0.0,
                min(
                    1.0,
                    hand_scalar + (-right_y) * float(HAND_CLOSE_SPEED_PER_SEC) * dt,
                ),
            )
            hand.send_q_target([hand_scalar] * 10)

            lb = bool(js.get_numbuttons() > 4 and js.get_button(4))
            rb = bool(js.get_numbuttons() > 5 and js.get_button(5))
            y_btn = bool(js.get_numbuttons() > 3 and js.get_button(3))
            a_btn = bool(js.get_numbuttons() > 0 and js.get_button(0))
            menu = bool(js.get_numbuttons() > 7 and js.get_button(7))
            lt = norm_trigger(float(js.get_axis(4))) if js.get_numaxes() > 4 else 0.0
            rt = norm_trigger(float(js.get_axis(5))) if js.get_numaxes() > 5 else 0.0
            lt_pressed = lt > 0.65
            rt_pressed = rt > 0.65

            if menu:
                break
            if lb and not lb_latch:
                distance_mm = max(0.0, distance_mm - 5.0)
            if rb and not rb_latch:
                distance_mm += 5.0
            if y_btn and not y_latch:
                twist_steps += 1
            if lt_pressed and not lt_latch:
                ref_idx = (ref_idx - 1) % len(ref_names)
            if rt_pressed and not rt_latch:
                ref_idx = (ref_idx + 1) % len(ref_names)
            lb_latch = lb
            rb_latch = rb
            y_latch = y_btn
            lt_latch = lt_pressed
            rt_latch = rt_pressed

            obj_pos, obj_quat = p.getBasePositionAndOrientation(obj_id)
            dir_world = [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
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
            if int(hit[0]) != int(obj_id):
                p.stepSimulation()
                time.sleep(dt)
                continue

            hit_pos = [float(hit[3][0]), float(hit[3][1]), float(hit[3][2])]
            hit_n = _normalize([float(hit[4][0]), float(hit[4][1]), float(hit[4][2])])
            p.resetBasePositionAndOrientation(point_id, hit_pos, [0.0, 0.0, 0.0, 1.0])

            # Build local plane frame at the grasp point.
            t0 = _orthonormal_tangent(hit_n, [0.0, 0.0, 1.0])
            t = _normalize(_rotate_vector_around_axis(t0, hit_n, float(twist_steps) * math.pi * 0.5))
            b = _normalize(_cross(hit_n, t))
            t = _normalize(_cross(b, hit_n))

            ref = refs[ref_names[ref_idx]]
            p_hand = [float(v) for v in ref["position_hand_xyz"]]
            n_hand = _normalize([float(v) for v in ref["normal_hand_xyz"]])
            t_hand = _orthonormal_tangent(n_hand, [float(v) for v in ref["tangent_hand_xyz"]])
            b_hand = _normalize(_cross(n_hand, t_hand))

            n_target = [-hit_n[0], -hit_n[1], -hit_n[2]]
            t_target = t
            b_target = _normalize(_cross(n_target, t_target))
            t_target = _normalize(_cross(b_target, n_target))

            r_world_target = _mat_from_basis(t_target, b_target, n_target)
            r_hand_local = _mat_from_basis(t_hand, b_hand, n_hand)
            r_hand_world = _mat_mul(r_world_target, _mat_transpose(r_hand_local))
            q_hand_world = _quat_from_mat(r_hand_world)

            target_ref = [
                hit_pos[0] + float(distance_mm) / 1000.0 * hit_n[0],
                hit_pos[1] + float(distance_mm) / 1000.0 * hit_n[1],
                hit_pos[2] + float(distance_mm) / 1000.0 * hit_n[2],
            ]
            offset = _mat_vec_mul(r_hand_world, p_hand)
            hand_pos = [target_ref[0] - offset[0], target_ref[1] - offset[1], target_ref[2] - offset[2]]
            p.resetBasePositionAndOrientation(hand_id, hand_pos, q_hand_world)

            if a_btn and not a_latch:
                pos_obj_local = _world_to_local_point(hit_pos, list(obj_pos), list(obj_quat))
                n_obj_local = _normalize(_rotate_world_to_local(hit_n, list(obj_quat)))
                t_obj_local = _normalize(_rotate_world_to_local(t, list(obj_quat)))
                out_yaml = f"artifacts/grasp_points_part_{int(part_id)}.yaml"
                gp_idx = _append_grasp_point(
                    path=out_yaml,
                    part_id=int(part_id),
                    part_urdf=str(benchmark_part_urdf(part_id)),
                    object_pose_world={
                        "position_xyz": [float(v) for v in obj_pos],
                        "orientation_xyzw": [float(v) for v in obj_quat],
                    },
                    grasp_point={
                        "position_obj_xyz": [round(v, 6) for v in pos_obj_local],
                        "normal_obj_xyz": [round(v, 6) for v in n_obj_local],
                        "tangent_obj_xyz": [round(v, 6) for v in t_obj_local],
                        "poses": {
                            str(ref_names[ref_idx]): {
                                "distance_mm": float(distance_mm),
                                "twist_deg": int((twist_steps % 4) * 90),
                            }
                        },
                    },
                )
                print(f"saved_grasp_point=gp_{gp_idx:03d} file={out_yaml}")
            a_latch = a_btn

            text_id = p.addUserDebugText(
                text=(
                    f"ref={ref_names[ref_idx]}  dist_mm={int(distance_mm)}  "
                    f"twist_deg={(twist_steps % 4) * 90}  hand={hand_scalar:.2f}"
                ),
                textPosition=[float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]) + 0.18],
                textColorRGB=[1.0, 1.0, 1.0],
                textSize=1.2,
                lifeTime=0.0,
                replaceItemUniqueId=text_id,
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
    parser.add_argument("--part-id", type=int, required=True)
    parser.add_argument("--hand-ref-yaml", default="artifacts/hand_reference_points.yaml")
    parser.add_argument("--initial-ref", default=None)
    parser.add_argument("--initial-distance-mm", type=float, default=20.0)
    parser.add_argument("--point-speed-deg", type=float, default=55.0)
    parser.add_argument("--deadzone", type=float, default=0.20)
    parser.add_argument("--obj-x", type=float, default=0.80)
    parser.add_argument("--obj-y", type=float, default=0.0)
    parser.add_argument("--pedestal-height-m", type=float, default=0.05)
    args = parser.parse_args()
    run(
        part_id=args.part_id,
        hand_ref_yaml=args.hand_ref_yaml,
        initial_ref=args.initial_ref,
        initial_distance_mm=args.initial_distance_mm,
        point_speed_deg=args.point_speed_deg,
        deadzone=args.deadzone,
        obj_x=args.obj_x,
        obj_y=args.obj_y,
        pedestal_h=args.pedestal_height_m,
    )


if __name__ == "__main__":
    main()
