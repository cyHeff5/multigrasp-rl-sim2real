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

POSES = ["medium_wrap", "tripod", "power_sphere", "thumb_1_finger", "lateral_pinch"]


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


def _linspace_open(n: int) -> list[float]:
    if n <= 1:
        return [0.5]
    return [(i + 1) / float(n + 1) for i in range(n)]


def _transform_local_to_world(point_local: list[float], pos_world: list[float], quat_world: list[float]) -> list[float]:
    p_world, _ = p.multiplyTransforms(pos_world, quat_world, point_local, [0.0, 0.0, 0.0, 1.0])
    return [float(p_world[0]), float(p_world[1]), float(p_world[2])]


def _world_to_local_point(point_world: list[float], pos_world: list[float], quat_world: list[float]) -> list[float]:
    inv_pos, inv_quat = p.invertTransform(pos_world, quat_world)
    p_local, _ = p.multiplyTransforms(inv_pos, inv_quat, point_world, [0.0, 0.0, 0.0, 1.0])
    return [float(p_local[0]), float(p_local[1]), float(p_local[2])]


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


def _rotate_world_to_local(vec_world: list[float], quat_world: list[float]) -> list[float]:
    m = p.getMatrixFromQuaternion(quat_world)
    x = float(vec_world[0])
    y = float(vec_world[1])
    z = float(vec_world[2])
    return [
        float(m[0]) * x + float(m[3]) * y + float(m[6]) * z,
        float(m[1]) * x + float(m[4]) * y + float(m[7]) * z,
        float(m[2]) * x + float(m[5]) * y + float(m[8]) * z,
    ]


def _create_sphere_marker(radius: float, rgba: list[float]) -> int:
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=[float(c) for c in rgba])
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_hand_reference(path: Path, hand_pose_world: dict, pose_name: str, p_local: list[float], n_local: list[float], t_local: list[float]) -> None:
    data = _load_yaml(path)
    refs = dict(data.get("reference_points", {}))
    refs[str(pose_name)] = {
        "position_hand_xyz": [round(float(v), 6) for v in p_local],
        "normal_hand_xyz": [round(float(v), 6) for v in n_local],
        "tangent_hand_xyz": [round(float(v), 6) for v in t_local],
    }
    out = {
        "hand_urdf": str(AR10_URDF),
        "hand_pose_world": hand_pose_world,
        "reference_points": refs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)


def _resolve_link_index_by_name(body_id: int, link_name: str) -> int:
    for j in range(p.getNumJoints(int(body_id))):
        info = p.getJointInfo(int(body_id), j)
        if info[12].decode("utf-8") == str(link_name):
            return int(j)
    raise RuntimeError(f"Link '{link_name}' not found in body {body_id}")


def _generate_palm_candidates(
    hand_id: int,
    palm_link_index: int,
    hand_pos: list[float],
    hand_quat: list[float],
    samples_u: int,
    samples_v: int,
    margin_ratio: float,
    probe_depth: float,
) -> list[dict]:
    # Use palm-link world AABB as proxy and probe from all 6 outward faces inward.
    # Important: getAABB returns WORLD coordinates.
    aabb_min, aabb_max = p.getAABB(int(hand_id), int(palm_link_index))
    mn = [float(v) for v in aabb_min]
    mx = [float(v) for v in aabb_max]
    ext = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]]
    mar = [max(1e-4, margin_ratio * e) for e in ext]
    us = _linspace_open(samples_u)
    vs = _linspace_open(samples_v)

    candidates = []
    faces = [
        # axis, sign, fixed value, u_axis, v_axis, nominal world tangent
        ("x", +1.0, mx[0], "y", "z", [0.0, 0.0, 1.0]),
        ("x", -1.0, mn[0], "y", "z", [0.0, 0.0, 1.0]),
        ("y", +1.0, mx[1], "x", "z", [0.0, 0.0, 1.0]),
        ("y", -1.0, mn[1], "x", "z", [0.0, 0.0, 1.0]),
        ("z", +1.0, mx[2], "x", "y", [1.0, 0.0, 0.0]),
        ("z", -1.0, mn[2], "x", "y", [1.0, 0.0, 0.0]),
    ]

    def axis_range(name: str) -> tuple[float, float]:
        if name == "x":
            return mn[0] + mar[0], mx[0] - mar[0]
        if name == "y":
            return mn[1] + mar[1], mx[1] - mar[1]
        return mn[2] + mar[2], mx[2] - mar[2]

    for axis, sign, fixed, u_axis, v_axis, t_local_nom in faces:
        u_min, u_max = axis_range(u_axis)
        v_min, v_max = axis_range(v_axis)
        n_world_nom = [0.0, 0.0, 0.0]
        if axis == "x":
            n_world_nom[0] = sign
        elif axis == "y":
            n_world_nom[1] = sign
        else:
            n_world_nom[2] = sign

        for u in us:
            for v in vs:
                p_world_nom = [0.0, 0.0, 0.0]
                if axis == "x":
                    p_world_nom[0] = fixed
                elif axis == "y":
                    p_world_nom[1] = fixed
                else:
                    p_world_nom[2] = fixed
                u_val = u_min + u * (u_max - u_min)
                v_val = v_min + v * (v_max - v_min)
                if u_axis == "x":
                    p_world_nom[0] = u_val
                elif u_axis == "y":
                    p_world_nom[1] = u_val
                else:
                    p_world_nom[2] = u_val
                if v_axis == "x":
                    p_world_nom[0] = v_val
                elif v_axis == "y":
                    p_world_nom[1] = v_val
                else:
                    p_world_nom[2] = v_val

                n_world_nom = _normalize(n_world_nom)
                t_world_nom = _orthonormal_tangent(n_world_nom, t_local_nom)

                ray_from = [
                    p_world_nom[0] + n_world_nom[0] * probe_depth,
                    p_world_nom[1] + n_world_nom[1] * probe_depth,
                    p_world_nom[2] + n_world_nom[2] * probe_depth,
                ]
                ray_to = [
                    p_world_nom[0] - n_world_nom[0] * probe_depth,
                    p_world_nom[1] - n_world_nom[1] * probe_depth,
                    p_world_nom[2] - n_world_nom[2] * probe_depth,
                ]
                hit = p.rayTest(ray_from, ray_to)[0]
                if int(hit[0]) != int(hand_id):
                    continue
                if int(hit[1]) != int(palm_link_index):
                    continue

                hit_pos = [float(hit[3][0]), float(hit[3][1]), float(hit[3][2])]
                hit_n_world = _normalize([float(hit[4][0]), float(hit[4][1]), float(hit[4][2])])
                if _dot(hit_n_world, n_world_nom) < 0.0:
                    hit_n_world = [-hit_n_world[0], -hit_n_world[1], -hit_n_world[2]]
                hit_t_world = _orthonormal_tangent(hit_n_world, t_world_nom)

                p_local = _world_to_local_point(hit_pos, hand_pos, hand_quat)
                n_local = _normalize(_rotate_world_to_local(hit_n_world, hand_quat))
                t_local = _orthonormal_tangent(n_local, _rotate_world_to_local(hit_t_world, hand_quat))
                candidates.append(
                    {
                        "position_world": hit_pos,
                        "normal_world": hit_n_world,
                        "tangent_world": hit_t_world,
                        "position_local": p_local,
                        "normal_local": n_local,
                        "tangent_local": t_local,
                    }
                )

    # Deduplicate near-identical hits from neighboring rays.
    unique = []
    for c in candidates:
        keep = True
        for u in unique:
            dx = c["position_world"][0] - u["position_world"][0]
            dy = c["position_world"][1] - u["position_world"][1]
            dz = c["position_world"][2] - u["position_world"][2]
            if (dx * dx + dy * dy + dz * dz) <= (0.004 * 0.004):
                keep = False
                break
        if keep:
            unique.append(c)
    return unique


def run(
    output_yaml: str,
    hand_xyz: list[float],
    hand_rpy_deg: list[float],
    samples_u: int,
    samples_v: int,
    margin_ratio: float,
    probe_depth: float,
    marker_radius_mm: float,
    hz: float,
) -> None:
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError("pygame is required. Install with: pip install pygame") from exc

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        raise RuntimeError("No gamepad found. Connect controller and retry.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Controller: {js.get_name()}")

    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf")

    hand_quat = p.getQuaternionFromEuler([math.radians(float(v)) for v in hand_rpy_deg])
    hand_id = p.loadURDF(
        str(AR10_URDF),
        basePosition=[float(v) for v in hand_xyz],
        baseOrientation=[float(v) for v in hand_quat],
        useFixedBase=True,
    )
    HandModel(hand_id).send_q_target([0.0] * 10)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.55,
        cameraYaw=35.0,
        cameraPitch=-25.0,
        cameraTargetPosition=[float(hand_xyz[0]), float(hand_xyz[1]), float(hand_xyz[2]) + 0.02],
    )

    hand_pos, hand_quat_now = p.getBasePositionAndOrientation(hand_id)
    palm_link_index = _resolve_link_index_by_name(hand_id, "palm")
    candidates = _generate_palm_candidates(
        hand_id=hand_id,
        palm_link_index=palm_link_index,
        hand_pos=list(hand_pos),
        hand_quat=list(hand_quat_now),
        samples_u=int(samples_u),
        samples_v=int(samples_v),
        margin_ratio=float(margin_ratio),
        probe_depth=float(probe_depth),
    )
    if not candidates:
        # Robust fallback: synthesize a small front-face grid from base-link AABB.
        aabb_min, aabb_max = p.getAABB(int(hand_id), int(palm_link_index))
        mn = [float(v) for v in aabb_min]
        mx = [float(v) for v in aabb_max]
        us = _linspace_open(max(2, int(samples_u)))
        vs = _linspace_open(max(2, int(samples_v)))
        y_front = mx[1]
        for u in us:
            for v in vs:
                p_local = [
                    mn[0] + u * (mx[0] - mn[0]),
                    y_front,
                    mn[2] + v * (mx[2] - mn[2]),
                ]
                p_world = [float(v) for v in p_local]
                n_world = [0.0, 1.0, 0.0]
                t_world = [0.0, 0.0, 1.0]
                p_local_hand = _world_to_local_point(p_world, list(hand_pos), list(hand_quat_now))
                n_local = _normalize(_rotate_world_to_local(n_world, list(hand_quat_now)))
                t_local = _orthonormal_tangent(n_local, _rotate_world_to_local(t_world, list(hand_quat_now)))
                candidates.append(
                    {
                        "position_world": p_world,
                        "normal_world": n_world,
                        "tangent_world": t_world,
                        "position_local": p_local_hand,
                        "normal_local": n_local,
                        "tangent_local": t_local,
                    }
                )
        print("warning=no_raycast_candidates_using_aabb_fallback")

    marker_radius = float(marker_radius_mm) / 1000.0
    marker_ids = [_create_sphere_marker(marker_radius, [1.0, 0.0, 0.0, 1.0]) for _ in candidates]
    for i, c in enumerate(candidates):
        p.resetBasePositionAndOrientation(marker_ids[i], c["position_world"], [0.0, 0.0, 0.0, 1.0])

    selected_idx = 0
    pose_idx = 0
    tangent_line = -1
    normal_line = -1
    text_id = -1

    def _refresh() -> None:
        nonlocal tangent_line, normal_line, text_id
        for i, mid in enumerate(marker_ids):
            if i == selected_idx:
                p.changeVisualShape(mid, -1, rgbaColor=[0.1, 0.95, 0.2, 1.0])
            else:
                p.changeVisualShape(mid, -1, rgbaColor=[1.0, 0.0, 0.0, 1.0])
        c = candidates[selected_idx]
        pw = c["position_world"]
        nw = c["normal_world"]
        tw = c["tangent_world"]
        if normal_line >= 0:
            p.removeUserDebugItem(normal_line)
        if tangent_line >= 0:
            p.removeUserDebugItem(tangent_line)
        normal_line = p.addUserDebugLine(
            pw,
            [pw[0] + 0.04 * nw[0], pw[1] + 0.04 * nw[1], pw[2] + 0.04 * nw[2]],
            [1.0, 1.0, 0.0],
            2.0,
            0.0,
        )
        tangent_line = p.addUserDebugLine(
            pw,
            [pw[0] + 0.04 * tw[0], pw[1] + 0.04 * tw[1], pw[2] + 0.04 * tw[2]],
            [1.0, 0.0, 1.0],
            2.0,
            0.0,
        )
        if text_id >= 0:
            p.removeUserDebugItem(text_id)
        text_id = p.addUserDebugText(
            text=f"candidate={selected_idx + 1}/{len(candidates)} pose={POSES[pose_idx]}",
            textPosition=[float(hand_xyz[0]), float(hand_xyz[1]), float(hand_xyz[2]) + 0.18],
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.2,
            lifeTime=0.0,
        )

    _refresh()
    print(f"num_candidates={len(candidates)}")
    print("Controls:")
    print("  RB/LB: next/prev candidate")
    print("  RT/LT: next/prev grasp pose")
    print("  A: save selected candidate to hand_reference_points.yaml")
    print("  Menu: quit")

    rb_latch = False
    lb_latch = False
    rt_latch = False
    lt_latch = False
    a_latch = False

    def _norm_trigger(raw: float) -> float:
        if raw < 0.0:
            return max(0.0, min(1.0, (raw + 1.0) * 0.5))
        return max(0.0, min(1.0, raw))

    out_path = Path(output_yaml)
    dt = 1.0 / float(hz)
    try:
        while p.isConnected(client_id):
            pygame.event.pump()
            lb = bool(js.get_numbuttons() > 4 and js.get_button(4))
            rb = bool(js.get_numbuttons() > 5 and js.get_button(5))
            a_btn = bool(js.get_numbuttons() > 0 and js.get_button(0))
            menu = bool(js.get_numbuttons() > 7 and js.get_button(7))
            lt = _norm_trigger(float(js.get_axis(4))) if js.get_numaxes() > 4 else 0.0
            rt = _norm_trigger(float(js.get_axis(5))) if js.get_numaxes() > 5 else 0.0
            lt_pressed = lt > 0.65
            rt_pressed = rt > 0.65

            if menu:
                break

            changed = False
            if rb and not rb_latch:
                selected_idx = (selected_idx + 1) % len(candidates)
                changed = True
            if lb and not lb_latch:
                selected_idx = (selected_idx - 1) % len(candidates)
                changed = True
            if rt_pressed and not rt_latch:
                pose_idx = (pose_idx + 1) % len(POSES)
                changed = True
            if lt_pressed and not lt_latch:
                pose_idx = (pose_idx - 1) % len(POSES)
                changed = True

            if a_btn and not a_latch:
                c = candidates[selected_idx]
                _save_hand_reference(
                    path=out_path,
                    hand_pose_world={
                        "position_xyz": [float(v) for v in hand_pos],
                        "orientation_xyzw": [float(v) for v in hand_quat_now],
                    },
                    pose_name=POSES[pose_idx],
                    p_local=[float(v) for v in c["position_local"]],
                    n_local=[float(v) for v in c["normal_local"]],
                    t_local=[float(v) for v in c["tangent_local"]],
                )
                print(f"saved pose={POSES[pose_idx]} -> {out_path}")

            if changed:
                _refresh()

            rb_latch = rb
            lb_latch = lb
            rt_latch = rt_pressed
            lt_latch = lt_pressed
            a_latch = a_btn
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
    parser.add_argument("--hand-x", type=float, default=0.30)
    parser.add_argument("--hand-y", type=float, default=0.00)
    parser.add_argument("--hand-z", type=float, default=0.18)
    parser.add_argument("--hand-roll-deg", type=float, default=90.0)
    parser.add_argument("--hand-pitch-deg", type=float, default=0.0)
    parser.add_argument("--hand-yaw-deg", type=float, default=0.0)
    parser.add_argument("--samples-u", type=int, default=5)
    parser.add_argument("--samples-v", type=int, default=4)
    parser.add_argument("--margin-ratio", type=float, default=0.10)
    parser.add_argument("--probe-depth", type=float, default=0.12)
    parser.add_argument("--marker-radius-mm", type=float, default=4.0)
    parser.add_argument("--hz", type=float, default=120.0)
    args = parser.parse_args()
    run(
        output_yaml=args.output_yaml,
        hand_xyz=[args.hand_x, args.hand_y, args.hand_z],
        hand_rpy_deg=[args.hand_roll_deg, args.hand_pitch_deg, args.hand_yaw_deg],
        samples_u=args.samples_u,
        samples_v=args.samples_v,
        margin_ratio=args.margin_ratio,
        probe_depth=args.probe_depth,
        marker_radius_mm=args.marker_radius_mm,
        hz=args.hz,
    )


if __name__ == "__main__":
    main()
