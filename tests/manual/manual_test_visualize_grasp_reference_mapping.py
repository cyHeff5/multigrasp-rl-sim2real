from __future__ import annotations

import argparse
import math
import re
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


def _rotate_vector_around_axis(v: list[float], axis: list[float], angle_rad: float) -> list[float]:
    a = _normalize(axis)
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    term1 = [float(v[0]) * c, float(v[1]) * c, float(v[2]) * c]
    cross = _cross(a, v)
    term2 = [float(cross[0]) * s, float(cross[1]) * s, float(cross[2]) * s]
    ad = _dot(a, v) * (1.0 - c)
    term3 = [float(a[0]) * ad, float(a[1]) * ad, float(a[2]) * ad]
    return [term1[0] + term2[0] + term3[0], term1[1] + term2[1] + term3[1], term1[2] + term2[2] + term3[2]]


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


def _part_id_from_filename(path: Path) -> int | None:
    m = re.search(r"grasp_points_part_(\d+)\.yaml$", path.name)
    if m is None:
        return None
    return int(m.group(1))


def _discover_object_yaml_files(artifacts_dir: Path) -> list[Path]:
    files = []
    for pth in artifacts_dir.glob("grasp_points_part_*.yaml"):
        pid = _part_id_from_filename(pth)
        if pid is not None:
            files.append((pid, pth))
    files.sort(key=lambda x: x[0])
    return [pth for _, pth in files]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _choose_pose_name_for_gp(gp: dict, preferred_pose: str, hand_refs: dict) -> str:
    poses = gp.get("poses", {}) or {}
    if preferred_pose in poses and preferred_pose in hand_refs:
        return str(preferred_pose)
    for name in poses.keys():
        if str(name) in hand_refs:
            return str(name)
    if preferred_pose in hand_refs:
        return str(preferred_pose)
    if hand_refs:
        return str(next(iter(hand_refs.keys())))
    raise RuntimeError("No hand reference points found.")


def run(
    artifacts_dir: str,
    hand_yaml: str,
    preferred_pose: str,
    object_x: float,
    object_y: float,
    hand_x: float,
    hand_y: float,
    hand_z: float,
    hz: float,
    patch_size_mm: float,
    patch_thickness_mm: float,
    marker_radius_mm: float,
) -> None:
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError("pygame is required. Install with: pip install pygame") from exc

    obj_yaml_paths = _discover_object_yaml_files(Path(artifacts_dir))
    if not obj_yaml_paths:
        raise RuntimeError(f"No grasp_points_part_*.yaml found in: {artifacts_dir}")
    hand_data = _load_yaml(Path(hand_yaml))
    hand_refs = dict(hand_data.get("reference_points", {}))
    if not hand_refs:
        raise RuntimeError(f"No reference_points in {hand_yaml}")

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
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=45.0,
        cameraPitch=-25.0,
        cameraTargetPosition=[(float(object_x) + float(hand_x)) * 0.5, float(object_y), 0.10],
    )

    hand_quat = hand_data.get("hand_pose_world", {}).get("orientation_xyzw", [0.70710678, 0.0, 0.0, 0.70710678])
    hand_id = p.loadURDF(
        str(AR10_URDF),
        basePosition=[float(hand_x), float(object_y), 0.05],
        baseOrientation=[float(v) for v in hand_quat],
        useFixedBase=True,
    )
    HandModel(hand_id).send_q_target([0.0] * 10)

    object_index = 0
    gp_index = 0
    object_id = -1
    hand_rest_pos = [float(hand_x), float(object_y), 0.05]
    hand_rest_quat = [float(v) for v in hand_quat]
    hand_pregrasp_active = False
    object_data: dict = {}
    gp_list: list[dict] = []
    vis_bodies: list[int] = []
    vis_lines: list[int] = []
    text_id = -1

    def clear_visuals() -> None:
        nonlocal vis_bodies, vis_lines, text_id
        for bid in vis_bodies:
            try:
                p.removeBody(int(bid))
            except Exception:
                pass
        vis_bodies = []
        for lid in vis_lines:
            try:
                p.removeUserDebugItem(int(lid))
            except Exception:
                pass
        vis_lines = []
        if text_id >= 0:
            try:
                p.removeUserDebugItem(int(text_id))
            except Exception:
                pass
            text_id = -1

    def load_current_object() -> None:
        nonlocal object_id, object_data, gp_list, gp_index, hand_rest_pos, hand_rest_quat, hand_pregrasp_active
        if object_id >= 0:
            p.removeBody(int(object_id))
        object_data = _load_yaml(obj_yaml_paths[object_index])
        gp_list = list(object_data.get("grasp_points", []))
        gp_index = 0 if gp_list else -1
        part_id = int(object_data["part_id"])
        obj_pose = object_data.get("object_pose_world", {}) or {}
        obj_pos_yaml = [float(v) for v in obj_pose.get("position_xyz", [0.8, 0.0, 0.05])]
        obj_quat_yaml = [float(v) for v in obj_pose.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])]
        object_id = p.loadURDF(
            str(benchmark_part_urdf(part_id)),
            basePosition=[float(object_x), float(object_y), float(obj_pos_yaml[2])],
            baseOrientation=obj_quat_yaml,
            useFixedBase=True,
        )
        # Keep hand on same world Y/Z as object; X stays user-defined for side-by-side view.
        p.resetBasePositionAndOrientation(
            hand_id,
            [float(hand_x), float(object_y), float(obj_pos_yaml[2])],
            [float(v) for v in hand_quat],
        )
        hand_rest_pos = [float(hand_x), float(object_y), float(obj_pos_yaml[2])]
        hand_rest_quat = [float(v) for v in hand_quat]
        hand_pregrasp_active = False

    def compute_selected_pregrasp_hand_pose() -> tuple[list[float], list[float], str] | None:
        if object_id < 0 or gp_index < 0 or not gp_list:
            return None
        obj_pos, obj_quat = p.getBasePositionAndOrientation(object_id)
        gp = gp_list[gp_index]
        gp_world = _transform_local_to_world([float(v) for v in gp["position_obj_xyz"]], list(obj_pos), list(obj_quat))
        n_obj_world = _normalize(_rotate_local_to_world([float(v) for v in gp["normal_obj_xyz"]], list(obj_quat)))
        t_obj_world = _orthonormal_tangent(
            n_obj_world,
            _rotate_local_to_world([float(v) for v in gp["tangent_obj_xyz"]], list(obj_quat)),
        )
        b_obj_world = _normalize(_cross(n_obj_world, t_obj_world))
        t_obj_world = _normalize(_cross(b_obj_world, n_obj_world))

        pose_name = _choose_pose_name_for_gp(gp, preferred_pose=preferred_pose, hand_refs=hand_refs)
        pose_data = (gp.get("poses", {}) or {}).get(pose_name, {})
        distance_mm = float(pose_data.get("distance_mm", 20.0))
        twist_deg = float(pose_data.get("twist_deg", 0.0))
        d = distance_mm / 1000.0
        target_ref_world = [
            gp_world[0] + d * n_obj_world[0],
            gp_world[1] + d * n_obj_world[1],
            gp_world[2] + d * n_obj_world[2],
        ]
        t_obj_world = _normalize(_rotate_vector_around_axis(t_obj_world, n_obj_world, math.radians(twist_deg)))

        ref = hand_refs[pose_name]
        p_hand = [float(v) for v in ref["position_hand_xyz"]]
        n_hand = _normalize([float(v) for v in ref["normal_hand_xyz"]])
        t_hand = _orthonormal_tangent(n_hand, [float(v) for v in ref["tangent_hand_xyz"]])
        b_hand = _normalize(_cross(n_hand, t_hand))
        t_hand = _normalize(_cross(b_hand, n_hand))

        n_target = [-n_obj_world[0], -n_obj_world[1], -n_obj_world[2]]
        t_target = t_obj_world
        b_target = _normalize(_cross(n_target, t_target))
        t_target = _normalize(_cross(b_target, n_target))

        r_world_target = _mat_from_basis(t_target, b_target, n_target)
        r_hand_local = _mat_from_basis(t_hand, b_hand, n_hand)
        r_hand_world = _mat_mul(r_world_target, _mat_transpose(r_hand_local))
        hand_quat_world = _quat_from_basis(
            [r_hand_world[0][0], r_hand_world[1][0], r_hand_world[2][0]],
            [r_hand_world[0][1], r_hand_world[1][1], r_hand_world[2][1]],
            [r_hand_world[0][2], r_hand_world[1][2], r_hand_world[2][2]],
        )
        hand_offset_world = _mat_vec_mul(r_hand_world, p_hand)
        hand_pos_world = [
            target_ref_world[0] - hand_offset_world[0],
            target_ref_world[1] - hand_offset_world[1],
            target_ref_world[2] - hand_offset_world[2],
        ]
        return hand_pos_world, hand_quat_world, pose_name

    def render_state() -> None:
        nonlocal text_id, vis_bodies, vis_lines
        clear_visuals()
        if object_id < 0 or gp_index < 0 or not gp_list:
            return

        obj_pos, obj_quat = p.getBasePositionAndOrientation(object_id)
        gp = gp_list[gp_index]
        gp_world = _transform_local_to_world([float(v) for v in gp["position_obj_xyz"]], list(obj_pos), list(obj_quat))
        n_obj_world = _normalize(_rotate_local_to_world([float(v) for v in gp["normal_obj_xyz"]], list(obj_quat)))
        t_obj_world = _orthonormal_tangent(
            n_obj_world,
            _rotate_local_to_world([float(v) for v in gp["tangent_obj_xyz"]], list(obj_quat)),
        )
        b_obj_world = _normalize(_cross(n_obj_world, t_obj_world))
        t_obj_world = _normalize(_cross(b_obj_world, n_obj_world))

        pose_name = _choose_pose_name_for_gp(gp, preferred_pose=preferred_pose, hand_refs=hand_refs)
        pose_data = (gp.get("poses", {}) or {}).get(pose_name, {})
        distance_mm = float(pose_data.get("distance_mm", 20.0))
        twist_deg = float(pose_data.get("twist_deg", 0.0))

        d = distance_mm / 1000.0
        t_obj_world = _normalize(_rotate_vector_around_axis(t_obj_world, n_obj_world, math.radians(twist_deg)))
        plane_center = [
            gp_world[0] + d * n_obj_world[0],
            gp_world[1] + d * n_obj_world[1],
            gp_world[2] + d * n_obj_world[2],
        ]
        plane_q = _quat_from_basis(t_obj_world, b_obj_world, n_obj_world)

        vis_bodies.append(_create_sphere_marker(gp_world, marker_radius_mm / 1000.0, [0.0, 0.85, 0.2, 1.0]))
        vis_bodies.append(
            _create_plane_patch(
                center=plane_center,
                quat=plane_q,
                size=patch_size_mm / 1000.0,
                thickness=patch_thickness_mm / 1000.0,
                rgba=[0.3, 0.6, 1.0, 0.45],
            )
        )
        vis_lines.append(
            p.addUserDebugLine(gp_world, plane_center, [0.2, 0.8, 1.0], lineWidth=2.0, lifeTime=0.0)
        )
        vis_lines.append(
            p.addUserDebugLine(
                gp_world,
                [gp_world[0] + 0.03 * t_obj_world[0], gp_world[1] + 0.03 * t_obj_world[1], gp_world[2] + 0.03 * t_obj_world[2]],
                [1.0, 1.0, 0.0],
                lineWidth=2.0,
                lifeTime=0.0,
            )
        )

        hand_pos, hand_quat_now = p.getBasePositionAndOrientation(hand_id)
        for name, ref in hand_refs.items():
            p_hand = [float(v) for v in ref["position_hand_xyz"]]
            ref_world = _transform_local_to_world(p_hand, list(hand_pos), list(hand_quat_now))
            if str(name) == pose_name:
                color = [1.0, 0.45, 0.05, 1.0]
                rad = marker_radius_mm / 1000.0 * 1.35
            else:
                color = [0.65, 0.65, 0.65, 0.75]
                rad = marker_radius_mm / 1000.0 * 0.9
            vis_bodies.append(_create_sphere_marker(ref_world, rad, color))

        sel_ref = hand_refs[pose_name]
        p_hand = [float(v) for v in sel_ref["position_hand_xyz"]]
        n_hand = _normalize([float(v) for v in sel_ref["normal_hand_xyz"]])
        t_hand = _orthonormal_tangent(n_hand, [float(v) for v in sel_ref["tangent_hand_xyz"]])
        n_hand_world = _normalize(_rotate_local_to_world(n_hand, list(hand_quat_now)))
        t_hand_world = _orthonormal_tangent(n_hand_world, _rotate_local_to_world(t_hand, list(hand_quat_now)))
        b_hand_world = _normalize(_cross(n_hand_world, t_hand_world))
        t_hand_world = _normalize(_cross(b_hand_world, n_hand_world))
        ref_world = _transform_local_to_world(p_hand, list(hand_pos), list(hand_quat_now))
        hand_plane_q = _quat_from_basis(t_hand_world, b_hand_world, n_hand_world)
        vis_bodies.append(
            _create_plane_patch(
                center=ref_world,
                quat=hand_plane_q,
                size=patch_size_mm / 1000.0 * 0.8,
                thickness=patch_thickness_mm / 1000.0,
                rgba=[1.0, 0.45, 0.05, 0.35],
            )
        )

        text = (
            f"part={int(object_data.get('part_id', -1))} "
            f"gp={gp.get('id', '?')} ({gp_index + 1}/{len(gp_list)}) "
            f"pose={pose_name} dist_mm={int(round(distance_mm))} twist_deg={int(round(twist_deg))}"
        )
        text_id = p.addUserDebugText(
            text=text,
            textPosition=[(float(object_x) + float(hand_x)) * 0.5, float(object_y), 0.23],
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.3,
            lifeTime=0.0,
        )

    load_current_object()
    render_state()
    print("Controls:")
    print("  RT/LT: next/prev benchmark object")
    print("  RB/LB: next/prev grasp point on active object")
    print("  Y: toggle hand pregrasp pose for active mapping")
    print("  Menu: quit")

    def norm_trigger(raw: float) -> float:
        if raw < 0.0:
            return max(0.0, min(1.0, (raw + 1.0) * 0.5))
        return max(0.0, min(1.0, raw))

    rb_latch = False
    lb_latch = False
    rt_latch = False
    lt_latch = False
    y_latch = False
    dt = 1.0 / float(hz)
    try:
        while p.isConnected(client_id):
            pygame.event.pump()
            rb = bool(js.get_numbuttons() > 5 and js.get_button(5))
            lb = bool(js.get_numbuttons() > 4 and js.get_button(4))
            y_btn = bool(js.get_numbuttons() > 3 and js.get_button(3))
            menu = bool(js.get_numbuttons() > 7 and js.get_button(7))
            rt = norm_trigger(float(js.get_axis(5))) if js.get_numaxes() > 5 else 0.0
            lt = norm_trigger(float(js.get_axis(4))) if js.get_numaxes() > 4 else 0.0
            rt_pressed = rt > 0.65
            lt_pressed = lt > 0.65

            changed = False
            if menu:
                break
            if rt_pressed and not rt_latch:
                object_index = (object_index + 1) % len(obj_yaml_paths)
                load_current_object()
                changed = True
            if lt_pressed and not lt_latch:
                object_index = (object_index - 1) % len(obj_yaml_paths)
                load_current_object()
                changed = True
            if gp_list:
                if rb and not rb_latch:
                    gp_index = (gp_index + 1) % len(gp_list)
                    changed = True
                if lb and not lb_latch:
                    gp_index = (gp_index - 1) % len(gp_list)
                    changed = True
            if y_btn and not y_latch:
                if hand_pregrasp_active:
                    p.resetBasePositionAndOrientation(hand_id, hand_rest_pos, hand_rest_quat)
                    hand_pregrasp_active = False
                else:
                    pre = compute_selected_pregrasp_hand_pose()
                    if pre is not None:
                        hpos, hquat, _ = pre
                        p.resetBasePositionAndOrientation(hand_id, hpos, hquat)
                        hand_pregrasp_active = True

            if changed:
                render_state()
                if hand_pregrasp_active:
                    pre = compute_selected_pregrasp_hand_pose()
                    if pre is not None:
                        hpos, hquat, _ = pre
                        p.resetBasePositionAndOrientation(hand_id, hpos, hquat)

            rb_latch = rb
            lb_latch = lb
            rt_latch = rt_pressed
            lt_latch = lt_pressed
            y_latch = y_btn

            p.stepSimulation()
            time.sleep(dt)
    finally:
        clear_visuals()
        try:
            pygame.quit()
        except Exception:
            pass
        if p.isConnected(client_id):
            p.disconnect(client_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--hand-yaml", default="artifacts/hand_reference_points.yaml")
    parser.add_argument("--preferred-pose", default="tripod")
    parser.add_argument("--object-x", type=float, default=0.80)
    parser.add_argument("--object-y", type=float, default=0.00)
    parser.add_argument("--hand-x", type=float, default=0.30)
    parser.add_argument("--hand-y", type=float, default=0.00)
    parser.add_argument("--hand-z", type=float, default=0.18)
    parser.add_argument("--hz", type=float, default=120.0)
    parser.add_argument("--patch-size-mm", type=float, default=35.0)
    parser.add_argument("--patch-thickness-mm", type=float, default=1.0)
    parser.add_argument("--marker-radius-mm", type=float, default=5.0)
    args = parser.parse_args()
    run(
        artifacts_dir=args.artifacts_dir,
        hand_yaml=args.hand_yaml,
        preferred_pose=args.preferred_pose,
        object_x=args.object_x,
        object_y=args.object_y,
        hand_x=args.hand_x,
        hand_y=args.hand_y,
        hand_z=args.hand_z,
        hz=args.hz,
        patch_size_mm=args.patch_size_mm,
        patch_thickness_mm=args.patch_thickness_mm,
        marker_radius_mm=args.marker_radius_mm,
    )


if __name__ == "__main__":
    main()
