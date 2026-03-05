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


def _world_to_local_point(point_world: list[float], pos_world: list[float], quat_world: list[float]) -> list[float]:
    inv_pos, inv_quat = p.invertTransform(pos_world, quat_world)
    p_local, _ = p.multiplyTransforms(inv_pos, inv_quat, point_world, [0.0, 0.0, 0.0, 1.0])
    return [float(p_local[0]), float(p_local[1]), float(p_local[2])]


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


def _transform_local_to_world(point_local: list[float], pos_world: list[float], quat_world: list[float]) -> list[float]:
    p_world, _ = p.multiplyTransforms(pos_world, quat_world, point_local, [0.0, 0.0, 0.0, 1.0])
    return [float(p_world[0]), float(p_world[1]), float(p_world[2])]


def _rotate_vector_around_axis(v: list[float], axis: list[float], angle_rad: float) -> list[float]:
    a = _normalize(axis)
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    term1 = [v[0] * c, v[1] * c, v[2] * c]
    cross = _cross(a, v)
    term2 = [cross[0] * s, cross[1] * s, cross[2] * s]
    ad = _dot(a, v) * (1.0 - c)
    term3 = [a[0] * ad, a[1] * ad, a[2] * ad]
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


def _create_plane_patch(center: list[float], normal: list[float], tangent: list[float], size: float, thickness: float) -> int:
    bitangent = _normalize(_cross(normal, tangent))
    tangent = _normalize(_cross(bitangent, normal))
    q = _quat_from_basis(tangent, bitangent, normal)
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[float(size) * 0.5, float(size) * 0.5, float(thickness) * 0.5],
        rgbaColor=[0.25, 0.55, 1.0, 0.35],
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[float(v) for v in center],
        baseOrientation=[float(v) for v in q],
    )


def _create_sphere_marker(radius: float, rgba: list[float]) -> int:
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=[float(c) for c in rgba])
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )


def _set_body_visibility(body_id: int, visible: bool) -> None:
    rgba = [1.0, 1.0, 1.0, 1.0 if visible else 0.0]
    p.changeVisualShape(int(body_id), -1, rgbaColor=rgba)
    for j in range(p.getNumJoints(int(body_id))):
        p.changeVisualShape(int(body_id), int(j), rgbaColor=rgba)


def _raise_object_above_ground(body_id: int, clearance_m: float = 0.001) -> None:
    aabb_min, _ = p.getAABB(int(body_id))
    min_z = float(aabb_min[2])
    if min_z < float(clearance_m):
        pos, quat = p.getBasePositionAndOrientation(int(body_id))
        p.resetBasePositionAndOrientation(
            int(body_id),
            [float(pos[0]), float(pos[1]), float(pos[2]) + (float(clearance_m) - min_z)],
            [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
        )


def _snap_from_nominal(
    body_id: int,
    nominal_world: list[float],
    nominal_normal: list[float],
    nominal_tangent: list[float],
    ray_len: float,
) -> tuple[list[float], list[float], list[float]] | None:
    ray_from = [
        nominal_world[0] + nominal_normal[0] * ray_len,
        nominal_world[1] + nominal_normal[1] * ray_len,
        nominal_world[2] + nominal_normal[2] * ray_len,
    ]
    ray_to = [
        nominal_world[0] - nominal_normal[0] * ray_len,
        nominal_world[1] - nominal_normal[1] * ray_len,
        nominal_world[2] - nominal_normal[2] * ray_len,
    ]
    hit = p.rayTest(ray_from, ray_to)[0]
    if int(hit[0]) != int(body_id):
        return None
    hit_pos = [float(hit[3][0]), float(hit[3][1]), float(hit[3][2])]
    hit_normal = _normalize([float(hit[4][0]), float(hit[4][1]), float(hit[4][2])])
    if _dot(hit_normal, nominal_normal) < 0.0:
        hit_normal = [-hit_normal[0], -hit_normal[1], -hit_normal[2]]
    hit_tangent = _orthonormal_tangent(hit_normal, nominal_tangent)
    return hit_pos, hit_normal, hit_tangent


def _generate_candidates_box_hull(
    body_id: int,
    samples_u: int,
    samples_v: int,
    edge_margin_ratio: float,
    ray_len: float,
) -> list[dict]:
    aabb_min, aabb_max = p.getAABB(int(body_id))
    mn = [float(v) for v in aabb_min]
    mx = [float(v) for v in aabb_max]
    ext = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]]
    mar = [max(1e-4, edge_margin_ratio * e) for e in ext]
    us = _linspace_open(samples_u)
    vs = _linspace_open(samples_v)

    faces = [
        ([+1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [mx[0], mn[1] + mar[1], mn[2] + mar[2]], [0.0, ext[1] - 2 * mar[1], 0.0], [0.0, 0.0, ext[2] - 2 * mar[2]]),
        ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [mn[0], mn[1] + mar[1], mn[2] + mar[2]], [0.0, ext[1] - 2 * mar[1], 0.0], [0.0, 0.0, ext[2] - 2 * mar[2]]),
        ([0.0, +1.0, 0.0], [1.0, 0.0, 0.0], [mn[0] + mar[0], mx[1], mn[2] + mar[2]], [ext[0] - 2 * mar[0], 0.0, 0.0], [0.0, 0.0, ext[2] - 2 * mar[2]]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [mn[0] + mar[0], mn[1], mn[2] + mar[2]], [ext[0] - 2 * mar[0], 0.0, 0.0], [0.0, 0.0, ext[2] - 2 * mar[2]]),
        ([0.0, 0.0, +1.0], [1.0, 0.0, 0.0], [mn[0] + mar[0], mn[1] + mar[1], mx[2]], [ext[0] - 2 * mar[0], 0.0, 0.0], [0.0, ext[1] - 2 * mar[1], 0.0]),
        ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [mn[0] + mar[0], mn[1] + mar[1], mn[2]], [ext[0] - 2 * mar[0], 0.0, 0.0], [0.0, ext[1] - 2 * mar[1], 0.0]),
    ]

    out = []
    for _, (n_nom, t_nom, origin, ax_u, ax_v) in enumerate(faces):
        for u in us:
            for v in vs:
                nominal = [
                    origin[0] + u * ax_u[0] + v * ax_v[0],
                    origin[1] + u * ax_u[1] + v * ax_v[1],
                    origin[2] + u * ax_u[2] + v * ax_v[2],
                ]
                snapped = _snap_from_nominal(
                    body_id=body_id,
                    nominal_world=nominal,
                    nominal_normal=n_nom,
                    nominal_tangent=t_nom,
                    ray_len=ray_len,
                )
                if snapped is None:
                    continue
                p_w, n_w, t_w = snapped
                out.append({"position_world": p_w, "normal_world": n_w, "tangent_world": t_w})
    return out


def _load_hand_refs(path: str) -> dict[str, dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    refs = dict(data.get("reference_points", {}))
    if not refs:
        raise RuntimeError(f"No reference_points in {path}")
    return refs


def _next_gp_id(existing: list[dict]) -> str:
    max_idx = 0
    for gp in existing:
        gp_id = str(gp.get("id", ""))
        if gp_id.startswith("gp_"):
            try:
                max_idx = max(max_idx, int(gp_id.split("_")[1]))
            except Exception:
                pass
    return f"gp_{max_idx + 1:03d}"


def _save_selected_gp(
    path: Path,
    part_id: int,
    obj_pos: list[float],
    obj_quat: list[float],
    p_local: list[float],
    n_local: list[float],
    t_local: list[float],
    pose_name: str,
    distance_mm: float,
    twist_deg: float,
) -> str:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    gp_list = list(data.get("grasp_points", []))
    gp_id = _next_gp_id(gp_list)
    gp_list.append(
        {
            "id": gp_id,
            "position_obj_xyz": [round(float(v), 6) for v in p_local],
            "normal_obj_xyz": [round(float(v), 6) for v in n_local],
            "tangent_obj_xyz": [round(float(v), 6) for v in t_local],
            "poses": {
                str(pose_name): {
                    "distance_mm": float(distance_mm),
                    "twist_deg": float(twist_deg),
                }
            },
        }
    )
    out = {
        "part_id": int(part_id),
        "part_urdf": str(benchmark_part_urdf(part_id)),
        "object_pose_world": {
            "position_xyz": [float(v) for v in obj_pos],
            "orientation_xyzw": [float(v) for v in obj_quat],
        },
        "grasp_points": gp_list,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    return gp_id


def run(
    part_id: int,
    object_xyz: list[float],
    object_rpy_deg: list[float],
    samples_u: int,
    samples_v: int,
    edge_margin_ratio: float,
    ray_len: float,
    patch_distance_mm: float,
    patch_size_mm: float,
    marker_radius_mm: float,
    hand_ref_yaml: str,
) -> None:
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError("pygame is required. Install with: pip install pygame") from exc

    refs = _load_hand_refs(hand_ref_yaml)
    ref_names = list(refs.keys())
    ref_idx = ref_names.index("tripod") if "tripod" in ref_names else 0
    selected_distance_mm = float(patch_distance_mm)
    twist_deg = 0.0
    hand_visible = False

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
    p.resetDebugVisualizerCamera(
        cameraDistance=0.9,
        cameraYaw=45.0,
        cameraPitch=-25.0,
        cameraTargetPosition=[float(object_xyz[0]), float(object_xyz[1]), float(object_xyz[2]) + 0.05],
    )

    obj_quat = p.getQuaternionFromEuler([math.radians(float(v)) for v in object_rpy_deg])
    object_id = p.loadURDF(
        str(benchmark_part_urdf(part_id)),
        basePosition=[float(v) for v in object_xyz],
        baseOrientation=[float(v) for v in obj_quat],
        useFixedBase=True,
    )
    _raise_object_above_ground(object_id, clearance_m=0.001)

    candidates = _generate_candidates_box_hull(
        body_id=object_id,
        samples_u=int(samples_u),
        samples_v=int(samples_v),
        edge_margin_ratio=float(edge_margin_ratio),
        ray_len=float(ray_len),
    )
    if not candidates:
        raise RuntimeError("No grasp point candidates found.")
    selected_idx = 0

    obj_pos, obj_quat_now = p.getBasePositionAndOrientation(object_id)
    for c in candidates:
        pw = [float(v) for v in c["position_world"]]
        nw = _normalize([float(v) for v in c["normal_world"]])
        tw = _orthonormal_tangent(nw, [float(v) for v in c["tangent_world"]])
        c["position_obj"] = _world_to_local_point(pw, list(obj_pos), list(obj_quat_now))
        c["normal_obj"] = _normalize(_rotate_world_to_local(nw, list(obj_quat_now)))
        c["tangent_obj"] = _orthonormal_tangent(c["normal_obj"], _rotate_world_to_local(tw, list(obj_quat_now)))

    hand_id = p.loadURDF(
        str(AR10_URDF),
        basePosition=[float(object_xyz[0]) - 0.25, float(object_xyz[1]), float(object_xyz[2]) + 0.25],
        useFixedBase=True,
    )
    HandModel(hand_id).send_q_target([0.0] * 10)
    _set_body_visibility(hand_id, visible=False)

    marker_radius = float(marker_radius_mm) / 1000.0
    marker_ids = [_create_sphere_marker(marker_radius, [1.0, 0.0, 0.0, 1.0]) for _ in candidates]
    marker_sel_scale = 1.35
    for i, c in enumerate(candidates):
        p.resetBasePositionAndOrientation(marker_ids[i], c["position_world"], [0.0, 0.0, 0.0, 1.0])

    plane_id = -1
    tangent_line_id = -1
    normal_line_id = -1
    text_id = -1

    def _update_selected_visual() -> None:
        nonlocal plane_id, tangent_line_id, normal_line_id, text_id
        for i, mid in enumerate(marker_ids):
            if i == selected_idx:
                p.changeVisualShape(mid, -1, rgbaColor=[0.05, 0.9, 0.2, 1.0])
            else:
                p.changeVisualShape(mid, -1, rgbaColor=[1.0, 0.0, 0.0, 1.0])
        c = candidates[selected_idx]
        pw = [float(v) for v in c["position_world"]]
        nw = _normalize([float(v) for v in c["normal_world"]])
        tw = _orthonormal_tangent(nw, [float(v) for v in c["tangent_world"]])
        tw = _normalize(_rotate_vector_around_axis(tw, nw, math.radians(float(twist_deg))))

        if plane_id >= 0:
            p.removeBody(plane_id)
        patch_center = [
            pw[0] + (selected_distance_mm / 1000.0) * nw[0],
            pw[1] + (selected_distance_mm / 1000.0) * nw[1],
            pw[2] + (selected_distance_mm / 1000.0) * nw[2],
        ]
        plane_id = _create_plane_patch(
            center=patch_center,
            normal=nw,
            tangent=tw,
            size=float(patch_size_mm) / 1000.0,
            thickness=0.001,
        )
        if tangent_line_id >= 0:
            p.removeUserDebugItem(tangent_line_id)
        if normal_line_id >= 0:
            p.removeUserDebugItem(normal_line_id)
        tangent_line_id = p.addUserDebugLine(
            pw, [pw[0] + 0.04 * tw[0], pw[1] + 0.04 * tw[1], pw[2] + 0.04 * tw[2]], [1.0, 0.0, 1.0], 2.0, 0.0
        )
        normal_line_id = p.addUserDebugLine(
            pw, [pw[0] + 0.04 * nw[0], pw[1] + 0.04 * nw[1], pw[2] + 0.04 * nw[2]], [1.0, 1.0, 0.0], 2.0, 0.0
        )
        if text_id >= 0:
            p.removeUserDebugItem(text_id)
        text_id = p.addUserDebugText(
            text=(
                f"part={part_id} sel={selected_idx + 1}/{len(candidates)} "
                f"ref={ref_names[ref_idx]} dist_mm={int(round(selected_distance_mm))} "
                f"twist_deg={int(round(twist_deg))}"
            ),
            textPosition=[float(object_xyz[0]), float(object_xyz[1]), float(object_xyz[2]) + 0.20],
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.2,
            lifeTime=0.0,
        )

    def _place_hand_to_selected() -> None:
        c = candidates[selected_idx]
        pw = [float(v) for v in c["position_world"]]
        nw = _normalize([float(v) for v in c["normal_world"]])
        tw_raw = _orthonormal_tangent(nw, [float(v) for v in c["tangent_world"]])
        tw = _normalize(_rotate_vector_around_axis(tw_raw, nw, math.radians(float(twist_deg))))
        bw = _normalize(_cross(nw, tw))
        tw = _normalize(_cross(bw, nw))

        ref = refs[ref_names[ref_idx]]
        p_hand = [float(v) for v in ref["position_hand_xyz"]]
        n_hand = _normalize([float(v) for v in ref["normal_hand_xyz"]])
        t_hand = _orthonormal_tangent(n_hand, [float(v) for v in ref["tangent_hand_xyz"]])
        b_hand = _normalize(_cross(n_hand, t_hand))
        t_hand = _normalize(_cross(b_hand, n_hand))

        n_target = [-nw[0], -nw[1], -nw[2]]
        t_target = tw
        b_target = _normalize(_cross(n_target, t_target))
        t_target = _normalize(_cross(b_target, n_target))

        r_world_target = _mat_from_basis(t_target, b_target, n_target)
        r_hand_local = _mat_from_basis(t_hand, b_hand, n_hand)
        r_hand_world = _mat_mul(r_world_target, _mat_transpose(r_hand_local))
        q_hand_world = _quat_from_basis(
            [r_hand_world[0][0], r_hand_world[1][0], r_hand_world[2][0]],
            [r_hand_world[0][1], r_hand_world[1][1], r_hand_world[2][1]],
            [r_hand_world[0][2], r_hand_world[1][2], r_hand_world[2][2]],
        )

        target_ref = [
            pw[0] + (selected_distance_mm / 1000.0) * nw[0],
            pw[1] + (selected_distance_mm / 1000.0) * nw[1],
            pw[2] + (selected_distance_mm / 1000.0) * nw[2],
        ]
        offset = _mat_vec_mul(r_hand_world, p_hand)
        hand_pos = [target_ref[0] - offset[0], target_ref[1] - offset[1], target_ref[2] - offset[2]]
        p.resetBasePositionAndOrientation(hand_id, hand_pos, q_hand_world)

    print(f"part_id={part_id}")
    print(f"num_candidates={len(candidates)}")
    print("Controls:")
    print("  LB/RB: previous/next candidate grasp point")
    print("  Right stick Y: distance +/- 5 mm")
    print("  X: switch hand reference point")
    print("  RT/LT: hand twist +/- 15 deg")
    print("  Y: spawn/place AR10 hand on selected plane")
    print("  A: save selected grasp point to artifacts/grasp_points_part_x.yaml")
    print("  Menu: quit")

    _update_selected_visual()

    lb_latch = False
    rb_latch = False
    x_latch = False
    y_latch = False
    a_latch = False
    lt_latch = False
    rt_latch = False
    next_distance_update = 0.0

    def _norm_trigger(raw: float) -> float:
        if raw < 0.0:
            return max(0.0, min(1.0, (raw + 1.0) * 0.5))
        return max(0.0, min(1.0, raw))

    out_path = Path(f"artifacts/grasp_points_part_{int(part_id)}.yaml")
    dt = 1.0 / 120.0
    try:
        while p.isConnected(client_id):
            pygame.event.pump()
            lb = bool(js.get_numbuttons() > 4 and js.get_button(4))
            rb = bool(js.get_numbuttons() > 5 and js.get_button(5))
            x_btn = bool(js.get_numbuttons() > 2 and js.get_button(2))
            y_btn = bool(js.get_numbuttons() > 3 and js.get_button(3))
            a_btn = bool(js.get_numbuttons() > 0 and js.get_button(0))
            menu = bool(js.get_numbuttons() > 7 and js.get_button(7))
            r_stick_y = float(js.get_axis(3)) if js.get_numaxes() > 3 else 0.0
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

            now = time.time()
            if abs(r_stick_y) > 0.55 and now >= next_distance_update:
                selected_distance_mm = max(0.0, selected_distance_mm + (5.0 if r_stick_y < 0.0 else -5.0))
                next_distance_update = now + 0.16
                changed = True

            if x_btn and not x_latch:
                ref_idx = (ref_idx + 1) % len(ref_names)
                changed = True

            if rt_pressed and not rt_latch:
                twist_deg += 15.0
                changed = True
            if lt_pressed and not lt_latch:
                twist_deg -= 15.0
                changed = True

            if y_btn and not y_latch:
                hand_visible = not hand_visible
                _set_body_visibility(hand_id, visible=hand_visible)
                if hand_visible:
                    _place_hand_to_selected()
            if hand_visible and changed:
                _place_hand_to_selected()

            if a_btn and not a_latch:
                c = candidates[selected_idx]
                gp_id = _save_selected_gp(
                    path=out_path,
                    part_id=part_id,
                    obj_pos=list(obj_pos),
                    obj_quat=list(obj_quat_now),
                    p_local=[float(v) for v in c["position_obj"]],
                    n_local=[float(v) for v in c["normal_obj"]],
                    t_local=[float(v) for v in c["tangent_obj"]],
                    pose_name=str(ref_names[ref_idx]),
                    distance_mm=float(selected_distance_mm),
                    twist_deg=float(twist_deg),
                )
                print(f"saved={gp_id} file={out_path} ref={ref_names[ref_idx]} dist_mm={round(selected_distance_mm,1)} twist_deg={round(twist_deg,1)}")

            if changed:
                _update_selected_visual()

            lb_latch = lb
            rb_latch = rb
            x_latch = x_btn
            y_latch = y_btn
            a_latch = a_btn
            lt_latch = lt_pressed
            rt_latch = rt_pressed

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
    parser.add_argument("--part-id", type=int, required=True, help="Benchmark part id, e.g. 1..14")
    parser.add_argument("--obj-x", type=float, default=0.80)
    parser.add_argument("--obj-y", type=float, default=0.00)
    parser.add_argument("--obj-z", type=float, default=0.05)
    parser.add_argument("--obj-roll-deg", type=float, default=0.0)
    parser.add_argument("--obj-pitch-deg", type=float, default=0.0)
    parser.add_argument("--obj-yaw-deg", type=float, default=0.0)
    parser.add_argument("--samples-u", type=int, default=4)
    parser.add_argument("--samples-v", type=int, default=4)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.08)
    parser.add_argument("--ray-len", type=float, default=0.20)
    parser.add_argument("--patch-distance-mm", type=float, default=20.0)
    parser.add_argument("--patch-size-mm", type=float, default=22.0)
    parser.add_argument("--marker-radius-mm", type=float, default=3.2)
    parser.add_argument("--hand-ref-yaml", default="artifacts/hand_reference_points.yaml")
    args = parser.parse_args()
    run(
        part_id=args.part_id,
        object_xyz=[args.obj_x, args.obj_y, args.obj_z],
        object_rpy_deg=[args.obj_roll_deg, args.obj_pitch_deg, args.obj_yaw_deg],
        samples_u=args.samples_u,
        samples_v=args.samples_v,
        edge_margin_ratio=args.edge_margin_ratio,
        ray_len=args.ray_len,
        patch_distance_mm=args.patch_distance_mm,
        patch_size_mm=args.patch_size_mm,
        marker_radius_mm=args.marker_radius_mm,
        hand_ref_yaml=args.hand_ref_yaml,
    )


if __name__ == "__main__":
    main()
