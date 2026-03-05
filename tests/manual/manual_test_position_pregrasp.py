from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pybullet as p
import pybullet_data
import yaml

from src.sim.assets import AR10_URDF, SAWYER_URDF, benchmark_part_urdf
from src.sim.hand_model import HandModel
from src.sim.mounting import apply_hand_friction, mount_hand_to_arm
from src.sim.sawyer_arm import SawyerHelper


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


def _mat_from_basis(x_axis: list[float], y_axis: list[float], z_axis: list[float]) -> list[list[float]]:
    return [
        [float(x_axis[0]), float(y_axis[0]), float(z_axis[0])],
        [float(x_axis[1]), float(y_axis[1]), float(z_axis[1])],
        [float(x_axis[2]), float(y_axis[2]), float(z_axis[2])],
    ]


def _mat_vec_mul(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [
        float(mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2]),
        float(mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2]),
        float(mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2]),
    ]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = float(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
    return out


def _mat_transpose(a: list[list[float]]) -> list[list[float]]:
    return [
        [float(a[0][0]), float(a[1][0]), float(a[2][0])],
        [float(a[0][1]), float(a[1][1]), float(a[2][1])],
        [float(a[0][2]), float(a[1][2]), float(a[2][2])],
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


def _quat_to_rpy(quat: list[float]) -> list[float]:
    return [float(v) for v in p.getEulerFromQuaternion(quat)]


def _mat_from_quat(quat: list[float]) -> list[list[float]]:
    m = p.getMatrixFromQuaternion(quat)
    return [
        [float(m[0]), float(m[1]), float(m[2])],
        [float(m[3]), float(m[4]), float(m[5])],
        [float(m[6]), float(m[7]), float(m[8])],
    ]


def _compute_target_hand_rotation(
    mode: str,
    current_hand_rot: list[list[float]],
    n_obj_world: list[float],
    t_obj_world: list[float],
    n_hand_local: list[float],
    t_hand_local: list[float],
    b_hand_local: list[float],
) -> list[list[float]]:
    current_n_world = _normalize(_mat_vec_mul(current_hand_rot, n_hand_local))
    current_t_world = _normalize(_mat_vec_mul(current_hand_rot, t_hand_local))

    if mode == "position_only":
        return current_hand_rot

    if mode == "twist_only":
        n_hand_world_target = current_n_world
        t_hand_world_target = _orthonormal_tangent(n_hand_world_target, t_obj_world)
    else:
        n_hand_world_target = [-n_obj_world[0], -n_obj_world[1], -n_obj_world[2]]
        if mode == "normal_only":
            t_hand_world_target = _orthonormal_tangent(n_hand_world_target, current_t_world)
        elif mode == "full_pose":
            t_hand_world_target = t_obj_world
        else:
            raise ValueError(f"Unsupported mode for target rotation: {mode}")

    b_hand_world_target = _normalize(_cross(n_hand_world_target, t_hand_world_target))
    t_hand_world_target = _normalize(_cross(b_hand_world_target, n_hand_world_target))

    r_world_target = _mat_from_basis(t_hand_world_target, b_hand_world_target, n_hand_world_target)
    r_hand_local = _mat_from_basis(t_hand_local, b_hand_local, n_hand_local)
    return _mat_mul(r_world_target, _mat_transpose(r_hand_local))


def _move_hand_reference_to_target(
    helper: SawyerHelper,
    hand_id: int,
    p_hand_local: list[float],
    n_hand_local: list[float],
    t_hand_local: list[float],
    b_hand_local: list[float],
    hand_to_ee_pos: tuple[float, float, float],
    hand_to_ee_quat: tuple[float, float, float, float],
    target_ref_world: list[float],
    n_obj_world: list[float],
    t_obj_world: list[float],
    mode: str,
    move_seconds: float,
    label: str,
    gain: float,
    force: float,
    settle_seconds: float,
) -> tuple[list[float], list[float], list[float]]:
    cur_hand_pos, cur_hand_quat = p.getBasePositionAndOrientation(hand_id)
    current_hand_rot = _mat_from_quat(list(cur_hand_quat))
    r_hand_base_world = _compute_target_hand_rotation(
        mode=mode,
        current_hand_rot=current_hand_rot,
        n_obj_world=n_obj_world,
        t_obj_world=t_obj_world,
        n_hand_local=n_hand_local,
        t_hand_local=t_hand_local,
        b_hand_local=b_hand_local,
    )
    hand_base_quat_target = _quat_from_mat(r_hand_base_world)
    hand_ref_offset_world = _mat_vec_mul(r_hand_base_world, p_hand_local)
    hand_base_pos_target = [
        target_ref_world[0] - hand_ref_offset_world[0],
        target_ref_world[1] - hand_ref_offset_world[1],
        target_ref_world[2] - hand_ref_offset_world[2],
    ]
    ee_pos_target, ee_quat_target = p.multiplyTransforms(
        hand_base_pos_target,
        hand_base_quat_target,
        hand_to_ee_pos,
        hand_to_ee_quat,
    )
    ee_rpy_target = _quat_to_rpy(list(ee_quat_target))
    print(f"{label}_target_ee_xyz={[round(float(v), 5) for v in ee_pos_target]}")
    print(f"{label}_target_ee_rpy_deg={[round(math.degrees(v), 2) for v in ee_rpy_target]}")
    helper.move_to_pose_blocking(
        xyz=[float(v) for v in ee_pos_target],
        rpy=[float(v) for v in ee_rpy_target],
        seconds=float(move_seconds),
        hz=240,
        gain=float(gain),
        force=float(force),
    )
    _hold_sawyer_target(
        robot_id=helper.robot_id,
        helper=helper,
        ee_pos_target=[float(v) for v in ee_pos_target],
        ee_rpy_target=[float(v) for v in ee_rpy_target],
        hold_seconds=float(settle_seconds),
        gain=float(gain),
        force=float(force),
    )
    return [float(v) for v in ee_pos_target], [float(v) for v in ee_rpy_target], hand_base_pos_target


def _create_sphere_marker(pos: list[float], radius: float, rgba: list[float]) -> int:
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=[float(c) for c in rgba])
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[float(v) for v in pos],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )


def _reset_marker(marker_id: int, pos: list[float]) -> None:
    p.resetBasePositionAndOrientation(
        int(marker_id),
        [float(v) for v in pos],
        [0.0, 0.0, 0.0, 1.0],
    )


def _print_joint_report(robot_id: int, helper: SawyerHelper, ee_pos_target: list[float], ee_rpy_target: list[float]) -> None:
    q_target = helper._ik(ee_pos_target, ee_rpy_target)
    q_reached = [float(p.getJointState(robot_id, int(j))[0]) for j in helper.joints]
    q_target_list = [float(v) for v in q_target.tolist()]
    q_target_deg = [math.degrees(v) for v in q_target_list]
    q_reached_deg = [math.degrees(v) for v in q_reached]
    print(f"sawyer_joints={helper.joints}")
    print(f"target_q_rad={[round(v, 6) for v in q_target_list]}")
    print(f"target_q_deg={[round(v, 2) for v in q_target_deg]}")
    print(f"reached_q_rad={[round(v, 6) for v in q_reached]}")
    print(f"reached_q_deg={[round(v, 2) for v in q_reached_deg]}")


def _hold_sawyer_target(
    robot_id: int,
    helper: SawyerHelper,
    ee_pos_target: list[float],
    ee_rpy_target: list[float],
    hold_seconds: float,
    gain: float,
    force: float,
    hz: int = 240,
) -> None:
    steps = max(1, int(float(hold_seconds) * float(hz)))
    q_target = helper._ik(ee_pos_target, ee_rpy_target).tolist()
    for _ in range(steps):
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=helper.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[float(v) for v in q_target],
            positionGains=[float(gain)] * len(helper.joints),
            forces=[float(force)] * len(helper.joints),
        )
        p.stepSimulation()


def _compute_ref_error_mm(hand_id: int, p_hand_local: list[float], target_ref_world: list[float]) -> tuple[float, list[float]]:
    hand_pos, hand_quat = p.getBasePositionAndOrientation(hand_id)
    current_ref_world = _transform_local_to_world(p_hand_local, list(hand_pos), list(hand_quat))
    err_mm = 1000.0 * math.sqrt(sum((a - b) * (a - b) for a, b in zip(current_ref_world, target_ref_world)))
    return float(err_mm), [float(v) for v in current_ref_world]


def _set_body_collision_enabled(body_id: int, enabled: bool) -> None:
    group = 1 if enabled else 0
    mask = 1 if enabled else 0
    num_joints = p.getNumJoints(body_id)
    for link_idx in range(-1, num_joints):
        p.setCollisionFilterGroupMask(
            bodyUniqueId=int(body_id),
            linkIndexA=int(link_idx),
            collisionFilterGroup=int(group),
            collisionFilterMask=int(mask),
        )


def _load_object_grasp_point(yaml_path: str, grasp_point_id: str | None) -> tuple[int, list[float], list[float], dict]:
    with Path(yaml_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    grasp_points = list(data.get("grasp_points", []))
    if not grasp_points:
        raise RuntimeError(f"No grasp_points in {yaml_path}")
    if grasp_point_id is None:
        gp = grasp_points[0]
    else:
        gp = next((item for item in grasp_points if str(item.get("id")) == str(grasp_point_id)), None)
        if gp is None:
            raise RuntimeError(f"Grasp point '{grasp_point_id}' not found in {yaml_path}")
    return int(data["part_id"]), list(data["object_pose_world"]["position_xyz"]), list(data["object_pose_world"]["orientation_xyzw"]), gp


def _load_hand_reference(yaml_path: str, grasp_pose: str) -> dict:
    with Path(yaml_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    refs = dict(data.get("reference_points", {}))
    if grasp_pose not in refs:
        raise RuntimeError(f"Hand reference '{grasp_pose}' not found in {yaml_path}")
    return refs[grasp_pose]


def _home_sawyer(robot_id: int, helper: SawyerHelper) -> None:
    home_q = [
        -0.19999511870154918,
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
        positionGains=[0.2] * len(helper.joints),
        forces=[900.0] * len(helper.joints),
    )
    for _ in range(240):
        p.stepSimulation()


def run(
    object_yaml: str,
    hand_yaml: str,
    grasp_pose: str,
    grasp_point_id: str | None,
    distance_mm: float,
    move_seconds: float,
    mode: str,
    joint_gain: float,
    joint_force: float,
    settle_seconds: float,
    final_replans: int,
    fine_tune_threshold_mm: float,
    fine_tune_max_passes: int,
    object_ghost_at_start: bool,
    restore_object_collision_after_move: bool,
) -> None:
    part_id, obj_pos, obj_quat, gp = _load_object_grasp_point(object_yaml, grasp_point_id)
    hand_ref = _load_hand_reference(hand_yaml, grasp_pose)

    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(
        cameraDistance=0.85,
        cameraYaw=55.0,
        cameraPitch=-28.0,
        cameraTargetPosition=[float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]) + 0.05],
    )

    object_id = p.loadURDF(
        str(benchmark_part_urdf(part_id)),
        basePosition=[float(v) for v in obj_pos],
        baseOrientation=[float(v) for v in obj_quat],
        useFixedBase=True,
    )
    if bool(object_ghost_at_start):
        _set_body_collision_enabled(object_id, enabled=False)
        print("object_collision=disabled_at_start")

    robot_id = p.loadURDF(str(SAWYER_URDF), basePosition=[0.0, 0.0, 0.0], useFixedBase=True)
    hand_id = p.loadURDF(str(AR10_URDF))
    mount_hand_to_arm(robot_id, hand_id, ee_link="right_l6")
    apply_hand_friction(hand_id)
    HandModel(hand_id).reset_open_pose()

    helper = SawyerHelper(robot_id, ee_link_name="right_l6")
    _home_sawyer(robot_id, helper)

    ee_xyz, ee_rpy = helper.ee_pose()
    ee_quat = p.getQuaternionFromEuler(ee_rpy)
    cur_hand_pos, cur_hand_quat = p.getBasePositionAndOrientation(hand_id)
    ee_to_hand_pos, ee_to_hand_quat = p.multiplyTransforms(
        *p.invertTransform(ee_xyz.tolist(), ee_quat),
        cur_hand_pos,
        cur_hand_quat,
    )
    hand_to_ee_pos, hand_to_ee_quat = p.invertTransform(ee_to_hand_pos, ee_to_hand_quat)

    p_obj = [float(v) for v in gp["position_obj_xyz"]]
    n_obj = _normalize([float(v) for v in gp["normal_obj_xyz"]])
    t_obj_hint = _normalize([float(v) for v in gp["tangent_obj_xyz"]])
    p_obj_world = _transform_local_to_world(p_obj, obj_pos, obj_quat)
    n_obj_world = _normalize(_rotate_local_to_world(n_obj, obj_quat))
    t_obj_world = _orthonormal_tangent(n_obj_world, _rotate_local_to_world(t_obj_hint, obj_quat))
    b_obj_world = _normalize(_cross(n_obj_world, t_obj_world))

    p_hand_local = [float(v) for v in hand_ref["position_hand_xyz"]]
    n_hand_local = _normalize([float(v) for v in hand_ref["normal_hand_xyz"]])
    t_hand_local_hint = _normalize([float(v) for v in hand_ref["tangent_hand_xyz"]])
    t_hand_local = _orthonormal_tangent(n_hand_local, t_hand_local_hint)
    b_hand_local = _normalize(_cross(n_hand_local, t_hand_local))

    target_ref_world = [
        p_obj_world[0] + (float(distance_mm) / 1000.0) * n_obj_world[0],
        p_obj_world[1] + (float(distance_mm) / 1000.0) * n_obj_world[1],
        p_obj_world[2] + (float(distance_mm) / 1000.0) * n_obj_world[2],
    ]

    _create_sphere_marker(p_obj_world, radius=0.005, rgba=[0.1, 0.9, 0.1, 1.0])
    _create_sphere_marker(target_ref_world, radius=0.005, rgba=[1.0, 0.5, 0.1, 1.0])
    current_ref_marker = _create_sphere_marker(target_ref_world, radius=0.004, rgba=[1.0, 0.0, 1.0, 1.0])
    p.addUserDebugLine(
        p_obj_world,
        target_ref_world,
        [0.2, 0.6, 1.0],
        lineWidth=2.0,
        lifeTime=0.0,
    )
    p.addUserDebugLine(
        target_ref_world,
        [
            target_ref_world[0] + 0.03 * n_obj_world[0],
            target_ref_world[1] + 0.03 * n_obj_world[1],
            target_ref_world[2] + 0.03 * n_obj_world[2],
        ],
        [1.0, 1.0, 0.0],
        lineWidth=2.0,
        lifeTime=0.0,
    )

    print(f"part_id={part_id}")
    print(f"grasp_point_id={gp.get('id')}")
    print(f"grasp_pose={grasp_pose}")
    print(f"mode={mode}")
    print(f"target_ref_world={[round(v, 5) for v in target_ref_world]}")

    if mode == "staged":
        stage_seconds = max(0.3, float(move_seconds) / 3.0)
        ee_pos_target, ee_rpy_target, _ = _move_hand_reference_to_target(
            helper=helper,
            hand_id=hand_id,
            p_hand_local=p_hand_local,
            n_hand_local=n_hand_local,
            t_hand_local=t_hand_local,
            b_hand_local=b_hand_local,
            hand_to_ee_pos=hand_to_ee_pos,
            hand_to_ee_quat=hand_to_ee_quat,
            target_ref_world=target_ref_world,
            n_obj_world=n_obj_world,
            t_obj_world=t_obj_world,
            mode="position_only",
            move_seconds=stage_seconds,
            label="stage1_position",
            gain=joint_gain,
            force=joint_force,
            settle_seconds=settle_seconds,
        )
        ee_pos_target, ee_rpy_target, _ = _move_hand_reference_to_target(
            helper=helper,
            hand_id=hand_id,
            p_hand_local=p_hand_local,
            n_hand_local=n_hand_local,
            t_hand_local=t_hand_local,
            b_hand_local=b_hand_local,
            hand_to_ee_pos=hand_to_ee_pos,
            hand_to_ee_quat=hand_to_ee_quat,
            target_ref_world=target_ref_world,
            n_obj_world=n_obj_world,
            t_obj_world=t_obj_world,
            mode="normal_only",
            move_seconds=stage_seconds,
            label="stage2_normal",
            gain=joint_gain,
            force=joint_force,
            settle_seconds=settle_seconds,
        )
        ee_pos_target, ee_rpy_target, _ = _move_hand_reference_to_target(
            helper=helper,
            hand_id=hand_id,
            p_hand_local=p_hand_local,
            n_hand_local=n_hand_local,
            t_hand_local=t_hand_local,
            b_hand_local=b_hand_local,
            hand_to_ee_pos=hand_to_ee_pos,
            hand_to_ee_quat=hand_to_ee_quat,
            target_ref_world=target_ref_world,
            n_obj_world=n_obj_world,
            t_obj_world=t_obj_world,
            mode="full_pose",
            move_seconds=stage_seconds,
            label="stage3_full",
            gain=joint_gain,
            force=joint_force,
            settle_seconds=settle_seconds,
        )
        for i in range(max(0, int(final_replans))):
            ee_pos_target, ee_rpy_target, _ = _move_hand_reference_to_target(
                helper=helper,
                hand_id=hand_id,
                p_hand_local=p_hand_local,
                n_hand_local=n_hand_local,
                t_hand_local=t_hand_local,
                b_hand_local=b_hand_local,
                hand_to_ee_pos=hand_to_ee_pos,
                hand_to_ee_quat=hand_to_ee_quat,
                target_ref_world=target_ref_world,
                n_obj_world=n_obj_world,
                t_obj_world=t_obj_world,
                mode="full_pose",
                move_seconds=max(0.25, stage_seconds * 0.6),
                label=f"stage3_replan_{i + 1}",
                gain=joint_gain,
                force=joint_force,
                settle_seconds=settle_seconds,
            )
    else:
        ee_pos_target, ee_rpy_target, _ = _move_hand_reference_to_target(
            helper=helper,
            hand_id=hand_id,
            p_hand_local=p_hand_local,
            n_hand_local=n_hand_local,
            t_hand_local=t_hand_local,
            b_hand_local=b_hand_local,
            hand_to_ee_pos=hand_to_ee_pos,
            hand_to_ee_quat=hand_to_ee_quat,
            target_ref_world=target_ref_world,
            n_obj_world=n_obj_world,
            t_obj_world=t_obj_world,
            mode=mode,
            move_seconds=float(move_seconds),
            label="final",
            gain=joint_gain,
            force=joint_force,
            settle_seconds=settle_seconds,
        )
        if mode == "full_pose":
            for i in range(max(0, int(final_replans))):
                ee_pos_target, ee_rpy_target, _ = _move_hand_reference_to_target(
                    helper=helper,
                    hand_id=hand_id,
                    p_hand_local=p_hand_local,
                    n_hand_local=n_hand_local,
                    t_hand_local=t_hand_local,
                    b_hand_local=b_hand_local,
                    hand_to_ee_pos=hand_to_ee_pos,
                    hand_to_ee_quat=hand_to_ee_quat,
                    target_ref_world=target_ref_world,
                    n_obj_world=n_obj_world,
                    t_obj_world=t_obj_world,
                    mode="full_pose",
                    move_seconds=max(0.25, float(move_seconds) * 0.35),
                    label=f"final_replan_{i + 1}",
                    gain=joint_gain,
                    force=joint_force,
                    settle_seconds=settle_seconds,
                )

    ref_error_mm, current_ref_world = _compute_ref_error_mm(hand_id, p_hand_local, target_ref_world)
    fine_tune_pass = 0
    while ref_error_mm > float(fine_tune_threshold_mm) and fine_tune_pass < int(fine_tune_max_passes):
        fine_tune_pass += 1
        print(f"fine_tune_pass={fine_tune_pass} trigger_ref_error_mm={round(ref_error_mm, 3)}")
        ee_pos_target, ee_rpy_target, _ = _move_hand_reference_to_target(
            helper=helper,
            hand_id=hand_id,
            p_hand_local=p_hand_local,
            n_hand_local=n_hand_local,
            t_hand_local=t_hand_local,
            b_hand_local=b_hand_local,
            hand_to_ee_pos=hand_to_ee_pos,
            hand_to_ee_quat=hand_to_ee_quat,
            target_ref_world=target_ref_world,
            n_obj_world=n_obj_world,
            t_obj_world=t_obj_world,
            mode="full_pose" if mode in ("staged", "full_pose") else mode,
            move_seconds=max(0.25, float(move_seconds) * 0.35),
            label=f"fine_tune_{fine_tune_pass}",
            gain=joint_gain,
            force=joint_force,
            settle_seconds=settle_seconds,
        )
        ref_error_mm, current_ref_world = _compute_ref_error_mm(hand_id, p_hand_local, target_ref_world)

    _reset_marker(current_ref_marker, current_ref_world)
    reached_ee_xyz, reached_ee_rpy = helper.ee_pose()
    p.addUserDebugLine(
        current_ref_world,
        target_ref_world,
        [1.0, 0.0, 1.0],
        lineWidth=2.0,
        lifeTime=0.0,
    )
    print(f"current_ref_world={[round(v, 5) for v in current_ref_world]}")
    print(f"ref_error_mm={round(ref_error_mm, 3)}")
    print(f"reached_ee_xyz={[round(float(v), 5) for v in reached_ee_xyz]}")
    print(f"reached_ee_rpy_deg={[round(math.degrees(v), 2) for v in reached_ee_rpy]}")
    print(
        "ee_pos_error_mm="
        f"{round(1000.0 * math.sqrt(sum((float(a) - float(b)) * (float(a) - float(b)) for a, b in zip(reached_ee_xyz, ee_pos_target))), 3)}"
    )
    _print_joint_report(robot_id=robot_id, helper=helper, ee_pos_target=ee_pos_target, ee_rpy_target=ee_rpy_target)
    if bool(object_ghost_at_start) and bool(restore_object_collision_after_move):
        _set_body_collision_enabled(object_id, enabled=True)
        print("object_collision=restored_after_move")

    print("Pregrasp pose reached. Press Ctrl+C in terminal to quit.")
    try:
        while p.isConnected(client_id):
            current_hand_pos, current_hand_quat = p.getBasePositionAndOrientation(hand_id)
            current_ref_world = _transform_local_to_world(p_hand_local, list(current_hand_pos), list(current_hand_quat))
            _reset_marker(current_ref_marker, current_ref_world)
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
    parser.add_argument("--hand-yaml", required=True, help="Saved hand reference-point yaml")
    parser.add_argument("--grasp-pose", required=True, help="Hand grasp pose key, e.g. tripod")
    parser.add_argument("--grasp-point-id", default=None, help="Object grasp point id; defaults to first point")
    parser.add_argument("--distance-mm", type=float, default=0.0, help="Offset along object normal for pregrasp")
    parser.add_argument("--move-seconds", type=float, default=1.5, help="Duration of the approach motion")
    parser.add_argument("--joint-gain", type=float, default=0.24, help="Sawyer POSITION_CONTROL gain")
    parser.add_argument("--joint-force", type=float, default=1000.0, help="Sawyer POSITION_CONTROL max force")
    parser.add_argument("--settle-seconds", type=float, default=0.8, help="Hold time after each stage/move")
    parser.add_argument("--final-replans", type=int, default=2, help="Additional full_pose replans after final stage")
    parser.add_argument("--fine-tune-threshold-mm", type=float, default=10.0, help="Auto fine-tune if ref error above this")
    parser.add_argument("--fine-tune-max-passes", type=int, default=3, help="Max automatic fine-tune passes")
    parser.add_argument(
        "--object-ghost-at-start",
        action="store_true",
        default=True,
        help="Disable object collisions during positioning so hand cannot get stuck",
    )
    parser.add_argument(
        "--no-object-ghost-at-start",
        dest="object_ghost_at_start",
        action="store_false",
        help="Keep object collisions enabled from the beginning",
    )
    parser.add_argument(
        "--restore-object-collision-after-move",
        action="store_true",
        default=True,
        help="Re-enable object collision after pregrasp positioning is finished",
    )
    parser.add_argument(
        "--no-restore-object-collision-after-move",
        dest="restore_object_collision_after_move",
        action="store_false",
        help="Keep object collision disabled after positioning",
    )
    parser.add_argument(
        "--mode",
        choices=["position_only", "normal_only", "twist_only", "full_pose", "staged"],
        default="position_only",
        help="IK target mode. 'staged' executes position -> normal -> full pose sequentially.",
    )
    args = parser.parse_args()
    run(
        object_yaml=args.object_yaml,
        hand_yaml=args.hand_yaml,
        grasp_pose=args.grasp_pose,
        grasp_point_id=args.grasp_point_id,
        distance_mm=args.distance_mm,
        move_seconds=args.move_seconds,
        mode=args.mode,
        joint_gain=args.joint_gain,
        joint_force=args.joint_force,
        settle_seconds=args.settle_seconds,
        final_replans=args.final_replans,
        fine_tune_threshold_mm=args.fine_tune_threshold_mm,
        fine_tune_max_passes=args.fine_tune_max_passes,
        object_ghost_at_start=args.object_ghost_at_start,
        restore_object_collision_after_move=args.restore_object_collision_after_move,
    )


if __name__ == "__main__":
    main()
