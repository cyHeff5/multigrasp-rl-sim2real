"""
Manual test: cycle through all grasp points on a benchmark part with the Sawyer arm.

The object is spawned in ghost mode (no collisions), so the hand can move freely
through it to any pregrasp pose without knocking it over.

Controls (gamepad):
  RT: next grasp point
  LT: previous grasp point
  Menu/Start: quit

Controls (keyboard, no gamepad):
  → or D: next grasp point
  ← or A: previous grasp point
  Q / Escape: quit
"""
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
from src.sim.mounting import apply_hand_friction, apply_physics_defaults, mount_hand_to_arm
from src.sim.sawyer_arm import SawyerHelper
from tests.manual.manual_test_position_pregrasp import (
    _cross,
    _normalize,
    _orthonormal_tangent,
    _rotate_local_to_world,
    _transform_local_to_world,
    _mat_from_basis,
    _mat_mul,
    _mat_transpose,
    _mat_vec_mul,
    _quat_from_mat,
    _load_hand_reference,
    _set_body_collision_enabled,
    _home_sawyer,
    _create_sphere_marker,
    _reset_marker,
)

_TRIGGER_THRESHOLD = 0.5
_HAND_YAML = "artifacts/hand_reference_points.yaml"


def _load_all_grasp_points(part_id: int) -> tuple[list[float], list[float], list[dict]]:
    """Load object pose and all grasp points for a benchmark part."""
    yaml_path = Path(f"artifacts/grasp_points_part_{part_id}.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"No grasp point file for part {part_id}: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    obj_pos = [float(v) for v in data["object_pose_world"]["position_xyz"]]
    obj_quat = [float(v) for v in data["object_pose_world"]["orientation_xyzw"]]
    gps = list(data.get("grasp_points", []))
    if not gps:
        raise RuntimeError(f"No grasp points in {yaml_path}")
    return obj_pos, obj_quat, gps


def _rotate_about_axis(v: list[float], axis: list[float], angle_rad: float) -> list[float]:
    """Rotate vector v around axis by angle_rad (Rodrigues)."""
    axis = _normalize(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    d = sum(axis[i] * v[i] for i in range(3))
    cr = _cross(axis, v)
    return [v[i] * c + cr[i] * s + axis[i] * d * (1.0 - c) for i in range(3)]


def _read_triggers(js) -> tuple[float, float]:
    n = js.get_numaxes()
    if n >= 6:
        lt = 0.5 * (float(js.get_axis(4)) + 1.0)
        rt = 0.5 * (float(js.get_axis(5)) + 1.0)
    elif n >= 3:
        a = float(js.get_axis(2))
        lt, rt = max(0.0, -a), max(0.0, a)
    else:
        lt, rt = 0.0, 0.0
    return max(0.0, min(1.0, lt)), max(0.0, min(1.0, rt))


def _clear_debug_items(item_ids: list[int]) -> list[int]:
    for iid in item_ids:
        try:
            p.removeUserDebugItem(iid)
        except Exception:
            pass
    return []


def _compute_target_ee_pose(
    gp: dict,
    obj_pos: list[float],
    obj_quat: list[float],
    hand_ref: dict,
    distance_mm: float,
    twist_deg: float,
    hand_to_ee_pos,
    hand_to_ee_quat,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Compute target EE pose (xyz, rpy) from grasp point geometry.
    Returns (ee_xyz, ee_rpy, gp_world, n_world, target_ref_world).
    """
    gp_world = _transform_local_to_world(
        [float(v) for v in gp["position_obj_xyz"]], obj_pos, obj_quat
    )
    n_world = _normalize(_rotate_local_to_world(
        _normalize([float(v) for v in gp["normal_obj_xyz"]]), obj_quat
    ))
    t_world = _orthonormal_tangent(
        n_world,
        _rotate_local_to_world(_normalize([float(v) for v in gp["tangent_obj_xyz"]]), obj_quat),
    )
    if abs(twist_deg) > 1e-9:
        t_world = _orthonormal_tangent(
            n_world, _rotate_about_axis(t_world, n_world, math.radians(twist_deg))
        )

    target_ref_world = [gp_world[i] + (distance_mm / 1000.0) * n_world[i] for i in range(3)]

    # Compute target hand base rotation (full_pose: hand normal opposes object normal)
    n_target = [-n_world[i] for i in range(3)]
    t_target = _orthonormal_tangent(n_target, t_world)
    b_target = _normalize(_cross(n_target, t_target))
    t_target = _normalize(_cross(b_target, n_target))
    r_world_target = _mat_from_basis(t_target, b_target, n_target)

    n_hand = _normalize([float(v) for v in hand_ref["normal_hand_xyz"]])
    t_hand = _orthonormal_tangent(n_hand, _normalize([float(v) for v in hand_ref["tangent_hand_xyz"]]))
    b_hand = _normalize(_cross(n_hand, t_hand))
    t_hand = _normalize(_cross(b_hand, n_hand))
    r_hand_local = _mat_from_basis(t_hand, b_hand, n_hand)

    r_hand_world = _mat_mul(r_world_target, _mat_transpose(r_hand_local))
    hand_quat = _quat_from_mat(r_hand_world)
    p_hand = [float(v) for v in hand_ref["position_hand_xyz"]]
    hand_offset = _mat_vec_mul(r_hand_world, p_hand)
    hand_pos = [target_ref_world[i] - hand_offset[i] for i in range(3)]

    ee_pos, ee_quat = p.multiplyTransforms(hand_pos, hand_quat, hand_to_ee_pos, hand_to_ee_quat)
    ee_rpy = list(p.getEulerFromQuaternion(ee_quat))
    return list(ee_pos), ee_rpy, gp_world, n_world, target_ref_world


def _joint_space_move(helper: SawyerHelper, q_target, move_seconds: float, hz: int = 240) -> None:
    """Smoothly interpolate in joint space from current config to q_target."""
    q_start = [float(p.getJointState(helper.robot_id, j)[0]) for j in helper.joints]
    q_end = [float(v) for v in q_target]
    steps = max(1, int(move_seconds * hz))
    for i in range(1, steps + 1):
        a = i / float(steps)
        q_i = [q_start[j] * (1.0 - a) + q_end[j] * a for j in range(len(helper.joints))]
        p.setJointMotorControlArray(
            bodyUniqueId=helper.robot_id,
            jointIndices=helper.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_i,
            positionGains=[0.5] * len(helper.joints),
            forces=[500.0] * len(helper.joints),
        )
        p.stepSimulation()


def _move_to_grasp_point(
    gp: dict,
    obj_pos: list[float],
    obj_quat: list[float],
    helper: SawyerHelper,
    hand_to_ee_pos,
    hand_to_ee_quat,
    move_seconds: float,
    ik_attempts: int = 150,
) -> tuple[list[float], list[float], list[float], str, float]:
    """Compute pregrasp for gp, move arm there via random IK + joint-space interpolation.
    Returns (gp_world, n_world, target_ref_world, grasp_type, distance_mm).
    """
    poses = dict(gp.get("poses", {}))
    if not poses:
        raise RuntimeError(f"Grasp point {gp.get('id')} has no poses")
    grasp_type = next(iter(poses))
    pose_cfg = poses[grasp_type]
    distance_mm = float(pose_cfg.get("distance_mm", 35.0))
    twist_deg = float(pose_cfg.get("twist_deg", 0.0))

    hand_ref = _load_hand_reference(_HAND_YAML, grasp_type)

    ee_xyz, ee_rpy, gp_world, n_world, target_ref_world = _compute_target_ee_pose(
        gp, obj_pos, obj_quat, hand_ref, distance_mm, twist_deg, hand_to_ee_pos, hand_to_ee_quat
    )

    # Find best reachable joint config via random restarts
    q_target = helper.find_random_ik(ee_xyz, ee_rpy, n_attempts=ik_attempts)
    if q_target is None:
        print(f"[warn] {gp.get('id')}: no IK solution found after {ik_attempts} attempts — skipping")
        return gp_world, n_world, target_ref_world, grasp_type, distance_mm

    # Move smoothly in joint space (much more reliable than Cartesian IK interpolation)
    _joint_space_move(helper, q_target, move_seconds=move_seconds)

    return gp_world, n_world, target_ref_world, grasp_type, distance_mm


def _draw_grasp_point_debug(
    gp_world: list[float],
    n_world: list[float],
    target_ref_world: list[float],
    gp_id: str,
    grasp_type: str,
    distance_mm: float,
    idx: int,
    total: int,
) -> list[int]:
    ids = []
    ids.append(p.addUserDebugLine(gp_world, target_ref_world, [0.2, 0.8, 1.0], lineWidth=2.0, lifeTime=0.0))
    ids.append(p.addUserDebugLine(
        target_ref_world,
        [target_ref_world[i] + 0.04 * n_world[i] for i in range(3)],
        [1.0, 1.0, 0.0], lineWidth=2.0, lifeTime=0.0,
    ))
    ids.append(p.addUserDebugText(
        f"[{idx + 1}/{total}] {gp_id} | {grasp_type} | {distance_mm:.0f}mm",
        [target_ref_world[0], target_ref_world[1], target_ref_world[2] + 0.10],
        textColorRGB=[1.0, 1.0, 1.0],
        textSize=1.4,
        lifeTime=0.0,
    ))
    return ids


def run(part_id: int, move_seconds: float, robot_base_rpy_deg: list[float]) -> None:
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError("pygame is required: pip install pygame") from exc

    obj_pos, obj_quat, grasp_points = _load_all_grasp_points(part_id)
    print(f"part_id={part_id}  grasp_points={len(grasp_points)}")

    # --- PyBullet setup ---
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    apply_physics_defaults()
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(
        cameraDistance=1.1,
        cameraYaw=55.0,
        cameraPitch=-28.0,
        cameraTargetPosition=[float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]) + 0.05],
    )

    # Pedestal (4cm diameter cylinder, 4cm tall)
    pedestal_h = 0.04
    pedestal_r = 0.02
    ped_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=pedestal_r, height=pedestal_h)
    ped_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=pedestal_r, length=pedestal_h, rgbaColor=[0.75, 0.75, 0.75, 1.0])
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=ped_col,
        baseVisualShapeIndex=ped_vis,
        basePosition=[float(obj_pos[0]), float(obj_pos[1]), pedestal_h * 0.5],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )

    # Spawn object as ghost (no collisions so hand passes through it)
    obj_pos_spawned = [float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]) + pedestal_h]
    object_id = p.loadURDF(
        str(benchmark_part_urdf(part_id)),
        basePosition=obj_pos_spawned,
        baseOrientation=[float(v) for v in obj_quat],
        useFixedBase=True,
    )
    _set_body_collision_enabled(object_id, enabled=False)

    # Sawyer + AR10 hand
    robot_base_quat = p.getQuaternionFromEuler([math.radians(v) for v in robot_base_rpy_deg])
    robot_id = p.loadURDF(
        str(SAWYER_URDF), basePosition=[0, 0, 0], baseOrientation=robot_base_quat, useFixedBase=True
    )
    hand_id = p.loadURDF(str(AR10_URDF))
    mount_hand_to_arm(robot_id, hand_id, ee_link="right_l6")
    apply_hand_friction(hand_id)
    HandModel(hand_id).reset_open_pose()

    # Use the raised position for all grasp point computations
    obj_pos = obj_pos_spawned

    helper = SawyerHelper(robot_id, ee_link_name="right_l6")
    _home_sawyer(robot_id, helper)

    # Compute hand-to-EE transform offset (needed by _move_hand_reference_to_target)
    ee_xyz, ee_rpy = helper.ee_pose()
    ee_quat = p.getQuaternionFromEuler([float(v) for v in ee_rpy])
    cur_hand_pos, cur_hand_quat = p.getBasePositionAndOrientation(hand_id)
    ee_to_hand_pos, ee_to_hand_quat = p.multiplyTransforms(
        *p.invertTransform([float(v) for v in ee_xyz], ee_quat),
        cur_hand_pos, cur_hand_quat,
    )
    hand_to_ee_pos, hand_to_ee_quat = p.invertTransform(ee_to_hand_pos, ee_to_hand_quat)

    # --- Pygame / input ---
    pygame.init()
    pygame.joystick.init()
    js = None
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"controller={js.get_name()}")
        print("Controls:  RT=next  LT=prev  Menu=quit")
    else:
        pygame.display.set_mode((300, 80))
        pygame.display.set_caption("Benchmark Grasp Points")
        print("No gamepad — using keyboard.")
        print("Controls:  →/D=next  ←/A=prev  Q/Esc=quit")

    # --- Move to first grasp point ---
    current_idx = 0
    gp_marker = _create_sphere_marker(obj_pos, radius=0.006, rgba=[0.1, 0.9, 0.1, 1.0])
    debug_ids: list[int] = []

    gp_world, n_world, target_ref_world, grasp_type, distance_mm = _move_to_grasp_point(
        grasp_points[current_idx], obj_pos, obj_quat, helper,
        hand_to_ee_pos, hand_to_ee_quat, move_seconds,
    )
    _reset_marker(gp_marker, gp_world)
    debug_ids = _draw_grasp_point_debug(
        gp_world, n_world, target_ref_world,
        str(grasp_points[current_idx].get("id")), grasp_type, distance_mm,
        current_idx, len(grasp_points),
    )
    print(f"[{current_idx + 1}/{len(grasp_points)}] {grasp_points[current_idx].get('id')} | {grasp_type} | {distance_mm:.0f}mm")

    rt_latch = lt_latch = False

    try:
        while p.isConnected(client_id):
            do_next = do_prev = do_quit = False

            if js is not None:
                pygame.event.pump()
                lt, rt = _read_triggers(js)
                rt_pressed = rt > _TRIGGER_THRESHOLD
                lt_pressed = lt > _TRIGGER_THRESHOLD
                do_next = rt_pressed and not rt_latch
                do_prev = lt_pressed and not lt_latch
                do_quit = bool(js.get_numbuttons() > 7 and js.get_button(7))
                rt_latch = rt_pressed
                lt_latch = lt_pressed
            else:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_RIGHT, pygame.K_d):
                            do_next = True
                        elif event.key in (pygame.K_LEFT, pygame.K_a):
                            do_prev = True
                        elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                            do_quit = True
                    elif event.type == pygame.QUIT:
                        do_quit = True

            if do_quit:
                break

            if do_next or do_prev:
                current_idx = (current_idx + (1 if do_next else -1)) % len(grasp_points)
                debug_ids = _clear_debug_items(debug_ids)

                gp_world, n_world, target_ref_world, grasp_type, distance_mm = _move_to_grasp_point(
                    grasp_points[current_idx], obj_pos, obj_quat, helper,
                    hand_to_ee_pos, hand_to_ee_quat, move_seconds,
                )
                _reset_marker(gp_marker, gp_world)
                debug_ids = _draw_grasp_point_debug(
                    gp_world, n_world, target_ref_world,
                    str(grasp_points[current_idx].get("id")), grasp_type, distance_mm,
                    current_idx, len(grasp_points),
                )
                print(f"[{current_idx + 1}/{len(grasp_points)}] {grasp_points[current_idx].get('id')} | {grasp_type} | {distance_mm:.0f}mm")

            p.stepSimulation()
            time.sleep(1.0 / 60.0)
    finally:
        try:
            pygame.quit()
        except Exception:
            pass
        if p.isConnected(client_id):
            p.disconnect(client_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cycle through benchmark part grasp points with Sawyer arm.")
    parser.add_argument("--part-id", type=int, required=True, help="Benchmark part ID (1-14)")
    parser.add_argument("--move-seconds", type=float, default=2.0, help="Arm movement duration per grasp point")
    parser.add_argument(
        "--robot-base-rpy-deg", type=float, nargs=3, default=[0.0, 0.0, 270.0],
        metavar=("R", "P", "Y"), help="Sawyer base orientation in degrees (default: 0 0 270)",
    )
    args = parser.parse_args()
    run(
        part_id=args.part_id,
        move_seconds=args.move_seconds,
        robot_base_rpy_deg=args.robot_base_rpy_deg,
    )


if __name__ == "__main__":
    main()
