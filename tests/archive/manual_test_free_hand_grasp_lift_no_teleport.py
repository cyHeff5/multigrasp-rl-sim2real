from __future__ import annotations

import argparse
import math
import time

import pybullet as p
import pybullet_data
import pygame

from src.sim.assets import AR10_URDF
from src.sim.hand_model import HandModel
from src.sim.mounting import apply_hand_friction, apply_physics_defaults


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _norm_trigger(raw: float) -> float:
    # Xbox trigger axis in many pygame mappings: [-1, 1]
    return _clamp((float(raw) + 1.0) * 0.5, 0.0, 1.0)


def run(hz: float, deadzone: float, move_speed: float, rot_speed_deg: float, hold_force: float) -> None:
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
    apply_physics_defaults(num_solver_iterations=200, fixed_time_step=1.0 / 240.0, num_substeps=6)
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(
        cameraDistance=0.9,
        cameraYaw=52.0,
        cameraPitch=-30.0,
        cameraTargetPosition=[0.55, 0.0, 0.08],
    )

    # Grasp object: light cube with elevated friction for contact stability.
    obj_half = 0.025
    obj_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[obj_half, obj_half, obj_half])
    obj_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[obj_half, obj_half, obj_half], rgbaColor=[0.95, 0.95, 0.95, 1.0])
    obj_id = p.createMultiBody(
        baseMass=0.08,
        baseCollisionShapeIndex=obj_col,
        baseVisualShapeIndex=obj_vis,
        basePosition=[0.58, 0.0, obj_half],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )
    p.changeDynamics(obj_id, -1, lateralFriction=1.6, spinningFriction=1.1, rollingFriction=0.001, frictionAnchor=True)

    hand_start_pos = [0.58, 0.08, 0.09]
    hand_start_quat = p.getQuaternionFromEuler([math.radians(90.0), 0.0, math.radians(90.0)])
    hand_id = p.loadURDF(
        str(AR10_URDF),
        basePosition=hand_start_pos,
        baseOrientation=hand_start_quat,
        useFixedBase=False,
    )
    apply_hand_friction(hand_id, lateral=1.8, spinning=1.3, rolling=0.0005, use_anchor=True)
    hand = HandModel(hand_id)
    hand.reset_open_pose(force=18.0)

    # Robust alternative to world-constraint: use an invisible fixed anchor body.
    anchor_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.002, rgbaColor=[1.0, 1.0, 1.0, 0.0])
    anchor_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=anchor_vis,
        basePosition=hand_start_pos,
        baseOrientation=hand_start_quat,
    )
    hand_constraint = p.createConstraint(
        parentBodyUniqueId=anchor_id,
        parentLinkIndex=-1,
        childBodyUniqueId=hand_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0.0, 0.0, 0.0],
        parentFramePosition=[0.0, 0.0, 0.0],
        childFramePosition=[0.0, 0.0, 0.0],
        parentFrameOrientation=[0.0, 0.0, 0.0, 1.0],
        childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
    )
    p.changeConstraint(hand_constraint, maxForce=float(hold_force), erp=0.95)

    target_pos = [float(v) for v in hand_start_pos]
    target_rpy = [math.radians(90.0), 0.0, math.radians(90.0)]
    hand_scalar = 0.0
    auto_lift = False

    print("Controls:")
    print("  Left stick: XY translate hand")
    print("  A/B: Z up/down")
    print("  Right stick Y: close/open hand")
    print("  Right stick X: yaw")
    print("  LT/RT: roll")
    print("  Y: toggle auto-lift (+Z)")
    print("  X: reset hand pose")
    print("  Menu or ESC: quit")

    dt = 1.0 / float(hz)
    rot_speed = math.radians(float(rot_speed_deg))
    y_latch = False
    x_latch = False
    esc_key = 27
    try:
        while p.isConnected(cid):
            pygame.event.pump()
            keys = p.getKeyboardEvents()
            if esc_key in keys and (keys[esc_key] & p.KEY_WAS_TRIGGERED):
                break

            left_x = float(js.get_axis(0)) if js.get_numaxes() > 0 else 0.0
            left_y = float(js.get_axis(1)) if js.get_numaxes() > 1 else 0.0
            right_x = float(js.get_axis(2)) if js.get_numaxes() > 2 else 0.0
            right_y = float(js.get_axis(3)) if js.get_numaxes() > 3 else 0.0
            lt = _norm_trigger(float(js.get_axis(4))) if js.get_numaxes() > 4 else 0.0
            rt = _norm_trigger(float(js.get_axis(5))) if js.get_numaxes() > 5 else 0.0

            if abs(left_x) < float(deadzone):
                left_x = 0.0
            if abs(left_y) < float(deadzone):
                left_y = 0.0
            if abs(right_x) < float(deadzone):
                right_x = 0.0
            if abs(right_y) < float(deadzone):
                right_y = 0.0

            a_btn = bool(js.get_numbuttons() > 0 and js.get_button(0))
            b_btn = bool(js.get_numbuttons() > 1 and js.get_button(1))
            x_btn = bool(js.get_numbuttons() > 2 and js.get_button(2))
            y_btn = bool(js.get_numbuttons() > 3 and js.get_button(3))
            menu_btn = bool(js.get_numbuttons() > 7 and js.get_button(7))
            if menu_btn:
                break

            target_pos[0] += (-left_y) * float(move_speed) * dt
            target_pos[1] += (+left_x) * float(move_speed) * dt
            if a_btn:
                target_pos[2] += float(move_speed) * dt
            if b_btn:
                target_pos[2] -= float(move_speed) * dt
            target_pos[2] = _clamp(target_pos[2], 0.03, 0.35)

            target_rpy[2] += (-right_x) * rot_speed * dt
            target_rpy[0] += (rt - lt) * rot_speed * dt

            if y_btn and not y_latch:
                auto_lift = not auto_lift
            if auto_lift:
                target_pos[2] = _clamp(target_pos[2] + 0.04 * dt, 0.03, 0.35)

            if x_btn and not x_latch:
                target_pos = [float(v) for v in hand_start_pos]
                target_rpy = [math.radians(90.0), 0.0, math.radians(90.0)]
                auto_lift = False

            hand_scalar = _clamp(hand_scalar + (-right_y) * 1.1 * dt, 0.0, 1.0)
            hand.send_q_target([hand_scalar] * 10, force=20.0)

            target_quat = p.getQuaternionFromEuler(target_rpy)
            p.resetBasePositionAndOrientation(anchor_id, target_pos, target_quat)
            p.changeConstraint(
                hand_constraint,
                maxForce=float(hold_force),
            )

            p.stepSimulation()
            time.sleep(dt)
            y_latch = y_btn
            x_latch = x_btn
    finally:
        try:
            pygame.quit()
        except Exception:
            pass
        if p.isConnected(cid):
            p.disconnect(cid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=float, default=120.0)
    parser.add_argument("--deadzone", type=float, default=0.20)
    parser.add_argument("--move-speed", type=float, default=0.20, help="Hand target speed in m/s")
    parser.add_argument("--rot-speed-deg", type=float, default=90.0, help="Hand rotation speed deg/s")
    parser.add_argument("--hold-force", type=float, default=3000.0, help="Constraint hold force")
    args = parser.parse_args()
    run(
        hz=args.hz,
        deadzone=args.deadzone,
        move_speed=args.move_speed,
        rot_speed_deg=args.rot_speed_deg,
        hold_force=args.hold_force,
    )


if __name__ == "__main__":
    main()
