from __future__ import annotations

import argparse
import math
import time

import pybullet as p
import pybullet_data
import pygame

from src.sim.assets import AR10_URDF, UR5_URDF
from src.sim.hand_model import HandModel
from src.sim.mounting import apply_hand_friction, mount_hand_to_arm
from src.sim.ur5_arm import UR5Helper


def _home_ur5(robot_id: int, helper: UR5Helper) -> None:
    # Neutral-ish UR5 pose for visual checks.
    home_q = [0.0, -1.2, 1.8, -1.6, -1.57, 0.0]
    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=helper.joints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=home_q,
        positionGains=[0.2] * len(helper.joints),
        forces=[400.0] * len(helper.joints),
    )
    for _ in range(360):
        p.stepSimulation()


def run(hz: float, demo_motion: bool, pos_speed: float, deadzone: float, rot_speed_deg: float) -> None:
    pygame.init()
    pygame.joystick.init()
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(
        cameraDistance=1.35,
        cameraYaw=45.0,
        cameraPitch=-28.0,
        cameraTargetPosition=[0.0, 0.0, 0.25],
    )

    robot_id = p.loadURDF(str(UR5_URDF), basePosition=[0.0, 0.0, 0.0], useFixedBase=True)
    hand_id = p.loadURDF(str(AR10_URDF), basePosition=[0.5, 0.1, 0.3], useFixedBase=False)
    mount_cid = mount_hand_to_arm(robot_id=robot_id, hand_id=hand_id, ee_link=9)
    apply_hand_friction(hand_id)
    hand = HandModel(hand_id)
    hand.reset_open_pose()
    hand_scalar = 0.0

    helper = UR5Helper(robot_id, ee_link=9)
    _home_ur5(robot_id, helper)
    target_xyz, target_rpy = helper.ee_pose()
    target_xyz = target_xyz.copy()
    target_rpy = target_rpy.copy()
    target_q = [p.getJointState(robot_id, j)[0] for j in helper.joints]

    ee_xyz, ee_rpy = helper.ee_pose()
    print(f"mounted_constraint_id={mount_cid}")
    print(f"ur5_joints={helper.joints}")
    print(f"ee_xyz={[round(float(v), 4) for v in ee_xyz]}")
    print(f"ee_rpy_deg={[round(math.degrees(float(v)), 2) for v in ee_rpy]}")
    print("Manual mount test running.")
    print("Controls:")
    print("  Keyboard O/C: open/close AR10 hand")
    print("  Gamepad left stick: move UR5 end-effector in XY")
    print("  Gamepad A/B: move UR5 end-effector +Z/-Z")
    print("  Gamepad right stick: pitch/yaw of end-effector")
    print("  Gamepad LT/RT: roll of end-effector")
    print("  Gamepad X/Y: close/open hand")
    print("  Gamepad Menu: quit")
    print("  ESC: quit")
    if demo_motion:
        print("  demo_motion=on (small UR5 oscillation)")

    dt = 1.0 / float(hz)
    rot_speed = math.radians(float(rot_speed_deg))
    t = 0.0
    esc_key = 27
    try:
        while p.isConnected(client_id):
            if joystick is not None:
                pygame.event.pump()
            keys = p.getKeyboardEvents()
            if esc_key in keys and (keys[esc_key] & p.KEY_WAS_TRIGGERED):
                break
            if ord("O") in keys and (keys[ord("O")] & p.KEY_WAS_TRIGGERED):
                hand.send_q_target([0.0] * 10)
                hand_scalar = 0.0
            if ord("C") in keys and (keys[ord("C")] & p.KEY_WAS_TRIGGERED):
                hand.send_q_target([1.0] * 10)
                hand_scalar = 1.0

            if joystick is not None:
                left_x = float(joystick.get_axis(0)) if joystick.get_numaxes() > 0 else 0.0
                left_y = float(joystick.get_axis(1)) if joystick.get_numaxes() > 1 else 0.0
                right_x = float(joystick.get_axis(2)) if joystick.get_numaxes() > 2 else 0.0
                right_y = float(joystick.get_axis(3)) if joystick.get_numaxes() > 3 else 0.0
                lt_raw = float(joystick.get_axis(4)) if joystick.get_numaxes() > 4 else -1.0
                rt_raw = float(joystick.get_axis(5)) if joystick.get_numaxes() > 5 else -1.0
                lt = max(0.0, min(1.0, (lt_raw + 1.0) * 0.5))
                rt = max(0.0, min(1.0, (rt_raw + 1.0) * 0.5))
                if abs(left_x) < float(deadzone):
                    left_x = 0.0
                if abs(left_y) < float(deadzone):
                    left_y = 0.0
                if abs(right_x) < float(deadzone):
                    right_x = 0.0
                if abs(right_y) < float(deadzone):
                    right_y = 0.0

                target_xyz[0] += (-left_y) * float(pos_speed) * dt
                target_xyz[1] += (+left_x) * float(pos_speed) * dt
                # Orientation control of mounted hand via UR5 IK target orientation.
                target_rpy[1] += (-right_y) * rot_speed * dt  # pitch
                target_rpy[2] += (-right_x) * rot_speed * dt  # yaw
                target_rpy[0] += (rt - lt) * rot_speed * dt   # roll

                a_btn = bool(joystick.get_numbuttons() > 0 and joystick.get_button(0))
                b_btn = bool(joystick.get_numbuttons() > 1 and joystick.get_button(1))
                x_btn = bool(joystick.get_numbuttons() > 2 and joystick.get_button(2))
                y_btn = bool(joystick.get_numbuttons() > 3 and joystick.get_button(3))
                menu_btn = bool(joystick.get_numbuttons() > 7 and joystick.get_button(7))
                if a_btn:
                    target_xyz[2] += float(pos_speed) * dt
                if b_btn:
                    target_xyz[2] -= float(pos_speed) * dt
                if x_btn:
                    hand_scalar = 1.0
                    hand.send_q_target([1.0] * 10)
                if y_btn:
                    hand_scalar = 0.0
                    hand.send_q_target([0.0] * 10)
                if menu_btn:
                    break

                if not demo_motion:
                    q = helper._ik(target_xyz, target_rpy)
                    target_q = q.tolist()
                    p.setJointMotorControlArray(
                        bodyUniqueId=robot_id,
                        jointIndices=helper.joints,
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=target_q,
                        positionGains=[0.2] * len(helper.joints),
                        forces=[400.0] * len(helper.joints),
                    )

            if demo_motion:
                # Small periodic elbow/wrist motion to verify fixed mount.
                q = [p.getJointState(robot_id, j)[0] for j in helper.joints]
                q[2] = 1.8 + 0.15 * math.sin(0.8 * t)
                q[4] = -1.57 + 0.20 * math.sin(1.1 * t)
                p.setJointMotorControlArray(
                    bodyUniqueId=robot_id,
                    jointIndices=helper.joints,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=q,
                    positionGains=[0.2] * len(helper.joints),
                    forces=[400.0] * len(helper.joints),
                )
                t += dt

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
    parser.add_argument("--hz", type=float, default=120.0)
    parser.add_argument("--demo-motion", action="store_true", help="Apply a small UR5 oscillation to verify mount")
    parser.add_argument("--pos-speed", type=float, default=0.20, help="UR5 EE translation speed in m/s")
    parser.add_argument("--deadzone", type=float, default=0.20, help="Gamepad axis deadzone")
    parser.add_argument("--rot-speed-deg", type=float, default=90.0, help="EE orientation speed in deg/s")
    args = parser.parse_args()
    run(
        hz=args.hz,
        demo_motion=args.demo_motion,
        pos_speed=args.pos_speed,
        deadzone=args.deadzone,
        rot_speed_deg=args.rot_speed_deg,
    )


if __name__ == "__main__":
    main()
