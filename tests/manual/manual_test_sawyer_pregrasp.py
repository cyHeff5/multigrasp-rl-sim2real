"""
Manual test: Sawyer arm moves AR10 hand to pregrasp pose for a random object.

The arm automatically moves to the correct pregrasp after each reset.
This is the basis for offline IK calibration: the joint angles printed after
each reset can be stored and replayed on the real Sawyer without IK.

Controls (gamepad):
  X: reset (new random object + move arm to pregrasp)
  Y: find alternative IK solution for current object
  Menu/Start: quit

Controls (keyboard, no gamepad):
  R: reset
  Q / Escape: quit
"""
from __future__ import annotations

import argparse
import time

import pybullet as p

from src.envs.grasp_env import GraspEnv


_SIM_CONFIG = {
    "grasp_type": "tripod",
    "world": {
        "gui": True,
        "robot_type": "sawyer",
        # Real setup: Sawyer base is on the floor, table is 70 cm high.
        # z=0 in sim = table surface. Sawyer base goes at z=-0.70.
        "robot_base_position_xyz": [0.0, 0.0, -0.70],
        "robot_base_rpy_deg": [0.0, 0.0, 270.0],
        "spawn_on_pedestal": True,
        "pedestal_height_m": 0.04,
        "pedestal_shape": "cylinder",
        "pedestal_diameter_m": 0.04,
        "pedestal_position_xy": [0.80, 0.00],
    },
    "pregrasp_hand_reference_yaml": "artifacts/hand_reference_points.yaml",
    "pregrasp_distance_mm": 40.0,
    "pregrasp_twist_deg": 270.0,
    "pregrasp_settle_steps": 0,
    "pregrasp_no_collision_steps": 150,
    "action_scale": 0.05,
    "action_mode": "close_only",
    "frame_skip": 4,
    "lift_success_z": 0.03,
    "lift_check_enabled": False,
    "overgrip_threshold": 0.90,
    "overgrip_penalty": 0.20,
    "max_tilt_rad": 0.8,
    "object_sampler": {
        "shapes": ["sphere", "cube", "cylinder"],
        "size_cm": {"min": 2.0, "max": 5.0},
        "mass_kg": {"min": 0.10, "max": 0.20},
        "lateral_friction": 0.4,
    },
    "spawn": {
        "position_xyz": [0.80, 0.00, 0.04],
        "jitter_xyz": [0.001, 0.001, 0.0],
        "yaw_range_deg": [-180.0, 180.0],
    },
}


def _get_pregrasp_pos_rpy(env: GraspEnv) -> tuple[list, list] | None:
    pregrasp = env.last_pregrasp
    if not pregrasp or "hand_base_world" not in pregrasp or "hand_quat_world" not in pregrasp:
        print(f"[warn] pregrasp data missing: {pregrasp}")
        return None
    pos = pregrasp["hand_base_world"]
    rpy = list(p.getEulerFromQuaternion(pregrasp["hand_quat_world"]))
    return pos, rpy


def _move_to_pregrasp(env: GraspEnv, move_seconds: float) -> None:
    result = _get_pregrasp_pos_rpy(env)
    if result is None:
        return
    pos, rpy = result
    print(f"[pregrasp] moving to pos={[f'{v:.3f}' for v in pos]}")
    env.world.arm.move_to_pose_blocking(pos, rpy, seconds=move_seconds, hz=240)
    env.world.sync_arm_hold_target_to_current()
    joints = [p.getJointState(env.world.robot_id, j)[0] for j in env.world.arm.joints]
    print(f"[pregrasp] reached  joints={[f'{v:.4f}' for v in joints]}")


def _find_and_move_alternative_ik(env: GraspEnv, move_seconds: float) -> None:
    result = _get_pregrasp_pos_rpy(env)
    if result is None:
        return
    pos, rpy = result
    print("[alt_ik] searching for alternative solution...")
    q = env.world.arm.find_random_ik(pos, rpy, n_attempts=50)
    if q is None:
        print("[alt_ik] no alternative solution found")
        return
    import numpy as np
    p.setJointMotorControlArray(
        bodyUniqueId=env.world.robot_id,
        jointIndices=env.world.arm.joints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=q.tolist(),
        positionGains=[0.25] * len(env.world.arm.joints),
        forces=[200.0] * len(env.world.arm.joints),
    )
    steps = max(1, int(move_seconds * 240))
    for _ in range(steps):
        p.stepSimulation()
    env.world.sync_arm_hold_target_to_current()
    print(f"[alt_ik] reached  joints={[f'{v:.4f}' for v in q.tolist()]}")


def _poll_gamepad(js) -> tuple[bool, bool, bool]:
    """Return (do_reset, do_alt_ik, do_quit) from gamepad state."""
    x_btn = bool(js.get_numbuttons() > 2 and js.get_button(2))
    y_btn = bool(js.get_numbuttons() > 3 and js.get_button(3))
    menu_btn = bool(js.get_numbuttons() > 7 and js.get_button(7))
    return x_btn, y_btn, menu_btn


def _poll_keyboard(pygame) -> tuple[bool, bool, bool]:
    """Return (do_reset, do_alt_ik, do_quit) from keyboard events."""
    do_reset = do_alt_ik = do_quit = False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                do_reset = True
            elif event.key == pygame.K_i:
                do_alt_ik = True
            elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                do_quit = True
        elif event.type == pygame.QUIT:
            do_quit = True
    return do_reset, do_alt_ik, do_quit


def _disable_object_collisions(env: GraspEnv) -> None:
    """Permanently disable hand and arm collisions with the current object."""
    if env.world.object_id is None:
        return
    env.world.set_hand_object_collision(env.world.object_id, False)
    env.world.set_arm_object_collision(env.world.object_id, False)


def run(cfg: dict, hz: float, move_seconds: float, part_id: int | None = None) -> None:
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError("pygame is required. Install with: pip install pygame") from exc

    env = GraspEnv(cfg)

    pygame.init()
    pygame.joystick.init()

    js = None
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"controller={js.get_name()}")
        print("Controls:  X=reset  Y=alt IK  Menu/Start=quit")
    else:
        pygame.display.set_mode((300, 100))
        pygame.display.set_caption("Sawyer Pregrasp Test")
        print("No gamepad found — using keyboard.")
        print("Controls:  R=reset  I=alt IK  Q/Escape=quit")

    if part_id is not None:
        obs, info = env.reset_benchmark(part_id)
        print(f"reset_done benchmark part_id={info.get('part_id')}")
    else:
        obs, info = env.reset()
        print(f"reset_done shape={info.get('shape')} size_cm={info.get('size_cm', 0.0):.2f}")
    _disable_object_collisions(env)
    _move_to_pregrasp(env, move_seconds)

    dt = 1.0 / max(1.0, float(hz))
    reset_latch = False
    alt_ik_latch = False

    try:
        while p.isConnected(env.world.client_id):
            if js is not None:
                pygame.event.pump()
                do_reset, do_alt_ik, do_quit = _poll_gamepad(js)
            else:
                do_reset, do_alt_ik, do_quit = _poll_keyboard(pygame)

            if do_quit:
                break

            if do_reset and not reset_latch:
                if part_id is not None:
                    obs, info = env.reset_benchmark(part_id)
                    print(f"reset_done benchmark part_id={info.get('part_id')}")
                else:
                    obs, info = env.reset()
                    print(f"reset_done shape={info.get('shape')} size_cm={info.get('size_cm', 0.0):.2f}")
                _disable_object_collisions(env)
                _move_to_pregrasp(env, move_seconds)

            if do_alt_ik and not alt_ik_latch:
                _find_and_move_alternative_ik(env, move_seconds)

            reset_latch = do_reset
            alt_ik_latch = do_alt_ik

            env.world.step(1)
            time.sleep(dt)
    finally:
        try:
            pygame.quit()
        except Exception:
            pass
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sawyer pregrasp test: arm moves to pregrasp for random object.")
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--move-seconds", type=float, default=2.0, help="Duration of arm movement to pregrasp")
    parser.add_argument("--grasp-type", type=str, default=None, help="Override grasp type (e.g. tripod, medium_wrap)")
    parser.add_argument("--part-id", type=int, default=None, help="Benchmark part id (1..14). If set, spawns this part instead of a random object.")
    args = parser.parse_args()

    cfg = dict(_SIM_CONFIG)
    if args.grasp_type is not None:
        cfg["grasp_type"] = args.grasp_type

    run(cfg=cfg, hz=args.hz, move_seconds=args.move_seconds, part_id=args.part_id)


if __name__ == "__main__":
    main()
