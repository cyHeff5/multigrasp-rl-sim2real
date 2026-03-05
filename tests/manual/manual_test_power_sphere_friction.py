"""
Manual test: power_sphere grasp with a large sphere object.

Controls:
  Left stick: move hand in X/Y
  A: move hand up
  B: move hand down
  RT: close hand
  LT: open hand
  X: call env.reset()
  Menu/Start: quit
"""
from __future__ import annotations

import argparse
import time

import pybullet as p

from src.envs.grasp_env import GraspEnv


_SIM_CONFIG = {
    "grasp_type": "power_sphere",
    "world": {
        "gui": True,
        "robot_type": "free_hand",
        "free_hand_pregrasp_position_xyz": [0.40, 0.00, 0.12],
        "free_hand_pregrasp_rpy_deg": [90.0, 0.0, 90.0],
        "free_hand_constraint_force": 3000.0,
        "spawn_on_pedestal": True,
        "pedestal_height_m": 0.04,
        "pedestal_shape": "cylinder",
        "pedestal_diameter_m": 0.04,
        "pedestal_position_xy": [0.40, 0.00],
    },
    "pregrasp_hand_reference_yaml": "artifacts/hand_reference_points.yaml",
    "pregrasp_distance_mm": 10.0,
    "pregrasp_twist_deg": 0.0,
    "pregrasp_settle_steps": 30,
    "pregrasp_no_collision_steps": 150,
    "action_scale": 0.05,
    "action_mode": "close_only",
    "frame_skip": 4,
    "lift_success_z": 0.03,
    "lift_check_enabled": True,
    "lift_check_dz": 0.06,
    "lift_check_hold_steps": 30,
    "lift_check_debug": True,
    "lift_check_contact_links_min": 2,
    "overgrip_threshold": 0.90,
    "overgrip_penalty": 0.20,
    "max_tilt_rad": 0.8,
    # Always spawn a large sphere
    "object_sampler": {
        "shapes": ["sphere"],
        "size_cm": 7.0,
        "mass_kg": 0.15,
        "lateral_friction": 0.6,
    },
    "spawn": {
        "position_xyz": [0.40, 0.00, 0.04],
        "jitter_xyz": [0.0, 0.0, 0.0],
        "yaw_range_deg": [0.0, 0.0],
    },
}


def _read_trigger_values(js) -> tuple[float, float]:
    n_axes = int(js.get_numaxes())
    if n_axes >= 6:
        lt = 0.5 * (float(js.get_axis(4)) + 1.0)
        rt = 0.5 * (float(js.get_axis(5)) + 1.0)
        return max(0.0, min(1.0, lt)), max(0.0, min(1.0, rt))
    if n_axes >= 3:
        a = float(js.get_axis(2))
        return max(0.0, min(1.0, -a)), max(0.0, min(1.0, a))
    return 0.0, 0.0


_STICK_DEADZONE = 0.1


def _read_left_stick(js) -> tuple[float, float]:
    """Return (sx, sy) from left stick in [-1, 1] with deadzone applied."""
    if js.get_numaxes() < 2:
        return 0.0, 0.0
    sx = float(js.get_axis(0))
    sy = float(js.get_axis(1))
    sx = sx if abs(sx) > _STICK_DEADZONE else 0.0
    sy = sy if abs(sy) > _STICK_DEADZONE else 0.0
    return sx, sy


def run(hz: float, dz_speed_m_s: float, grip_speed_per_s: float) -> None:
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError("pygame is required. Install with: pip install pygame") from exc

    env = GraspEnv(_SIM_CONFIG)

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        env.close()
        raise RuntimeError("No gamepad found. Connect controller and retry.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"controller={js.get_name()}")

    obs, info = env.reset()
    print(f"reset_done shape={info.get('shape')} size_cm={info.get('size_cm'):.2f}")

    dt = 1.0 / max(1.0, float(hz))
    x_latch = False

    print("Controls:")
    print("  Left stick: move hand in X/Y")
    print("  A: move hand up")
    print("  B: move hand down")
    print("  RT: close hand")
    print("  LT: open hand")
    print("  X: call env.reset()")
    print("  Menu/Start: quit")

    try:
        while p.isConnected(env.world.client_id):
            pygame.event.pump()

            a_btn = bool(js.get_numbuttons() > 0 and js.get_button(0))
            b_btn = bool(js.get_numbuttons() > 1 and js.get_button(1))
            x_btn = bool(js.get_numbuttons() > 2 and js.get_button(2))
            menu_btn = bool(js.get_numbuttons() > 7 and js.get_button(7))

            if menu_btn:
                break

            if x_btn and not x_latch:
                obs, info = env.reset()
                print(f"reset_done shape={info.get('shape')} size_cm={info.get('size_cm'):.2f}")

            x_latch = x_btn

            dz = 0.0
            if a_btn:
                dz += float(dz_speed_m_s) * dt
            if b_btn:
                dz -= float(dz_speed_m_s) * dt

            sx, sy = _read_left_stick(js)
            dx = sx * float(dz_speed_m_s) * dt
            dy = -sy * float(dz_speed_m_s) * dt  # stick up (negative) → positive world Y

            if abs(dx) > 1e-12 or abs(dy) > 1e-12 or abs(dz) > 1e-12:
                env.world.lift_grasping_hand_blocking(
                    dz=dz, dx=dx, dy=dy, seconds=dt, hz=max(1, int(hz))
                )

            lt, rt = _read_trigger_values(js)
            grip_delta = float(grip_speed_per_s) * dt * (rt - lt)
            if abs(grip_delta) > 1e-9:
                env.world.hand.apply_delta_q_target(
                    [float(grip_delta)] * 10,
                    max_delta=max(0.001, abs(float(grip_delta))),
                    force=20.0,
                )

            env.world.step(1)
            time.sleep(dt)
    finally:
        try:
            pygame.quit()
        except Exception:
            pass
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual friction test: large sphere + power_sphere grasp.")
    parser.add_argument("--hz", type=float, default=120.0)
    parser.add_argument("--dz-speed-m-s", type=float, default=0.10)
    parser.add_argument("--grip-speed-per-s", type=float, default=0.8)
    args = parser.parse_args()
    run(hz=args.hz, dz_speed_m_s=args.dz_speed_m_s, grip_speed_per_s=args.grip_speed_per_s)


if __name__ == "__main__":
    main()
