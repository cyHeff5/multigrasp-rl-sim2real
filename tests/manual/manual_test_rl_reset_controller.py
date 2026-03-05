from __future__ import annotations

import argparse
import time
from pathlib import Path

import pybullet as p
import yaml

from src.envs.grasp_env import GraspEnv


def _load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_trigger_values(js) -> tuple[float, float]:
    """Return (lt, rt) in [0, 1] for common gamepad axis layouts."""
    n_axes = int(js.get_numaxes())
    if n_axes >= 6:
        # Typical Xbox layout on Windows via pygame:
        # LT axis 4, RT axis 5, both in [-1, 1]
        lt = 0.5 * (float(js.get_axis(4)) + 1.0)
        rt = 0.5 * (float(js.get_axis(5)) + 1.0)
        return max(0.0, min(1.0, lt)), max(0.0, min(1.0, rt))
    if n_axes >= 3:
        # Fallback combined trigger axis (negative=LT, positive=RT).
        a = float(js.get_axis(2))
        lt = max(0.0, -a)
        rt = max(0.0, a)
        return max(0.0, min(1.0, lt)), max(0.0, min(1.0, rt))
    return 0.0, 0.0


def run(sim_config: str, hz: float, dz_speed_m_s: float, grip_speed_per_s: float) -> None:
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError("pygame is required. Install with: pip install pygame") from exc

    sim_cfg = _load_yaml(sim_config)
    sim_cfg.setdefault("world", {})
    sim_cfg["world"]["gui"] = True

    env = GraspEnv(sim_cfg)

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        env.close()
        raise RuntimeError("No gamepad found. Connect controller and retry.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"controller={js.get_name()}")

    obs, info = env.reset()
    print(f"reset_done shape={info.get('shape')} size_cm={info.get('size_cm')}")

    dt = 1.0 / max(1.0, float(hz))
    x_latch = False

    print("Controls:")
    print("  A: move hand up")
    print("  B: move hand down")
    print("  RT: close hand")
    print("  LT: open hand")
    print("  X: call RL reset()")
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
                print(f"reset_done shape={info.get('shape')} size_cm={info.get('size_cm')}")

            x_latch = x_btn

            dz = 0.0
            if a_btn:
                dz += float(dz_speed_m_s) * dt
            if b_btn:
                dz -= float(dz_speed_m_s) * dt

            if abs(dz) > 1e-12:
                if env.world.robot_type == "free_hand":
                    hand_pos, hand_quat = p.getBasePositionAndOrientation(env.world.hand_id)
                    new_pos = [float(hand_pos[0]), float(hand_pos[1]), float(hand_pos[2] + dz)]
                    env.world.set_free_hand_pose(new_pos, [float(v) for v in hand_quat])
                else:
                    env.world.lift_grasping_hand_blocking(dz=dz, seconds=dt, hz=max(1, int(hz)))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-config", default="configs/sim/tripod.yaml")
    parser.add_argument("--hz", type=float, default=120.0)
    parser.add_argument("--dz-speed-m-s", type=float, default=0.10, help="Vertical hand speed when holding A/B")
    parser.add_argument("--grip-speed-per-s", type=float, default=0.8, help="Hand open/close speed from LT/RT")
    args = parser.parse_args()
    run(
        sim_config=args.sim_config,
        hz=args.hz,
        dz_speed_m_s=args.dz_speed_m_s,
        grip_speed_per_s=args.grip_speed_per_s,
    )


if __name__ == "__main__":
    main()
