from __future__ import annotations

import argparse
import time
from pathlib import Path

import pybullet as p
import yaml

from src.envs.grasp_env import GraspEnv


def _load_sim_config_with_gui(sim_config_path: str, force_gui: bool) -> dict:
    with Path(sim_config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("world", {})
    if force_gui:
        cfg["world"]["gui"] = True
    return cfg


def run_reset_button_test(sim_config: str, hz: float, force_gui: bool) -> None:
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError("pygame is required for controller input. Install with: pip install pygame") from exc

    pygame.init()
    pygame.joystick.init()
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Controller connected: {joystick.get_name()}")
    else:
        print("No controller connected. Keyboard R still works.")

    cfg = _load_sim_config_with_gui(sim_config, force_gui=force_gui)
    env = GraspEnv(cfg)
    obs, info = env.reset()
    _ = (obs, info)
    reset_count = 1
    print("Reset test started.")
    print("Press keyboard R or controller RB to call reset().")
    print("Press ESC or Menu to quit.")
    print("Important: click into the PyBullet window so keyboard input is captured.")

    dt = 1.0 / float(hz)
    esc_key = 27

    try:
        while True:
            do_reset = False
            quit_req = False

            # Keyboard input from PyBullet window
            keys = p.getKeyboardEvents()
            r_pressed = False
            for key_code in (ord("r"), ord("R")):
                if key_code in keys and (keys[key_code] & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN)):
                    r_pressed = True
                    break
            if r_pressed:
                do_reset = True
            if esc_key in keys and (keys[esc_key] & p.KEY_WAS_TRIGGERED):
                quit_req = True

            # Controller input via pygame
            if joystick is not None:
                pygame.event.pump()
                rb_pressed = joystick.get_numbuttons() > 5 and bool(joystick.get_button(5))  # RB
                menu_pressed = joystick.get_numbuttons() > 7 and bool(joystick.get_button(7))  # Menu
                if rb_pressed:
                    do_reset = True
                if menu_pressed:
                    quit_req = True

            if quit_req:
                break

            if do_reset:
                obs, info = env.reset()
                _ = (obs, info)
                reset_count += 1
                print(f"reset_called={reset_count}")
                # Debounce to avoid multiple resets while button is held.
                time.sleep(0.2)
                continue

            # Keep simulation alive between resets.
            env.world.step(1)
            time.sleep(dt)
    finally:
        env.close()
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-config", default="configs/sim/tripod.yaml")
    parser.add_argument("--hz", type=float, default=120.0)
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI override (use config value)")
    args = parser.parse_args()
    run_reset_button_test(sim_config=args.sim_config, hz=args.hz, force_gui=not args.no_gui)


if __name__ == "__main__":
    main()

