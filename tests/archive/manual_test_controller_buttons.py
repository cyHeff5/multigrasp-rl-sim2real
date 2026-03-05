from __future__ import annotations

import argparse
import time

import pygame


XBOX_BUTTON_NAMES = {
    0: "A",
    1: "B",
    2: "X",
    3: "Y",
    4: "LB",
    5: "RB",
    6: "View",
    7: "Menu",
    8: "LStick",
    9: "RStick",
}

XBOX_AXIS_NAMES = {
    0: "LeftStick_X",
    1: "LeftStick_Y",
    2: "RightStick_X",
    3: "RightStick_Y",
    4: "LT_or_Axis4",
    5: "RT_or_Axis5",
}


def button_name(button_id: int) -> str:
    return XBOX_BUTTON_NAMES.get(button_id, f"Button{button_id}")


def axis_name(axis_id: int) -> str:
    return XBOX_AXIS_NAMES.get(axis_id, f"Axis{axis_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Joystick index")
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() <= args.index:
        raise RuntimeError(f"No controller at index {args.index}. Connected: {pygame.joystick.get_count()}")

    js = pygame.joystick.Joystick(args.index)
    js.init()
    print(f"Controller: {js.get_name()}")
    print("Press buttons. Ctrl+C to stop.")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"DOWN  {button_name(event.button)} ({event.button})")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"UP    {button_name(event.button)} ({event.button})")
                elif event.type == pygame.JOYHATMOTION:
                    print(f"HAT   {event.value}")
                elif event.type == pygame.JOYAXISMOTION:
                    # Filter tiny noise around zero to reduce spam.
                    if abs(float(event.value)) >= 0.05:
                        print(f"AXIS  {axis_name(event.axis)} ({event.axis}) = {event.value:+.3f}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
