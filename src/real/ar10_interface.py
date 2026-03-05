"""Real AR10 hand IO adapter.

q values throughout this module are normalized to [0, 1]:
  0.0 = fully open
  1.0 = fully closed

Internally, values are mapped to Pololu Maestro servo targets (units of 0.25 µs).

Channel and joint mapping (10 joints, channels 10-19):
  joint 0  thumb base   channel 10
  joint 1  thumb curl   channel 11
  joint 2  little base  channel 12
  joint 3  little curl  channel 13
  joint 4  ring base    channel 14
  joint 5  ring curl    channel 15
  joint 6  middle base  channel 16
  joint 7  middle curl  channel 17
  joint 8  index base   channel 18
  joint 9  index curl   channel 19
"""

import time
from typing import List, Optional

try:
    import serial
except ImportError:
    serial = None  # type: ignore[assignment]


_CHANNELS = list(range(10, 20))
_DEFAULT_SERVO_MIN = [4200] * 10  # fully open
_DEFAULT_SERVO_MAX = [7700] * 10  # fully closed


class AR10Interface:
    """
    Adapter for the real AR10 hand.

    Usage (hardware connected):
        iface = AR10Interface(com_port="COM3")
        iface.send_q_target([0.0] * 10)       # open hand
        q = iface.read_q_measured()            # read actual positions [0..1]
        err = iface.position_error_norm()      # scalar error in [0, 1]
        iface.close()

    Usage (no hardware / testing):
        iface = AR10Interface()                # com_port=None → mock mode
        q = iface.read_q_measured()            # returns last q_target
    """

    def __init__(
        self,
        com_port: Optional[str] = None,
        servo_min: Optional[List[int]] = None,
        servo_max: Optional[List[int]] = None,
        speed: int = 100,
        acceleration: int = 0,
    ):
        """
        Args:
            com_port:     Serial port of the Pololu Maestro (e.g. "COM3", "/dev/ttyACM0").
                          If None, the interface runs in mock mode (no hardware required).
            servo_min:    Per-joint minimum servo target (fully open). Defaults to 4200 for all.
            servo_max:    Per-joint maximum servo target (fully closed). Defaults to 7700 for all.
            speed:        Maestro channel speed (0 = unlimited). Must be < max for
                          read_q_measured() to return actual positions.
            acceleration: Maestro channel acceleration (0 = unlimited).
        """
        self._servo_min = list(servo_min) if servo_min is not None else list(_DEFAULT_SERVO_MIN)
        self._servo_max = list(servo_max) if servo_max is not None else list(_DEFAULT_SERVO_MAX)
        self._q_target: List[float] = [0.0] * 10
        self._usb: Optional[serial.Serial] = None

        if com_port is not None:
            if serial is None:
                raise ImportError("pyserial is required for hardware mode: pip install pyserial")
            self._usb = serial.Serial(com_port, baudrate=9600)
            for ch in _CHANNELS:
                self._set_channel_speed(ch, speed)
                time.sleep(0.05)
                self._set_channel_acceleration(ch, acceleration)
                time.sleep(0.05)

    # ── Pololu Maestro low-level protocol ─────────────────────────────────────

    def _send_command(self, *args: str) -> None:
        if self._usb is None:
            return
        msg = chr(0xAA) + chr(0x0C) + "".join(args)
        self._usb.write(msg.encode())

    def _set_channel_speed(self, channel: int, speed: int) -> None:
        lsb = speed & 0x7F
        msb = (speed >> 7) & 0x7F
        self._send_command(chr(0x07), chr(channel), chr(lsb), chr(msb))

    def _set_channel_acceleration(self, channel: int, accel: int) -> None:
        accel = max(0, min(255, accel))
        lsb = accel & 0x7F
        msb = (accel >> 7) & 0x7F
        self._send_command(chr(0x09), chr(channel), chr(lsb), chr(msb))

    def _set_all_channel_targets(self, targets: List[int]) -> None:
        """Send all 10 servo targets in a single Maestro command."""
        args = [chr(0x1F), chr(10), chr(10)]
        for i, t in enumerate(targets):
            t = max(self._servo_min[i], min(self._servo_max[i], t))
            args.append(chr(t & 0x7F))
            args.append(chr((t >> 7) & 0x7F))
        self._send_command(*args)

    def _read_channel_target(self, channel: int) -> int:
        """Read actual servo position from Maestro.
        Returns the actual position when speed < max, otherwise the commanded target.
        """
        if self._usb is None:
            return 0
        self._send_command(chr(0x10), chr(channel))
        lsb = ord(self._usb.read())
        msb = ord(self._usb.read())
        return (msb << 8) + lsb

    # ── normalization helpers ─────────────────────────────────────────────────

    def _to_servo(self, q_norm: float, joint_idx: int) -> int:
        lo = self._servo_min[joint_idx]
        hi = self._servo_max[joint_idx]
        return int(round(lo + max(0.0, min(1.0, q_norm)) * (hi - lo)))

    def _to_norm(self, servo_val: int, joint_idx: int) -> float:
        lo = self._servo_min[joint_idx]
        hi = self._servo_max[joint_idx]
        if hi == lo:
            return 0.0
        return max(0.0, min(1.0, (servo_val - lo) / (hi - lo)))

    # ── public interface ──────────────────────────────────────────────────────

    def send_q_target(self, q_target: List[float]) -> None:
        """Send 10 normalized [0, 1] joint targets to the real hand.

        Args:
            q_target: List of 10 values in [0, 1]. 0 = open, 1 = closed.
        """
        if len(q_target) != 10:
            raise ValueError(f"q_target must have 10 values, got {len(q_target)}.")
        self._q_target = [max(0.0, min(1.0, v)) for v in q_target]
        servo_targets = [self._to_servo(v, i) for i, v in enumerate(self._q_target)]
        self._set_all_channel_targets(servo_targets)

    def read_q_measured(self) -> List[float]:
        """Read actual joint positions from hardware, normalized to [0, 1].

        Reads the actual servo positions via the Maestro 'Get Position' command.
        Requires channel speed < max (non-zero). The default speed=100 satisfies this.

        In mock mode (com_port=None), returns the last commanded q_target.

        Returns:
            List of 10 floats in [0, 1].
        """
        if self._usb is None:
            return list(self._q_target)
        return [
            self._to_norm(self._read_channel_target(ch), i)
            for i, ch in enumerate(_CHANNELS)
        ]

    def position_error_norm(self) -> float:
        """Mean absolute error between q_target and q_measured, normalized to [0, 1].

        0.0 = hand has perfectly reached the commanded target.
        1.0 = maximum possible error (all joints at opposite extreme).

        Since both q_target and q_measured are in [0, 1], the per-joint absolute
        error is naturally bounded by [0, 1], and so is the mean.

        Returns:
            Scalar in [0, 1].
        """
        measured = self.read_q_measured()
        errors = [abs(t - m) for t, m in zip(self._q_target, measured)]
        return sum(errors) / len(errors)

    def close(self) -> None:
        """Close the serial connection to the Maestro."""
        if self._usb is not None:
            self._usb.close()
            self._usb = None
