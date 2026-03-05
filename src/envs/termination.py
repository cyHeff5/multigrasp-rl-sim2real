from __future__ import annotations


def check_termination(step_count: int, max_steps: int) -> tuple[bool, bool]:
    terminated = False
    truncated = step_count >= max_steps
    return terminated, truncated
