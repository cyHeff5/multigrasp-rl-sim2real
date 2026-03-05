from __future__ import annotations

from typing import Any

import gymnasium as gym


class GraspEnvBase(gym.Env):
    """Base env contract: obs=|q_target-q_measured|, action=delta_q_target."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[list[float], dict]:
        super().reset(seed=seed)
        # Subclasses should call this for seeding, then return their own (obs, info).
        return [], {}

    def step(self, action: list[float]) -> tuple[list[float], float, bool, bool, dict]:
        raise NotImplementedError
