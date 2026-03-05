import random


class ObjectSampler:
    def __init__(self, object_cfg: dict, spawn_cfg: dict | None = None):
        self.object_cfg = object_cfg or {}
        self.spawn_cfg = spawn_cfg or {}

    def _sample_spawn_position(self) -> list[float]:
        base = self.spawn_cfg.get("position_xyz", [0.6, 0.0, 0.04])
        jitter = self.spawn_cfg.get("jitter_xyz", [0.0, 0.0, 0.0])
        return [
            float(base[0]) + random.uniform(-float(jitter[0]), float(jitter[0])),
            float(base[1]) + random.uniform(-float(jitter[1]), float(jitter[1])),
            float(base[2]) + random.uniform(-float(jitter[2]), float(jitter[2])),
        ]

    @staticmethod
    def _sample_range(cfg, default_min: float, default_max: float) -> float:
        # Accept either a fixed scalar or a {min,max} dict.
        if isinstance(cfg, (int, float)):
            return float(cfg)
        if isinstance(cfg, dict):
            low = float(cfg.get("min", default_min))
            high = float(cfg.get("max", default_max))
            if high < low:
                low, high = high, low
            return random.uniform(low, high)
        return random.uniform(float(default_min), float(default_max))

    def sample(self) -> dict:
        shapes = self.object_cfg.get("shapes", ["sphere"])
        size_cfg = self.object_cfg.get("size_cm", {"min": 2.0, "max": 5.0})
        thickness_cfg = self.object_cfg.get("thickness_cm", size_cfg)
        height_cfg = self.object_cfg.get("height_cm", size_cfg)
        mass_cfg = self.object_cfg.get("mass_kg", {"min": 0.03, "max": 0.15})
        friction_cfg = self.object_cfg.get("lateral_friction", {"min": 0.25, "max": 0.7})

        yaw_range = self.spawn_cfg.get("yaw_range_deg", [-180.0, 180.0])
        yaw_deg = random.uniform(float(yaw_range[0]), float(yaw_range[1]))

        return {
            "shape": random.choice(shapes),
            "size_cm": self._sample_range(size_cfg, 2.0, 5.0),
            "thickness_cm": self._sample_range(thickness_cfg, 2.0, 5.0),
            "height_cm": self._sample_range(height_cfg, 2.0, 5.0),
            "mass_kg": self._sample_range(mass_cfg, 0.03, 0.15),
            "lateral_friction": self._sample_range(friction_cfg, 0.25, 0.7),
            "position_xyz": self._sample_spawn_position(),
            "rpy": [0.0, 0.0, yaw_deg * 3.141592653589793 / 180.0],
        }
