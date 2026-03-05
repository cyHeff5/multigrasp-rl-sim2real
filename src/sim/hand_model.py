from __future__ import annotations

import pybullet as p


class HandModel:
    """AR10 helper with normalized [0, 1] joint interface."""

    def __init__(self, hand_id: int):
        self.hand_id = int(hand_id)
        self.control_joint_names = [
            "servo0",
            "servo1",
            "servo2",
            "servo3",
            "servo4",
            "servo5",
            "servo6",
            "servo7",
            "servo8",
            "servo9",
        ]
        self.mimic_map = {
            "servo3": ["tip1"],
            "servo5": ["tip2"],
            "servo7": ["tip3"],
            "servo9": ["tip4"],
        }
        self.joint_name_to_index = {
            "servo0": 28,
            "servo1": 31,
            "servo2": 3,
            "servo3": 4,
            "servo4": 9,
            "servo5": 10,
            "servo6": 15,
            "servo7": 16,
            "servo8": 21,
            "servo9": 22,
            "tip1": 5,
            "tip2": 11,
            "tip3": 17,
            "tip4": 23,
            "thumb2joint": 29,
        }
        self.joint_limits = {}
        self._load_joint_limits(self.control_joint_names)
        for mimic_names in self.mimic_map.values():
            self._load_joint_limits(mimic_names)
        self._q_target = [0.0] * len(self.control_joint_names)

    def _load_joint_limits(self, joint_names: list[str]) -> None:
        for name in joint_names:
            if name in self.joint_limits:
                continue
            joint_index = self.joint_name_to_index[name]
            info = p.getJointInfo(self.hand_id, joint_index)
            lower, upper = float(info[8]), float(info[9])
            if lower >= upper:
                lower, upper = 0.0, 1.3
            self.joint_limits[name] = (lower, upper)

    def apply_joint_command_vector(self, vector: list[float], force: float = 10.0) -> None:
        if len(vector) != len(self.control_joint_names):
            raise ValueError(f"Expected {len(self.control_joint_names)} values, got {len(vector)}")

        joint_targets = {}
        for idx, name in enumerate(self.control_joint_names):
            norm = float(vector[idx])
            norm = max(0.0, min(1.0, norm))
            low, high = self.joint_limits[name]
            target = low + norm * (high - low)
            joint_targets[name] = target

            for mimic_name in self.mimic_map.get(name, []):
                m_low, m_high = self.joint_limits[mimic_name]
                m_target = m_low + norm * (m_high - m_low)
                joint_targets[mimic_name] = m_target

        for joint_name, target in joint_targets.items():
            p.setJointMotorControl2(
                bodyIndex=self.hand_id,
                jointIndex=self.joint_name_to_index[joint_name],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=float(force),
            )

    def send_q_target(self, q_target: list[float], force: float = 10.0) -> None:
        q = [max(0.0, min(1.0, float(v))) for v in q_target]
        self._q_target = q
        self.apply_joint_command_vector(q, force=force)

    def apply_delta_q_target(self, delta_q: list[float], max_delta: float = 0.05, force: float = 10.0) -> list[float]:
        clipped_delta = [max(-max_delta, min(float(v), max_delta)) for v in delta_q]
        next_q = [max(0.0, min(1.0, q + dq)) for q, dq in zip(self._q_target, clipped_delta)]
        self.send_q_target(next_q, force=force)
        return next_q

    def get_q_measured(self) -> list[float]:
        measured = []
        for name in self.control_joint_names:
            joint_index = self.joint_name_to_index[name]
            pos = float(p.getJointState(self.hand_id, joint_index)[0])
            low, high = self.joint_limits[name]
            norm = (pos - low) / (high - low)
            measured.append(max(0.0, min(1.0, norm)))
        return measured

    def get_q_target(self) -> list[float]:
        # Normalized target command in [0, 1] per controlled joint.
        return [float(v) for v in self._q_target]

    def get_tip_link_indices(self) -> list[int]:
        return [self.joint_name_to_index[name] for name in ["tip1", "tip2", "tip3", "tip4"]]

    def get_contact_link_indices(
        self,
        mode: str = "tips",
        link_names: list[str] | None = None,
    ) -> list[int]:
        """Return link indices used for grasp-contact checks."""
        if link_names:
            indices = []
            for name in link_names:
                if name not in self.joint_name_to_index:
                    raise ValueError(f"Unknown hand link/joint name for contact check: {name}")
                idx = int(self.joint_name_to_index[name])
                if idx not in indices:
                    indices.append(idx)
            return indices

        mode = str(mode).lower()
        if mode == "tips":
            names = ["tip1", "tip2", "tip3", "tip4"]
        elif mode in ("tips_plus_distal", "tips_plus_under"):
            # Distal phalanx joints directly below the fingertip links.
            names = ["tip1", "tip2", "tip3", "tip4", "servo3", "servo5", "servo7", "servo9"]
        elif mode == "tips_plus_middle":
            names = ["tip1", "tip2", "tip3", "tip4", "servo2", "servo4", "servo6", "servo8"]
        else:
            raise ValueError(f"Unsupported contact link mode: {mode}")
        return [int(self.joint_name_to_index[n]) for n in names]

    def reset_open_pose(self, force: float = 10.0) -> None:
        self.send_q_target([0.0] * len(self.control_joint_names), force=force)
