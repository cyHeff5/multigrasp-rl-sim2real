# Env wrapper: reset(), evaluate(q10), reward.
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pybullet as p

from sim.world import World, WorldConfig


@dataclass
class PocEnvConfig:
    pre_grasp: Tuple[float, ...] = (0.0,) * 10
    settle_steps: int = 120
    lift_dz: float = 0.08
    lift_success_z: float = 0.03
    contact_link_reward: float = 0.08
    com_shift_penalty: float = 2.0
    ang_vel_penalty: float = 0.05
    slip_penalty: float = 1.0
    lift_reward: float = 1.0
    fail_reward: float = -0.2
    tip_over_penalty: float = -1.0
    max_tilt_rad: float = 0.8
    grasp_steps: int = 200


class PocEnv:
    """
    Simple PoC environment:
      - reset: spawn world + set hand pre-grasp
      - evaluate(q10): apply grasp, lift, compute reward
    """

    def __init__(self, world: Optional[World] = None, config: Optional[PocEnvConfig] = None):
        self.config = config or PocEnvConfig()
        self.world = world or World(WorldConfig(gui=False, robot_type="ur5"))

    def reset(self):
        self.world.reset_world()
        # Apply the configured hand pre-grasp pose.
        self.world.hand.apply_joint_command_vector(list(self.config.pre_grasp))
        self.world.step(self.config.settle_steps)

    def evaluate(self, q10) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a 10D grasp vector and return (reward, info).
        """
        q = np.asarray(q10, dtype=float).tolist()
        if len(q) != 10:
            raise ValueError("q10 must be length 10")

        # Apply grasp with a short ramp to avoid snapping.
        start_pos, _ = p.getBasePositionAndOrientation(self.world.benchmark_id)
        start_q = self.world.hand.get_current_joint_command_vector()
        steps = max(1, int(self.config.grasp_steps))
        for i in range(1, steps + 1):
            a = i / float(steps)
            q_i = [min(1.0, max(0.0, (1.0 - a) * s + a * t)) for s, t in zip(start_q, q)]
            self.world.hand.apply_joint_command_vector(q_i)
            self.world.step(1)
        self.world.step(self.config.settle_steps)

        # Contact check
        contact = self._has_contact(self.world.hand_id, self.world.benchmark_id)
        contact_links = self._contact_link_count()
        end_pos, _ = p.getBasePositionAndOrientation(self.world.benchmark_id)
        com_shift = float(np.linalg.norm(np.array(end_pos) - np.array(start_pos)))
        obj_lin_vel, obj_ang_vel = p.getBaseVelocity(self.world.benchmark_id)
        hand_lin_vel, _ = p.getBaseVelocity(self.world.hand_id)
        ang_speed = float(np.linalg.norm(np.array(obj_ang_vel)))
        slip_speed = float(np.linalg.norm(np.array(obj_lin_vel) - np.array(hand_lin_vel)))

        # Lift attempt
        start_z = self._object_z()
        self.world.arm.lift_dz_blocking(self.config.lift_dz)
        end_z = self._object_z()

        lifted = (end_z - start_z) >= self.config.lift_success_z
        tipped = self._is_tipped()

        reward = 0.0
        reward += contact_links * self.config.contact_link_reward
        reward -= self.config.com_shift_penalty * com_shift
        reward -= self.config.ang_vel_penalty * ang_speed
        reward -= self.config.slip_penalty * slip_speed
        if lifted:
            reward += self.config.lift_reward
        else:
            reward += self.config.fail_reward
        if tipped:
            reward += self.config.tip_over_penalty

        info = {
            "contact": float(contact),
            "contact_links": float(contact_links),
            "com_shift": float(com_shift),
            "ang_speed": float(ang_speed),
            "slip_speed": float(slip_speed),
            "lifted": float(lifted),
            "tipped": float(tipped),
            "start_z": float(start_z),
            "end_z": float(end_z),
            "reward": float(reward),
        }
        return reward, info

    def _has_contact(self, body_a: int, body_b: int) -> bool:
        return len(p.getContactPoints(bodyA=body_a, bodyB=body_b)) > 0

    def _contact_link_count(self) -> int:
        count = 0
        for link_index in self.world.hand.get_tip_link_indices():
            if p.getContactPoints(
                bodyA=self.world.hand_id,
                bodyB=self.world.benchmark_id,
                linkIndexA=link_index,
            ):
                count += 1
        return count

    def _object_z(self) -> float:
        pos, _ = p.getBasePositionAndOrientation(self.world.benchmark_id)
        return float(pos[2])

    def _is_tipped(self) -> bool:
        _, quat = p.getBasePositionAndOrientation(self.world.benchmark_id)
        rpy = p.getEulerFromQuaternion(quat)
        return abs(rpy[0]) > self.config.max_tilt_rad or abs(rpy[1]) > self.config.max_tilt_rad


__all__ = ["PocEnv", "PocEnvConfig"]
