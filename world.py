# PyBullet connection, physics params, timestep.
import math
from dataclasses import dataclass
from typing import Optional

import pybullet as p
import pybullet_data

from sim.assets import AR10_URDF, SAWYER_URDF, UR5_URDF, benchmark_part_urdf
from sim.sawyer_arm import SawyerHelper
from sim.ur5_arm import UR5Helper
from sim.hand import AR10Helper
from sim.mounting import (
    apply_hand_friction_like_sim_py,
    apply_physics_params_like_sim_py,
    mount_hand_to_arm,
)


@dataclass
class WorldConfig:
    robot_type: str = "ur5"  # "sawyer" or "ur5"
    gui: bool = True
    gravity: float = -9.81
    ee_link: Optional[object] = None
    use_fixed_base: bool = True
    settle_steps: int = 240
    benchmark_part_id: int = 3
    benchmark_pos: tuple = (0.6, 0.0, 0.05)
    benchmark_rpy: tuple = (0.0, 0.0, 0.0)
    home_joint_positions: Optional[tuple] = None
    home_position_gain: float = 0.1
    home_steps: int = 120
    use_pla_pla_friction: bool = True
    pla_pla_lateral_mu: float = 0.32
    pla_pla_spinning_friction: float = 0.001
    pla_pla_rolling_friction: float = 0.0005

    def __post_init__(self):
        if self.ee_link is None:
            self.ee_link = 9 if self.robot_type == "ur5" else "right_l6"
        if self.home_joint_positions is None:
            if self.robot_type == "ur5":
                self.home_joint_positions = (
                    -3.526193839290861,
                    -1.5340938701773221,
                    2.0985749444290285,
                    -0.4472659626584632,
                    1.1955911555227652,
                    3.0898584023854077,
                )
            else:
                self.home_joint_positions = (
                    -0.19999511870154918,
                    -1.449988765591175,
                    -0.03997614956493058,
                    1.3400699248337562,
                    -1.520061695852127,
                    1.1699913835803561,
                    3.2498732514288604,
                )


class World:
    """
    Owns the PyBullet connection and core scene objects.

    Responsibilities:
      - connect / disconnect
      - load plane, Sawyer, AR10, mount hand
      - apply default physics + friction parameters
      - expose robot/hand IDs and helper instances
    """

    def __init__(self, config: Optional[WorldConfig] = None):
        self.config = config or WorldConfig()
        self.client_id = None
        self.plane_id = None
        self.robot_id = None
        self.ur5_id = None
        self.sawyer_id = None
        self.hand_id = None
        self.benchmark_id = None
        self.arm = None
        self.hand = None
        self._connected = False

    def connect(self):
        if self._connected:
            return
        self.client_id = p.connect(p.GUI if self.config.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._connected = True

    def disconnect(self):
        if self._connected:
            p.disconnect()
            self._connected = False

    def reset_world(self):
        """
        Reset the entire simulation and (re)spawn assets.
        """
        if not self._connected:
            self.connect()

        p.resetSimulation()
        p.setGravity(0.0, 0.0, float(self.config.gravity))
        apply_physics_params_like_sim_py()

        self.plane_id = p.loadURDF("plane.urdf")

        if self.config.robot_type == "ur5":
            self.robot_id = p.loadURDF(
                str(UR5_URDF),
                basePosition=[0, 0, 0],
                useFixedBase=bool(self.config.use_fixed_base),
            )
            self.ur5_id = self.robot_id
            arm_helper = UR5Helper(self.robot_id, ee_link=int(self.config.ee_link))
        else:
            self.robot_id = p.loadURDF(
                str(SAWYER_URDF),
                basePosition=[0, 0, 0],
                useFixedBase=bool(self.config.use_fixed_base),
            )
            self.sawyer_id = self.robot_id
            arm_helper = SawyerHelper(self.robot_id, ee_link_name=self.config.ee_link)

        # Move arm joints into a stable home pose before mounting the hand.
        revolute_indices = arm_helper.joints
        if len(revolute_indices) != len(self.config.home_joint_positions):
            raise RuntimeError(
                "home_joint_positions must match arm joints ({}).".format(len(revolute_indices))
            )
        p.setJointMotorControlArray(
            self.robot_id,
            revolute_indices,
            p.POSITION_CONTROL,
            targetPositions=list(self.config.home_joint_positions),
            positionGains=[float(self.config.home_position_gain)] * len(revolute_indices),
        )
        for _ in range(max(0, int(self.config.home_steps))):
            p.stepSimulation()

        self.hand_id = p.loadURDF(str(AR10_URDF))

        # Mount hand to arm and apply friction
        mount_hand_to_arm(self.robot_id, self.hand_id, ee_link=self.config.ee_link)
        if self.config.use_pla_pla_friction:
            mu_lat = math.sqrt(max(0.0, float(self.config.pla_pla_lateral_mu)))
            mu_spin = math.sqrt(max(0.0, float(self.config.pla_pla_spinning_friction)))
            mu_roll = math.sqrt(max(0.0, float(self.config.pla_pla_rolling_friction)))
            apply_hand_friction_like_sim_py(
                self.hand_id,
                lateral=mu_lat,
                spinning=mu_spin,
                rolling=mu_roll,
            )
        else:
            apply_hand_friction_like_sim_py(self.hand_id)

        # Helpers for convenience
        self.arm = arm_helper
        self.hand = AR10Helper(self.hand_id)
        # Keep the hand opened right after spawn to avoid initial jitters.
        self.hand.apply_joint_command_vector([0.0] * 10)

        # Load benchmark part after the arm + hand are in place.
        bench_urdf = benchmark_part_urdf(self.config.benchmark_part_id)
        bench_quat = p.getQuaternionFromEuler([float(x) for x in self.config.benchmark_rpy])
        self.benchmark_id = p.loadURDF(
            str(bench_urdf),
            basePosition=[float(x) for x in self.config.benchmark_pos],
            baseOrientation=bench_quat,
        )
        if self.config.use_pla_pla_friction:
            mu_lat = math.sqrt(max(0.0, float(self.config.pla_pla_lateral_mu)))
            mu_spin = math.sqrt(max(0.0, float(self.config.pla_pla_spinning_friction)))
            mu_roll = math.sqrt(max(0.0, float(self.config.pla_pla_rolling_friction)))
            for link_index in range(-1, p.getNumJoints(self.benchmark_id)):
                p.changeDynamics(
                    self.benchmark_id,
                    link_index,
                    lateralFriction=mu_lat,
                    spinningFriction=mu_spin,
                    rollingFriction=mu_roll,
                    frictionAnchor=True,
                )

        # Temporarily disable hand↔benchmark collisions to avoid spawn kick.
        for hand_link in range(-1, p.getNumJoints(self.hand_id)):
            p.setCollisionFilterPair(self.hand_id, self.benchmark_id, hand_link, -1, 0)
        self.step(60)
        for hand_link in range(-1, p.getNumJoints(self.hand_id)):
            p.setCollisionFilterPair(self.hand_id, self.benchmark_id, hand_link, -1, 1)

        self.settle()

    def settle(self, steps: Optional[int] = None):
        """
        Step the simulation for a short period to let contacts settle.
        """
        n = self.config.settle_steps if steps is None else int(steps)
        for _ in range(max(0, n)):
            p.stepSimulation()

    def step(self, steps: int = 1):
        for _ in range(max(1, int(steps))):
            p.stepSimulation()

    def close(self):
        self.disconnect()


__all__ = ["World", "WorldConfig"]
