from __future__ import annotations

import pybullet as p
import pybullet_data

from src.sim.assets import AR10_URDF, SAWYER_URDF, UR5_URDF
from src.sim.hand_model import HandModel
from src.sim.mounting import apply_hand_friction, apply_physics_defaults, mount_hand_to_arm
from src.sim.sawyer_arm import SawyerHelper
from src.sim.ur5_arm import UR5Helper


class PyBulletWorld:
    """World wrapper for robot + mounted AR10 + spawned grasp object."""

    def __init__(
        self,
        gui: bool = False,
        robot_type: str = "sawyer",
        arm_pregrasp_joint_positions: list[float] | None = None,
        arm_pregrasp_settle_steps: int = 20,
        arm_hold_soft_steps: int = 80,
        arm_hold_soft_gain: float = 0.03,
        arm_hold_soft_force: float = 80.0,
        arm_hold_gain: float = 0.10,
        arm_hold_force: float = 250.0,
        arm_hold_velocity_gain: float = 1.0,
        arm_kinematic_lock: bool = True,
        spawn_on_pedestal: bool = False,
        pedestal_height_m: float = 0.05,
        pedestal_shape: str = "box",
        pedestal_diameter_m: float = 0.04,
        pedestal_half_extents_xy: tuple[float, float] = (0.08, 0.08),
        pedestal_position_xy: tuple[float, float] = (0.40, 0.00),
        free_hand_pregrasp_position_xyz: tuple[float, float, float] = (0.40, 0.00, 0.12),
        free_hand_pregrasp_rpy_deg: tuple[float, float, float] = (90.0, 0.0, 90.0),
        free_hand_constraint_force: float = 3000.0,
        robot_base_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.gui = bool(gui)
        self.robot_type = robot_type
        self.arm_pregrasp_joint_positions = arm_pregrasp_joint_positions
        self.arm_pregrasp_settle_steps = int(arm_pregrasp_settle_steps)
        self.arm_hold_soft_steps = int(arm_hold_soft_steps)
        self.arm_hold_soft_gain = float(arm_hold_soft_gain)
        self.arm_hold_soft_force = float(arm_hold_soft_force)
        self.arm_hold_gain = float(arm_hold_gain)
        self.arm_hold_force = float(arm_hold_force)
        self.arm_hold_velocity_gain = float(arm_hold_velocity_gain)
        self.arm_kinematic_lock = bool(arm_kinematic_lock)
        self.spawn_on_pedestal = bool(spawn_on_pedestal)
        self.pedestal_height_m = float(pedestal_height_m)
        self.pedestal_shape = str(pedestal_shape).lower()
        self.pedestal_diameter_m = float(pedestal_diameter_m)
        self.pedestal_half_extents_xy = (
            float(pedestal_half_extents_xy[0]),
            float(pedestal_half_extents_xy[1]),
        )
        self.pedestal_position_xy = (
            float(pedestal_position_xy[0]),
            float(pedestal_position_xy[1]),
        )
        self.free_hand_pregrasp_position_xyz = (
            float(free_hand_pregrasp_position_xyz[0]),
            float(free_hand_pregrasp_position_xyz[1]),
            float(free_hand_pregrasp_position_xyz[2]),
        )
        self.free_hand_pregrasp_rpy_deg = (
            float(free_hand_pregrasp_rpy_deg[0]),
            float(free_hand_pregrasp_rpy_deg[1]),
            float(free_hand_pregrasp_rpy_deg[2]),
        )
        self.free_hand_constraint_force = float(free_hand_constraint_force)
        self.robot_base_rpy_deg = (
            float(robot_base_rpy_deg[0]),
            float(robot_base_rpy_deg[1]),
            float(robot_base_rpy_deg[2]),
        )
        self.client_id = None
        self.plane_id = None
        self.pedestal_id = None
        self.robot_id = None
        self.hand_id = None
        self.object_id = None
        self.arm = None
        self.hand = None
        self._arm_hold_target = None
        self._arm_hold_soft_steps_left = 0
        self._pregrasp_collision_steps_left = 0
        self._pregrasp_collision_object_id = None
        self._pregrasp_disabled_pairs: list[tuple[int, int]] = []
        self._free_hand_anchor_id: int | None = None
        self._free_hand_constraint_id: int | None = None

    def connect(self) -> None:
        if self.client_id is not None:
            return
        self.client_id = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=50.0,
                cameraPitch=-35.0,
                cameraTargetPosition=[-0.01, 0.37, -0.24],
            )

    def reset(self) -> None:
        self.connect()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        try:
            p.resetSimulation()
            p.setGravity(0.0, 0.0, -9.81)
            apply_physics_defaults()
            self.plane_id = p.loadURDF("plane.urdf")
            if self.spawn_on_pedestal:
                self._spawn_pedestal()
            self.robot_id = None
            self.arm = None
            self._free_hand_anchor_id = None
            self._free_hand_constraint_id = None

            robot_base_quat = p.getQuaternionFromEuler(
                [v * 3.141592653589793 / 180.0 for v in self.robot_base_rpy_deg]
            )
            if self.robot_type == "sawyer":
                self.robot_id = p.loadURDF(
                    str(SAWYER_URDF),
                    basePosition=[0, 0, 0],
                    baseOrientation=robot_base_quat,
                    useFixedBase=True,
                )
                self.arm = SawyerHelper(self.robot_id, ee_link_name="right_l6")
                ee_link = "right_l6"
            elif self.robot_type == "ur5":
                self.robot_id = p.loadURDF(
                    str(UR5_URDF),
                    basePosition=[0, 0, 0],
                    baseOrientation=robot_base_quat,
                    useFixedBase=True,
                )
                self.arm = UR5Helper(self.robot_id, ee_link=9)
                ee_link = 9
            elif self.robot_type == "free_hand":
                self._spawn_free_hand_with_anchor()
            else:
                raise ValueError(f"Unsupported robot_type: {self.robot_type}")

            if self.robot_type in ("sawyer", "ur5"):
                self.hand_id = p.loadURDF(str(AR10_URDF))
                mount_hand_to_arm(self.robot_id, self.hand_id, ee_link=ee_link)
                apply_hand_friction(self.hand_id)
                self.hand = HandModel(self.hand_id)
                self.hand.reset_open_pose()
                self._apply_arm_pregrasp_if_configured(reapply_after_mount=True)
            else:
                apply_hand_friction(self.hand_id)
                self.hand = HandModel(self.hand_id)
                self.hand.reset_open_pose()
        finally:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def _spawn_free_hand_with_anchor(self) -> None:
        start_pos = [float(v) for v in self.free_hand_pregrasp_position_xyz]
        start_quat = p.getQuaternionFromEuler([v * 3.141592653589793 / 180.0 for v in self.free_hand_pregrasp_rpy_deg])
        self.hand_id = p.loadURDF(
            str(AR10_URDF),
            basePosition=start_pos,
            baseOrientation=start_quat,
            useFixedBase=False,
        )
        self._free_hand_anchor_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=start_pos,
            baseOrientation=start_quat,
        )
        self._free_hand_constraint_id = p.createConstraint(
            parentBodyUniqueId=int(self._free_hand_anchor_id),
            parentLinkIndex=-1,
            childBodyUniqueId=int(self.hand_id),
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=[0.0, 0.0, 0.0],
            childFramePosition=[0.0, 0.0, 0.0],
            parentFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
        )
        p.changeConstraint(
            int(self._free_hand_constraint_id),
            maxForce=float(self.free_hand_constraint_force),
            erp=0.95,
        )

    def set_free_hand_pose(self, position_xyz: list[float], orientation_xyzw: list[float]) -> None:
        """Set free-hand pose by moving the anchor and syncing the hand base."""
        if self.robot_type != "free_hand":
            return
        if self._free_hand_anchor_id is None or self._free_hand_constraint_id is None or self.hand_id is None:
            return
        pos = [float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2])]
        quat = [
            float(orientation_xyzw[0]),
            float(orientation_xyzw[1]),
            float(orientation_xyzw[2]),
            float(orientation_xyzw[3]),
        ]
        p.resetBasePositionAndOrientation(int(self._free_hand_anchor_id), pos, quat)
        p.resetBasePositionAndOrientation(int(self.hand_id), pos, quat)
        p.changeConstraint(int(self._free_hand_constraint_id), maxForce=float(self.free_hand_constraint_force), erp=0.95)

    def _apply_arm_pregrasp_if_configured(self, reapply_after_mount: bool = False) -> None:
        if self.arm is None or self.arm_pregrasp_joint_positions is None:
            return
        target = [float(v) for v in self.arm_pregrasp_joint_positions]
        if len(target) != len(self.arm.joints):
            raise ValueError(
                f"arm_pregrasp_joint_positions length {len(target)} does not match "
                f"{self.robot_type} joints {len(self.arm.joints)}"
            )

        # Hard-set the robot state to pre-grasp.
        for j, q in zip(self.arm.joints, target):
            p.resetJointState(self.robot_id, j, q)

        # Keep motors at the same pose for stabilization.
        self._arm_hold_target = list(target)
        if reapply_after_mount:
            # After adding the mounted hand constraint, hold immediately with full strength
            # to avoid visible "jump then catch" behavior.
            self._arm_hold_soft_steps_left = 0
        else:
            self._arm_hold_soft_steps_left = max(0, self.arm_hold_soft_steps)
        self._apply_arm_hold_control()
        if self.arm_pregrasp_settle_steps > 0 and not reapply_after_mount:
            self.step(self.arm_pregrasp_settle_steps)

    def _apply_arm_hold_control(self) -> None:
        if self.arm is None or self._arm_hold_target is None:
            return
        if self.arm_kinematic_lock:
            self._hard_lock_arm_to_target()
            return
        if self._arm_hold_soft_steps_left > 0:
            gain = self.arm_hold_soft_gain
            force = self.arm_hold_soft_force
        else:
            gain = self.arm_hold_gain
            force = self.arm_hold_force
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.arm.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self._arm_hold_target,
            targetVelocities=[0.0] * len(self.arm.joints),
            positionGains=[gain] * len(self.arm.joints),
            velocityGains=[self.arm_hold_velocity_gain] * len(self.arm.joints),
            forces=[force] * len(self.arm.joints),
        )

    def _hard_lock_arm_to_target(self) -> None:
        if self.arm is None or self._arm_hold_target is None:
            return
        for j, q in zip(self.arm.joints, self._arm_hold_target):
            p.resetJointState(self.robot_id, j, float(q), targetVelocity=0.0)

    def sync_arm_hold_target_to_current(self) -> None:
        """Update arm hold target to current joint states (used after external arm motions)."""
        if self.arm is None or self.robot_id is None:
            return
        self._arm_hold_target = [float(p.getJointState(self.robot_id, j)[0]) for j in self.arm.joints]

    @staticmethod
    def _set_collision_enabled_between_bodies(body_a: int, body_b: int, enabled: bool) -> None:
        for link_a in range(-1, p.getNumJoints(body_a)):
            for link_b in range(-1, p.getNumJoints(body_b)):
                p.setCollisionFilterPair(body_a, body_b, link_a, link_b, int(bool(enabled)))

    def disable_pregrasp_collisions_temporarily(self, object_id: int, steps: int) -> None:
        if self.hand_id is None:
            return
        n = max(0, int(steps))
        if n <= 0:
            return

        pairs: list[tuple[int, int]] = [(self.hand_id, int(object_id))]
        if self.robot_id is not None:
            pairs.extend(
                [
                    (self.robot_id, self.hand_id),
                    (self.robot_id, int(object_id)),
                ]
            )
        if self.plane_id is not None:
            pairs.append((self.hand_id, int(self.plane_id)))
        if self.pedestal_id is not None:
            pairs.append((self.hand_id, int(self.pedestal_id)))

        for body_a, body_b in pairs:
            self._set_collision_enabled_between_bodies(body_a, body_b, False)

        self._pregrasp_collision_object_id = int(object_id)
        self._pregrasp_disabled_pairs = pairs
        self._pregrasp_collision_steps_left = n

    def _maybe_restore_pregrasp_collisions(self) -> None:
        if self._pregrasp_collision_steps_left <= 0:
            return
        self._pregrasp_collision_steps_left -= 1
        if self._pregrasp_collision_steps_left > 0:
            return

        for body_a, body_b in self._pregrasp_disabled_pairs:
            self._set_collision_enabled_between_bodies(body_a, body_b, True)
        self._pregrasp_disabled_pairs = []
        self._pregrasp_collision_object_id = None

    def wait_until_pregrasp_collisions_restored(self, max_steps: int | None = None) -> None:
        """Block until temporary pregrasp collision disabling has fully ended."""
        if self._pregrasp_collision_steps_left <= 0:
            return
        if max_steps is None:
            max_steps = self._pregrasp_collision_steps_left
        remaining_budget = max(0, int(max_steps))
        while self._pregrasp_collision_steps_left > 0 and remaining_budget > 0:
            self.step(1)
            remaining_budget -= 1

    def spawn_primitive_object(self, spec: dict) -> int:
        shape = spec.get("shape", "sphere")
        size_cm = float(spec.get("size_cm", 3.0))
        size_m = size_cm / 100.0
        thickness_cm = float(spec.get("thickness_cm", size_cm))
        height_cm = float(spec.get("height_cm", size_cm))
        thickness_m = thickness_cm / 100.0
        height_m = height_cm / 100.0
        mass = float(spec.get("mass_kg", 0.08))
        pos = list(spec.get("position_xyz", [0.6, 0.0, size_m * 0.5]))
        rgba = spec.get("rgba", [0.8, 0.2, 0.2, 1.0])

        if shape == "sphere":
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=size_m * 0.5)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=size_m * 0.5, rgbaColor=rgba)
        elif shape == "cube":
            half = size_m * 0.5
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=rgba)
        elif shape == "cylinder":
            radius = thickness_m * 0.5
            height = height_m
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba)
        elif shape == "rect_cylinder":
            hx = thickness_m * 0.5
            hy = thickness_m * 0.5
            hz = height_m * 0.5
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=rgba)
        else:
            raise ValueError(f"Unsupported shape: {shape}")

        if self.spawn_on_pedestal:
            half_h = self._object_half_height_m(shape=shape, size_m=size_m, height_m=height_m)
            pos[2] = self.pedestal_height_m + half_h

        orn = p.getQuaternionFromEuler(spec.get("rpy", [0.0, 0.0, 0.0]))
        self.object_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[float(x) for x in pos],
            baseOrientation=orn,
        )
        p.changeDynamics(
            self.object_id,
            -1,
            lateralFriction=float(spec.get("lateral_friction", 0.4)),
            spinningFriction=float(spec.get("spinning_friction", 0.001)),
            rollingFriction=float(spec.get("rolling_friction", 0.0005)),
            frictionAnchor=True,
        )
        return self.object_id

    def _spawn_pedestal(self) -> None:
        half_z = self.pedestal_height_m * 0.5
        if self.pedestal_shape == "cylinder":
            radius = max(1e-6, self.pedestal_diameter_m * 0.5)
            col = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=self.pedestal_height_m,
            )
            vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=self.pedestal_height_m,
                rgbaColor=[0.75, 0.75, 0.75, 1.0],
            )
        else:
            half_x, half_y = self.pedestal_half_extents_xy
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[half_x, half_y, half_z],
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[half_x, half_y, half_z],
                rgbaColor=[0.75, 0.75, 0.75, 1.0],
            )
        self.pedestal_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[self.pedestal_position_xy[0], self.pedestal_position_xy[1], half_z],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
        )

    @staticmethod
    def _object_half_height_m(shape: str, size_m: float, height_m: float | None = None) -> float:
        if shape in ("sphere", "cube"):
            return size_m * 0.5
        if shape in ("cylinder", "rect_cylinder"):
            if height_m is not None:
                return height_m * 0.5
            return size_m * 0.5
        return size_m * 0.5

    def step(self, steps: int = 1) -> None:
        for _ in range(max(1, int(steps))):
            if self.arm_kinematic_lock and self._arm_hold_target is not None:
                self._hard_lock_arm_to_target()
            else:
                self._apply_arm_hold_control()
            p.stepSimulation()
            if self._arm_hold_soft_steps_left > 0:
                self._arm_hold_soft_steps_left -= 1
            self._maybe_restore_pregrasp_collisions()

    def lift_grasping_hand_blocking(
        self, dz: float, seconds: float = 0.6, hz: int = 240, dx: float = 0.0, dy: float = 0.0
    ) -> None:
        if self.arm is not None:
            self.arm.lift_dz_blocking(dz)
            self.sync_arm_hold_target_to_current()
            return
        if self.robot_type != "free_hand":
            raise RuntimeError("lift_grasping_hand_blocking requires arm or robot_type=free_hand")
        if self._free_hand_anchor_id is None or self._free_hand_constraint_id is None:
            return
        pos, quat = p.getBasePositionAndOrientation(int(self._free_hand_anchor_id))
        target = [float(pos[0]) + float(dx), float(pos[1]) + float(dy), float(pos[2]) + float(dz)]
        n_steps = max(1, int(float(seconds) * float(hz)))
        for i in range(1, n_steps + 1):
            a = i / float(n_steps)
            p_i = [
                float(pos[0]) + a * (target[0] - float(pos[0])),
                float(pos[1]) + a * (target[1] - float(pos[1])),
                float(pos[2]) + a * (target[2] - float(pos[2])),
            ]
            p.resetBasePositionAndOrientation(int(self._free_hand_anchor_id), p_i, quat)
            p.changeConstraint(int(self._free_hand_constraint_id), maxForce=float(self.free_hand_constraint_force))
            p.stepSimulation()
            self._maybe_restore_pregrasp_collisions()

    def close(self) -> None:
        if self.client_id is not None:
            p.disconnect()
            self.client_id = None
