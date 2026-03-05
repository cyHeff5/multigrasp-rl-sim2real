# UR5 controller (IK + blocking moves).
import math
import numpy as np
import pybullet as p


class UR5Helper:
    """
    Minimal helper for the UR5 arm tailored to my RL use-case.
    """

    def __init__(self, robot_id, ee_link=9, joint_indices=None):
        self.robot_id = robot_id
        self.ee_link = ee_link

        # If joint indices are not provided, detect all revolute joints (UR5 has 6)
        self.joints = joint_indices if joint_indices is not None else self._detect_revolute_joints()
        if len(self.joints) != 6:
            raise RuntimeError("UR5Helper expects 6 revolute joints, got: {}".format(self.joints))

        # Cache joint limits once (fallback to +/- 2*pi if URDF uses inf)
        self.limits = self._read_limits(self.joints)

    # ---------------- discovery / limits ----------------

    def _detect_revolute_joints(self):
        """Find all revolute joint indices for this body."""
        out = []
        for j in range(p.getNumJoints(self.robot_id)):
            if p.getJointInfo(self.robot_id, j)[2] == p.JOINT_REVOLUTE:
                out.append(j)
        return out

    def _read_limits(self, joints):
        """Read (lower, upper) angle limits for each joint."""
        lims = []
        for j in joints:
            lo, hi = p.getJointInfo(self.robot_id, j)[8:10]
            # Use a safe fallback if the URDF exposes invalid or infinite limits
            if not (hi > lo) or math.isinf(lo) or math.isinf(hi):
                lo, hi = -2 * math.pi, 2 * math.pi
            lims.append((float(lo), float(hi)))
        return lims

    # ---------------- basic state ----------------

    def ee_pose(self):
        """Return current end-effector pose as (xyz, rpy)."""
        ls = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        xyz = np.array(ls[4], dtype=float)
        rpy = np.array(p.getEulerFromQuaternion(ls[5]), dtype=float)
        return xyz, rpy

    # ---------------- IK helpers ----------------

    def _ik(self, xyz, rpy):
        """Compute a joint solution for the desired EE pose using PyBullet IK."""
        quat = p.getQuaternionFromEuler(rpy)
        sol = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link,
            targetPosition=xyz,
            targetOrientation=quat,
            maxNumIterations=100,
            residualThreshold=1e-3,
        )
        q = np.array(sol[:len(self.joints)], dtype=float)

        # Clamp to joint limits (keeps things stable near boundaries)
        lo = np.array([l for l, _ in self.limits])
        hi = np.array([h for _, h in self.limits])
        return np.clip(q, lo, hi)

    # ---------------- blocking servo moves (simple & smooth) ----------------

    def move_to_pose_blocking(self, xyz, rpy, seconds=1.0, hz=240, gain=0.25, force=200.0):
        """
        Move from current EE pose to (xyz, rpy) smoothly over 'seconds'.
        Simple linear interpolation in task space; each step:
          (interpolate pose) -> IK -> POSITION_CONTROL -> stepSimulation()
        """
        steps = max(1, int(seconds * hz))
        cur_xyz, cur_rpy = self.ee_pose()
        tgt_xyz = np.array(xyz, dtype=float)
        tgt_rpy = np.array(rpy, dtype=float)

        for i in range(1, steps + 1):
            a = i / float(steps)
            xyz_i = (1.0 - a) * cur_xyz + a * tgt_xyz
            rpy_i = (1.0 - a) * cur_rpy + a * tgt_rpy

            q = self._ik(xyz_i, rpy_i)
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=q.tolist(),
                positionGains=[gain] * len(self.joints),
                forces=[force] * len(self.joints),
            )
            p.stepSimulation()

    def lift_dz_blocking(self, dz, seconds=0.6, hz=240, gain=0.25, force=200.0):
        """
        Lift the end-effector upwards by 'dz' meters (relative move).
        I use a blocking, smooth move so I can evaluate the grasp cleanly afterwards.
        """
        xyz, rpy = self.ee_pose()
        target = xyz.copy()
        target[2] += float(dz)
        self.move_to_pose_blocking(target, rpy, seconds=seconds, hz=hz, gain=gain, force=force)
