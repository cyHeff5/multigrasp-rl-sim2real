# Sawyer controller (IK + blocking moves).
import math

import numpy as np
import pybullet as p


class SawyerHelper:
    """
    Minimal helper for the Sawyer arm tailored to this project.
    """

    def __init__(self, robot_id, ee_link_name="right_hand", joint_names=None, joint_indices=None):
        self.robot_id = robot_id
        self.ee_link = self._resolve_link_index(ee_link_name)

        if joint_indices is None:
            if joint_names is None:
                joint_names = [
                    "right_j0",
                    "right_j1",
                    "right_j2",
                    "right_j3",
                    "right_j4",
                    "right_j5",
                    "right_j6",
                ]
            joint_indices = self._resolve_joint_indices(joint_names)

        self.joints = joint_indices
        if len(self.joints) != 7:
            raise RuntimeError("SawyerHelper expects 7 arm joints, got: {}".format(self.joints))

        self.limits = self._read_limits(self.joints)

    # ---------------- discovery / limits ----------------

    def _resolve_link_index(self, link_name):
        if isinstance(link_name, int):
            return link_name
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            name = info[12].decode("utf-8")
            if name == link_name:
                return j
        raise RuntimeError("Link not found: {}".format(link_name))

    def _resolve_joint_indices(self, joint_names):
        name_to_index = {}
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            name = info[1].decode("utf-8")
            if name in joint_names:
                name_to_index[name] = j
        missing = [n for n in joint_names if n not in name_to_index]
        if missing:
            raise RuntimeError("Missing Sawyer joints: {}".format(missing))
        return [name_to_index[n] for n in joint_names]

    def _read_limits(self, joints):
        """Read (lower, upper) angle limits for each joint."""
        lims = []
        for j in joints:
            lo, hi = p.getJointInfo(self.robot_id, j)[8:10]
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
        lo = [l for l, _ in self.limits]
        hi = [h for _, h in self.limits]
        ranges = [h - l for l, h in self.limits]
        rest = [p.getJointState(self.robot_id, j)[0] for j in self.joints]

        sol = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link,
            targetPosition=xyz,
            targetOrientation=quat,
            lowerLimits=lo,
            upperLimits=hi,
            jointRanges=ranges,
            restPoses=rest,
            maxNumIterations=200,
            residualThreshold=1e-4,
        )
        q = np.array(sol[:len(self.joints)], dtype=float)
        return np.clip(q, np.array(lo, dtype=float), np.array(hi, dtype=float))

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
        """
        xyz, rpy = self.ee_pose()
        target = xyz.copy()
        target[2] += float(dz)
        self.move_to_pose_blocking(target, rpy, seconds=seconds, hz=hz, gain=gain, force=force)


__all__ = ["SawyerHelper"]
