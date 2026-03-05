import math

import numpy as np
import pybullet as p


class UR5Helper:
    def __init__(self, robot_id, ee_link=9, joint_indices=None):
        self.robot_id = robot_id
        self.ee_link = ee_link
        self.joints = joint_indices if joint_indices is not None else self._detect_revolute_joints()
        if len(self.joints) != 6:
            raise RuntimeError(f"UR5Helper expects 6 revolute joints, got: {self.joints}")
        self.limits = self._read_limits(self.joints)
        self._ik_solution_index_map = self._build_ik_solution_index_map()

    def _detect_revolute_joints(self):
        out = []
        for j in range(p.getNumJoints(self.robot_id)):
            if p.getJointInfo(self.robot_id, j)[2] == p.JOINT_REVOLUTE:
                out.append(j)
        return out

    def _read_limits(self, joints):
        limits = []
        for j in joints:
            low, high = p.getJointInfo(self.robot_id, j)[8:10]
            if not (high > low) or math.isinf(low) or math.isinf(high):
                low, high = -2 * math.pi, 2 * math.pi
            limits.append((float(low), float(high)))
        return limits

    def _build_ik_solution_index_map(self):
        movable = []
        for j in range(p.getNumJoints(self.robot_id)):
            jt = p.getJointInfo(self.robot_id, j)[2]
            if jt in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                movable.append(j)
        idx_map = {}
        for joint_idx in self.joints:
            if joint_idx not in movable:
                raise RuntimeError(f"Joint {joint_idx} is not movable and cannot be controlled by IK.")
            idx_map[joint_idx] = movable.index(joint_idx)
        return idx_map

    def ee_pose(self):
        ls = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        xyz = np.array(ls[4], dtype=float)
        rpy = np.array(p.getEulerFromQuaternion(ls[5]), dtype=float)
        return xyz, rpy

    def _ik(self, xyz, rpy):
        quat = p.getQuaternionFromEuler(rpy)
        sol = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link,
            targetPosition=xyz,
            targetOrientation=quat,
            maxNumIterations=100,
            residualThreshold=1e-3,
        )
        max_sol_idx = max(self._ik_solution_index_map.values())
        if len(sol) <= max_sol_idx:
            raise RuntimeError(
                f"IK solution length {len(sol)} is too short for IK map {self._ik_solution_index_map}"
            )
        q = np.array([sol[self._ik_solution_index_map[j]] for j in self.joints], dtype=float)
        low = np.array([l for l, _ in self.limits])
        high = np.array([h for _, h in self.limits])
        return np.clip(q, low, high)

    def move_to_pose_blocking(self, xyz, rpy, seconds=1.0, hz=240, gain=0.25, force=200.0):
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
        xyz, rpy = self.ee_pose()
        target = xyz.copy()
        target[2] += float(dz)
        self.move_to_pose_blocking(target, rpy, seconds=seconds, hz=hz, gain=gain, force=force)
