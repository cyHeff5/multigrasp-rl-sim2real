import math

import numpy as np
import pybullet as p


class SawyerHelper:
    def __init__(self, robot_id, ee_link_name="right_l6", joint_names=None, joint_indices=None):
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
            raise RuntimeError(f"SawyerHelper expects 7 arm joints, got: {self.joints}")

        self.limits = self._read_limits(self.joints)
        self._ik_solution_index_map = self._build_ik_solution_index_map()

    def _resolve_link_index(self, link_name):
        if isinstance(link_name, int):
            return link_name
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            name = info[12].decode("utf-8")
            if name == link_name:
                return j
        raise RuntimeError(f"Link not found: {link_name}")

    def _resolve_joint_indices(self, joint_names):
        name_to_index = {}
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            name = info[1].decode("utf-8")
            if name in joint_names:
                name_to_index[name] = j
        missing = [n for n in joint_names if n not in name_to_index]
        if missing:
            raise RuntimeError(f"Missing Sawyer joints: {missing}")
        return [name_to_index[n] for n in joint_names]

    def _read_limits(self, joints):
        limits = []
        for j in joints:
            low, high = p.getJointInfo(self.robot_id, j)[8:10]
            if not (high > low) or math.isinf(low) or math.isinf(high):
                low, high = -2 * math.pi, 2 * math.pi
            limits.append((float(low), float(high)))
        return limits

    def _build_ik_solution_index_map(self):
        # PyBullet IK solution is ordered by non-fixed joints, not absolute joint indices.
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

    def _ik(self, xyz, rpy, rest=None):
        quat = p.getQuaternionFromEuler(rpy)
        low = [l for l, _ in self.limits]
        high = [h for _, h in self.limits]
        ranges = [h - l for l, h in self.limits]
        if rest is None:
            rest = [p.getJointState(self.robot_id, j)[0] for j in self.joints]

        sol = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link,
            targetPosition=xyz,
            targetOrientation=quat,
            lowerLimits=low,
            upperLimits=high,
            jointRanges=ranges,
            restPoses=rest,
            maxNumIterations=500,
            residualThreshold=1e-4,
        )
        max_sol_idx = max(self._ik_solution_index_map.values())
        if len(sol) <= max_sol_idx:
            raise RuntimeError(
                f"IK solution length {len(sol)} is too short for IK map {self._ik_solution_index_map}"
            )
        q = np.array([sol[self._ik_solution_index_map[j]] for j in self.joints], dtype=float)
        return np.clip(q, np.array(low, dtype=float), np.array(high, dtype=float))

    def find_random_ik(
        self, xyz, rpy, n_attempts: int = 30, pos_tol: float = 2e-2, rng=None
    ) -> np.ndarray | None:
        """Try random rest poses to find an alternative IK solution.

        Returns a joint config (7,) whose EE position is within pos_tol of xyz,
        or None if no valid solution was found in n_attempts.
        """
        if rng is None:
            rng = np.random.default_rng()
        low = np.array([l for l, _ in self.limits])
        high = np.array([h for _, h in self.limits])

        # First attempt: use current joint state as rest pose
        current = [p.getJointState(self.robot_id, j)[0] for j in self.joints]
        rest_poses = [current] + [rng.uniform(low, high).tolist() for _ in range(n_attempts - 1)]

        for rest in rest_poses:
            q = self._ik(xyz, rpy, rest=rest)

            # Temporarily set joints to check EE position via FK.
            saved = [p.getJointState(self.robot_id, j)[0] for j in self.joints]
            for j, qi in zip(self.joints, q.tolist()):
                p.resetJointState(self.robot_id, j, qi)
            ls = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
            ee_pos = np.array(ls[4])
            for j, qi in zip(self.joints, saved):
                p.resetJointState(self.robot_id, j, qi)

            if np.linalg.norm(ee_pos - np.array(xyz)) < pos_tol:
                return q

        return None

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
