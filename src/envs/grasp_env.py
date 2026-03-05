from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import yaml
from gymnasium import spaces

from src.envs.grasp_env_base import GraspEnvBase
from src.envs.reward import RewardConfig, compute_reward_terms
from src.policies.wrappers import clip_action_delta
from src.sim.object_sampler import ObjectSampler
from src.sim.pybullet_world import PyBulletWorld


class GraspEnv(GraspEnvBase):
    """Step-based grasp env with obs=|q_target-q_measured| and action=delta_q_target."""

    def __init__(self, config: dict | str):
        cfg = self._load_config(config)
        super().__init__(cfg)

        world_cfg = cfg.get("world", {})
        self.world = PyBulletWorld(
            gui=bool(world_cfg.get("gui", False)),
            robot_type=str(world_cfg.get("robot_type", "sawyer")),
            arm_pregrasp_joint_positions=world_cfg.get("arm_pregrasp_joint_positions"),
            arm_pregrasp_settle_steps=int(world_cfg.get("arm_pregrasp_settle_steps", 20)),
            arm_hold_soft_steps=int(world_cfg.get("arm_hold_soft_steps", 80)),
            arm_hold_soft_gain=float(world_cfg.get("arm_hold_soft_gain", 0.03)),
            arm_hold_soft_force=float(world_cfg.get("arm_hold_soft_force", 80.0)),
            arm_hold_gain=float(world_cfg.get("arm_hold_gain", 0.10)),
            arm_hold_force=float(world_cfg.get("arm_hold_force", 250.0)),
            arm_hold_velocity_gain=float(world_cfg.get("arm_hold_velocity_gain", 1.0)),
            arm_kinematic_lock=bool(world_cfg.get("arm_kinematic_lock", True)),
            spawn_on_pedestal=bool(world_cfg.get("spawn_on_pedestal", False)),
            pedestal_height_m=float(world_cfg.get("pedestal_height_m", 0.05)),
            pedestal_shape=str(world_cfg.get("pedestal_shape", "box")),
            pedestal_diameter_m=float(world_cfg.get("pedestal_diameter_m", 0.04)),
            pedestal_half_extents_xy=tuple(world_cfg.get("pedestal_half_extents_xy", [0.08, 0.08])),
            pedestal_position_xy=tuple(world_cfg.get("pedestal_position_xy", [0.40, 0.00])),
            free_hand_pregrasp_position_xyz=tuple(world_cfg.get("free_hand_pregrasp_position_xyz", [0.40, 0.00, 0.12])),
            free_hand_pregrasp_rpy_deg=tuple(world_cfg.get("free_hand_pregrasp_rpy_deg", [90.0, 0.0, 90.0])),
            free_hand_constraint_force=float(world_cfg.get("free_hand_constraint_force", 3000.0)),
            robot_base_rpy_deg=tuple(world_cfg.get("robot_base_rpy_deg", [0.0, 0.0, 0.0])),
        )
        self.sampler = ObjectSampler(cfg.get("object_sampler", {}), cfg.get("spawn", {}))
        self.reward_cfg = RewardConfig(
            lift_success_z=float(cfg.get("lift_success_z", 0.03)),
            max_tilt_rad=float(cfg.get("max_tilt_rad", 0.8)),
            overgrip_threshold=float(cfg.get("overgrip_threshold", 0.90)),
            overgrip_penalty=float(cfg.get("overgrip_penalty", 0.20)),
            close_contact_link_reward=float(cfg.get("close_contact_link_reward", 0.04)),
            close_com_shift_penalty=float(cfg.get("close_com_shift_penalty", 1.0)),
            close_slip_penalty=float(cfg.get("close_slip_penalty", 0.4)),
            close_ang_vel_penalty=float(cfg.get("close_ang_vel_penalty", 0.02)),
            close_overgrip_penalty=float(cfg.get("close_overgrip_penalty", 0.10)),
            close_tip_over_penalty=float(cfg.get("close_tip_over_penalty", -1.0)),
            lift_success_reward=float(cfg.get("lift_success_reward", 5.0)),
            lift_fail_penalty=float(cfg.get("lift_fail_penalty", -2.0)),
            lift_slip_penalty=float(cfg.get("lift_slip_penalty", 1.0)),
            lift_ang_vel_penalty=float(cfg.get("lift_ang_vel_penalty", 0.05)),
            lift_tip_over_penalty=float(cfg.get("lift_tip_over_penalty", -2.0)),
        )
        self.grasp_type = str(cfg.get("grasp_type", "tripod")).lower()
        self.pregrasp_hand_reference_yaml = Path(
            str(cfg.get("pregrasp_hand_reference_yaml", "artifacts/hand_reference_points.yaml"))
        )
        distance_default_by_pose = {
            "tripod": 35.0,
            "medium_wrap": 20.0,
            "power_sphere": 10.0,
            "thumb_1_finger": 25.0,
            "lateral_pinch": 20.0,
        }
        distance_by_pose_cfg = cfg.get("pregrasp_distance_mm_by_pose", {})
        pose_distance_default = float(distance_default_by_pose.get(self.grasp_type, 20.0))
        if isinstance(distance_by_pose_cfg, dict) and self.grasp_type in distance_by_pose_cfg:
            pose_distance_default = float(distance_by_pose_cfg[self.grasp_type])
        self.pregrasp_distance_mm = float(cfg.get("pregrasp_distance_mm", pose_distance_default))
        self.pregrasp_twist_deg = float(cfg.get("pregrasp_twist_deg", 0.0))
        self.pregrasp_settle_steps = int(cfg.get("pregrasp_settle_steps", 30))
        self.pregrasp_hand_ref = self._load_hand_reference_point(self.pregrasp_hand_reference_yaml, self.grasp_type)

        self.frame_skip = int(cfg.get("frame_skip", 4))
        self.max_delta = float(cfg.get("action_scale", 0.05))
        self.max_steps = int(math.ceil(1.0 / max(1e-9, self.max_delta)) * 1.2)
        self.close_phase_steps = int(cfg.get("close_phase_steps", math.ceil((1.0 / max(1e-9, self.max_delta)) * 1.5)))
        self.max_steps = max(self.max_steps, self.close_phase_steps + 1)
        self.hand_dof = int(cfg.get("hand_dof", 10))
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
        self.active_joint_indices = self._resolve_active_joint_indices(cfg.get("action_active_joints"))
        self.action_dim = len(self.active_joint_indices)
        self.pregrasp_no_collision_steps = int(cfg.get("pregrasp_no_collision_steps", 30))
        self.action_mode = str(cfg.get("action_mode", "bidirectional")).lower()
        if self.action_mode not in ("bidirectional", "close_only"):
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")
        self.lift_check_enabled = bool(cfg.get("lift_check_enabled", True))
        self.lift_check_dz = float(cfg.get("lift_check_dz", 0.06))
        self.lift_check_hold_steps = int(cfg.get("lift_check_hold_steps", 30))
        self.lift_check_debug = bool(cfg.get("lift_check_debug", False))
        self.reward_debug_print = bool(cfg.get("reward_debug_print", False))
        self.lift_check_contact_links_min = int(cfg.get("lift_check_contact_links_min", 2))
        self.lift_check_contact_mode = str(cfg.get("lift_check_contact_mode", "tips")).lower()
        raw_contact_names = cfg.get("lift_check_contact_link_names")
        self.lift_check_contact_link_names = list(raw_contact_names) if isinstance(raw_contact_names, list) else None

        if self.action_mode == "close_only":
            action_low = 0.0
            action_high = self.max_delta
        else:
            action_low = -self.max_delta
            action_high = self.max_delta
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.hand_dof,),
            dtype=np.float32,
        )

        self.object_id = None
        self.step_count = 0
        self.object_start_z = 0.0
        self.last_obj_pos = None
        self.lift_check_done = False
        self.lift_check_lifted = False
        self.lift_check_delta_z = 0.0
        self.lift_check_link_indices: list[int] = []
        self.phase = "close"
        self.episode_reward_sum = 0.0
        self.episode_reward_terms: dict[str, float] = {}
        self.last_pregrasp: dict[str, Any] = {}

    def _load_config(self, config: dict | str) -> dict:
        if isinstance(config, dict):
            return config
        path = Path(config)
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _resolve_active_joint_indices(self, raw_active: list[int | str] | None) -> list[int]:
        if raw_active is None:
            return list(range(self.hand_dof))
        if not isinstance(raw_active, list) or not raw_active:
            raise ValueError("action_active_joints must be a non-empty list of joint indices or joint names")

        name_to_index = {name: idx for idx, name in enumerate(self.control_joint_names)}
        active: list[int] = []
        for item in raw_active:
            if isinstance(item, str):
                if item not in name_to_index:
                    raise ValueError(f"Unknown joint name in action_active_joints: {item}")
                idx = name_to_index[item]
            else:
                idx = int(item)
            if idx < 0 or idx >= self.hand_dof:
                raise ValueError(f"Invalid joint index in action_active_joints: {idx}")
            if idx not in active:
                active.append(idx)
        return active

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.world.reset()
        self.world.hand.reset_open_pose()
        self.lift_check_link_indices = self.world.hand.get_contact_link_indices(
            mode=self.lift_check_contact_mode,
            link_names=self.lift_check_contact_link_names,
        )

        obj_spec = self.sampler.sample()
        self.object_id = self.world.spawn_primitive_object(obj_spec)
        self.world.disable_pregrasp_collisions_temporarily(
            object_id=self.object_id,
            steps=self.pregrasp_no_collision_steps,
        )
        self.last_pregrasp = self._place_hand_to_pregrasp(obj_spec)
        if self.pregrasp_settle_steps > 0:
            self.world.step(self.pregrasp_settle_steps)
        self.world.wait_until_pregrasp_collisions_restored()

        pos, _ = p.getBasePositionAndOrientation(self.object_id)
        self.last_obj_pos = pos
        self.object_start_z = float(pos[2])
        self.step_count = 0
        self.lift_check_done = False
        self.lift_check_lifted = False
        self.lift_check_delta_z = 0.0
        self.phase = "close"
        self.episode_reward_sum = 0.0
        self.episode_reward_terms = {}

        obs = self._build_observation()
        info = {
            "shape": obj_spec.get("shape"),
            "size_cm": float(obj_spec.get("size_cm", 0.0)),
            "q_target": self.world.hand.get_q_target(),
            "q_measured": self.world.hand.get_q_measured(),
            "obs_delta": obs.tolist(),
            "pregrasp": dict(self.last_pregrasp),
        }
        return obs, info

    def step(self, action: list[float] | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_list = np.asarray(action, dtype=np.float32).reshape(-1).tolist()
        if len(action_list) != self.action_dim:
            raise ValueError(f"Expected action length {self.action_dim}, got {len(action_list)}")

        sparse_delta = clip_action_delta(action_list, self.max_delta)
        delta = [0.0] * self.hand_dof
        for a_idx, j_idx in enumerate(self.active_joint_indices):
            delta[j_idx] = float(sparse_delta[a_idx])
        if self.action_mode == "close_only":
            delta = [max(0.0, v) for v in delta]
        q_target = self.world.hand.apply_delta_q_target(delta, max_delta=self.max_delta)
        self.world.step(self.frame_skip)

        self.step_count += 1
        if self.lift_check_enabled and not self.lift_check_done and self.step_count >= self.close_phase_steps:
            self.phase = "lift"
            if self.lift_check_debug:
                print(
                    f"[lift_check] trigger step={self.step_count} "
                    f"reason=fixed_schedule close_phase_steps={self.close_phase_steps} dz={self.lift_check_dz}"
                )
            self._run_lift_check()
            self.phase = "lift_done"
        obs = self._build_observation()

        metrics = self._collect_metrics()
        reward, reward_terms = compute_reward_terms(metrics, self.reward_cfg, phase=self.phase)
        terminated = bool(metrics["tipped"] or self.lift_check_done)
        truncated = bool(self.step_count >= self.max_steps)
        self.episode_reward_sum += float(reward)
        for k, v in reward_terms.items():
            self.episode_reward_terms[k] = float(self.episode_reward_terms.get(k, 0.0) + float(v))

        info = dict(metrics)
        info["q_target"] = q_target
        info["q_measured"] = self.world.hand.get_q_measured()
        info["obs_delta"] = obs.tolist()
        info["reward"] = reward
        info["reward_terms"] = dict(reward_terms)
        info["phase"] = str(self.phase)
        info["close_phase_steps"] = int(self.close_phase_steps)
        info["contact_links_now"] = float(self._contact_link_count())
        info["lift_check_done"] = bool(self.lift_check_done)
        info["lift_check_delta_z"] = float(self.lift_check_delta_z)
        if (terminated or truncated) and self.reward_debug_print:
            parts = [f"{k}={self.episode_reward_terms[k]:.3f}" for k in sorted(self.episode_reward_terms.keys())]
            print(
                "[episode_reward] "
                f"steps={self.step_count} total={self.episode_reward_sum:.3f} "
                f"terminated={terminated} truncated={truncated} "
                + " ".join(parts)
            )
        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        q_target = np.asarray(self.world.hand.get_q_target(), dtype=np.float32)
        q_measured = np.asarray(self.world.hand.get_q_measured(), dtype=np.float32)
        delta = np.abs(q_target - q_measured)
        # Both vectors are already normalized [0, 1], so delta is in [0, 1].
        return np.clip(delta, 0.0, 1.0).astype(np.float32)

    def _load_hand_reference_point(self, yaml_path: Path, grasp_pose: str) -> dict[str, list[float]]:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        refs = dict(data.get("reference_points", {}))
        ref = refs.get(str(grasp_pose))
        if not isinstance(ref, dict):
            raise RuntimeError(f"Hand reference '{grasp_pose}' not found in {yaml_path}")
        return {
            "position_hand_xyz": [float(v) for v in ref.get("position_hand_xyz", [0.0, 0.0, 0.0])],
            "normal_hand_xyz": [float(v) for v in ref.get("normal_hand_xyz", [0.0, 0.0, 1.0])],
            "tangent_hand_xyz": [float(v) for v in ref.get("tangent_hand_xyz", [1.0, 0.0, 0.0])],
        }

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

    @staticmethod
    def _cross(a: list[float], b: list[float]) -> list[float]:
        return [
            float(a[1] * b[2] - a[2] * b[1]),
            float(a[2] * b[0] - a[0] * b[2]),
            float(a[0] * b[1] - a[1] * b[0]),
        ]

    @staticmethod
    def _normalize(v: list[float]) -> list[float]:
        n = math.sqrt(float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
        if n <= 1e-12:
            return [0.0, 0.0, 0.0]
        return [float(v[0] / n), float(v[1] / n), float(v[2] / n)]

    @classmethod
    def _orthonormal_tangent(cls, normal: list[float], tangent_hint: list[float]) -> list[float]:
        d = cls._dot(tangent_hint, normal)
        t = [
            float(tangent_hint[0] - d * normal[0]),
            float(tangent_hint[1] - d * normal[1]),
            float(tangent_hint[2] - d * normal[2]),
        ]
        t = cls._normalize(t)
        if cls._dot(t, t) < 1e-12:
            c = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
            t = cls._normalize(cls._cross(c, normal))
        return t

    @staticmethod
    def _mat_from_basis(x_axis: list[float], y_axis: list[float], z_axis: list[float]) -> list[list[float]]:
        return [
            [float(x_axis[0]), float(y_axis[0]), float(z_axis[0])],
            [float(x_axis[1]), float(y_axis[1]), float(z_axis[1])],
            [float(x_axis[2]), float(y_axis[2]), float(z_axis[2])],
        ]

    @staticmethod
    def _mat_transpose(a: list[list[float]]) -> list[list[float]]:
        return [
            [float(a[0][0]), float(a[1][0]), float(a[2][0])],
            [float(a[0][1]), float(a[1][1]), float(a[2][1])],
            [float(a[0][2]), float(a[1][2]), float(a[2][2])],
        ]

    @staticmethod
    def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        out = [[0.0, 0.0, 0.0] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                out[i][j] = float(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
        return out

    @staticmethod
    def _mat_vec_mul(mat: list[list[float]], vec: list[float]) -> list[float]:
        return [
            float(mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2]),
            float(mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2]),
            float(mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2]),
        ]

    @staticmethod
    def _quat_from_basis(tangent: list[float], bitangent: list[float], normal: list[float]) -> list[float]:
        m = [
            tangent[0],
            bitangent[0],
            normal[0],
            tangent[1],
            bitangent[1],
            normal[1],
            tangent[2],
            bitangent[2],
            normal[2],
        ]
        tr = m[0] + m[4] + m[8]
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[7] - m[5]) / s
            qy = (m[2] - m[6]) / s
            qz = (m[3] - m[1]) / s
        elif (m[0] > m[4]) and (m[0] > m[8]):
            s = math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0
            qw = (m[7] - m[5]) / s
            qx = 0.25 * s
            qy = (m[1] + m[3]) / s
            qz = (m[2] + m[6]) / s
        elif m[4] > m[8]:
            s = math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0
            qw = (m[2] - m[6]) / s
            qx = (m[1] + m[3]) / s
            qy = 0.25 * s
            qz = (m[5] + m[7]) / s
        else:
            s = math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0
            qw = (m[3] - m[1]) / s
            qx = (m[2] + m[6]) / s
            qy = (m[5] + m[7]) / s
            qz = 0.25 * s
        return [float(qx), float(qy), float(qz), float(qw)]

    @classmethod
    def _rotate_about_axis(cls, vec: list[float], axis_unit: list[float], angle_rad: float) -> list[float]:
        c = math.cos(float(angle_rad))
        s = math.sin(float(angle_rad))
        dot_va = cls._dot(vec, axis_unit)
        axv = cls._cross(axis_unit, vec)
        return [
            float(vec[0] * c + axv[0] * s + axis_unit[0] * dot_va * (1.0 - c)),
            float(vec[1] * c + axv[1] * s + axis_unit[1] * dot_va * (1.0 - c)),
            float(vec[2] * c + axv[2] * s + axis_unit[2] * dot_va * (1.0 - c)),
        ]

    @staticmethod
    def _shape_grasp_point_local(spec: dict) -> tuple[list[float], list[float], list[float]]:
        shape = str(spec.get("shape", "sphere"))
        size_m = float(spec.get("size_cm", 3.0)) / 100.0
        height_m = float(spec.get("height_cm", float(spec.get("size_cm", 3.0)))) / 100.0
        if shape in ("sphere", "cube"):
            hz = 0.5 * size_m
        elif shape in ("cylinder", "rect_cylinder"):
            hz = 0.5 * height_m
        else:
            hz = 0.5 * size_m
        return [0.0, 0.0, hz], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]

    @classmethod
    def _pose_world_from_obj_local(
        cls,
        obj_pos_world: list[float],
        obj_quat_world: list[float],
        p_local: list[float],
        n_local: list[float],
        t_local: list[float],
    ) -> tuple[list[float], list[float], list[float]]:
        rot9 = p.getMatrixFromQuaternion(obj_quat_world)
        r = [
            [float(rot9[0]), float(rot9[1]), float(rot9[2])],
            [float(rot9[3]), float(rot9[4]), float(rot9[5])],
            [float(rot9[6]), float(rot9[7]), float(rot9[8])],
        ]
        p_off = cls._mat_vec_mul(r, p_local)
        p_world = [
            float(obj_pos_world[0] + p_off[0]),
            float(obj_pos_world[1] + p_off[1]),
            float(obj_pos_world[2] + p_off[2]),
        ]
        n_world = cls._normalize(cls._mat_vec_mul(r, n_local))
        t_world = cls._orthonormal_tangent(n_world, cls._mat_vec_mul(r, t_local))
        return p_world, n_world, t_world

    def _compute_pregrasp_pose(self, obj_spec: dict) -> tuple[list[float], list[float], dict[str, Any]]:
        obj_pos, obj_quat = p.getBasePositionAndOrientation(self.object_id)
        gp_local, n_local, t_local = self._shape_grasp_point_local(obj_spec)
        gp_world, n_world, t_world = self._pose_world_from_obj_local(
            [float(v) for v in obj_pos],
            [float(v) for v in obj_quat],
            gp_local,
            n_local,
            t_local,
        )
        if abs(float(self.pregrasp_twist_deg)) > 1e-9:
            t_world = self._orthonormal_tangent(
                n_world,
                self._rotate_about_axis(t_world, n_world, math.radians(float(self.pregrasp_twist_deg))),
            )

        d_m = float(self.pregrasp_distance_mm) / 1000.0
        target_ref_world = [
            float(gp_world[0] + d_m * n_world[0]),
            float(gp_world[1] + d_m * n_world[1]),
            float(gp_world[2] + d_m * n_world[2]),
        ]

        p_hand = [float(v) for v in self.pregrasp_hand_ref["position_hand_xyz"]]
        n_hand = self._normalize([float(v) for v in self.pregrasp_hand_ref["normal_hand_xyz"]])
        t_hand = self._orthonormal_tangent(n_hand, [float(v) for v in self.pregrasp_hand_ref["tangent_hand_xyz"]])
        b_hand = self._normalize(self._cross(n_hand, t_hand))
        t_hand = self._normalize(self._cross(b_hand, n_hand))

        n_target = [-n_world[0], -n_world[1], -n_world[2]]
        t_target = self._orthonormal_tangent(n_target, t_world)
        b_target = self._normalize(self._cross(n_target, t_target))
        t_target = self._normalize(self._cross(b_target, n_target))

        r_world_target = self._mat_from_basis(t_target, b_target, n_target)
        r_hand_local = self._mat_from_basis(t_hand, b_hand, n_hand)
        r_hand_world = self._mat_mul(r_world_target, self._mat_transpose(r_hand_local))
        hand_quat_world = self._quat_from_basis(
            [r_hand_world[0][0], r_hand_world[1][0], r_hand_world[2][0]],
            [r_hand_world[0][1], r_hand_world[1][1], r_hand_world[2][1]],
            [r_hand_world[0][2], r_hand_world[1][2], r_hand_world[2][2]],
        )
        hand_offset_world = self._mat_vec_mul(r_hand_world, p_hand)
        hand_base_world = [
            float(target_ref_world[0] - hand_offset_world[0]),
            float(target_ref_world[1] - hand_offset_world[1]),
            float(target_ref_world[2] - hand_offset_world[2]),
        ]
        debug = {
            "grasp_type": str(self.grasp_type),
            "distance_mm": float(self.pregrasp_distance_mm),
            "twist_deg": float(self.pregrasp_twist_deg),
            "target_ref_world": [float(v) for v in target_ref_world],
            "hand_base_world": [float(v) for v in hand_base_world],
            "hand_quat_world": [float(v) for v in hand_quat_world],
            "normal_world": [float(v) for v in n_world],
            "tangent_world": [float(v) for v in t_world],
        }
        return hand_base_world, hand_quat_world, debug

    def _place_hand_to_pregrasp(self, obj_spec: dict) -> dict[str, Any]:
        hand_pos, hand_quat, debug = self._compute_pregrasp_pose(obj_spec)
        if self.world.robot_type == "free_hand":
            self.world.set_free_hand_pose(hand_pos, hand_quat)
            return debug
        return {"skipped": True, "reason": f"robot_type={self.world.robot_type}", **debug}

    def close(self) -> None:
        self.world.close()

    def _contact_link_count(self) -> int:
        count = 0
        for link_index in self.lift_check_link_indices:
            if p.getContactPoints(
                bodyA=self.world.hand_id,
                bodyB=self.object_id,
                linkIndexA=link_index,
            ):
                count += 1
        return count

    def _run_lift_check(self) -> None:
        start_z = self._object_z()
        self.world.lift_grasping_hand_blocking(self.lift_check_dz)
        if self.lift_check_hold_steps > 0:
            self.world.step(self.lift_check_hold_steps)
        end_z = self._object_z()
        self.lift_check_delta_z = float(end_z - start_z)
        self.lift_check_lifted = bool(self.lift_check_delta_z >= self.reward_cfg.lift_success_z)
        self.lift_check_done = True

    def _object_z(self) -> float:
        pos, _ = p.getBasePositionAndOrientation(self.object_id)
        return float(pos[2])

    def _collect_metrics(self) -> dict:
        contact = len(p.getContactPoints(bodyA=self.world.hand_id, bodyB=self.object_id)) > 0

        contact_links = self._contact_link_count()

        pos, quat = p.getBasePositionAndOrientation(self.object_id)
        com_shift = 0.0
        if self.last_obj_pos is not None:
            dx = float(pos[0] - self.last_obj_pos[0])
            dy = float(pos[1] - self.last_obj_pos[1])
            dz = float(pos[2] - self.last_obj_pos[2])
            com_shift = (dx * dx + dy * dy + dz * dz) ** 0.5
        self.last_obj_pos = pos

        obj_lin_vel, obj_ang_vel = p.getBaseVelocity(self.object_id)
        hand_lin_vel, _ = p.getBaseVelocity(self.world.hand_id)
        ang_speed = (float(obj_ang_vel[0]) ** 2 + float(obj_ang_vel[1]) ** 2 + float(obj_ang_vel[2]) ** 2) ** 0.5
        slip_vx = float(obj_lin_vel[0]) - float(hand_lin_vel[0])
        slip_vy = float(obj_lin_vel[1]) - float(hand_lin_vel[1])
        slip_vz = float(obj_lin_vel[2]) - float(hand_lin_vel[2])
        slip_speed = (slip_vx * slip_vx + slip_vy * slip_vy + slip_vz * slip_vz) ** 0.5

        rpy = p.getEulerFromQuaternion(quat)
        tipped = abs(float(rpy[0])) > self.reward_cfg.max_tilt_rad or abs(float(rpy[1])) > self.reward_cfg.max_tilt_rad

        end_z = float(pos[2])
        lifted_geom = (end_z - self.object_start_z) >= self.reward_cfg.lift_success_z
        lifted = bool(self.lift_check_lifted) if self.lift_check_done else bool(lifted_geom)
        q_measured = self.world.hand.get_q_measured()
        hand_closure_mean = float(sum(q_measured) / max(1, len(q_measured)))
        overgrip_excess = max(0.0, hand_closure_mean - float(self.reward_cfg.overgrip_threshold))

        return {
            "contact": bool(contact),
            "contact_links": float(contact_links),
            "com_shift": float(com_shift),
            "ang_speed": float(ang_speed),
            "slip_speed": float(slip_speed),
            "lifted": bool(lifted),
            "tipped": bool(tipped),
            "hand_closure_mean": float(hand_closure_mean),
            "overgrip_excess": float(overgrip_excess),
            "start_z": float(self.object_start_z),
            "end_z": float(end_z),
        }

