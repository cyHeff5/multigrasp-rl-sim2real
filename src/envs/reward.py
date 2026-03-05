from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    # Close-phase shaping (small magnitude)
    close_contact_link_reward: float = 0.02
    close_com_shift_penalty: float = 0.7
    close_slip_penalty: float = 0.4
    close_ang_vel_penalty: float = 0.02
    close_overgrip_penalty: float = 0.10
    close_tip_over_penalty: float = -1.0
    # Lift-phase outcome (dominant magnitude)
    lift_success_reward: float = 5.0
    lift_fail_penalty: float = -2.0
    lift_slip_penalty: float = 1.0
    lift_ang_vel_penalty: float = 0.05
    lift_tip_over_penalty: float = -2.0
    # Shared thresholds
    max_tilt_rad: float = 0.8
    lift_success_z: float = 0.03
    overgrip_threshold: float = 0.90
    overgrip_penalty: float = 0.20


def compute_reward_terms(metrics: dict, cfg: RewardConfig | None = None, phase: str = "close") -> tuple[float, dict]:
    cfg = cfg or RewardConfig()
    tipped = bool(metrics.get("tipped", False))
    lifted = bool(metrics.get("lifted", False))
    contact_links = float(metrics.get("contact_links", 0.0))
    com_shift = float(metrics.get("com_shift", 0.0))
    ang_speed = float(metrics.get("ang_speed", 0.0))
    slip_speed = float(metrics.get("slip_speed", 0.0))
    overgrip_excess = float(metrics.get("overgrip_excess", 0.0))

    phase = str(phase).lower()
    reward = 0.0
    terms: dict[str, float] = {}
    if phase == "close":
        terms["close_contact"] = contact_links * cfg.close_contact_link_reward
        terms["close_com_shift"] = -cfg.close_com_shift_penalty * com_shift
        terms["close_ang_vel"] = -cfg.close_ang_vel_penalty * ang_speed
        terms["close_slip"] = -cfg.close_slip_penalty * slip_speed
        terms["close_overgrip"] = -cfg.close_overgrip_penalty * overgrip_excess
        if tipped:
            terms["close_tip_over"] = cfg.close_tip_over_penalty
        reward = float(sum(terms.values()))
        return reward, terms

    # Lift phase and post-lift outcome.
    terms["lift_outcome"] = cfg.lift_success_reward if lifted else cfg.lift_fail_penalty
    terms["lift_ang_vel"] = -cfg.lift_ang_vel_penalty * ang_speed
    terms["lift_slip"] = -cfg.lift_slip_penalty * slip_speed
    terms["lift_overgrip"] = -cfg.overgrip_penalty * overgrip_excess
    if tipped:
        terms["lift_tip_over"] = cfg.lift_tip_over_penalty
    reward = float(sum(terms.values()))
    return reward, terms


def compute_reward(metrics: dict, cfg: RewardConfig | None = None, phase: str = "close") -> float:
    reward, _ = compute_reward_terms(metrics, cfg=cfg, phase=phase)
    return reward
