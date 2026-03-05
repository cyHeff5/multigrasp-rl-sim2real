import argparse
from pathlib import Path

import yaml

from src.envs.grasp_env import GraspEnv


def _load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sim-config", default="configs/sim/tripod.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
    except Exception as exc:
        raise RuntimeError(
            "stable-baselines3 is required for evaluation. Install with: pip install stable-baselines3"
        ) from exc

    # Always force GUI for evaluation runs.
    sim_cfg = _load_yaml(args.sim_config)
    sim_cfg.setdefault("world", {})
    sim_cfg["world"]["gui"] = True

    env = GraspEnv(sim_cfg)
    model = SAC.load(args.checkpoint, env=env, device="auto")

    successes = 0
    total_reward = 0.0

    for ep in range(max(1, int(args.episodes))):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        ep_success = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
            if bool(info.get("lifted", False)):
                ep_success = True

        successes += int(ep_success)
        total_reward += ep_reward

    success_rate = successes / float(max(1, int(args.episodes)))
    avg_reward = total_reward / float(max(1, int(args.episodes)))
    print(f"episodes={args.episodes}")
    print(f"successes={successes}")
    print(f"success_rate={success_rate:.4f}")
    print(f"avg_reward={avg_reward:.4f}")

    env.close()


if __name__ == "__main__":
    main()

