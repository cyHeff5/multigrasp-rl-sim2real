import argparse
from pathlib import Path

import yaml

from src.envs.grasp_env import GraspEnv


def _load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--output-dir", default="artifacts/models")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--gui", action="store_true", help="Force PyBullet GUI for this run")
    args = parser.parse_args()

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
    except Exception as exc:
        raise RuntimeError(
            "stable-baselines3 is required for training. Install with: pip install stable-baselines3"
        ) from exc

    train_cfg = _load_yaml(args.train_config)
    algo = str(train_cfg.get("algorithm", "sac")).lower()
    if algo != "sac":
        raise ValueError(f"Unsupported algorithm: {algo}. This trainer currently supports only 'sac'.")

    grasp_type = str(train_cfg.get("grasp_type", "tripod"))
    total_steps = int(args.total_steps) if args.total_steps is not None else int(train_cfg.get("total_steps", 200000))
    seed = int(train_cfg.get("seed", 42))
    learning_rate = float(train_cfg.get("learning_rate", 3e-4))
    batch_size = int(train_cfg.get("batch_size", 256))
    buffer_size = int(train_cfg.get("buffer_size", 100000))

    sim_cfg = _load_yaml(args.sim_config)
    if args.gui:
        sim_cfg.setdefault("world", {})
        sim_cfg["world"]["gui"] = True

    env = GraspEnv(sim_cfg)
    env = Monitor(env)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        verbose=1,
        device="auto",
    )

    model.learn(total_timesteps=total_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{grasp_type}_latest"
    model.save(str(out_path))
    print(f"saved_model={out_path}.zip")

    env.close()


if __name__ == "__main__":
    main()

