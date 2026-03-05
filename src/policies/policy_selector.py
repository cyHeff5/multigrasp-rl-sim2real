from pathlib import Path


def select_policy(grasp_type: str, model_dir: str = "artifacts/models") -> Path:
    return Path(model_dir) / f"{grasp_type}_latest.pt"
