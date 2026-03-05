from pathlib import Path


_HERE = Path(__file__).resolve().parent
ASSETS_DIR = _HERE / "assets"

UR5_DIR = ASSETS_DIR / "ur5"
UR5_URDF = UR5_DIR / "ur5.urdf"

SAWYER_DIR = ASSETS_DIR / "sawyer_robot" / "sawyer_description" / "urdf"
SAWYER_URDF = SAWYER_DIR / "sawyer_pybullet.urdf"

AR10_DIR = ASSETS_DIR / "ar10_description"
AR10_URDF = AR10_DIR / "urdf" / "ar10.urdf"

BENCHMARK_DIR = ASSETS_DIR / "benchmark_parts"


def benchmark_part_urdf(part_id: int):
    part_id = int(part_id)
    if not (1 <= part_id <= 14):
        raise ValueError("part_id must be in [1, 14]")
    return BENCHMARK_DIR / f"benchmark_part_{part_id}.urdf"
