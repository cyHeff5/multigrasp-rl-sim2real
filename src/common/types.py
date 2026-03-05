from dataclasses import dataclass
from typing import List


@dataclass
class StepResult:
    observation: List[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict
