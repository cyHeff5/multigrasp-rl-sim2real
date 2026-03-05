# multigrasp-rl-sim2real

Minimal scaffold for multimodal in-hand grasp RL with AR10 + Sawyer.

## Scope
- Pose-specific in-hand RL policies (Tripod, Medium Wrap, Lateral Pinch)
- Observation constrained to `q_measured`
- Action as joint target increments `delta_q_target`
- Sim training in PyBullet, deployment on real AR10 through a shared interface contract

## Quick start
1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Start with `scripts/train_tripod.ps1`.
