# sim/mounting.py
# Mount AR10 hand to Sawyer end-effector (right_l6) with a fixed constraint.
#
# Behavior matches your example:
# - hand is flipped 180 deg around X before mounting
# - fixed constraint at EE origin
# - child offset [0,0,-0.01]
# - child link index = 0

import math

import pybullet as p


def mount_hand_to_arm(
    robot_id,
    hand_id,
    ee_link="right_l6",
    *,
    # "spawn before mounting" pose (arbitrary, just so it exists somewhere visible)
    pre_mount_pos=(0.5, 0.1, 0.3),
    pre_mount_rpy=(math.pi, 0.0, 0.0),   # 180 deg flip around X (matches your example)
    # mounting transform (matches your example)
    child_link_index=0,
    child_frame_offset=(0.0, 0.0, -0.01),
    child_frame_rpy=(0.0, 0.0, 0.0),
    parent_frame_pos=(0.0, 0.0, 0.0),
    parent_frame_rpy=(0.0, 0.0, 0.0),
    # constraint stiffness
    max_force=1e6,
    erp=0.9,
):
    """
    Mounts the AR10 hand rigidly to the Sawyer end-effector (right_l6) with a fixed constraint.

    This mirrors the mounting logic from your sim.py / example:
      - reset hand base pose (pre-mount)
      - flip hand 180 deg around X
      - create JOINT_FIXED constraint from arm ee_link to hand child_link_index
      - child_frame_offset = [0,0,-0.01]

    Returns:
        constraint_id (int)
    """
    robot_id = int(robot_id)
    hand_id = int(hand_id)
    if isinstance(ee_link, str):
        ee_link = _resolve_link_index(robot_id, ee_link)
    ee_link = int(ee_link)
    child_link_index = int(child_link_index)

    # 1) Put the hand somewhere and flip it 180 deg around X (same as in your example)
    hand_pre_orn = p.getQuaternionFromEuler([float(x) for x in pre_mount_rpy])
    p.resetBasePositionAndOrientation(
        hand_id,
        [float(x) for x in pre_mount_pos],
        hand_pre_orn,
    )

    # 2) Build frame orientations
    child_orn = p.getQuaternionFromEuler([float(x) for x in child_frame_rpy])
    parent_orn = p.getQuaternionFromEuler([float(x) for x in parent_frame_rpy])

    # 3) Create fixed constraint (same parent/child frames as your example)
    cid = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=ee_link,
        childBodyUniqueId=hand_id,
        childLinkIndex=child_link_index,
        jointType=p.JOINT_FIXED,
        jointAxis=[0.0, 0.0, 0.0],
        parentFramePosition=[float(x) for x in parent_frame_pos],
        childFramePosition=[float(x) for x in child_frame_offset],
        parentFrameOrientation=parent_orn,
        childFrameOrientation=child_orn,
    )

    # 4) Make it stiff (helps with lift tests)
    try:
        p.changeConstraint(cid, maxForce=float(max_force), erp=float(erp))
    except TypeError:
        # Some builds don't accept 'erp' here.
        p.changeConstraint(cid, maxForce=float(max_force))

    return cid


def _resolve_link_index(robot_id, link_name):
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        name = info[12].decode("utf-8")
        if name == link_name:
            return j
    raise RuntimeError("Link not found: {}".format(link_name))


def apply_hand_friction_like_sim_py(
    hand_id,
    *,
    lateral=1.3,
    spinning=1.0,
    rolling=0.001,
    use_anchor=True,
):
    """
    Applies the same per-link friction settings you used in sim.py / example.
    """
    hand_id = int(hand_id)
    for link_index in range(p.getNumJoints(hand_id)):
        p.changeDynamics(
            hand_id,
            link_index,
            lateralFriction=float(lateral),
            spinningFriction=float(spinning),
            rollingFriction=float(rolling),
            frictionAnchor=bool(use_anchor),
        )


def apply_physics_params_like_sim_py(*, num_solver_iterations=150, fixed_time_step=1.0 / 240.0, num_substeps=4):
    """
    Applies the same physics parameters you used in sim.py / example.
    """
    p.setPhysicsEngineParameter(
        numSolverIterations=int(num_solver_iterations),
        fixedTimeStep=float(fixed_time_step),
        numSubSteps=int(num_substeps),
    )
