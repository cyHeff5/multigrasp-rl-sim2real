import math

import pybullet as p


def _resolve_link_index(robot_id: int, link_name: str) -> int:
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        name = info[12].decode("utf-8")
        if name == link_name:
            return j
    raise RuntimeError(f"Link not found: {link_name}")


def mount_hand_to_arm(
    robot_id: int,
    hand_id: int,
    ee_link="right_l6",
    pre_mount_pos=(0.5, 0.1, 0.3),
    pre_mount_rpy=(math.pi, 0.0, 0.0),
    child_link_index=0,
    child_frame_offset=(0.0, 0.0, -0.03),
    child_frame_rpy=(0.0, 0.0, 0.0),
    parent_frame_pos=(0.0, 0.0, 0.0),
    parent_frame_rpy=(0.0, 0.0, 0.0),
    max_force=1e6,
    erp=0.9,
) -> int:
    robot_id = int(robot_id)
    hand_id = int(hand_id)
    if isinstance(ee_link, str):
        ee_link = _resolve_link_index(robot_id, ee_link)

    hand_pre_orn = p.getQuaternionFromEuler([float(x) for x in pre_mount_rpy])
    p.resetBasePositionAndOrientation(
        hand_id,
        [float(x) for x in pre_mount_pos],
        hand_pre_orn,
    )

    child_orn = p.getQuaternionFromEuler([float(x) for x in child_frame_rpy])
    parent_orn = p.getQuaternionFromEuler([float(x) for x in parent_frame_rpy])

    cid = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=int(ee_link),
        childBodyUniqueId=hand_id,
        childLinkIndex=int(child_link_index),
        jointType=p.JOINT_FIXED,
        jointAxis=[0.0, 0.0, 0.0],
        parentFramePosition=[float(x) for x in parent_frame_pos],
        childFramePosition=[float(x) for x in child_frame_offset],
        parentFrameOrientation=parent_orn,
        childFrameOrientation=child_orn,
    )

    try:
        p.changeConstraint(cid, maxForce=float(max_force), erp=float(erp))
    except TypeError:
        p.changeConstraint(cid, maxForce=float(max_force))

    return cid


def apply_hand_friction(
    hand_id: int,
    lateral=1.3,
    spinning=1.0,
    rolling=0.001,
    use_anchor=True,
) -> None:
    for link_index in range(-1, p.getNumJoints(int(hand_id))):
        p.changeDynamics(
            int(hand_id),
            link_index,
            lateralFriction=float(lateral),
            spinningFriction=float(spinning),
            rollingFriction=float(rolling),
            frictionAnchor=bool(use_anchor),
        )


def apply_physics_defaults(
    num_solver_iterations=150,
    fixed_time_step=1.0 / 240.0,
    num_substeps=4,
) -> None:
    p.setPhysicsEngineParameter(
        numSolverIterations=int(num_solver_iterations),
        fixedTimeStep=float(fixed_time_step),
        numSubSteps=int(num_substeps),
    )
