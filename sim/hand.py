# AR10 controller (10D -> joint targets).
import pybullet as p
import time

class AR10Helper:
    def __init__(self, hand_id):
        self.hand_id = hand_id

        # Controllable servo joints
        self.control_joint_names = [
            "servo0", "servo1",
            "servo2", "servo3",
            "servo4", "servo5",
            "servo6", "servo7",
            "servo8", "servo9"
        ]

        # Mimic map: these joints follow the motion of a controlling servo
        self.mimic_map = {
            "servo3": ["tip1"],
            "servo5": ["tip2"],
            "servo7": ["tip3"],
            "servo9": ["tip4"]
        }

        # Joint name → PyBullet joint index
        self.joint_name_to_index = {
            "servo0": 28, "servo1": 31, "servo2": 3,  "servo3": 4,
            "servo4": 9,  "servo5": 10, "servo6": 15, "servo7": 16,
            "servo8": 21, "servo9": 22,
            "tip1": 5, "tip2": 11, "tip3": 17, "tip4": 23,
            "thumb2joint": 29
        }

        # Joint limits for all control and mimic joints
        self.joint_limits = {}
        self._load_joint_limits(self.control_joint_names)
        for mimics in self.mimic_map.values():
            self._load_joint_limits(mimics)

    def _load_joint_limits(self, joint_names):
        """Retrieve joint limits for a list of joint names."""
        for name in joint_names:
            if name in self.joint_limits:
                continue  # already loaded
            joint_index = self.joint_name_to_index[name]
            info = p.getJointInfo(self.hand_id, joint_index)
            lower, upper = info[8], info[9]
            if lower >= upper:
                lower, upper = 0.0, 1.3  # fallback default
            self.joint_limits[name] = (lower, upper)

    def apply_joint_command_vector(self, vector, force=10.0):
        """Applies a normalized [0,1] command vector to all controllable joints."""
        assert len(vector) == len(self.control_joint_names), \
            f"Input vector must have {len(self.control_joint_names)} values."

        joint_targets = {}

        for i, name in enumerate(self.control_joint_names):
            norm = vector[i]
            assert 0.0 <= norm <= 1.0, f"Value out of range: {norm}"

            min_val, max_val = self.joint_limits[name]
            value = min_val + norm * (max_val - min_val)
            joint_targets[name] = value

            # Handle mimic joints
            for mimic_name in self.mimic_map.get(name, []):
                mimic_min, mimic_max = self.joint_limits[mimic_name]
                mimic_value = mimic_min + norm * (mimic_max - mimic_min)
                joint_targets[mimic_name] = mimic_value

        # Apply all joint targets in PyBullet
        for joint_name, target in joint_targets.items():
            joint_index = self.joint_name_to_index[joint_name]
            p.setJointMotorControl2(
                bodyIndex=self.hand_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=force
            )

    def get_current_joint_command_vector(self):
        """Returns normalized [0,1] values of all controllable joints."""
        positions = []
        for name in self.control_joint_names:
            joint_index = self.joint_name_to_index[name]
            pos = p.getJointState(self.hand_id, joint_index)[0]
            min_val, max_val = self.joint_limits[name]
            norm_pos = (pos - min_val) / (max_val - min_val)
            positions.append(norm_pos)
        return positions

    def get_base_pose(self):
        """Returns the base (world) pose of the hand body as (position, quaternion)."""
        return p.getBasePositionAndOrientation(self.hand_id)

    def reset_pose(self):
        """Resets all servos to their open (0.0) position."""
        print("[AR10] Resetting pose to open hand...")
        self.apply_joint_command_vector([0.0] * len(self.control_joint_names))
        self.step(steps=100, real_time=True)

    def open_all_fingers(self):
        """Opens all fingers (0.0 for each servo)."""
        print("[AR10] Opening all fingers...")
        self.apply_joint_command_vector([0.0] * len(self.control_joint_names))

    def close_all_fingers(self):
        """Closes all fingers (1.0 for each servo)."""
        print("[AR10] Closing all fingers...")
        self.apply_joint_command_vector([1.0] * len(self.control_joint_names))

    def get_tip_link_indices(self):
        """Returns the link indices of the fingertip links (for collision detection)."""
        tip_names = ["tip1", "tip2", "tip3", "tip4"]
        return [self.joint_name_to_index[name] for name in tip_names]

    def set_finger_friction(self, lateral=1.2, spinning=1.0, rolling=0.0008):
        """Sets friction parameters for all links."""
        print(f"[AR10] Setting finger friction: lateral={lateral}, spinning={spinning}, rolling={rolling}")
        for link in self.joint_name_to_index.values():
            p.changeDynamics(
                self.hand_id, link,
                lateralFriction=lateral,
                spinningFriction=spinning,
                rollingFriction=rolling,
                frictionAnchor=True,
            )

    def step(self, steps=60, real_time=False):
        """Performs simulation steps (optional real-time sleep)."""
        for _ in range(steps):
            p.stepSimulation()
            if real_time:
                time.sleep(1. / 240)
