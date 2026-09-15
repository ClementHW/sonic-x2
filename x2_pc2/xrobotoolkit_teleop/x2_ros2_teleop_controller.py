import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import meshcat.transformations as tf
import numpy as np

from xrobotoolkit_teleop.common.base_hardware_teleop_controller import HardwareTeleopController
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD

try:
    from xrobotoolkit_teleop.hardware.interface.ros2_camera import Ros2CameraInterface
except ImportError:
    Ros2CameraInterface = None

try:
    import rclpy
    from aimdk_msgs.msg import (
        HandCommand,
        HandCommandArray,
        HandType,
        JointCommand,
        JointCommandArray,
        JointStateArray,
        MessageHeader,
    )
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import Twist
except ImportError as exc:  # pragma: no cover - depends on ROS2 runtime
    rclpy = None
    HandCommand = None
    HandCommandArray = None
    HandType = None
    JointCommand = None
    JointCommandArray = None
    JointStateArray = None
    MessageHeader = None
    Twist = None
    MultiThreadedExecutor = None
    Node = object
    DurabilityPolicy = None
    HistoryPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None
    _ROS2_IMPORT_ERROR = exc
else:
    _ROS2_IMPORT_ERROR = None

try:
    import ruckig
except ImportError as exc:  # pragma: no cover - depends on runtime environment
    ruckig = None
    _RUCKIG_IMPORT_ERROR = exc
else:
    _RUCKIG_IMPORT_ERROR = None


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_X2_UPPER_BODY_URDF_PATH = str(_find_repo_root() / "X2_URDF" / "x2_upper_body_no_waist.urdf")
DEFAULT_SCALE_FACTOR = 1.2

DEFAULT_X2_MANIPULATOR_CONFIG = {
    "left_arm": {
        "link_name": "left_wrist_roll_link",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "left_trigger",
            "joint_names": ["left_hand"],
            "open_pos": [1.0],
            "close_pos": [0.0],
        },
        "control_point_offset_xyz": [0.0, 0.0, -0.1],
        "activation_on_frames": 1,
        "activation_off_frames": 4,
        "input_linear_deadband_m": 0.003,
        "input_angular_deadband_rad": 0.04,
        "input_position_alpha": 0.35,
        "input_rotation_alpha": 0.25,
        "max_target_linear_step_m": 0.03,
        "max_target_angular_step_rad": 0.35,
        "workspace_min_z": -0.2,
    },
    "right_arm": {
        "link_name": "right_wrist_roll_link",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": ["right_hand"],
            "open_pos": [1.0],
            "close_pos": [0.0],
        },
        "control_point_offset_xyz": [0.0, 0.0, -0.1],
        "activation_on_frames": 1,
        "activation_off_frames": 4,
        "input_linear_deadband_m": 0.003,
        "input_angular_deadband_rad": 0.04,
        "input_position_alpha": 0.35,
        "input_rotation_alpha": 0.25,
        "max_target_linear_step_m": 0.03,
        "max_target_angular_step_rad": 0.35,
        "workspace_min_z": 0.0,
    },
}

DEFAULT_ARM_STATE_TOPIC = "/aima/hal/joint/arm/state"
DEFAULT_UPPER_BODY_COMMAND_TOPIC = "/upper_body/teleop_joint_states"
# Direct hardware takeover is the default X2 path.  The gravity-compensation
# node can still be used explicitly by passing DEFAULT_UPPER_BODY_COMMAND_TOPIC
# (or --arm-command-topic /upper_body/teleop_joint_states).
DEFAULT_ARM_COMMAND_TOPIC = "/aima/hal/joint/arm/command"
DEFAULT_HEAD_STATE_TOPIC = "/aima/hal/joint/head/state"
DEFAULT_HEAD_COMMAND_TOPIC = "/aima/hal/joint/head/command"
DEFAULT_HAND_COMMAND_TOPIC = "/aima/hal/joint/hand/command"
DEFAULT_WALK_COMMAND_TOPIC = "/x2/cmd_vel"
DEFAULT_X2_CAMERA_COLOR_TOPICS = (
    "head_front=/aima/hal/sensor/rgbd_head_front/rgb_image/compressed,"
    "right_wrist=/right/rgb/image_compressed,"
    "left_wrist=/left/rgb/image_compressed"
)
DEFAULT_X2_CAMERA_DEPTH_TOPICS = ""

EXPECTED_X2_CAMERA_NAMES = ("head_front", "right_wrist", "left_wrist")
DEFAULT_CAMERA_COMPLETENESS_WAIT_S = 2.0

if QoSProfile is not None:
    SUBSCRIBER_QOS = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )

    PUBLISHER_QOS = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )
else:
    SUBSCRIBER_QOS = None
    PUBLISHER_QOS = None

ARM_MODEL_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]

HEAD_MODEL_JOINT_NAMES = [
    "head_yaw_joint",
    "head_pitch_joint",
]

LEFT_ARM_MODEL_JOINT_NAMES = ARM_MODEL_JOINT_NAMES[:7]
RIGHT_ARM_MODEL_JOINT_NAMES = ARM_MODEL_JOINT_NAMES[7:]

# The real robot ROS2 topics exposed in motocontrol.py use a slightly different naming
# scheme from the X2 upper-body URDF used for IK. Keep the mapping configurable so it can
# be swapped out if the deployed firmware names already match the URDF.
DEFAULT_MODEL_TO_HARDWARE_JOINT_MAP = {
    "left_shoulder_pitch_joint": "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint": "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint": "left_shoulder_yaw_joint",
    "left_elbow_joint": "left_elbow_joint",
    "left_wrist_yaw_joint": "left_wrist_yaw_joint",
    "left_wrist_pitch_joint": "left_wrist_pitch_joint",
    "left_wrist_roll_joint": "left_wrist_roll_joint",
    "right_shoulder_pitch_joint": "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint": "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint": "right_shoulder_yaw_joint",
    "right_elbow_joint": "right_elbow_joint",
    "right_wrist_yaw_joint": "right_wrist_yaw_joint",
    "right_wrist_pitch_joint": "right_wrist_pitch_joint",
    "right_wrist_roll_joint": "right_wrist_roll_joint",
    "head_yaw_joint": "head_yaw_joint",
    "head_pitch_joint": "head_pitch_joint",
}


@dataclass(frozen=True)
class JointCommandSpec:
    lower_limit: float
    upper_limit: float
    kp: float
    kd: float


# Limits and gains come from motocontrol.py so the hardware topic interface stays aligned
# with the real robot side.
DEFAULT_HARDWARE_COMMAND_SPECS: Dict[str, JointCommandSpec] = {
    "left_shoulder_pitch_joint": JointCommandSpec(-3.08, 2.04, 40.0, 4.0),
    "left_shoulder_roll_joint": JointCommandSpec(-0.061, 2.993, 40.0, 4.0),
    "left_shoulder_yaw_joint": JointCommandSpec(-2.556, 2.556, 40.0, 4.0),
    "left_elbow_joint": JointCommandSpec(-2.3556, 0.0, 40.0, 4.0),
    "left_wrist_yaw_joint": JointCommandSpec(-2.556, 2.556, 20.0, 2.0),
    "left_wrist_pitch_joint": JointCommandSpec(-0.558, 0.558, 20.0, 2.0),
    "left_wrist_roll_joint": JointCommandSpec(-1.571, 0.724, 20.0, 2.0),
    "right_shoulder_pitch_joint": JointCommandSpec(-3.08, 2.04, 40.0, 4.0),
    "right_shoulder_roll_joint": JointCommandSpec(-2.993, 0.061, 40.0, 4.0),
    "right_shoulder_yaw_joint": JointCommandSpec(-2.556, 2.556, 40.0, 4.0),
    "right_elbow_joint": JointCommandSpec(-2.3556, 0.0, 40.0, 4.0),
    "right_wrist_yaw_joint": JointCommandSpec(-2.556, 2.556, 20.0, 2.0),
    "right_wrist_pitch_joint": JointCommandSpec(-0.558, 0.558, 20.0, 2.0),
    "right_wrist_roll_joint": JointCommandSpec(-0.724, 1.571, 20.0, 2.0),
    "head_yaw_joint": JointCommandSpec(-0.366, 0.366, 20.0, 2.0),
    "head_pitch_joint": JointCommandSpec(-0.3838, 0.3838, 20.0, 2.0),
}


class Ros2JointGroupInterface(Node):
    def __init__(
        self,
        node_name: str,
        state_topic: str,
        command_topic: str,
        joint_names: Iterable[str],
        enable_state_subscription: bool = True,
        enable_effort: bool = False,
    ):
        super().__init__(node_name)
        self.joint_names = list(joint_names)
        self._state_subscription_enabled = bool(enable_state_subscription)
        self._enable_effort = bool(enable_effort)
        self._lock = threading.Lock()
        self._state_condition = threading.Condition(self._lock)
        self._positions = {joint_name: 0.0 for joint_name in self.joint_names}
        self._velocities = {joint_name: 0.0 for joint_name in self.joint_names}
        self._efforts = {joint_name: 0.0 for joint_name in self.joint_names}
        self._state_event = threading.Event()
        self._state_seq = 0
        self.timestamp = -1.0

        if self._state_subscription_enabled:
            self._sub = self.create_subscription(
                JointStateArray,
                state_topic,
                self._state_callback,
                SUBSCRIBER_QOS,
            )
        else:
            self._sub = None
        self._pub = self.create_publisher(
            JointCommandArray,
            command_topic,
            PUBLISHER_QOS,
        )

    def _iter_joint_entries(self, msg) -> Iterable:
        for attr_name in ("joints", "joints_state", "states"):
            if hasattr(msg, attr_name):
                entries = getattr(msg, attr_name)
                if entries is not None:
                    return entries
        return []

    def _get_command_entries(self, msg):
        for attr_name in ("joints", "joint_commands", "commands"):
            if hasattr(msg, attr_name):
                return getattr(msg, attr_name)
        raise AttributeError("JointCommandArray message has no writable joint list field.")

    def _state_callback(self, msg):
        with self._state_condition:
            for joint in self._iter_joint_entries(msg):
                joint_name = getattr(joint, "name", "")
                if joint_name not in self._positions:
                    continue
                self._positions[joint_name] = float(getattr(joint, "position", 0.0))
                self._velocities[joint_name] = float(getattr(joint, "velocity", 0.0))
                if self._enable_effort:
                    self._efforts[joint_name] = float(getattr(joint, "effort", 0.0))
            self.timestamp = time.time()
            self._state_seq += 1
            self._state_event.set()
            self._state_condition.notify_all()

    def wait_for_state(self, timeout_s: float) -> bool:
        if not self._state_subscription_enabled:
            return True
        return self._state_event.wait(timeout_s)

    def get_joint_positions(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._positions)

    def get_joint_velocities(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._velocities)

    def get_joint_efforts(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._efforts)

    def get_state_seq(self) -> int:
        with self._lock:
            return int(self._state_seq)

    def wait_for_next_state(self, last_seq: int, timeout_s: float) -> bool:
        if not self._state_subscription_enabled:
            return True
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._state_condition:
            while self._state_seq <= int(last_seq):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._state_condition.wait(timeout=remaining)
            return True

    def has_state_subscription(self) -> bool:
        return self._state_subscription_enabled

    def publish_command(
        self,
        joint_targets: Dict[str, float],
        command_specs: Dict[str, JointCommandSpec],
        joint_velocities: Optional[Dict[str, float]] = None,
    ):
        msg = JointCommandArray()
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            msg.header.stamp = self.get_clock().now().to_msg()
        command_entries = self._get_command_entries(msg)

        for joint_name in self.joint_names:
            joint_msg = JointCommand()
            joint_msg.name = joint_name
            joint_msg.position = float(joint_targets[joint_name])
            joint_msg.velocity = float(joint_velocities.get(joint_name, 0.0) if joint_velocities else 0.0)
            joint_msg.effort = 0.0
            spec = command_specs[joint_name]
            joint_msg.stiffness = spec.kp
            joint_msg.damping = spec.kd
            command_entries.append(joint_msg)

        self._pub.publish(msg)

        # Optional fallback mode: if state subscription is disabled, mirror the
        # latest commanded targets into local state cache so downstream logic can
        # use deterministic references without reading stale zeros.
        if not self._state_subscription_enabled:
            with self._state_condition:
                for joint_name in self.joint_names:
                    self._positions[joint_name] = float(joint_targets[joint_name])
                    self._velocities[joint_name] = float(joint_velocities.get(joint_name, 0.0) if joint_velocities else 0.0)
                    if self._enable_effort:
                        self._efforts[joint_name] = 0.0
                self.timestamp = time.time()
                self._state_seq += 1
                self._state_event.set()
                self._state_condition.notify_all()


class OnlineRuckigSmoother:
    def __init__(
        self,
        group_name: str,
        joint_names: Iterable[str],
        dt: float,
        max_velocity: float,
        max_acceleration: float,
        max_jerk: float,
    ):
        if ruckig is None:
            raise RuntimeError(
                "Ruckig smoothing is enabled, but the Python module 'ruckig' is unavailable. "
                "Please build/source the ROS2 workspace that contains ruckig."
            ) from _RUCKIG_IMPORT_ERROR

        self.group_name = group_name
        self.joint_names = list(joint_names)
        self.dofs = len(self.joint_names)
        self.otg = ruckig.Ruckig(self.dofs, dt)
        self.input = ruckig.InputParameter(self.dofs)
        self.output = ruckig.OutputParameter(self.dofs)
        self.input.max_velocity = [float(max_velocity)] * self.dofs
        self.input.max_acceleration = [float(max_acceleration)] * self.dofs
        self.input.max_jerk = [float(max_jerk)] * self.dofs
        self._warned_result = False
        self._initialized = False
        self._resync_position_threshold_rad = 0.08
        self._resync_velocity_threshold_rad_s = max(0.5, float(max_velocity) * 0.75)

    def _reset_state_from_measurement(
        self,
        current_positions: Dict[str, float],
        current_velocities: Dict[str, float],
    ) -> None:
        self.input.current_position = [
            float(current_positions.get(joint_name, 0.0)) for joint_name in self.joint_names
        ]
        self.input.current_velocity = [
            float(current_velocities.get(joint_name, 0.0)) for joint_name in self.joint_names
        ]
        self.input.current_acceleration = [0.0] * self.dofs
        self._initialized = True

    def smooth_targets(
        self,
        current_positions: Dict[str, float],
        current_velocities: Dict[str, float],
        target_positions: Dict[str, float],
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        measured_positions = [
            float(current_positions.get(joint_name, 0.0)) for joint_name in self.joint_names
        ]
        measured_velocities = [
            float(current_velocities.get(joint_name, 0.0)) for joint_name in self.joint_names
        ]

        if not self._initialized:
            self._reset_state_from_measurement(current_positions, current_velocities)
        else:
            position_error = max(
                abs(measured_positions[idx] - float(self.input.current_position[idx]))
                for idx in range(self.dofs)
            )
            velocity_error = max(
                abs(measured_velocities[idx] - float(self.input.current_velocity[idx]))
                for idx in range(self.dofs)
            )
            if (
                position_error > self._resync_position_threshold_rad
                or velocity_error > self._resync_velocity_threshold_rad_s
            ):
                print(
                    f"Info: Resyncing Ruckig state for {self.group_name} "
                    f"(position_error={position_error:.4f}, velocity_error={velocity_error:.4f})."
                )
                self._reset_state_from_measurement(current_positions, current_velocities)

        self.input.target_position = [float(target_positions[joint_name]) for joint_name in self.joint_names]
        self.input.target_velocity = [0.0] * self.dofs
        self.input.target_acceleration = [0.0] * self.dofs

        result = self.otg.update(self.input, self.output)
        if result not in (ruckig.Result.Working, ruckig.Result.Finished):
            if not self._warned_result:
                print(f"Warning: Ruckig returned {result} for {self.group_name}. Falling back to raw targets.")
                self._warned_result = True
            zero_velocities = {joint_name: 0.0 for joint_name in self.joint_names}
            self._initialized = False
            return dict(target_positions), zero_velocities

        smoothed_positions = {
            joint_name: float(self.output.new_position[idx]) for idx, joint_name in enumerate(self.joint_names)
        }
        smoothed_velocities = {
            joint_name: float(self.output.new_velocity[idx]) for idx, joint_name in enumerate(self.joint_names)
        }
        self.output.pass_to_input(self.input)
        return smoothed_positions, smoothed_velocities


class Ros2HandCommandInterface(Node):
    def __init__(self, node_name: str, command_topic: str):
        super().__init__(node_name)
        self._pub = self.create_publisher(
            HandCommandArray,
            command_topic,
            PUBLISHER_QOS,
        )

    def publish_command(self, left_position: float, right_position: float):
        msg = HandCommandArray()
        msg.header = MessageHeader()
        if hasattr(msg.header, "stamp"):
            msg.header.stamp = self.get_clock().now().to_msg()

        left_hand = HandCommand()
        left_hand.name = "left_hand"
        left_hand.position = float(left_position)
        left_hand.velocity = 1.0
        left_hand.acceleration = 1.0
        left_hand.deceleration = 1.0
        left_hand.effort = 1.0

        right_hand = HandCommand()
        right_hand.name = "right_hand"
        right_hand.position = float(right_position)
        right_hand.velocity = 1.0
        right_hand.acceleration = 1.0
        right_hand.deceleration = 1.0
        right_hand.effort = 1.0

        msg.left_hand_type = HandType(value=HandType.CLAW)
        msg.right_hand_type = HandType(value=HandType.CLAW)
        msg.left_hands = [left_hand]
        msg.right_hands = [right_hand]
        self._pub.publish(msg)


class Ros2WalkCommandInterface(Node):
    """Publishes locomotion velocity commands (vx/vy/vyaw) as a Twist.

    This feeds the X2 walking RL policy node (``x2_deploy_walk``), which
    subscribes to ``/x2/cmd_vel`` and maps linear.x/y and angular.z to
    vx/vy/vyaw.  The walking policy node subscribes with RELIABLE QoS, so we
    publish with the same RELIABLE ``PUBLISHER_QOS`` used for joint commands.
    """

    def __init__(self, node_name: str, command_topic: str):
        super().__init__(node_name)
        self._pub = self.create_publisher(Twist, command_topic, PUBLISHER_QOS)

    def publish_command(self, vx: float, vy: float, vyaw: float):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(vyaw)
        self._pub.publish(msg)


class X2Ros2TeleopController(HardwareTeleopController):
    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_X2_UPPER_BODY_URDF_PATH,
        manipulator_config: dict = DEFAULT_X2_MANIPULATOR_CONFIG,
        arm_state_topic: str = DEFAULT_ARM_STATE_TOPIC,
        arm_command_topic: str = DEFAULT_ARM_COMMAND_TOPIC,
        head_state_topic: str = DEFAULT_HEAD_STATE_TOPIC,
        head_command_topic: str = DEFAULT_HEAD_COMMAND_TOPIC,
        hand_command_topic: str = DEFAULT_HAND_COMMAND_TOPIC,
        model_to_hardware_joint_map: Optional[Dict[str, str]] = None,
        command_specs: Optional[Dict[str, JointCommandSpec]] = None,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = False,
        control_rate_hz: int = 100,
        enable_log_data: bool = True,
        log_dir: str = "/logs/x2_hardware_logs",
        log_freq: float = 30,
        validate_log_before_save: bool = True,
        decode_images_on_log_validate: bool = True,
        enable_camera: bool = False,
        camera_fps: int = 30,
        show_camera_window: bool = True,
        camera_color_topics: str = DEFAULT_X2_CAMERA_COLOR_TOPICS,
        camera_depth_topics: str = DEFAULT_X2_CAMERA_DEPTH_TOPICS,
        camera_width: int = 424,
        camera_height: int = 240,
        camera_enable_depth: bool = False,
        camera_enable_compression: bool = True,
        camera_jpg_quality: int = 85,
        camera_raw_passthrough_for_logging: bool = False,
        enable_head_tracking: bool = False,
        enable_head_state_feedback: bool = False,
        head_yaw_scale: float = 1.0,
        head_pitch_scale: float = 1.0,
        initial_state_timeout_s: float = 10.0,
        software_estop_button: str = "right_menu_button",
        software_estop_hold_s: float = 0.5,
        max_arm_joint_step_rad: float = 1.0,
        inactive_arm_return_max_step_rad: float = 0.03,
        max_head_joint_step_rad: float = 0.05,
        debug_print_targets: bool = False,
        debug_print_hz: float = 2.0,
        enable_ruckig_smoothing: bool = False,
        arm_ruckig_max_velocity: float = 1.2,
        arm_ruckig_max_acceleration: float = 4.0,
        arm_ruckig_max_jerk: float = 40.0,
        head_ruckig_max_velocity: float = 0.8,
        head_ruckig_max_acceleration: float = 2.5,
        head_ruckig_max_jerk: float = 25.0,
        require_fresh_state_each_cycle: bool = False,
        fresh_state_timeout_s: float = 0.2,
        enable_log_torque: bool = False,
        enable_walk_control: bool = True,
        walk_command_topic: str = DEFAULT_WALK_COMMAND_TOPIC,
        walk_max_lin_vel: float = 0.2,
        walk_max_ang_vel: float = 0.3,
        walk_deadzone: float = 0.1,
        walk_deadman: str = "none",
        walk_flip_vx: bool = False,
        walk_flip_vy: bool = False,
        walk_flip_vyaw: bool = False,
        walk_debug_print: bool = False,
        episode_boundary_button: str = "X",
    ):
        if _ROS2_IMPORT_ERROR is not None:
            raise RuntimeError(
                "ROS2 dependencies are unavailable. Please source your ROS2 workspace "
                "so rclpy and aimdk_msgs can be imported."
            ) from _ROS2_IMPORT_ERROR

        self.arm_state_topic = arm_state_topic
        self.arm_command_topic = arm_command_topic
        self.head_state_topic = head_state_topic
        self.head_command_topic = head_command_topic
        self.hand_command_topic = hand_command_topic
        self.model_to_hardware_joint_map = model_to_hardware_joint_map or DEFAULT_MODEL_TO_HARDWARE_JOINT_MAP.copy()
        self.command_specs = command_specs or DEFAULT_HARDWARE_COMMAND_SPECS.copy()
        self.enable_head_tracking = enable_head_tracking
        self.enable_head_state_feedback = bool(enable_head_state_feedback)
        self.camera_color_topics = camera_color_topics
        self.camera_depth_topics = camera_depth_topics
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_enable_depth = bool(camera_enable_depth)
        self.camera_enable_compression = bool(camera_enable_compression)
        self.camera_jpg_quality = int(camera_jpg_quality)
        self.camera_raw_passthrough_for_logging = bool(camera_raw_passthrough_for_logging)
        self.head_yaw_scale = head_yaw_scale
        self.head_pitch_scale = head_pitch_scale
        self.initial_state_timeout_s = initial_state_timeout_s
        self.software_estop_button = software_estop_button
        self.software_estop_hold_s = max(0.0, float(software_estop_hold_s))
        self._software_estop_pressed_since: float | None = None
        self._software_estop_triggered = False
        self.max_arm_joint_step_rad = max(0.0, float(max_arm_joint_step_rad))
        self.inactive_arm_return_max_step_rad = max(0.0, float(inactive_arm_return_max_step_rad))
        self.max_head_joint_step_rad = max(0.0, float(max_head_joint_step_rad))
        self.debug_print_targets = bool(debug_print_targets)
        self.debug_print_hz = max(0.0, float(debug_print_hz))
        self._last_target_debug_print_time = 0.0
        self._last_joint_limit_report_time = {"arm": 0.0, "head": 0.0}
        self.enable_ruckig_smoothing = bool(enable_ruckig_smoothing)
        self.enable_log_torque = bool(enable_log_torque)
        self.enable_walk_control = bool(enable_walk_control)
        self.walk_command_topic = str(walk_command_topic)
        self.walk_max_lin_vel = float(walk_max_lin_vel)
        self.walk_max_ang_vel = float(walk_max_ang_vel)
        self.walk_deadzone = float(walk_deadzone)
        self.walk_deadman = (str(walk_deadman).strip().lower() or "none")
        if self.walk_deadman in ("none", "off", "0"):
            self.walk_deadman = None
        self.walk_vx_sign = -1.0 if walk_flip_vx else 1.0
        self.walk_vy_sign = -1.0 if walk_flip_vy else 1.0
        self.walk_vyaw_sign = -1.0 if walk_flip_vyaw else 1.0
        self.walk_debug_print = bool(walk_debug_print)
        self._last_walk_debug_time = 0.0
        self.walk_interface = None
        self._last_walk_command = (0.0, 0.0, 0.0)
        self.arm_ruckig_max_velocity = float(arm_ruckig_max_velocity)
        self.arm_ruckig_max_acceleration = float(arm_ruckig_max_acceleration)
        self.arm_ruckig_max_jerk = float(arm_ruckig_max_jerk)
        self.head_ruckig_max_velocity = float(head_ruckig_max_velocity)
        self.head_ruckig_max_acceleration = float(head_ruckig_max_acceleration)
        self.head_ruckig_max_jerk = float(head_ruckig_max_jerk)
        self.require_fresh_state_each_cycle = bool(require_fresh_state_each_cycle)
        self.fresh_state_timeout_s = max(0.001, float(fresh_state_timeout_s))
        self._enforce_fresh_state_each_cycle = False
        self._arm_last_state_seq = -1
        self._head_last_state_seq = -1
        self._last_fresh_state_warn_time = 0.0
        self._control_state_lock = threading.Lock()

        self.arm_model_joint_names = list(ARM_MODEL_JOINT_NAMES)
        self.head_model_joint_names = list(HEAD_MODEL_JOINT_NAMES)
        self.left_arm_model_joint_names = list(LEFT_ARM_MODEL_JOINT_NAMES)
        self.right_arm_model_joint_names = list(RIGHT_ARM_MODEL_JOINT_NAMES)
        self.arm_hardware_joint_names = [self.model_to_hardware_joint_map[name] for name in self.arm_model_joint_names]
        self.head_hardware_joint_names = [self.model_to_hardware_joint_map[name] for name in self.head_model_joint_names]
        self.left_arm_hardware_joint_names = [
            self.model_to_hardware_joint_map[name] for name in self.left_arm_model_joint_names
        ]
        self.right_arm_hardware_joint_names = [
            self.model_to_hardware_joint_map[name] for name in self.right_arm_model_joint_names
        ]
        self.arm_return_zero_targets = {
            joint_name: self._clip_target(joint_name, 0.0) for joint_name in self.arm_hardware_joint_names
        }
        self.arm_return_zero_targets.update(
            {
                "left_shoulder_pitch_joint": 0.04,
                "left_shoulder_roll_joint": 0.5,
                "left_shoulder_yaw_joint": 0.08,
                "left_elbow_joint": -1.5,
                "left_wrist_yaw_joint": 0.75,
                "left_wrist_pitch_joint": -0.29,
                "left_wrist_roll_joint": -1.15,

                "right_shoulder_pitch_joint": 0.04,
                "right_shoulder_roll_joint": -0.5,
                "right_shoulder_yaw_joint": -0.08,
                "right_elbow_joint": -1.5,
                "right_wrist_yaw_joint": -0.75,
                "right_wrist_pitch_joint": -0.29,
                "right_wrist_roll_joint": 1.15,
            }
        )
        self._validate_configuration()

        self._ros2_setup_complete = False
        self._executor = None
        self._spin_thread = None
        self.arm_interface: Optional[Ros2JointGroupInterface] = None
        self.head_interface: Optional[Ros2JointGroupInterface] = None
        self.hand_interface: Optional[Ros2HandCommandInterface] = None
        self.camera_interface: Optional[Ros2CameraInterface] = None
        self._prev_arm_targets: Optional[Dict[str, float]] = None
        self._prev_head_targets: Optional[Dict[str, float]] = None
        self._prev_hand_targets = {"left_hand": 1.0, "right_hand": 1.0}
        self._last_active_state = {name: False for name in manipulator_config.keys()}
        self.arm_joint_offsets: Dict[str, int] = {}
        self.head_joint_offsets: Dict[str, int] = {}
        self.head_target_positions = {joint_name: 0.0 for joint_name in self.head_hardware_joint_names}
        self.joints_task = None
        self.arm_smoother: Optional[OnlineRuckigSmoother] = None
        self.head_smoother: Optional[OnlineRuckigSmoother] = None

        if self.enable_ruckig_smoothing:
            self.arm_smoother = OnlineRuckigSmoother(
                group_name="arm",
                joint_names=self.arm_hardware_joint_names,
                dt=1.0 / control_rate_hz,
                max_velocity=self.arm_ruckig_max_velocity,
                max_acceleration=self.arm_ruckig_max_acceleration,
                max_jerk=self.arm_ruckig_max_jerk,
            )
            self.head_smoother = OnlineRuckigSmoother(
                group_name="head",
                joint_names=self.head_hardware_joint_names,
                dt=1.0 / control_rate_hz,
                max_velocity=self.head_ruckig_max_velocity,
                max_acceleration=self.head_ruckig_max_acceleration,
                max_jerk=self.head_ruckig_max_jerk,
            )

        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            R_headset_world=R_headset_world,
            floating_base=False,
            scale_factor=scale_factor,
            visualize_placo=visualize_placo,
            control_rate_hz=control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
            validate_log_before_save=validate_log_before_save,
            decode_images_on_log_validate=decode_images_on_log_validate,
            enable_camera=enable_camera,
            camera_fps=camera_fps,
            show_camera_window=show_camera_window,
            episode_boundary_button=episode_boundary_button,
        )

    def _validate_configuration(self):
        missing_specs = [
            joint_name
            for joint_name in self.arm_hardware_joint_names + self.head_hardware_joint_names
            if joint_name not in self.command_specs
        ]
        if missing_specs:
            raise ValueError(f"Missing command specs for joints: {missing_specs}")

    def _robot_setup(self):
        if self._ros2_setup_complete:
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        self.arm_interface = Ros2JointGroupInterface(
            node_name="x2_arm_ros2_interface",
            state_topic=self.arm_state_topic,
            command_topic=self.arm_command_topic,
            joint_names=self.arm_hardware_joint_names,
            enable_effort=self.enable_log_torque,
        )
        self.head_interface = Ros2JointGroupInterface(
            node_name="x2_head_ros2_interface",
            state_topic=self.head_state_topic,
            command_topic=self.head_command_topic,
            joint_names=self.head_hardware_joint_names,
            enable_state_subscription=self.enable_head_state_feedback,
        )
        self.hand_interface = Ros2HandCommandInterface(
            node_name="x2_hand_ros2_interface",
            command_topic=self.hand_command_topic,
        )

        if self.enable_walk_control:
            self.walk_interface = Ros2WalkCommandInterface(
                node_name="x2_walk_ros2_interface",
                command_topic=self.walk_command_topic,
            )

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.arm_interface)
        self._executor.add_node(self.head_interface)
        self._executor.add_node(self.hand_interface)
        if self.walk_interface is not None:
            self._executor.add_node(self.walk_interface)
        self._spin_thread = threading.Thread(target=self._executor.spin, name="x2_ros2_spin", daemon=True)
        self._spin_thread.start()

        print("Waiting for initial ROS2 joint states from X2 topics...")
        arm_ready = self.arm_interface.wait_for_state(self.initial_state_timeout_s)
        head_ready = self.head_interface.wait_for_state(self.initial_state_timeout_s)
        if not arm_ready:
            raise RuntimeError(
                f"Timed out waiting for arm joint state topic: {self.arm_state_topic}"
            )
        if self.enable_head_state_feedback and not head_ready:
            raise RuntimeError(
                f"Timed out waiting for head joint state topic: {self.head_state_topic}"
            )

        self._arm_last_state_seq = self.arm_interface.get_state_seq()
        self._head_last_state_seq = self.head_interface.get_state_seq()

        self._ros2_setup_complete = True
        print("ROS2 joint interfaces are ready.")

    def _placo_setup(self):
        super()._placo_setup()

        self.arm_joint_offsets = {
            joint_name: self.placo_robot.get_joint_offset(joint_name) for joint_name in self.arm_model_joint_names
        }
        self.head_joint_offsets = {
            joint_name: self.placo_robot.get_joint_offset(joint_name) for joint_name in self.head_model_joint_names
        }

        self.joints_task = self.solver.add_joints_task()
        default_joints = {
            "left_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -1.0,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.3,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -1.0,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
        }
        self.joints_task.set_joints(default_joints)
        self.joints_task.configure("joints_regularization", "soft", 1e-4)

        self._update_robot_state()
        self.placo_robot.update_kinematics()
        self.sync_end_effector_poses_to_placo_tasks()

    def _initialize_camera(self):
        self.camera_interface = None
        if not self.enable_camera:
            return

        if Ros2CameraInterface is None:
            print(
                "Camera display is enabled, but Ros2CameraInterface is unavailable. "
                "Skipping camera initialization."
            )
            self.enable_camera = False
            return

        camera_topics = self._build_camera_topics(
            color_topics_spec=self.camera_color_topics,
            depth_topics_spec=self.camera_depth_topics,
        )
        if not camera_topics:
            print("Camera display is enabled, but no valid camera topics were configured.")
            self.enable_camera = False
            return

        try:
            self.camera_interface = Ros2CameraInterface(
                node_name="x2_ros2_camera_interface",
                camera_topics=camera_topics,
                enable_depth=self.camera_enable_depth,
                width=self.camera_width,
                height=self.camera_height,
                enable_compression=self.camera_enable_compression,
                jpg_quality=self.camera_jpg_quality,
                raw_passthrough_for_logging=self.camera_raw_passthrough_for_logging,
            )
            self.camera_interface.start()
            if self._executor is not None:
                self._executor.add_node(self.camera_interface)
            print(f"Camera initialized successfully with topics: {camera_topics}")
        except Exception as exc:
            print(f"Error initializing camera interface: {exc}")
            if self.camera_interface is not None:
                try:
                    self.camera_interface.stop()
                except Exception:
                    pass
                try:
                    self.camera_interface.destroy_node()
                except Exception:
                    pass
            self.camera_interface = None
            self.enable_camera = False

    @staticmethod
    def _parse_camera_topic_spec(spec: str) -> Dict[str, str]:
        topics: Dict[str, str] = {}
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    f"Invalid camera topic item '{item}'. Expected format 'camera_name=/topic/name'."
                )
            camera_name, topic_name = item.split("=", 1)
            camera_name = camera_name.strip()
            topic_name = topic_name.strip()
            if not camera_name or not topic_name:
                raise ValueError(
                    f"Invalid camera topic item '{item}'. Camera name and topic must both be non-empty."
                )
            topics[camera_name] = topic_name
        return topics

    def _build_camera_topics(
        self,
        color_topics_spec: str,
        depth_topics_spec: str,
    ) -> Dict[str, Dict[str, str]]:
        color_topics = self._parse_camera_topic_spec(color_topics_spec)
        depth_topics = self._parse_camera_topic_spec(depth_topics_spec)
        camera_names = sorted(set(color_topics) | set(depth_topics))
        camera_topics: Dict[str, Dict[str, str]] = {}
        for camera_name in camera_names:
            streams: Dict[str, str] = {}
            if camera_name in color_topics:
                streams["color"] = color_topics[camera_name]
            if camera_name in depth_topics:
                streams["depth"] = depth_topics[camera_name]
            if streams:
                camera_topics[camera_name] = streams
        return camera_topics

    def _hardware_state_to_model_positions(
        self,
        model_joint_names: Iterable[str],
        hardware_positions: Dict[str, float],
    ) -> Dict[str, float]:
        positions = {}
        for model_joint_name in model_joint_names:
            hardware_joint_name = self.model_to_hardware_joint_map[model_joint_name]
            positions[model_joint_name] = float(hardware_positions.get(hardware_joint_name, 0.0))
        return positions

    def _update_robot_state(self):
        if self._enforce_fresh_state_each_cycle and self.require_fresh_state_each_cycle:
            arm_ready = self.arm_interface.wait_for_next_state(self._arm_last_state_seq, self.fresh_state_timeout_s)
            head_ready = True
            if self.enable_head_state_feedback:
                head_ready = self.head_interface.wait_for_next_state(self._head_last_state_seq, self.fresh_state_timeout_s)
            if not arm_ready or not head_ready:
                now = time.time()
                if (now - self._last_fresh_state_warn_time) >= 0.5:
                    print(
                        "Warning: fresh ROS2 joint state not received in time "
                        f"(arm_ready={arm_ready}, head_ready={head_ready}, timeout={self.fresh_state_timeout_s:.3f}s)."
                    )
                    self._last_fresh_state_warn_time = now
                raise RuntimeError("Fresh ROS2 joint states unavailable for IK cycle.")

            self._arm_last_state_seq = self.arm_interface.get_state_seq()
            self._head_last_state_seq = self.head_interface.get_state_seq()

        arm_positions = self.arm_interface.get_joint_positions()
        if self.enable_head_state_feedback:
            head_positions = self.head_interface.get_joint_positions()
        else:
            head_positions = dict(self.head_target_positions)

        model_arm_positions = self._hardware_state_to_model_positions(self.arm_model_joint_names, arm_positions)
        model_head_positions = self._hardware_state_to_model_positions(self.head_model_joint_names, head_positions)

        for joint_name, position in model_arm_positions.items():
            self.placo_robot.state.q[self.arm_joint_offsets[joint_name]] = position
        for joint_name, position in model_head_positions.items():
            self.placo_robot.state.q[self.head_joint_offsets[joint_name]] = position

    def _ik_thread(self, stop_event: threading.Event):
        """Run IK loop and require fresh ROS2 joint states every cycle."""
        while not stop_event.is_set():
            start_time = time.time()
            try:
                with self._control_state_lock:
                    self._update_gripper_target()
                    self._pre_ik_update()
                    self._update_ik()
                    self._handle_activation_edges()
                    if self.visualize_placo:
                        self._update_placo_viz()
            except RuntimeError as exc:
                if "Fresh ROS2 joint states unavailable" in str(exc):
                    elapsed_time = time.time() - start_time
                    sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue
                raise

            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("IK loop has stopped.")

    def run(self):
        self._enforce_fresh_state_each_cycle = self.require_fresh_state_each_cycle
        super().run()

    def _compute_joint_velocities(
        self,
        targets: Dict[str, float],
        prev_targets: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        if prev_targets is None:
            return {joint_name: 0.0 for joint_name in targets}
        return {
            joint_name: (targets[joint_name] - prev_targets.get(joint_name, targets[joint_name])) / self.dt
            for joint_name in targets
        }

    def _clip_target(self, hardware_joint_name: str, target: float) -> float:
        spec = self.command_specs[hardware_joint_name]
        return float(np.clip(target, spec.lower_limit, spec.upper_limit))

    def _apply_joint_step_limit(
        self,
        group_name: str,
        targets: Dict[str, float],
        reference_targets: Dict[str, float],
        max_joint_step_rad: float,
    ) -> Dict[str, float]:
        if max_joint_step_rad <= 0.0:
            return dict(targets)

        limited_targets = {}
        clipped_joints = []
        for joint_name, target in targets.items():
            ref_target = float(reference_targets.get(joint_name, target))
            limited_target = float(np.clip(target, ref_target - max_joint_step_rad, ref_target + max_joint_step_rad))
            if abs(limited_target - target) > 1.0e-9:
                clipped_joints.append(joint_name)
            limited_targets[joint_name] = limited_target

        now = time.time()
        if clipped_joints and (now - self._last_joint_limit_report_time[group_name]) >= 1.0:
            joint_list = ", ".join(clipped_joints)
            print(
                f"Joint step limiter clipped {group_name} targets "
                f"(max_step={max_joint_step_rad:.3f} rad): {joint_list}"
            )
            self._last_joint_limit_report_time[group_name] = now

        return limited_targets

    def _build_arm_targets(self) -> tuple[Dict[str, float], Dict[str, float]]:
        raw_targets = {}
        clipped_targets = {}
        for model_joint_name in self.arm_model_joint_names:
            hardware_joint_name = self.model_to_hardware_joint_map[model_joint_name]
            q_target = self.placo_robot.state.q[self.arm_joint_offsets[model_joint_name]]
            raw_targets[hardware_joint_name] = float(q_target)
            clipped_targets[hardware_joint_name] = self._clip_target(hardware_joint_name, q_target)
        return raw_targets, clipped_targets

    def _arm_is_active_or_pending(self, arm_name: str) -> bool:
        if self.active.get(arm_name, False):
            return True

        config = self.manipulator_config.get(arm_name, {})
        activation_mode = str(config.get("activation_mode", "analog")).strip().lower()
        if activation_mode in {"always_on", "always", "on"}:
            return True

        threshold = float(config.get("activation_threshold", 0.9))
        return self._get_control_signal_value(config.get("control_trigger")) > threshold

    def _apply_inactive_arm_zero_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        adjusted_targets = dict(targets)
        if not self._arm_is_active_or_pending("left_arm"):
            for joint_name in self.left_arm_hardware_joint_names:
                adjusted_targets[joint_name] = self.arm_return_zero_targets[joint_name]
        if not self._arm_is_active_or_pending("right_arm"):
            for joint_name in self.right_arm_hardware_joint_names:
                adjusted_targets[joint_name] = self.arm_return_zero_targets[joint_name]
        return adjusted_targets

    def _apply_inactive_arm_return_step_limit(
        self,
        targets: Dict[str, float],
        reference_targets: Dict[str, float],
    ) -> Dict[str, float]:
        if self.inactive_arm_return_max_step_rad <= 0.0:
            return dict(targets)

        limited_targets = dict(targets)
        if not self._arm_is_active_or_pending("left_arm"):
            for joint_name in self.left_arm_hardware_joint_names:
                ref_target = float(reference_targets.get(joint_name, limited_targets[joint_name]))
                limited_targets[joint_name] = float(
                    np.clip(
                        limited_targets[joint_name],
                        ref_target - self.inactive_arm_return_max_step_rad,
                        ref_target + self.inactive_arm_return_max_step_rad,
                    )
                )
        if not self._arm_is_active_or_pending("right_arm"):
            for joint_name in self.right_arm_hardware_joint_names:
                ref_target = float(reference_targets.get(joint_name, limited_targets[joint_name]))
                limited_targets[joint_name] = float(
                    np.clip(
                        limited_targets[joint_name],
                        ref_target - self.inactive_arm_return_max_step_rad,
                        ref_target + self.inactive_arm_return_max_step_rad,
                    )
                )
        return limited_targets

    @staticmethod
    def _format_debug_targets(targets: Dict[str, float]) -> str:
        ordered = ", ".join(f"{joint_name}={targets[joint_name]:+.3f}" for joint_name in sorted(targets))
        return "{" + ordered + "}"

    @staticmethod
    def _format_debug_comparison(targets: Dict[str, float], currents: Dict[str, float]) -> str:
        ordered_joints = sorted(targets)
        ordered = ", ".join(
            f"{joint_name}: cur={currents.get(joint_name, 0.0):+.3f}, "
            f"tgt={targets[joint_name]:+.3f}, "
            f"d={targets[joint_name] - currents.get(joint_name, 0.0):+.3f}"
            for joint_name in ordered_joints
        )
        return "{" + ordered + "}"

    @staticmethod
    def _format_debug_transition(source: Dict[str, float], target: Dict[str, float]) -> str:
        ordered_joints = sorted(source)
        ordered = ", ".join(
            f"{joint_name}: src={source[joint_name]:+.3f}, "
            f"dst={target.get(joint_name, 0.0):+.3f}, "
            f"d={target.get(joint_name, 0.0) - source[joint_name]:+.3f}"
            for joint_name in ordered_joints
        )
        return "{" + ordered + "}"

    def _maybe_print_targets(
        self,
        raw_arm_targets: Dict[str, float],
        clipped_arm_targets: Dict[str, float],
        arm_targets: Dict[str, float],
        head_targets: Dict[str, float],
        arm_currents: Dict[str, float],
        head_currents: Dict[str, float],
    ) -> None:
        if not self.debug_print_targets:
            return

        now = time.time()
        min_period = 0.0 if self.debug_print_hz <= 0.0 else 1.0 / self.debug_print_hz
        if min_period > 0.0 and (now - self._last_target_debug_print_time) < min_period:
            return

        self._last_target_debug_print_time = now
        print(f"ARM raw->clip: {self._format_debug_transition(raw_arm_targets, clipped_arm_targets)}")
        print(f"ARM clip->final: {self._format_debug_transition(clipped_arm_targets, arm_targets)}")
        print(f"ARM current/target/delta: {self._format_debug_comparison(arm_targets, arm_currents)}")
        print(f"HEAD current/target/delta: {self._format_debug_comparison(head_targets, head_currents)}")

    def _normalize_headset_angles(self, yaw: float, pitch: float) -> tuple[float, float]:
        if yaw > np.pi / 2 and yaw < np.pi:
            yaw -= np.pi
        if yaw < -np.pi / 2:
            yaw = np.pi + yaw

        if pitch < -np.pi / 2:
            pitch = -pitch - np.pi
        if pitch > np.pi / 2:
            pitch = np.pi - pitch
        return yaw, pitch

    def _update_head_targets(self):
        if self.enable_head_state_feedback:
            current_head_positions = self.head_interface.get_joint_positions()
        else:
            current_head_positions = dict(self.head_target_positions)
        targets = {
            joint_name: float(current_head_positions.get(joint_name, 0.0))
            for joint_name in self.head_hardware_joint_names
        }

        if not self.enable_head_tracking:
            self.head_target_positions = {
                joint_name: self._clip_target(joint_name, 0.0)
                for joint_name in self.head_hardware_joint_names
            }
            return

        try:
            head_pose = self.xr_client.get_pose_by_name("headset")
        except Exception as exc:
            print(f"Failed to read XR headset pose for head tracking: {exc}")
            self.head_target_positions = targets
            return

        quat = np.array([head_pose[6], head_pose[3], head_pose[4], head_pose[5]])
        rot_matrix = tf.quaternion_matrix(quat)[:3, :3]
        euler = tf.euler_from_matrix(rot_matrix, "rzxy")
        yaw, pitch = self._normalize_headset_angles(euler[2], euler[1])

        targets["head_yaw_joint"] = self._clip_target("head_yaw_joint", yaw * self.head_yaw_scale)
        targets["head_pitch_joint"] = self._clip_target("head_pitch_joint", pitch * self.head_pitch_scale)
        self.head_target_positions = targets

    def _pre_ik_update(self):
        self._update_head_targets()
        self._update_walk_command()

    # ── 下半身行走速度指令（Pico 摇杆 → /x2/cmd_vel Twist）────────────────
    # 线性线速度控制：摇杆偏转与线速度成线性比例（满偏 = walk_max_lin_vel）。
    # 不使用角速度控制：vyaw 恒为 0，右摇杆不映射任何指令。
    #   左摇杆 Y  → vx (前推=前进, +)
    #   左摇杆 X  → vy (左推=左移, -X → +vy)

    def _apply_walk_deadzone(self, value: float) -> float:
        value = float(value)
        if abs(value) <= self.walk_deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        magnitude = (abs(value) - self.walk_deadzone) / (1.0 - self.walk_deadzone)
        return sign * min(1.0, magnitude)

    def _is_walk_deadman_active(self) -> bool:
        name = self.walk_deadman
        try:
            if name in ("left_trigger", "right_trigger", "left_grip", "right_grip"):
                return float(self.xr_client.get_key_value_by_name(name)) > 0.5
            return bool(self.xr_client.get_button_state_by_name(name))
        except Exception:
            return False

    def _update_walk_command(self):
        if not self.enable_walk_control or self.walk_interface is None:
            return

        try:
            left = self.xr_client.get_joystick_state("left")
        except Exception as exc:
            print(f"Failed to read joystick state for walk control: {exc}")
            return

        lx = self._apply_walk_deadzone(left[0])
        ly = self._apply_walk_deadzone(left[1])

        vx = ly * self.walk_max_lin_vel * self.walk_vx_sign
        vy = -lx * self.walk_max_lin_vel * self.walk_vy_sign
        vyaw = 0.0

        if self.walk_deadman is not None and not self._is_walk_deadman_active():
            vx = vy = 0.0

        self._last_walk_command = (float(vx), float(vy), float(vyaw))
        self.walk_interface.publish_command(vx, vy, vyaw)

        if self.walk_debug_print:
            now = time.time()
            if now - self._last_walk_debug_time >= 0.5:
                self._last_walk_debug_time = now
                print(
                    f"[walk] L=({left[0]:+.3f},{left[1]:+.3f}) "
                    f"-> vx={vx:+.3f} vy={vy:+.3f} (angular control disabled) "
                    f"deadman={'ON' if (self.walk_deadman and self._is_walk_deadman_active()) else '-'}"
                )

    def _handle_activation_edges(self):
        for arm_name in ("left_arm", "right_arm"):
            is_active = bool(self.active.get(arm_name, False))
            if is_active and not self._last_active_state.get(arm_name, False):
                self._prev_arm_targets = None
            self._last_active_state[arm_name] = is_active

    def _check_software_estop(self) -> bool:
        if self._software_estop_triggered:
            return True

        try:
            pressed = self.xr_client.get_button_state_by_name(self.software_estop_button)
        except Exception as exc:
            print(f"Failed to read software e-stop button '{self.software_estop_button}': {exc}")
            return False

        now = time.time()
        if pressed:
            if self._software_estop_pressed_since is None:
                self._software_estop_pressed_since = now
            elif (now - self._software_estop_pressed_since) >= self.software_estop_hold_s:
                self._activate_software_estop()
        else:
            self._software_estop_pressed_since = None

        return self._software_estop_triggered

    def _activate_software_estop(self):
        if self._software_estop_triggered:
            return

        self._software_estop_triggered = True
        self._software_estop_pressed_since = None

        arm_hold_targets = self.arm_interface.get_joint_positions()
        if self.enable_head_state_feedback:
            head_hold_targets = self.head_interface.get_joint_positions()
        else:
            head_hold_targets = dict(self.head_target_positions)
        zero_arm_velocities = {joint_name: 0.0 for joint_name in arm_hold_targets}

        self.arm_interface.publish_command(
            joint_targets=arm_hold_targets,
            joint_velocities=zero_arm_velocities,
            command_specs=self.command_specs,
        )

        self._prev_arm_targets = dict(arm_hold_targets)
        self._prev_head_targets = dict(head_hold_targets)
        self.head_target_positions = dict(head_hold_targets)

        print(
            "SOFTWARE E-STOP triggered. "
            f"Held current joint positions via '{self.software_estop_button}' and stopping teleoperation."
        )
        self._stop_event.set()

    def _send_command(self):
        with self._control_state_lock:
            if self._check_software_estop():
                return

            current_arm_positions = self.arm_interface.get_joint_positions()
            current_arm_velocities = self.arm_interface.get_joint_velocities()
            raw_arm_targets, arm_targets = self._build_arm_targets()
            arm_targets = self._apply_inactive_arm_zero_targets(arm_targets)
            clipped_arm_targets = dict(arm_targets)
            if self.arm_smoother is not None:
                arm_targets, arm_velocities = self.arm_smoother.smooth_targets(
                    current_positions=current_arm_positions,
                    current_velocities=current_arm_velocities,
                    target_positions=arm_targets,
                )
            else:
                arm_velocities = None
            arm_reference = self._prev_arm_targets or current_arm_positions
            prelimit_arm_targets = dict(arm_targets)
            arm_targets = self._apply_joint_step_limit(
                "arm",
                arm_targets,
                arm_reference,
                self.max_arm_joint_step_rad,
            )
            arm_targets = self._apply_inactive_arm_return_step_limit(arm_targets, arm_reference)
            if arm_velocities is None or any(
                abs(arm_targets[joint_name] - prelimit_arm_targets[joint_name]) > 1.0e-9
                for joint_name in arm_targets
            ):
                arm_velocities = self._compute_joint_velocities(arm_targets, self._prev_arm_targets)
            self._prev_arm_targets = arm_targets

            if self.enable_head_state_feedback:
                current_head_positions = self.head_interface.get_joint_positions()
                current_head_velocities = self.head_interface.get_joint_velocities()
            else:
                current_head_positions = dict(self.head_target_positions)
                current_head_velocities = {
                    joint_name: 0.0 for joint_name in self.head_hardware_joint_names
                }
            head_targets = dict(self.head_target_positions)
            if self.head_smoother is not None:
                head_targets, head_velocities = self.head_smoother.smooth_targets(
                    current_positions=current_head_positions,
                    current_velocities=current_head_velocities,
                    target_positions=head_targets,
                )
            else:
                head_velocities = None
            head_reference = self._prev_head_targets or current_head_positions
            prelimit_head_targets = dict(head_targets)
            limited_head_targets = self._apply_joint_step_limit(
                "head",
                head_targets,
                head_reference,
                self.max_head_joint_step_rad,
            )
            self.head_target_positions = dict(limited_head_targets)
            if head_velocities is None or any(
                abs(self.head_target_positions[joint_name] - prelimit_head_targets[joint_name]) > 1.0e-9
                for joint_name in self.head_target_positions
            ):
                head_velocities = self._compute_joint_velocities(self.head_target_positions, self._prev_head_targets)
            self.arm_interface.publish_command(
                joint_targets=arm_targets,
                joint_velocities=arm_velocities,
                command_specs=self.command_specs,
            )
            hand_targets = self._build_hand_targets()
            if self.hand_interface is not None:
                self.hand_interface.publish_command(
                    left_position=hand_targets["left_hand"],
                    right_position=hand_targets["right_hand"],
                )
            self._prev_hand_targets = dict(hand_targets)
            self._maybe_print_targets(
                raw_arm_targets,
                clipped_arm_targets,
                arm_targets,
                self.head_target_positions,
                current_arm_positions,
                current_head_positions,
            )
            self._prev_head_targets = dict(self.head_target_positions)

    def _build_hand_targets(self) -> Dict[str, float]:
        hand_targets = {"left_hand": 1.0, "right_hand": 1.0}
        for arm_name, config in self.manipulator_config.items():
            gripper_config = config.get("gripper_config")
            if not gripper_config:
                continue
            joint_name = gripper_config["joint_names"][0]
            target_value = float(self.gripper_pos_target[arm_name][joint_name])
            if arm_name == "left_arm":
                hand_targets["left_hand"] = float(np.clip(target_value, 0.0, 1.0))
            elif arm_name == "right_arm":
                hand_targets["right_hand"] = float(np.clip(target_value, 0.0, 1.0))
        return hand_targets

    def _get_robot_state_for_logging(self) -> Dict:
        arm_state = self.arm_interface.get_joint_positions()
        arm_command = self._prev_arm_targets if self._prev_arm_targets else arm_state
        result = {
            "arm_state": arm_state,
            "arm_velocity": self.arm_interface.get_joint_velocities(),
            "arm_command": dict(arm_command),
            "head_state": (
                self.head_interface.get_joint_positions()
                if self.enable_head_state_feedback
                else dict(self.head_target_positions)
            ),
            "head_velocity": (
                self.head_interface.get_joint_velocities()
                if self.enable_head_state_feedback
                else {joint_name: 0.0 for joint_name in self.head_hardware_joint_names}
            ),
            "head_command": self._prev_head_targets or {},
            "hand_command": dict(self._prev_hand_targets),
            "hand_trigger_raw": {
                "left_hand": self.gripper_trigger_value.get("left_arm"),
                "right_hand": self.gripper_trigger_value.get("right_arm"),
            },
        }
        if self.enable_log_torque:
            result["arm_effort"] = self.arm_interface.get_joint_efforts()
        if self.enable_walk_control:
            vx, vy, vyaw = self._last_walk_command
            result["walk_command"] = {"vx": float(vx), "vy": float(vy), "vyaw": float(vyaw)}
        return result

    def _get_camera_frame_for_logging(self) -> Dict:
        if not self.camera_interface:
            return {}

        if self.camera_interface.enable_compression:
            frames = self.camera_interface.get_compressed_frames()
        else:
            frames = self.camera_interface.get_frames()
        return frames if frames else {}

    def _shutdown_robot(self):
        if self.camera_interface is not None:
            try:
                self.camera_interface.stop()
            except Exception:
                pass

        if self._executor is not None:
            for interface in (
                self.camera_interface,
                self.arm_interface,
                self.head_interface,
                self.hand_interface,
                self.walk_interface,
            ):
                if interface is not None:
                    try:
                        self._executor.remove_node(interface)
                    except Exception:
                        pass
            self._executor.shutdown()
            self._executor = None

        for interface in (
            self.camera_interface,
            self.arm_interface,
            self.head_interface,
            self.hand_interface,
            self.walk_interface,
        ):
            if interface is not None:
                interface.destroy_node()
        self.camera_interface = None
        self.walk_interface = None

        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self._spin_thread = None

        if rclpy.ok():
            rclpy.shutdown()
