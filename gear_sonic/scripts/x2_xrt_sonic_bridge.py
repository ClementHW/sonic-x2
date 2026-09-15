#!/usr/bin/env python3
"""XRoboToolkit -> SONIC teleop bridge for AgiBot X2 (runs on PC2).

This process is the XR-driven operator front-end for the SONIC whole-body
stack. It *reuses* the XRoboToolkit teleop controller (Placo IK, grip
activation, inactive-arm return-to-default ramp, joystick reads, software
e-stop) but replaces its hardware sinks: instead of publishing ROS2 joint
commands onto ``/aima/hal/joint/arm/command`` (which the SONIC deploy owns
while it is running), every computed target is republished onto two ZMQ
wires consumed by the PC2-local merge planner (``pc2_kplanner_onnx.py``,
which owns every SUB bind):

    arm_targets / hand_finger_cmd / stream_mode  (msgpack, PUB-connect :5564)
        -> planner's ArmTargetIngest overlays the 14 arm joints + 2x10 hand
           joints onto its own 31-DOF pose stream (slew-limited, held until
           a passthrough frame clears them)

    planner_cmd  (JSON, PUB-connect :5563)
        -> planner's command thread. Locomotion payload carries
           ``target_velocity: [yaw_rate, vel_x, vel_z, hip_h]`` directly
           (``source="vr"`` with the same ownership / 2 s-silence-release
           semantics as the Quest 3 manager). The joystick drives LINEAR
           velocity only, at a FIXED speed past the deadzone (no
           proportional scaling); angular-velocity (turn) control is
           intentionally not wired -- channel 0 is always 0.

Topology (everything on PC2; every socket here connects OUT to the
planner, so start order between bridge and planner does not matter):

    Pico controllers -> XRoboToolkit PC Service (:60061)
      -> this bridge
         |-- arm_targets/hand_finger_cmd --> 127.0.0.1:5564 (planner SUB)
         `-- planner_cmd ------------------> 127.0.0.1:5563 (planner SUB)
    planner (legs/waist/head + arm/hand overlay)
      -> pose :5556 -> pose watchdog :5558 -> SONIC deploy -> robot.

Behaviour notes
---------------
* Grip released -> the inherited XRoboToolkit ramp returns that arm to
  the XR default pose (``arm_return_zero_targets``), exactly like the
  direct-hardware teleop. The ramp values are what gets published, so
  "no input -> back to default" is preserved end-to-end.
* Bridge start: the first published frames ramp from the robot's
  *current* arm pose (SONIC stand) to the XR default over ~1-2 s.
* Left stick: forward/back = walk fwd/back, left/right = strafe.
  Fixed-speed response (default): any deflection past the deadzone
  commands the full ``--walk-max-lin-vel`` / ``--walk-max-side-mps``
  speed in that direction -- the stick is a direction switch, not a
  throttle. Pass ``--walk-proportional`` for the legacy
  deflection-scaled response. Right stick is ignored (no turn control).
* Ctrl-C: a final ``passthrough_arm_targets=True`` frame hands the arms
  back to the planner's stand pose and a ``vr_release`` planner_cmd
  frees the planner. Software e-stop (hold right_menu_button 0.5 s)
  instead holds the current arm pose (no passthrough) and stops.
* Nothing is published to ``/aima/hal/joint/arm/command``,
  ``/aima/hal/joint/hand/command`` or ``/x2/cmd_vel`` -- the SONIC
  deploy stays the single hardware commander. The bridge only *reads*
  ``/aima/hal/joint/arm/state`` for IK feedback.

The first section of this file (wire helpers + hand mapping) is
deliberately free of ROS / XRoboToolkit imports so it can be unit-tested
off-robot (``tests/test_x2_xrt_sonic_bridge.py``). Everything below the
``_import_xrt_modules`` call only runs on PC2.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Wire-format helpers (pure python + numpy + msgpack; no ROS / XRT imports).
# ---------------------------------------------------------------------------

# 7-DOF arm joint order on the wire. Matches BOTH the SONIC MuJoCo 31-DOF
# body_q slices (15..21 left / 22..28 right, see
# gear_sonic/utils/planner/constants.py MUJOCO_JOINT_NAMES) and the
# XRoboToolkit ARM_MODEL_JOINT_NAMES order, so a plain dict lookup in this
# order is the whole mapping -- no permutation needed.
ARM_JOINT_ORDER_LEFT: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)
ARM_JOINT_ORDER_RIGHT: tuple[str, ...] = tuple(
    name.replace("left_", "right_", 1) for name in ARM_JOINT_ORDER_LEFT
)
ARM_DOF_PER_SIDE: int = 7

NUM_HAND_DOF_PER_SIDE: int = 10

# ── OmniHand grasp anchors (degrees) ────────────────────────────────────────
# Vendored from gear_sonic/utils/teleop/x2_hand_retarget.py
# (HAND_GRASP_OPEN/CLOSED_{LEFT,RIGHT}_DEG, 2026-06-03 revision) so this
# file stays self-contained on PC2 (the gear_sonic package is not
# installed there). tests/test_x2_xrt_sonic_bridge.py asserts the mapping
# stays byte-identical with the source of truth -- if you change one,
# change both. Joint order is the firmware-canonical
# HAND_FINGER_NAMES_PER_SIDE:
#   thumb_roll thumb_abad thumb_mcp index_abad index_pip
#   middle_pip ring_abad ring_pip pinky_abad pinky_pip
_HAND_OPEN_LEFT_DEG: tuple[float, ...] = (
    -12.0, 55.0, -5.0, 0.0, 5.0, 5.0, 0.0, 5.0, 0.0, 5.0,
)
_HAND_CLOSED_LEFT_DEG: tuple[float, ...] = (
    -40.0, 80.0, -40.0, 6.0, 88.0, 88.0, -5.0, 88.0, -5.0, 88.0,
)
_HAND_OPEN_RIGHT_DEG: tuple[float, ...] = (
    12.0, -55.0, 5.0, 0.0, 5.0, 5.0, 0.0, 5.0, 0.0, 5.0,
)
_HAND_CLOSED_RIGHT_DEG: tuple[float, ...] = (
    40.0, -80.0, 40.0, -6.0, 88.0, 88.0, 5.0, 88.0, 5.0, 88.0,
)
_HAND_LIMITS_LEFT_DEG: tuple[tuple[float, float], ...] = (
    (-50.0, 10.0), (0.0, 100.0), (-49.0, 0.0),
    (0.0, 12.0), (0.0, 90.0), (0.0, 90.0),
    (-10.0, 0.0), (0.0, 90.0), (-10.0, 0.0), (0.0, 90.0),
)
_HAND_LIMITS_RIGHT_DEG: tuple[tuple[float, float], ...] = (
    (-10.0, 50.0), (-100.0, 0.0), (0.0, 49.0),
    (-12.0, 0.0), (0.0, 90.0), (0.0, 90.0),
    (0.0, 10.0), (0.0, 90.0), (0.0, 10.0), (0.0, 90.0),
)


def _deg_table_to_rad(table: Iterable[float]) -> np.ndarray:
    return np.asarray([math.radians(v) for v in table], dtype=np.float64)


def _deg_limits_to_rad(
    table: Iterable[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(math.radians(lo), math.radians(hi)) for lo, hi in table]
    lo = np.asarray([p[0] for p in pairs], dtype=np.float64)
    hi = np.asarray([p[1] for p in pairs], dtype=np.float64)
    return lo, hi


def grasp_command_from_ratio(side: str, ratio: float) -> np.ndarray:
    """Map a 0..1 grasp ratio (0=open, 1=closed) to 10 OmniHand joints (rad).

    Mirrors ``gear_sonic.utils.teleop.x2_hand_retarget.grasp_command_from_ratio``
    (lerp OPEN->CLOSED then clamp to firmware limits). Kept in sync by test.
    """
    if side == "left":
        open_q = _deg_table_to_rad(_HAND_OPEN_LEFT_DEG)
        closed_q = _deg_table_to_rad(_HAND_CLOSED_LEFT_DEG)
        lo, hi = _deg_limits_to_rad(_HAND_LIMITS_LEFT_DEG)
    elif side == "right":
        open_q = _deg_table_to_rad(_HAND_OPEN_RIGHT_DEG)
        closed_q = _deg_table_to_rad(_HAND_CLOSED_RIGHT_DEG)
        lo, hi = _deg_limits_to_rad(_HAND_LIMITS_RIGHT_DEG)
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    r = float(np.clip(ratio, 0.0, 1.0))
    cmd = (1.0 - r) * open_q + r * closed_q
    return np.clip(cmd, lo, hi)


def split_arm_targets(
    joint_targets: Dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Split a 14-joint {name: rad} dict into (left[7], right[7]) wire order."""
    left = np.asarray(
        [float(joint_targets[name]) for name in ARM_JOINT_ORDER_LEFT],
        dtype=np.float64,
    )
    right = np.asarray(
        [float(joint_targets[name]) for name in ARM_JOINT_ORDER_RIGHT],
        dtype=np.float64,
    )
    return left, right


def pack_arm_targets(
    left_q_rad: np.ndarray,
    right_q_rad: np.ndarray,
    *,
    is_engaged: bool,
    tick: int,
    passthrough_arm_targets: bool = False,
) -> bytes:
    """msgpack payload for the planner's ``arm_targets`` ingest topic.

    Keys match ``ArmTargetIngest._loop`` in pc2_kplanner_onnx.py.
    """
    import msgpack

    return msgpack.packb(
        {
            "left_q_rad": [float(v) for v in left_q_rad],
            "right_q_rad": [float(v) for v in right_q_rad],
            "is_engaged": bool(is_engaged),
            "passthrough_arm_targets": bool(passthrough_arm_targets),
            "tick": int(tick),
            "ts": time.time(),
        },
        use_bin_type=True,
    )


def pack_hand_finger_cmd(
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    *,
    tick: int,
) -> bytes:
    """msgpack payload for the planner's ``hand_finger_cmd`` ingest topic."""
    import msgpack

    return msgpack.packb(
        {
            "left_hand_q": [float(v) for v in left_hand_q],
            "right_hand_q": [float(v) for v in right_hand_q],
            "tick": int(tick),
            "ts": time.time(),
        },
        use_bin_type=True,
    )


def pack_stream_mode(mode: str, *, tick: int) -> bytes:
    """msgpack payload for the ``stream_mode`` topic (operator display)."""
    import msgpack

    return msgpack.packb(
        {"mode": str(mode), "tick": int(tick), "ts": time.time()},
        use_bin_type=True,
    )


# ── planner_cmd (JSON) builders — keys exactly match what the planner's
# command thread parses (``pc2_kplanner_onnx._zmq_command_thread``). ─────────

# Planner-side default hip height (pc2_kplanner_onnx._HIP_HEIGHT_M). The
# 4-D velocity intent always carries it explicitly.
DEFAULT_HIP_HEIGHT_M: float = 0.687


def build_planner_cmd_locomotion(
    vel_x: float,
    vel_z: float,
    hip_height_m: float = DEFAULT_HIP_HEIGHT_M,
) -> Dict[str, Any]:
    """Continuous direct-velocity locomotion command.

    ``vel_x`` = lateral m/s (+left / -right), ``vel_z`` = forward m/s
    (+fwd / -back). The angular channel is pinned to 0: turn control is
    deliberately not wired (linear-velocity-only operation).
    """
    return {
        "intent": "locomotion",
        "magnitude": "continuous",
        "source": "vr",
        "target_velocity": [
            0.0,
            float(vel_x),
            float(vel_z),
            float(hip_height_m),
        ],
    }


def build_planner_cmd_idle() -> Dict[str, Any]:
    """Stand-still command that KEEPS vr ownership (sticks centred)."""
    return {"intent": "idle", "magnitude": "default", "source": "vr"}


def build_planner_cmd_vr_release() -> Dict[str, Any]:
    """The ONLY message that hands planner ownership back from vr to pad."""
    return {
        "intent": "idle",
        "magnitude": "default",
        "source": "vr",
        "vr_release": True,
    }


def linear_velocity_from_sticks(
    lx: float,
    ly: float,
    *,
    max_lin_vel: float,
    max_side_mps: float,
    flip_vx: bool = False,
    flip_vy: bool = False,
    proportional: bool = False,
) -> tuple[float, float]:
    """Map post-deadzone stick deflections to (vel_x, vel_z) in m/s.

    Fixed-speed walking (default): any deflection past the deadzone
    commands the FULL speed in that direction -- the deflection
    magnitude is ignored, only its sign picks the direction. The stick
    behaves as a per-axis 3-position switch (centre = stop, either
    direction = full speed). Pass ``proportional=True`` for the legacy
    deflection-scaled response.

    Linear-velocity-only walking: there is NO angular channel -- the
    planner's yaw intent stays 0 and the right stick is never read.

    Frame conventions (planner velocity intent):
      ``vel_z`` forward m/s (+fwd):  ``ly``  (push forward -> walk forward)
      ``vel_x`` lateral m/s (+left): ``-lx`` (push right   -> strafe right)
    """
    vx_sign = -1.0 if flip_vx else 1.0
    vy_sign = -1.0 if flip_vy else 1.0
    if proportional:
        vel_z = float(ly) * float(max_lin_vel) * vx_sign
        vel_x = -float(lx) * float(max_side_mps) * vy_sign
        return vel_x, vel_z
    fwd = float(max_lin_vel) * vx_sign
    side = float(max_side_mps) * vy_sign
    if ly > 0.0:
        vel_z = fwd
    elif ly < 0.0:
        vel_z = -fwd
    else:
        vel_z = 0.0
    if lx > 0.0:
        vel_x = -side
    elif lx < 0.0:
        vel_x = side
    else:
        vel_x = 0.0
    return vel_x, vel_z


class WalkZeroConfirm:
    """Reject brief zero-axis dropouts without delaying an explicit stop.

    The Pico/XRoboToolkit service can emit a transient ``[0, 0]`` sample
    while the controller stream reconnects. Sending that one sample straight
    through changes a held walk command to planner ``idle`` and repeatedly
    restarts the slow-walk state machine. Keep the last non-zero velocity for
    a small, bounded confirmation window; a sustained centred stick then
    stops normally. Call :meth:`reset` for safety-critical stops (deadman,
    malformed input, e-stop, or shutdown) so those are never delayed.
    """

    def __init__(self, *, zero_confirm_s: float) -> None:
        self.zero_confirm_s = float(zero_confirm_s)
        if not math.isfinite(self.zero_confirm_s) or self.zero_confirm_s < 0.0:
            raise ValueError("zero_confirm_s must be a finite value >= 0")
        self._last_nonzero: Optional[tuple[float, float]] = None
        self._last_nonzero_at_s: Optional[float] = None

    def reset(self) -> None:
        """Forget any held walk command and make the next zero immediate."""
        self._last_nonzero = None
        self._last_nonzero_at_s = None

    def update(
        self,
        *,
        vel_x: float,
        vel_z: float,
        now_s: float,
    ) -> tuple[float, float, bool]:
        """Return ``(vel_x, vel_z, holding_zero_glitch)`` for this tick."""
        vel_x = float(vel_x)
        vel_z = float(vel_z)
        now_s = float(now_s)
        if not all(math.isfinite(value) for value in (vel_x, vel_z, now_s)):
            self.reset()
            return 0.0, 0.0, False
        if vel_x != 0.0 or vel_z != 0.0:
            self._last_nonzero = (vel_x, vel_z)
            self._last_nonzero_at_s = now_s
            return vel_x, vel_z, False
        if (
            self._last_nonzero is not None
            and self._last_nonzero_at_s is not None
            and now_s - self._last_nonzero_at_s < self.zero_confirm_s
        ):
            return *self._last_nonzero, True
        self.reset()
        return 0.0, 0.0, False


# Default stream_mode the planner displays while this bridge drives the
# arms. Cosmetic only (the merge gate keys off arm_targets presence).
DEFAULT_STREAM_MODE = "ARM_MANIPULATION"

# Topic names (must match the planner's SUB filters).
ARM_TARGETS_TOPIC = "arm_targets"
HAND_FINGER_CMD_TOPIC = "hand_finger_cmd"
STREAM_MODE_TOPIC = "stream_mode"
PLANNER_CMD_TOPIC = "planner_cmd"

# Default ports: the planner binds its SUB sockets here (it is launched
# with ``--cmd-bind --cmd-port 5563 --arm-port 5564``); this bridge
# PUB-connects to both over loopback.
DEFAULT_ARM_HANDS_PORT = 5564
DEFAULT_PLANNER_CMD_PORT = 5563


class SonicZmqSink:
    """Owns the two PUB sockets and serialises every outbound wire message.

    Both sockets CONNECT to the PC2-local merge planner (which owns the
    SUB binds), so the bridge and planner can start in any order and
    either one reconnects automatically after a restart.

    Thread-safety: the arm/hands socket is driven from the teleop control
    thread, the planner socket from the IK thread (plus the shutdown hook).
    Each socket has its own lock so a send never interleaves mid-message.
    """

    def __init__(
        self,
        *,
        arm_hands_host: str = "127.0.0.1",
        arm_hands_port: int = DEFAULT_ARM_HANDS_PORT,
        planner_cmd_host: str = "127.0.0.1",
        planner_cmd_port: int = DEFAULT_PLANNER_CMD_PORT,
        arm_hands_topic: str = ARM_TARGETS_TOPIC,
        hand_topic: str = HAND_FINGER_CMD_TOPIC,
        stream_mode_topic: str = STREAM_MODE_TOPIC,
        planner_topic: str = PLANNER_CMD_TOPIC,
        stream_mode: str = DEFAULT_STREAM_MODE,
    ) -> None:
        import zmq

        self._ctx = zmq.Context.instance()
        self._arm_hands_sock = self._ctx.socket(zmq.PUB)
        self._arm_hands_sock.setsockopt(zmq.LINGER, 1000)
        self._arm_hands_sock.connect(f"tcp://{arm_hands_host}:{arm_hands_port}")
        self._planner_sock = self._ctx.socket(zmq.PUB)
        self._planner_sock.setsockopt(zmq.LINGER, 1000)
        self._planner_sock.connect(f"tcp://{planner_cmd_host}:{planner_cmd_port}")

        self._arm_hands_lock = threading.Lock()
        self._planner_lock = threading.Lock()
        self._arm_topic = arm_hands_topic.encode("ascii")
        self._hand_topic = hand_topic.encode("ascii")
        self._stream_topic = stream_mode_topic.encode("ascii")
        self._planner_topic = planner_topic.encode("ascii")
        self._stream_mode = stream_mode
        self._tick = 0
        self._last_walk_idle = True
        self.arm_hands_endpoint = f"tcp://{arm_hands_host}:{arm_hands_port}"
        self.planner_cmd_endpoint = f"tcp://{planner_cmd_host}:{planner_cmd_port}"

    def _next_tick(self) -> int:
        self._tick += 1
        return self._tick

    # ── control-thread publishers (arm socket) ──────────────────────────
    def publish_arm_targets(
        self,
        joint_targets: Dict[str, float],
        *,
        is_engaged: bool,
    ) -> None:
        left, right = split_arm_targets(joint_targets)
        payload = pack_arm_targets(
            left, right, is_engaged=is_engaged, tick=self._next_tick()
        )
        with self._arm_hands_lock:
            self._arm_hands_sock.send_multipart([self._arm_topic, payload])

    def publish_hand_targets(
        self,
        *,
        left_claw: float,
        right_claw: float,
    ) -> None:
        """``*_claw`` follows the XRoboToolkit convention: 1.0=open, 0.0=closed."""
        left_q = grasp_command_from_ratio("left", 1.0 - float(left_claw))
        right_q = grasp_command_from_ratio("right", 1.0 - float(right_claw))
        payload = pack_hand_finger_cmd(left_q, right_q, tick=self._tick)
        with self._arm_hands_lock:
            self._arm_hands_sock.send_multipart([self._hand_topic, payload])

    def publish_stream_mode(self) -> None:
        payload = pack_stream_mode(self._stream_mode, tick=self._tick)
        with self._arm_hands_lock:
            self._arm_hands_sock.send_multipart([self._stream_topic, payload])

    def publish_arm_passthrough(self) -> None:
        """One-shot 'no arm override' sentinel -> planner arms flow through."""
        zeros = np.zeros(ARM_DOF_PER_SIDE, dtype=np.float64)
        payload = pack_arm_targets(
            zeros,
            zeros,
            is_engaged=False,
            tick=self._next_tick(),
            passthrough_arm_targets=True,
        )
        with self._arm_hands_lock:
            self._arm_hands_sock.send_multipart([self._arm_topic, payload])

    # ── IK-thread publisher (planner socket) ────────────────────────────
    def publish_walk(
        self,
        *,
        vel_x: float,
        vel_z: float,
        hip_height_m: float = DEFAULT_HIP_HEIGHT_M,
    ) -> None:
        """Steady-rate walk heartbeat: locomotion while any axis is live.

        ``vel_x`` lateral m/s (+left), ``vel_z`` forward m/s (+fwd); both
        are direct velocity setpoints (fixed magnitude past the stick
        deadzone unless the proportional response is enabled).
        """
        if abs(vel_x) > 0.0 or abs(vel_z) > 0.0:
            payload: Dict[str, Any] = build_planner_cmd_locomotion(
                vel_x, vel_z, hip_height_m
            )
            self._last_walk_idle = False
        else:
            payload = build_planner_cmd_idle()
            self._last_walk_idle = True
        blob = json.dumps(payload).encode("utf-8")
        with self._planner_lock:
            self._planner_sock.send_multipart([self._planner_topic, blob])

    def publish_walk_release(self) -> None:
        blob = json.dumps(build_planner_cmd_vr_release()).encode("utf-8")
        with self._planner_lock:
            self._planner_sock.send_multipart([self._planner_topic, blob])

    def close(self) -> None:
        for sock, lock in (
            (self._arm_hands_sock, self._arm_hands_lock),
            (self._planner_sock, self._planner_lock),
        ):
            with lock:
                try:
                    sock.close(linger=1000)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# PC2-only section: XRoboToolkit + ROS2 glue. Imported lazily so the helpers
# above stay importable on any machine (unit tests, no ROS).
# ---------------------------------------------------------------------------

DEFAULT_XRT_REPO = Path("/agibot/data/home/agi/XRoboToolkit-Teleop-Sample-Python")

# Private command topics the bridge's arm/head interfaces bind their (never
# used) ROS2 publishers to. Nobody subscribes; the real hardware topics
# stay untouched so the SONIC deploy remains the sole commander.
_PRIVATE_ARM_COMMAND_TOPIC = "/x2_sonic_bridge/arm_command_unused"
_PRIVATE_HEAD_COMMAND_TOPIC = "/x2_sonic_bridge/head_command_unused"


def _import_xrt_modules(xrt_repo: Path):
    """Put the XRoboToolkit teleop repo on sys.path and import the controller."""
    if str(xrt_repo) not in sys.path:
        sys.path.insert(0, str(xrt_repo))
    from xrobotoolkit_teleop.hardware.x2_ros2_teleop_controller import (  # noqa: E402
        DEFAULT_X2_MANIPULATOR_CONFIG,
        DEFAULT_X2_UPPER_BODY_URDF_PATH,
        Ros2JointGroupInterface,
        X2Ros2TeleopController,
    )

    return (
        X2Ros2TeleopController,
        Ros2JointGroupInterface,
        DEFAULT_X2_MANIPULATOR_CONFIG,
        DEFAULT_X2_UPPER_BODY_URDF_PATH,
    )


def build_controller_class(xrt_repo: Path):
    """Build the SonicBridgeController class bound to the XRT modules."""
    (
        X2Ros2TeleopController,
        Ros2JointGroupInterface,
        _DEFAULT_MANIPULATOR_CONFIG,
        _DEFAULT_URDF,
    ) = _import_xrt_modules(xrt_repo)

    class _SonicArmInterface(Ros2JointGroupInterface):
        """Arm interface whose command publisher is redirected to the ZMQ sink.

        The ROS2 publisher still exists (on a private unused topic) but
        ``publish_command`` never emits ROS traffic -- the final,
        fully-limited joint targets (post IK, post inactive-arm
        return-to-default ramp, post step limiters, post e-stop hold) are
        forwarded to the planner's arm-ingest wire instead.
        """

        def __init__(self, *args, sink: SonicZmqSink, engaged_fn: Callable[[], bool], **kwargs):
            super().__init__(*args, **kwargs)
            self._sonic_sink = sink
            self._engaged_fn = engaged_fn

        def publish_command(
            self,
            joint_targets: Dict[str, float],
            command_specs: Dict[str, Any],
            joint_velocities: Optional[Dict[str, float]] = None,
        ) -> None:
            self._sonic_sink.publish_arm_targets(
                joint_targets, is_engaged=self._engaged_fn()
            )

    class _SonicHandStub:
        """Duck-typed replacement for Ros2HandCommandInterface.

        Receives the XRoboToolkit claw setpoints (1.0=open / 0.0=closed)
        and republishes them as 10-DOF OmniHand frames on the planner's
        ingest wire. ``destroy_node`` is a no-op so the parent's
        ``_shutdown_robot`` loop treats it like a ROS node.
        """

        def __init__(self, sink: SonicZmqSink):
            self._sonic_sink = sink

        def publish_command(self, left_position: float, right_position: float) -> None:
            self._sonic_sink.publish_hand_targets(
                left_claw=left_position, right_claw=right_position
            )

        def destroy_node(self) -> None:  # parity with rclpy Node API
            pass

    class SonicBridgeController(X2Ros2TeleopController):
        """X2Ros2TeleopController with SONIC ZMQ sinks instead of ROS cmd topics."""

        # Bridge-only walk knobs the base controller does not know about
        # (set from CLI args after construction). False = fixed-speed
        # response (default), True = legacy deflection-scaled response.
        walk_proportional = False

        def __init__(self, *args, sonic_sink: SonicZmqSink, **kwargs):
            self._sonic_sink = sonic_sink
            # Replaced from CLI configuration after construction. A zero
            # window is the safe default for any external construction path.
            self.walk_zero_confirm = WalkZeroConfirm(zero_confirm_s=0.0)
            super().__init__(*args, **kwargs)

        # ── hardware setup: state subscription only, no hardware pubs ──
        def _robot_setup(self):
            if self._ros2_setup_complete:
                return

            import rclpy
            from rclpy.executors import MultiThreadedExecutor

            if not rclpy.ok():
                rclpy.init(args=None)

            self.arm_interface = _SonicArmInterface(
                node_name="x2_sonic_arm_interface",
                state_topic=self.arm_state_topic,
                command_topic=_PRIVATE_ARM_COMMAND_TOPIC,
                joint_names=self.arm_hardware_joint_names,
                enable_effort=self.enable_log_torque,
                sink=self._sonic_sink,
                engaged_fn=lambda: bool(
                    self.active.get("left_arm") or self.active.get("right_arm")
                ),
            )
            if self.enable_head_state_feedback:
                self.head_interface = Ros2JointGroupInterface(
                    node_name="x2_sonic_head_interface",
                    state_topic=self.head_state_topic,
                    command_topic=_PRIVATE_HEAD_COMMAND_TOPIC,
                    joint_names=self.head_hardware_joint_names,
                )
            else:
                self.head_interface = None
            self.hand_interface = _SonicHandStub(self._sonic_sink)
            self.walk_interface = None

            self._executor = MultiThreadedExecutor()
            self._executor.add_node(self.arm_interface)
            if self.head_interface is not None:
                self._executor.add_node(self.head_interface)
            self._spin_thread = threading.Thread(
                target=self._executor.spin, name="x2_sonic_ros2_spin", daemon=True
            )
            self._spin_thread.start()

            print("[sonic-bridge] waiting for arm joint state topic: "
                  f"{self.arm_state_topic} ...", flush=True)
            arm_ready = self.arm_interface.wait_for_state(self.initial_state_timeout_s)
            if not arm_ready:
                raise RuntimeError(
                    "Timed out waiting for arm joint state topic: "
                    f"{self.arm_state_topic}"
                )
            if self.enable_head_state_feedback and self.head_interface is not None:
                head_ready = self.head_interface.wait_for_state(
                    self.initial_state_timeout_s
                )
                if not head_ready:
                    raise RuntimeError(
                        "Timed out waiting for head joint state topic: "
                        f"{self.head_state_topic}"
                    )

            self._arm_last_state_seq = self.arm_interface.get_state_seq()
            self._head_last_state_seq = (
                self.head_interface.get_state_seq()
                if self.head_interface is not None
                else -1
            )
            self._ros2_setup_complete = True
            print("[sonic-bridge] ROS2 state subscription ready.", flush=True)

        # ── joystick -> direct linear-velocity planner_cmd ─────────────
        def _update_walk_command(self):
            if not self.enable_walk_control:
                return
            try:
                left = self.xr_client.get_joystick_state("left")
            except Exception as exc:
                print(f"[sonic-bridge] failed to read joystick state: {exc}")
                # A read failure is not a zero-sample glitch. Fail closed
                # instead of leaving the planner's prior velocity live until
                # its ownership watchdog expires.
                self.walk_zero_confirm.reset()
                self._last_walk_command = (0.0, 0.0, 0.0)
                self._sonic_sink.publish_walk(vel_x=0.0, vel_z=0.0)
                return

            try:
                raw_lx, raw_ly = float(left[0]), float(left[1])
            except (IndexError, TypeError, ValueError) as exc:
                print(f"[sonic-bridge] invalid left joystick sample: {exc}")
                self.walk_zero_confirm.reset()
                self._last_walk_command = (0.0, 0.0, 0.0)
                self._sonic_sink.publish_walk(vel_x=0.0, vel_z=0.0)
                return
            if not all(math.isfinite(value) for value in (raw_lx, raw_ly)):
                print("[sonic-bridge] non-finite left joystick sample; stopping walk")
                self.walk_zero_confirm.reset()
                self._last_walk_command = (0.0, 0.0, 0.0)
                self._sonic_sink.publish_walk(vel_x=0.0, vel_z=0.0)
                return

            lx = self._apply_walk_deadzone(raw_lx)
            ly = self._apply_walk_deadzone(raw_ly)

            # Fixed-speed walk by default: any deflection past the
            # deadzone commands the full speed in that direction;
            # angular (turn) control is intentionally not wired.
            vel_x, vel_z = linear_velocity_from_sticks(
                lx,
                ly,
                max_lin_vel=self.walk_max_lin_vel,
                max_side_mps=self.walk_max_side_mps,
                flip_vx=self.walk_vx_sign < 0.0,
                flip_vy=self.walk_vy_sign < 0.0,
                proportional=self.walk_proportional,
            )

            deadman_inactive = (
                self.walk_deadman is not None
                and not self._is_walk_deadman_active()
            )
            if deadman_inactive:
                # Deadman release is an explicit operator stop, not a noisy
                # XR zero. It must always take effect on this tick.
                self.walk_zero_confirm.reset()
                vel_x = vel_z = 0.0
                holding_zero_glitch = False
            else:
                vel_x, vel_z, holding_zero_glitch = self.walk_zero_confirm.update(
                    vel_x=vel_x,
                    vel_z=vel_z,
                    now_s=time.monotonic(),
                )

            self._last_walk_command = (float(vel_z), float(vel_x), 0.0)
            self._sonic_sink.publish_walk(
                vel_x=vel_x,
                vel_z=vel_z,
                hip_height_m=self.walk_hip_height_m,
            )

            if self.walk_debug_print:
                now = time.time()
                if now - self._last_walk_debug_time >= 0.5:
                    self._last_walk_debug_time = now
                    print(
                        f"[sonic-bridge walk] L=({raw_lx:+.3f},{raw_ly:+.3f}) -> "
                        f"vel_z={vel_z:+.3f} m/s vel_x={vel_x:+.3f} m/s "
                        "(linear-only, no yaw; "
                        f"zero_confirm={'HOLD' if holding_zero_glitch else 'LIVE'})"
                    )

        # ── signal handling: rclpy.init swallows SIGINT/SIGTERM ─────────
        # rclpy registers its own SIGINT/SIGTERM handlers at init time
        # (they wake its guard conditions but never raise
        # KeyboardInterrupt), so the stock ``run()`` never sees Ctrl-C when
        # the process runs under ssh/tmux/nohup. Reinstall plain handlers
        # AFTER rclpy.init (run() is always post-init) that drive the same
        # cooperative shutdown path the software e-stop uses.
        def run(self):
            import signal as _signal

            self._external_stop_requested = False

            def _on_external_signal(signum, _frame):
                self._external_stop_requested = True

            _signal.signal(_signal.SIGINT, _on_external_signal)
            _signal.signal(_signal.SIGTERM, _on_external_signal)
            super().run()

        def _should_keep_running(self) -> bool:
            if getattr(self, "_external_stop_requested", False):
                return False
            return super()._should_keep_running()

        # ── periodic stream_mode heartbeat on the arm socket ────────────
        def _send_command(self):
            # The parent's _send_command drives our intercepted publishers
            # every tick; stream_mode rides along at ~1 Hz for the
            # operator display.
            if self._sonic_sink._tick % max(1, int(self.control_rate_hz)) == 0:
                self._sonic_sink.publish_stream_mode()
            super()._send_command()

        # ── shutdown: final wire messages, then parent teardown ─────────
        def _shutdown_robot(self):
            estop = bool(getattr(self, "_software_estop_triggered", False))
            try:
                if estop:
                    # E-stop path already published a hold-pose arm frame;
                    # leave the planner's arm cache untouched so the arms
                    # FREEZE instead of migrating back to the stand pose.
                    print(
                        "[sonic-bridge] e-stop shutdown: arms hold current "
                        "pose (no passthrough sent).",
                        flush=True,
                    )
                else:
                    # Clean exit: hand arms back to the planner stand pose.
                    self._sonic_sink.publish_arm_passthrough()
                    print(
                        "[sonic-bridge] clean shutdown: arm passthrough sent "
                        "(planner stand pose takes over).",
                        flush=True,
                    )
                # Either way release the planner so walking stops now
                # instead of after the 2 s VR-silence timeout.
                self._sonic_sink.publish_walk_release()
                # Give the PUB sockets a moment to flush before the process
                # tears them down (both sockets carry LINGER=1000 ms; they
                # are intentionally NOT closed here -- the IK thread may
                # still be mid-publish, daemon, exits on stop_event).
                time.sleep(0.4)
            except Exception as exc:
                print(f"[sonic-bridge] shutdown wire messages failed: {exc}")
            super()._shutdown_robot()

    return SonicBridgeController


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--xrt-repo",
        type=Path,
        default=DEFAULT_XRT_REPO,
        help="Path to the XRoboToolkit-Teleop-Sample-Python checkout on PC2.",
    )
    parser.add_argument(
        "--arm-state-topic",
        default="/aima/hal/joint/arm/state",
        help="ROS2 topic with the real arm joint feedback (read-only).",
    )
    parser.add_argument(
        "--head-state-topic",
        default="/aima/hal/joint/head/state",
        help="ROS2 topic with the real head joint feedback (read-only).",
    )
    parser.add_argument("--arm-hands-host", default="127.0.0.1",
                        help="Planner arm-ingest host (the planner SUB-binds "
                             "--arm-port there; this bridge PUB-connects).")
    parser.add_argument("--arm-hands-port", type=int, default=DEFAULT_ARM_HANDS_PORT)
    parser.add_argument("--planner-cmd-host", default="127.0.0.1",
                        help="Planner command-thread host (the planner "
                             "SUB-binds --cmd-port there with --cmd-bind).")
    parser.add_argument("--planner-cmd-port", type=int, default=DEFAULT_PLANNER_CMD_PORT)
    parser.add_argument("--control-rate-hz", type=int, default=30)
    parser.add_argument("--scale-factor", type=float, default=1.2)
    parser.add_argument(
        "--software-estop-button", default="right_menu_button",
    )
    parser.add_argument("--software-estop-hold-s", type=float, default=0.5)
    parser.add_argument("--max-arm-joint-step-rad", type=float, default=1.0)
    parser.add_argument(
        "--inactive-arm-return-max-step-rad", type=float, default=0.03,
        help="Ramp rate (rad/cycle) of the inactive-arm return-to-default.",
    )
    parser.add_argument("--enable-ruckig-smoothing", action="store_true")
    parser.add_argument("--walk-deadzone", type=float, default=0.1)
    parser.add_argument(
        "--walk-max-lin-vel", type=float, default=0.22,
        help="Forward/backward speed (m/s) commanded at ANY deflection "
             "past the deadzone (fixed-speed response).",
    )
    parser.add_argument(
        "--walk-max-side-mps", type=float, default=0.22,
        help="Lateral speed (m/s) commanded at ANY deflection past the "
             "deadzone (fixed-speed response).",
    )
    parser.add_argument(
        "--walk-proportional", action="store_true",
        help="Restore the legacy deflection-scaled speed response "
             "(default: fixed full speed past the deadzone).",
    )
    parser.add_argument(
        "--hip-height-m", type=float, default=DEFAULT_HIP_HEIGHT_M,
        help="Hip-height channel of the velocity intent (planner default "
             "0.687 m).",
    )
    parser.add_argument(
        "--walk-deadman",
        default="none",
        help="Optional deadman input name (e.g. left_trigger) for walking.",
    )
    parser.add_argument(
        "--walk-zero-confirm-s", type=float, default=0.25,
        help="Require this many seconds of continuous centred joystick input "
             "before stopping a live walk command (masks brief XR reconnect "
             "zero samples; deadman/e-stop remain immediate).",
    )
    parser.add_argument("--walk-flip-vx", action="store_true")
    parser.add_argument("--walk-flip-vy", action="store_true")
    parser.add_argument("--walk-debug-print", action="store_true")
    parser.add_argument("--debug-print-targets", action="store_true")
    parser.add_argument(
        "--release-arms-once",
        action="store_true",
        help="Do not start teleop: emit a single arm-passthrough + "
             "walk-release frame and exit (clears a stale arm override "
             "left in the planner by a previous session).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    sink = SonicZmqSink(
        arm_hands_host=args.arm_hands_host,
        arm_hands_port=args.arm_hands_port,
        planner_cmd_host=args.planner_cmd_host,
        planner_cmd_port=args.planner_cmd_port,
    )
    print(
        f"[sonic-bridge] PUB-connect arm/hands {sink.arm_hands_endpoint} "
        f"topics=[arm_targets, hand_finger_cmd, stream_mode]; "
        f"planner_cmd {sink.planner_cmd_endpoint}",
        flush=True,
    )

    if args.release_arms_once:
        # Slow-joiner: a PUB-connect -> SUB-bind handshake can take ~1 s to
        # settle; one-shot frames must not race it. Continuous streams
        # self-heal, these do not.
        time.sleep(1.5)
        sink.publish_arm_passthrough()
        sink.publish_walk_release()
        time.sleep(0.6)
        sink.close()
        print("[sonic-bridge] release-arms: passthrough + walk release sent.")
        return 0

    if not args.xrt_repo.is_dir():
        raise SystemExit(
            f"XRoboToolkit repo not found at {args.xrt_repo}. "
            "Pass --xrt-repo /path/to/XRoboToolkit-Teleop-Sample-Python"
        )

    SonicBridgeController = build_controller_class(args.xrt_repo)
    (
        _ctrl_cls,
        _iface_cls,
        default_manipulator_config,
        default_urdf,
    ) = _import_xrt_modules(args.xrt_repo)

    controller = SonicBridgeController(
        robot_urdf_path=str(default_urdf),
        manipulator_config=default_manipulator_config,
        scale_factor=args.scale_factor,
        enable_log_data=False,
        visualize_placo=False,
        control_rate_hz=args.control_rate_hz,
        enable_camera=False,
        enable_head_tracking=False,
        enable_head_state_feedback=False,
        arm_state_topic=args.arm_state_topic,
        head_state_topic=args.head_state_topic,
        # Hardware command topics are pointed at private unused topics;
        # the sinks intercept every publish before ROS sees anything.
        arm_command_topic=_PRIVATE_ARM_COMMAND_TOPIC,
        head_command_topic=_PRIVATE_HEAD_COMMAND_TOPIC,
        hand_command_topic="/x2_sonic_bridge/hand_command_unused",
        initial_state_timeout_s=10.0,
        software_estop_button=args.software_estop_button,
        software_estop_hold_s=args.software_estop_hold_s,
        max_arm_joint_step_rad=args.max_arm_joint_step_rad,
        inactive_arm_return_max_step_rad=args.inactive_arm_return_max_step_rad,
        debug_print_targets=args.debug_print_targets,
        enable_ruckig_smoothing=args.enable_ruckig_smoothing,
        enable_walk_control=True,
        walk_command_topic="/x2_sonic_bridge/cmd_vel_unused",
        walk_max_lin_vel=args.walk_max_lin_vel,
        walk_deadzone=args.walk_deadzone,
        walk_deadman=args.walk_deadman,
        walk_flip_vx=args.walk_flip_vx,
        walk_flip_vy=args.walk_flip_vy,
        walk_debug_print=args.walk_debug_print,
        sonic_sink=sink,
    )
    # Bridge-only walk knobs the base controller does not know about.
    controller.walk_max_side_mps = float(args.walk_max_side_mps)
    controller.walk_hip_height_m = float(args.hip_height_m)
    controller.walk_proportional = bool(args.walk_proportional)
    try:
        controller.walk_zero_confirm = WalkZeroConfirm(
            zero_confirm_s=args.walk_zero_confirm_s
        )
    except ValueError as exc:
        raise SystemExit(f"--walk-zero-confirm-s: {exc}") from exc
    walk_response = (
        "proportional to deflection"
        if args.walk_proportional
        else "fixed speed past deadzone"
    )
    print(
        "[sonic-bridge] teleop running. Arms: hold L/R grip to drive; "
        "release to return to default. Walk: L-stick linear velocity "
        f"({walk_response}: fwd/back {args.walk_max_lin_vel} m/s, side "
        f"{args.walk_max_side_mps} m/s; no turn control; XR-zero confirm "
        f"{args.walk_zero_confirm_s:.2f}s). "
        f"E-stop: hold {args.software_estop_button} "
        f"{args.software_estop_hold_s}s. Ctrl-C to exit.",
        flush=True,
    )
    controller.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
