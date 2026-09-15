#!/usr/bin/env python3
"""PC2-local ZMQ pose to X2 AimDK hand-command bridge.

The planner wire carries canonical ten-joint hand poses.  At startup this
bridge reads the robot's latched ``HandStateArray`` and preserves each side's
actual AimDK ``HandType``.  X2 claw hands receive the one-joint 0..1 contract
(``1=open``, ``0=closed``); nimble hands receive the ten-joint contract.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Path-bootstrap so we can import the canonical decoder regardless of
# how the script is launched (direct ``python3 path/to/script.py``,
# colcon-installed script, etc).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    DecodedMessage,
    HEADER_SIZE,
    unpack_message,
)
# ─────────────────────────────────────────────────────────────────────
# Hardcoded AimDK / OmniHand constants. Mirrored verbatim from
# ``agitbot-x2-record-and-replay/src/x2_recorder/constants.py`` --
# kept inline so this script has zero build dep on the recorder repo.
# Source-of-truth for any change is that file (which itself mirrors
# the AgiBot Omnihand-2025-SDK API_PYTHON.md per-side joint ranges).
# ─────────────────────────────────────────────────────────────────────
NUM_HAND_DOF_PER_SIDE = 10

# Canonical ten-joint poses emitted by the PC2 XR bridge. This hardware
# bridge is installed independently in the ROS workspace, so the anchors are
# deliberately self-contained.
HAND_GRASP_OPEN_RAD_LEFT = tuple(
    math.radians(v) for v in (-12.0, 55.0, -5.0, 0.0, 5.0, 5.0, 0.0, 5.0, 0.0, 5.0)
)
HAND_GRASP_CLOSED_RAD_LEFT = tuple(
    math.radians(v) for v in (-40.0, 80.0, -40.0, 6.0, 88.0, 88.0, -5.0, 88.0, -5.0, 88.0)
)
HAND_GRASP_OPEN_RAD_RIGHT = tuple(
    math.radians(v) for v in (12.0, -55.0, 5.0, 0.0, 5.0, 5.0, 0.0, 5.0, 0.0, 5.0)
)
HAND_GRASP_CLOSED_RAD_RIGHT = tuple(
    math.radians(v) for v in (40.0, -80.0, 40.0, -6.0, 88.0, 88.0, 5.0, 88.0, 5.0, 88.0)
)

HAND_FINGER_NAMES_PER_SIDE: tuple[str, ...] = (
    "thumb_roll",
    "thumb_abad",
    "thumb_mcp",
    "index_abad",
    "index_pip",
    "middle_pip",
    "ring_abad",
    "ring_pip",
    "pinky_abad",
    "pinky_pip",
)

# HandType enum values from aimdk_msgs/HandType.msg.
HAND_TYPE_NONE = 0
HAND_TYPE_NIMBLE = 1
HAND_TYPE_CLAW = 2
HAND_TYPE_ERROR = 0xFF

HAND_STATE_TOPIC = "/aima/hal/joint/hand/state"
HAND_CMD_TOPIC = "/aima/hal/joint/hand/command"

# velocity=0.1 is what the OmniHand example sends; empirically motors
# fail to enable when velocity=0 even though the field is documented
# as ignored. Mirrors HAND_CMD_VELOCITY_HINT in the recorder.
HAND_CMD_VELOCITY_HINT = 0.1


def claw_position_from_hand_joints(side: str, positions: np.ndarray) -> float:
    """Recover the X2 claw scalar (1=open, 0=closed) from a 10-DOF pose."""
    q = np.asarray(positions, dtype=np.float64).reshape(-1)
    if q.shape != (NUM_HAND_DOF_PER_SIDE,):
        raise ValueError(f"{side} hand pose must contain 10 joints, got {q.shape}")
    if side == "left":
        open_q = np.asarray(HAND_GRASP_OPEN_RAD_LEFT, dtype=np.float64)
        closed_q = np.asarray(HAND_GRASP_CLOSED_RAD_LEFT, dtype=np.float64)
    elif side == "right":
        open_q = np.asarray(HAND_GRASP_OPEN_RAD_RIGHT, dtype=np.float64)
        closed_q = np.asarray(HAND_GRASP_CLOSED_RAD_RIGHT, dtype=np.float64)
    else:
        raise ValueError(f"side must be left or right, got {side!r}")
    span = closed_q - open_q
    grasp_ratio = float(np.dot(q - open_q, span) / np.dot(span, span))
    return float(np.clip(1.0 - grasp_ratio, 0.0, 1.0))

# Per-motor firmware-enforced angle ranges, degrees, motor-axis order
# (1..10), from Omnihand-2025-SDK/document/en/API_PYTHON.md. Asymmetric
# per side (L/R are mirror images). Mirrors the recorder's
# HAND_JOINT_RANGE_{LEFT,RIGHT}_DEG.
HAND_JOINT_RANGE_LEFT_DEG: tuple[tuple[float, float], ...] = (
    (-50.0,  10.0),  # thumb_roll
    (  0.0, 100.0),  # thumb_abad
    (-49.0,   0.0),  # thumb_mcp
    (  0.0,  12.0),  # index_abad
    (  0.0,  90.0),  # index_pip
    (  0.0,  90.0),  # middle_pip
    (-10.0,   0.0),  # ring_abad
    (  0.0,  90.0),  # ring_pip
    (-10.0,   0.0),  # pinky_abad
    (  0.0,  90.0),  # pinky_pip
)

HAND_JOINT_RANGE_RIGHT_DEG: tuple[tuple[float, float], ...] = (
    ( -10.0,  50.0),  # thumb_roll
    (-100.0,   0.0),  # thumb_abad
    (   0.0,  49.0),  # thumb_mcp
    ( -12.0,   0.0),  # index_abad
    (   0.0,  90.0),  # index_pip
    (   0.0,  90.0),  # middle_pip
    (   0.0,  10.0),  # ring_abad
    (   0.0,  90.0),  # ring_pip
    (   0.0,  10.0),  # pinky_abad
    (   0.0,  90.0),  # pinky_pip
)


def _hand_joint_limits_rad(side: str) -> list[tuple[float, float]]:
    """(lower, upper) in radians per motor for the requested side."""
    table = HAND_JOINT_RANGE_LEFT_DEG if side == "left" else HAND_JOINT_RANGE_RIGHT_DEG
    return [(math.radians(lo), math.radians(hi)) for lo, hi in table]


_HAND_LIMITS_LEFT_RAD = _hand_joint_limits_rad("left")
_HAND_LIMITS_RIGHT_RAD = _hand_joint_limits_rad("right")


def _clamp_per_finger(values: np.ndarray, side: str) -> tuple[np.ndarray, int]:
    """Per-motor clamp + count of joints that hit a limit.

    Inputs shorter than NUM_HAND_DOF_PER_SIDE are zero-padded; longer
    ones are truncated. Returns (clamped[NUM_HAND_DOF_PER_SIDE], n_hit).
    """
    limits = _HAND_LIMITS_LEFT_RAD if side == "left" else _HAND_LIMITS_RIGHT_RAD
    out = np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)
    n_hit = 0
    n = min(len(values), NUM_HAND_DOF_PER_SIDE)
    for i in range(n):
        lo, hi = limits[i]
        v = float(values[i])
        c = max(lo, min(hi, v))
        if c != v:
            n_hit += 1
        out[i] = c
    return out, n_hit


# ─────────────────────────────────────────────────────────────────────
# ZMQ subscriber thread
# ─────────────────────────────────────────────────────────────────────
@dataclass
class _LatestPose:
    """Thread-safe snapshot of the latest decoded pose frame.

    Only the hand fields are published downstream by this bridge; the
    rest of the ``DecodedMessage.fields`` dict is ignored.
    """

    left: Optional[np.ndarray] = None       # raw, pre-clamp (rad)
    right: Optional[np.ndarray] = None      # raw, pre-clamp (rad)
    recv_monotonic: float = 0.0             # monotonic seconds at receive
    frame_index: int = -1
    n_recv: int = 0                         # total decoded frames since start
    n_hand_present: int = 0                 # frames that included any hand field


class _ZmqPoseReader:
    """Background ZMQ SUB that decodes pose messages and updates _LatestPose.

    Importing pyzmq is deferred to construction time so ``--self-test``
    can run on hosts without pyzmq installed.
    """

    def __init__(self, host: str, port: int, topic: str, recv_timeout_ms: int = 200):
        import zmq

        self._zmq = zmq
        self._host = host
        self._port = port
        self._topic = topic
        self._recv_timeout_ms = recv_timeout_ms

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self._sock.setsockopt(zmq.RCVHWM, 8)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._sock.connect(f"tcp://{host}:{port}")

        self._lock = threading.Lock()
        self._latest = _LatestPose()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="x2-hand-bridge-zmq", daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._sock.close(linger=0)
        except Exception:
            pass

    def latest(self) -> _LatestPose:
        with self._lock:
            return _LatestPose(
                left=None if self._latest.left is None else self._latest.left.copy(),
                right=None if self._latest.right is None else self._latest.right.copy(),
                recv_monotonic=self._latest.recv_monotonic,
                frame_index=self._latest.frame_index,
                n_recv=self._latest.n_recv,
                n_hand_present=self._latest.n_hand_present,
            )

    def endpoint(self) -> str:
        return f"tcp://{self._host}:{self._port} (topic={self._topic!r})"

    def _run(self) -> None:
        zmq = self._zmq
        while not self._stop.is_set():
            try:
                raw = self._sock.recv()
            except zmq.error.Again:
                continue
            except Exception as exc:
                # Avoid stdout spam: print at most once per second on errors.
                print(f"[x2-hand-bridge] zmq recv error: {exc}", file=sys.stderr)
                time.sleep(0.5)
                continue

            try:
                msg: DecodedMessage = unpack_message(raw, expected_topic=self._topic)
            except ValueError as exc:
                # Malformed frame -- skip but count.
                print(f"[x2-hand-bridge] decode error: {exc}", file=sys.stderr)
                continue

            left = msg.fields.get("left_hand_joints")
            right = msg.fields.get("right_hand_joints")
            frame_idx_arr = msg.fields.get("frame_index")
            try:
                frame_idx = int(frame_idx_arr[0]) if frame_idx_arr is not None else -1
            except Exception:
                frame_idx = -1

            now = time.monotonic()
            with self._lock:
                self._latest.n_recv += 1
                if left is not None or right is not None:
                    self._latest.n_hand_present += 1
                if left is not None:
                    self._latest.left = np.asarray(left, dtype=np.float32).reshape(-1)
                if right is not None:
                    self._latest.right = np.asarray(right, dtype=np.float32).reshape(-1)
                self._latest.recv_monotonic = now
                self._latest.frame_index = frame_idx


# ─────────────────────────────────────────────────────────────────────
# Self-test (no ROS, no robot)
# ─────────────────────────────────────────────────────────────────────
def _build_synthetic_pose_packet(
    *,
    left: list[float] | None,
    right: list[float] | None,
    topic: str = "pose",
    frame_index: int = 42,
) -> bytes:
    """Construct a minimal valid pose packet matching pack_pose_message."""
    import json

    fields: list[dict] = []
    payload_chunks: list[bytes] = []

    if left is not None:
        fields.append({"name": "left_hand_joints", "dtype": "f32", "shape": [len(left)]})
        payload_chunks.append(np.asarray(left, dtype=np.float32).tobytes())
    if right is not None:
        fields.append({"name": "right_hand_joints", "dtype": "f32", "shape": [len(right)]})
        payload_chunks.append(np.asarray(right, dtype=np.float32).tobytes())

    fields.append({"name": "frame_index", "dtype": "i64", "shape": [1]})
    payload_chunks.append(np.asarray([frame_index], dtype=np.int64).tobytes())

    header_dict = {"v": 4, "endian": "le", "count": 1, "fields": fields}
    header_blob = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    if len(header_blob) > HEADER_SIZE:
        raise ValueError("synthetic header too large")
    header_blob = header_blob.ljust(HEADER_SIZE, b"\x00")

    return topic.encode("utf-8") + header_blob + b"".join(payload_chunks)


def _self_test() -> int:
    """Decode-and-clamp smoke test that needs no ROS and no robot."""
    print("[self-test] building synthetic pose packet (left half-curl, "
          "right deliberately out-of-range to exercise the clamp)...")
    left_in_deg = [-25.0, 30.0, -25.0, 6.0, 45.0, 45.0, -5.0, 45.0, -5.0, 45.0]
    # Right values are deliberately outside every per-joint range
    # in HAND_JOINT_RANGE_RIGHT_DEG (e.g. thumb_abad range is
    # -100..0, so -150 trips it; index_abad range is -12..0, so -50
    # trips it). The test asserts all 10 right joints get clamped.
    right_in_deg = [120.0, -150.0, 120.0, -50.0, 200.0, 200.0,
                    50.0, 200.0, 50.0, 200.0]

    left_in_rad = [math.radians(d) for d in left_in_deg]
    right_in_rad = [math.radians(d) for d in right_in_deg]

    packet = _build_synthetic_pose_packet(left=left_in_rad, right=right_in_rad)
    print(f"[self-test]   packet size: {len(packet)} bytes "
          f"(topic + {HEADER_SIZE}B header + payload)")

    msg = unpack_message(packet, expected_topic="pose")
    print(f"[self-test]   decoded: v={msg.version}, fields="
          f"{sorted(msg.fields.keys())}, frame_index="
          f"{int(msg.fields['frame_index'][0])}")

    left = np.asarray(msg.fields["left_hand_joints"]).reshape(-1)
    right = np.asarray(msg.fields["right_hand_joints"]).reshape(-1)
    left_c, n_left = _clamp_per_finger(left, "left")
    right_c, n_right = _clamp_per_finger(right, "right")

    print(f"[self-test] left  raw (deg) : "
          f"{[round(math.degrees(v), 1) for v in left]}")
    print(f"[self-test] left  clamp(deg): "
          f"{[round(math.degrees(v), 1) for v in left_c]}  "
          f"(joints clamped: {n_left})")
    print(f"[self-test] right raw (deg) : "
          f"{[round(math.degrees(v), 1) for v in right]}")
    print(f"[self-test] right clamp(deg): "
          f"{[round(math.degrees(v), 1) for v in right_c]}  "
          f"(joints clamped: {n_right})")

    if n_left != 0:
        print(f"[self-test] FAIL: left was within range but {n_left} joints "
              f"reported clamped.", file=sys.stderr)
        return 1
    if n_right != NUM_HAND_DOF_PER_SIDE:
        print(f"[self-test] FAIL: right was deliberately out-of-range; "
              f"expected all {NUM_HAND_DOF_PER_SIDE} joints clamped, "
              f"got {n_right}.", file=sys.stderr)
        return 1
    print("[self-test] OK")
    return 0


# ─────────────────────────────────────────────────────────────────────
# ROS 2 bridge node (constructed lazily so --self-test stays ROS-free)
# ─────────────────────────────────────────────────────────────────────
def _build_node_class():
    """Return a HandBridgeNode class with rclpy/aimdk_msgs imports satisfied.

    Importing rclpy at module top would force ROS + aimdk_msgs to be on
    PYTHONPATH for ``--self-test`` and ``--help`` invocations from a
    plain shell. Wrapping the import inside this factory keeps both
    those paths cheap and gives us a precise error message when ROS or
    aimdk_msgs is missing on a host that legitimately needs them.
    """
    try:
        import rclpy  # noqa: F401
        from rclpy.node import Node  # type: ignore
        from rclpy.qos import (  # type: ignore
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from aimdk_msgs.msg import (  # type: ignore
            HandCommand,
            HandCommandArray,
            HandStateArray,
        )
    except Exception as exc:
        raise RuntimeError(
            "rclpy + aimdk_msgs are required for live mode. Source the "
            "ROS 2 + aimdk colcon overlay first (e.g. "
            "`source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash`)\n"
            f"  underlying import error: {exc}"
        ) from exc

    _PUB_QOS = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )
    # Latched state frames are TRANSIENT_LOCAL on the publisher side
    # so late joiners see the most recent frame on connect; matching
    # here lets us auto-detect attached sides without waiting for the
    # next live frame.
    _STATE_QOS = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    class HandBridgeNode(Node):
        def __init__(
            self,
            *,
            reader: _ZmqPoseReader,
            sides: str,
            detect_timeout: float,
            engage_shots: int,
            engage_period_s: float,
            publish_hz: float,
            max_stale_s: float,
            stats_period_s: float,
        ):
            super().__init__("x2_hand_zmq_to_aimdk_bridge")

            self._reader = reader
            self._sides_request = sides
            self._engage_shots_total = engage_shots
            self._engage_shots_left = engage_shots
            self._engage_period_s = engage_period_s
            self._max_stale_s = max_stale_s
            self._stats_period_s = stats_period_s

            self._lock = threading.Lock()
            self._latest_state_left_type = HAND_TYPE_NONE
            self._latest_state_right_type = HAND_TYPE_NONE
            self._latest_state_left_pos: Optional[np.ndarray] = None
            self._latest_state_right_pos: Optional[np.ndarray] = None
            self._state_received = threading.Event()

            self._sides: tuple[bool, bool] = (False, False)
            self._stop = False

            # Last-good snapshot (post-clamp), refreshed every time we
            # successfully publish a wire-derived frame. Used for the
            # stale-wire fallback so we don't dribble zeros.
            self._last_good_left: Optional[np.ndarray] = None
            self._last_good_right: Optional[np.ndarray] = None

            # Counters for the periodic stats line.
            self._n_pub_total = 0
            self._n_pub_wire = 0
            self._n_pub_stale_hold = 0
            self._n_pub_engage = 0
            self._n_clamp_left = 0
            self._n_clamp_right = 0
            self._stats_last = time.monotonic()

            # Subscribe early so the latched (TRANSIENT_LOCAL) state
            # frame can land before we have to decide which sides to
            # enable in auto mode.
            self.create_subscription(
                HandStateArray, HAND_STATE_TOPIC,
                self._on_hand_state, _STATE_QOS,
            )

            self._left_type = HAND_TYPE_NONE
            self._right_type = HAND_TYPE_NONE
            self._publisher = self.create_publisher(
                HandCommandArray, HAND_CMD_TOPIC, _PUB_QOS,
            )

            # Resolve which sides to enable. Auto polls the latched
            # frame for up to detect_timeout seconds, spinning the
            # node ourselves so callbacks actually fire (super().__init__
            # is past so subscribers exist, but no executor is driving
            # them yet -- we have to do that explicitly here).
            sides_lower = sides.lower()
            if sides_lower == "off":
                left_on, right_on = False, False
            elif sides_lower == "left":
                left_on, right_on = True, False
            elif sides_lower == "right":
                left_on, right_on = False, True
            elif sides_lower == "both":
                left_on, right_on = True, True
            elif sides_lower == "auto":
                deadline = time.monotonic() + max(0.0, detect_timeout)
                while (
                    not self._state_received.is_set()
                    and time.monotonic() < deadline
                ):
                    rclpy.spin_once(self, timeout_sec=0.05)
                if not self._state_received.is_set():
                    self.get_logger().warn(
                        f"no HandStateArray within {detect_timeout:.1f}s; "
                        "auto-detect defaulting to BOTH SIDES OFF (pass "
                        "--sides left|right|both to override)"
                    )
                    left_on, right_on = False, False
                else:
                    left_on = self._latest_state_left_type != HAND_TYPE_NONE
                    right_on = self._latest_state_right_type != HAND_TYPE_NONE
                    self.get_logger().info(
                        f"auto-detect: left_type="
                        f"{self._latest_state_left_type}, right_type="
                        f"{self._latest_state_right_type}"
                    )
            else:
                raise ValueError(
                    f"--sides must be auto|left|right|both|off, got {sides!r}"
                )
            self._sides = (left_on, right_on)
            # Preserve the hardware contract reported by AimDK.  X2 claw
            # hands report type=CLAW and require one 0..1 command; publishing
            # ten NIMBLE joints under type=NIMBLE is silently ignored.
            self._left_type = (
                self._latest_state_left_type
                if self._latest_state_left_type not in (HAND_TYPE_NONE, HAND_TYPE_ERROR)
                else HAND_TYPE_CLAW
            )
            self._right_type = (
                self._latest_state_right_type
                if self._latest_state_right_type not in (HAND_TYPE_NONE, HAND_TYPE_ERROR)
                else HAND_TYPE_CLAW
            )
            self.get_logger().info(
                f"sides enabled: left={left_on} type={self._left_type} "
                f"right={right_on} type={self._right_type}"
            )

            if not (left_on or right_on):
                self.get_logger().warn(
                    "no sides enabled; bridge will idle (no commands published). "
                    "Plug in a hand or pass --sides explicitly."
                )

            # Engage burst -- 1 Hz timer that decrements ``_engage_shots_left``.
            self._engage_timer = self.create_timer(
                engage_period_s, self._on_engage_tick,
            )
            # Main publish loop. Starts immediately; the engage burst
            # gates the wire-publish branch via ``_engage_shots_left``.
            self._publish_period_s = 1.0 / max(1.0, publish_hz)
            self._publish_timer = self.create_timer(
                self._publish_period_s, self._on_publish_tick,
            )
            self._stats_timer = self.create_timer(
                stats_period_s, self._log_stats,
            )

            self.get_logger().info(
                f"reader={reader.endpoint()} publish_hz={publish_hz:.0f} "
                f"max_stale={max_stale_s*1000:.0f}ms engage_burst="
                f"{engage_shots}x@{engage_period_s:.1f}s"
            )

        # ── Hand state subscription ────────────────────────────────
        def _on_hand_state(self, msg) -> None:
            try:
                left_type = int(msg.left_hand_type.value)
                right_type = int(msg.right_hand_type.value)
            except Exception:
                return
            n_left = min(len(msg.left_hands), NUM_HAND_DOF_PER_SIDE)
            n_right = min(len(msg.right_hands), NUM_HAND_DOF_PER_SIDE)
            left_pos = np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)
            right_pos = np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)
            for i in range(n_left):
                left_pos[i] = float(msg.left_hands[i].position)
            for i in range(n_right):
                right_pos[i] = float(msg.right_hands[i].position)

            with self._lock:
                self._latest_state_left_type = left_type
                self._latest_state_right_type = right_type
                self._latest_state_left_pos = left_pos
                self._latest_state_right_pos = right_pos
            self._state_received.set()

        # ── Tick handlers ──────────────────────────────────────────
        def _on_engage_tick(self) -> None:
            if self._engage_shots_left <= 0:
                self._engage_timer.cancel()
                return
            left_on, right_on = self._sides
            if not (left_on or right_on):
                self._engage_shots_left = 0
                self._engage_timer.cancel()
                return
            self._publish_command(None, None,
                                  origin="engage")
            self._engage_shots_left -= 1
            self.get_logger().info(
                f"engage burst {self._engage_shots_total - self._engage_shots_left}"
                f"/{self._engage_shots_total} sent"
            )

        def _on_publish_tick(self) -> None:
            if self._stop:
                return
            left_on, right_on = self._sides
            if not (left_on or right_on):
                return
            # While the engage burst is still in flight, the engage
            # timer drives the publishes. Skip the wire branch so we
            # don't double-publish (matches the recorder's pattern).
            if self._engage_shots_left > 0:
                return

            snap = self._reader.latest()
            now = time.monotonic()
            wire_age = now - snap.recv_monotonic if snap.recv_monotonic > 0 else float("inf")
            wire_fresh = (
                snap.recv_monotonic > 0
                and wire_age < self._max_stale_s
                and (snap.left is not None or snap.right is not None)
            )

            if wire_fresh:
                self._publish_command(
                    snap.left if left_on else None,
                    snap.right if right_on else None,
                    origin="wire",
                )
            elif self._last_good_left is not None or self._last_good_right is not None:
                # Wire stalled -- republish last good positions so
                # motors stay enabled. We do NOT switch to the latest
                # observed encoder reading here (that's a follow-record
                # behaviour); we want the operator's last commanded
                # pose to hold so a transient ZMQ blip doesn't unfurl
                # a closed grasp.
                self._publish_command(
                    self._last_good_left if left_on else None,
                    self._last_good_right if right_on else None,
                    origin="stale_hold",
                )

        def _publish_command(
            self,
            left_positions: Optional[np.ndarray],
            right_positions: Optional[np.ndarray],
            *,
            origin: str,
        ) -> None:
            left_on, right_on = self._sides
            msg = HandCommandArray()
            msg.left_hand_type.value = self._left_type if left_on else HAND_TYPE_NONE
            msg.right_hand_type.value = self._right_type if right_on else HAND_TYPE_NONE

            if left_on:
                msg.left_hands, stored, n_hit = self._build_hardware_side(
                    "left", self._left_type, left_positions
                )
                self._n_clamp_left += n_hit
                if origin == "wire" and stored is not None:
                    self._last_good_left = stored
            else:
                msg.left_hands = []

            if right_on:
                msg.right_hands, stored, n_hit = self._build_hardware_side(
                    "right", self._right_type, right_positions
                )
                self._n_clamp_right += n_hit
                if origin == "wire" and stored is not None:
                    self._last_good_right = stored
            else:
                msg.right_hands = []

            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "x2_hand_zmq_to_aimdk_bridge"
            self._publisher.publish(msg)

            self._n_pub_total += 1
            if origin == "wire":
                self._n_pub_wire += 1
            elif origin == "stale_hold":
                self._n_pub_stale_hold += 1
            elif origin == "engage":
                self._n_pub_engage += 1

        @staticmethod
        def _build_nimble_cmd(clamped_positions: np.ndarray) -> list:
            out = []
            n = min(len(clamped_positions), NUM_HAND_DOF_PER_SIDE)
            for i in range(n):
                c = HandCommand()
                c.name = HAND_FINGER_NAMES_PER_SIDE[i]
                c.position = float(clamped_positions[i])
                c.velocity = HAND_CMD_VELOCITY_HINT
                c.acceleration = 0.0
                c.deceleration = 0.0
                c.effort = 0.0
                out.append(c)
            return out

        @staticmethod
        def _build_claw_cmd(side: str, claw_position: float) -> list:
            c = HandCommand()
            c.name = f"{side}_hand"
            c.position = float(np.clip(claw_position, 0.0, 1.0))
            c.velocity = 1.0
            c.acceleration = 1.0
            c.deceleration = 1.0
            c.effort = 1.0
            return [c]

        @classmethod
        def _build_hardware_side(
            cls,
            side: str,
            hand_type: int,
            positions: Optional[np.ndarray],
        ) -> tuple[list, Optional[np.ndarray], int]:
            if hand_type == HAND_TYPE_CLAW:
                # Engage with the documented open target.  Live ten-joint
                # poses are projected back onto their open->closed grasp axis.
                claw = (
                    1.0
                    if positions is None
                    else claw_position_from_hand_joints(side, positions)
                )
                stored = (
                    None
                    if positions is None
                    else np.asarray(positions, dtype=np.float64).copy()
                )
                return cls._build_claw_cmd(side, claw), stored, 0

            raw = (
                np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)
                if positions is None
                else np.asarray(positions, dtype=np.float64)
            )
            clamped, n_hit = _clamp_per_finger(raw, side)
            return cls._build_nimble_cmd(clamped), clamped, n_hit

        # ── Diagnostics ────────────────────────────────────────────
        def _log_stats(self) -> None:
            now = time.monotonic()
            dt = max(1e-3, now - self._stats_last)
            snap = self._reader.latest()
            left_on, right_on = self._sides
            wire_age_ms = (
                (now - snap.recv_monotonic) * 1000.0
                if snap.recv_monotonic > 0 else float("inf")
            )
            self.get_logger().info(
                f"stats: pub_total={self._n_pub_total} "
                f"(wire={self._n_pub_wire} stale_hold={self._n_pub_stale_hold} "
                f"engage={self._n_pub_engage}) "
                f"clamp_hits L={self._n_clamp_left} R={self._n_clamp_right} | "
                f"sides L={left_on} R={right_on} | "
                f"zmq_rx={snap.n_recv} hand_present={snap.n_hand_present} "
                f"frame_idx={snap.frame_index} wire_age_ms={wire_age_ms:.0f}"
            )
            self._stats_last = now

        def request_stop(self) -> None:
            self._stop = True

    return HandBridgeNode


# ─────────────────────────────────────────────────────────────────────
# Main / arg parsing
# ─────────────────────────────────────────────────────────────────────
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ZMQ pose -> AimDK HandCommandArray bridge (X2 OmniHand).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--zmq-host", default="127.0.0.1",
                   help="ZMQ pose publisher host (default: %(default)s).")
    p.add_argument("--zmq-port", type=int, default=5556,
                   help="ZMQ pose publisher port (default: %(default)d). "
                        "Matches the deploy/recorder/MuJoCo-bridge default.")
    p.add_argument("--zmq-topic", default="pose",
                   help="ZMQ topic prefix (default: %(default)s).")
    p.add_argument("--sides", default="auto",
                   choices=["auto", "left", "right", "both", "off"],
                   help="Which sides to drive. 'auto' uses the latched "
                        "HandStateArray to detect attached sides. (default: %(default)s)")
    p.add_argument("--detect-timeout", type=float, default=2.0,
                   help="Seconds to wait for the first HandStateArray "
                        "frame in --sides=auto mode (default: %(default).1fs).")
    p.add_argument("--engage-shots", type=int, default=3,
                   help="Number of position=0 engage shots to send at "
                        "startup so the OmniHand HAL enables motors. "
                        "Set 0 to skip (e.g. when piggy-backing on an "
                        "already-engaged HAL). (default: %(default)d)")
    p.add_argument("--engage-period-s", type=float, default=1.0,
                   help="Seconds between engage shots (default: %(default).1fs).")
    p.add_argument("--publish-hz", type=float, default=50.0,
                   help="Publish loop rate (default: %(default).0f Hz, "
                        "matches the deploy CONTROL tick).")
    p.add_argument("--max-stale-s", type=float, default=0.20,
                   help="Wire frame is considered stale after this many "
                        "seconds with no update (default: %(default).2fs). "
                        "Stale -> publish last-good positions instead of "
                        "dribbling zeros.")
    p.add_argument("--stats-period-s", type=float, default=5.0,
                   help="Seconds between stats log lines (default: %(default).1fs).")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Run for this many seconds then exit gracefully. "
                        "0 = unlimited (run until SIGINT). (default: %(default).0f)")
    p.add_argument("--self-test", action="store_true",
                   help="Decode-and-clamp smoke test (no ROS, no robot). Exits 0/1.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])

    if args.self_test:
        return _self_test()

    # Build ZMQ reader first so we can fail fast on a missing pyzmq.
    try:
        reader = _ZmqPoseReader(args.zmq_host, args.zmq_port, args.zmq_topic)
    except Exception as exc:
        print(f"[x2-hand-bridge] failed to set up ZMQ subscriber: {exc}",
              file=sys.stderr)
        return 1
    reader.start()

    try:
        HandBridgeNode = _build_node_class()
    except Exception as exc:
        reader.stop()
        print(f"[x2-hand-bridge] {exc}", file=sys.stderr)
        return 1

    import rclpy  # type: ignore
    rclpy.init()
    node = HandBridgeNode(
        reader=reader,
        sides=args.sides,
        detect_timeout=args.detect_timeout,
        engage_shots=max(0, args.engage_shots),
        engage_period_s=max(0.05, args.engage_period_s),
        publish_hz=max(1.0, args.publish_hz),
        max_stale_s=max(0.01, args.max_stale_s),
        stats_period_s=max(1.0, args.stats_period_s),
    )

    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    stop_evt = threading.Event()

    def _on_signal(signum, _frame):
        node.get_logger().info(f"signal {signum} received; shutting down")
        stop_evt.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        while not stop_evt.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                node.get_logger().info(
                    f"--duration {args.duration:.0f}s elapsed; exiting"
                )
                break
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.request_stop()
        try:
            node.destroy_node()
        finally:
            try:
                rclpy.shutdown()
            except Exception:
                pass
        reader.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
