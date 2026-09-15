"""Wire-format compatibility tests for the XRoboToolkit -> SONIC bridge.

The bridge (``gear_sonic/scripts/x2_xrt_sonic_bridge.py``) runs on PC2 and
PUB-connects to the PC2-local merge planner (``pc2_kplanner_onnx.py``),
which owns every SUB bind. These tests pin the byte-level contract against
the *actual* consumers:

* ``pc2_kplanner_onnx.ArmTargetIngest`` (arm_targets / hand_finger_cmd
  msgpack topics), and
* ``pc2_kplanner_onnx._zmq_command_thread``'s JSON payload expectations
  (planner_cmd topic, ``target_velocity`` direct-velocity locomotion).

They run entirely off-robot -- no ROS2, no XRoboToolkit SDK, no PC2.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from gear_sonic.scripts.x2_xrt_sonic_bridge import (
    ARM_DOF_PER_SIDE,
    ARM_JOINT_ORDER_LEFT,
    ARM_JOINT_ORDER_RIGHT,
    DEFAULT_HIP_HEIGHT_M,
    NUM_HAND_DOF_PER_SIDE,
    SonicZmqSink,
    build_planner_cmd_idle,
    build_planner_cmd_locomotion,
    build_planner_cmd_vr_release,
    grasp_command_from_ratio,
    linear_velocity_from_sticks,
    pack_arm_targets,
    pack_hand_finger_cmd,
    split_arm_targets,
)
from gear_sonic.utils.planner.constants import MUJOCO_JOINT_NAMES


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _sub_bind_socket(ctx, port: int, topics: list[str]):
    import zmq

    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    for topic in topics:
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.bind(f"tcp://*:{port}")
    return sock


# ---------------------------------------------------------------------------
# Joint ordering: bridge wire order == SONIC MuJoCo body_q arm slices.
# ---------------------------------------------------------------------------

def test_arm_joint_order_matches_mujoco_slices():
    assert list(MUJOCO_JOINT_NAMES[15:22]) == list(ARM_JOINT_ORDER_LEFT)
    assert list(MUJOCO_JOINT_NAMES[22:29]) == list(ARM_JOINT_ORDER_RIGHT)
    assert len(ARM_JOINT_ORDER_LEFT) == ARM_DOF_PER_SIDE
    assert len(ARM_JOINT_ORDER_RIGHT) == ARM_DOF_PER_SIDE


def test_split_arm_targets_maps_dict_to_wire_order():
    targets = {name: float(i) for i, name in enumerate(MUJOCO_JOINT_NAMES)}
    left, right = split_arm_targets(targets)
    # MUJOCO indices 15..21 / 22..28 are exactly the values 15..21 / 22..28.
    np.testing.assert_array_equal(left, np.arange(15.0, 22.0))
    np.testing.assert_array_equal(right, np.arange(22.0, 29.0))


# ---------------------------------------------------------------------------
# Hand mapping: vendored anchors must stay identical to the source of truth.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75, 1.0, 1.1, -0.2])
def test_grasp_mapping_matches_source_of_truth(side: str, ratio: float):
    from gear_sonic.utils.teleop.x2_hand_retarget import (
        grasp_command_from_ratio as reference_impl,
    )

    ours = grasp_command_from_ratio(side, ratio)
    theirs = reference_impl(side, ratio)
    assert ours.shape == (NUM_HAND_DOF_PER_SIDE,)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


# ---------------------------------------------------------------------------
# Joystick -> linear velocity mapping (fixed speed past deadzone; no angular).
# ---------------------------------------------------------------------------

def test_linear_velocity_fixed_speed_past_deadzone():
    # Default response: deflection magnitude is ignored -- any push past
    # the deadzone commands the full speed in that direction.
    vel_x, vel_z = linear_velocity_from_sticks(
        0.0, 0.5, max_lin_vel=0.22, max_side_mps=0.22
    )
    assert vel_z == pytest.approx(0.22)
    assert vel_x == pytest.approx(0.0)
    # A tiny post-deadzone deflection is still full speed.
    _, vel_z = linear_velocity_from_sticks(
        0.0, 0.05, max_lin_vel=0.22, max_side_mps=0.22
    )
    assert vel_z == pytest.approx(0.22)
    vel_x, _ = linear_velocity_from_sticks(
        -0.02, 0.0, max_lin_vel=0.22, max_side_mps=0.22
    )
    assert vel_x == pytest.approx(0.22)
    # Centred axes stay zero.
    vel_x, vel_z = linear_velocity_from_sticks(
        0.0, 0.0, max_lin_vel=0.22, max_side_mps=0.22
    )
    assert vel_x == 0.0
    assert vel_z == 0.0


def test_linear_velocity_backward_and_strafe_signs():
    # Pull back -> negative forward speed.
    _, vel_z = linear_velocity_from_sticks(
        0.0, -1.0, max_lin_vel=0.35, max_side_mps=0.30
    )
    assert vel_z == pytest.approx(-0.35)
    # Push right -> strafe right = negative vel_x (planner frame is +left).
    vel_x, vel_z = linear_velocity_from_sticks(
        1.0, 0.0, max_lin_vel=0.35, max_side_mps=0.30
    )
    assert vel_x == pytest.approx(-0.30)
    assert vel_z == pytest.approx(0.0)
    # Push left -> positive vel_x.
    vel_x, _ = linear_velocity_from_sticks(
        -1.0, 0.0, max_lin_vel=0.35, max_side_mps=0.30
    )
    assert vel_x == pytest.approx(0.30)


def test_linear_velocity_flip_flags():
    # Fixed speed with both axes flipped: forward channel by flip_vx,
    # lateral channel by flip_vy (sign of the push still picks direction).
    vel_x, vel_z = linear_velocity_from_sticks(
        0.25, -0.5, max_lin_vel=0.35, max_side_mps=0.30,
        flip_vx=True, flip_vy=True,
    )
    assert vel_z == pytest.approx(0.35)
    assert vel_x == pytest.approx(0.30)


def test_linear_velocity_legacy_proportional_response():
    # Opt-in legacy behaviour: speed scales with deflection.
    vel_x, vel_z = linear_velocity_from_sticks(
        0.0, 0.5, max_lin_vel=0.35, max_side_mps=0.30, proportional=True
    )
    assert vel_z == pytest.approx(0.175)
    assert vel_x == pytest.approx(0.0)
    vel_x, vel_z = linear_velocity_from_sticks(
        0.25, -0.5, max_lin_vel=0.35, max_side_mps=0.30,
        flip_vx=True, flip_vy=True, proportional=True,
    )
    assert vel_z == pytest.approx(0.175)
    assert vel_x == pytest.approx(0.075)


# ---------------------------------------------------------------------------
# planner_cmd JSON: fields the planner's _zmq_command_thread reads.
# ---------------------------------------------------------------------------

def _planner_parse(blob: bytes) -> dict:
    """Mirror of pc2_kplanner_onnx._zmq_command_thread's json field reads."""
    payload = json.loads(blob.decode("utf-8"))
    tv = payload.get("target_velocity")
    return {
        "intent": str(payload["intent"]),
        "magnitude": str(payload.get("magnitude", "default")),
        "source": str(payload.get("source", "pad")),
        "target_velocity": [float(v) for v in tv] if tv is not None else None,
        "vr_release": bool(payload.get("vr_release", False)),
    }


def test_planner_cmd_locomotion_carries_direct_velocity():
    cmd = build_planner_cmd_locomotion(-0.2, 0.3)
    parsed = _planner_parse(json.dumps(cmd).encode("utf-8"))
    assert parsed["intent"] == "locomotion"
    assert parsed["magnitude"] == "continuous"
    assert parsed["source"] == "vr"
    # Layout [yaw_rate, vel_x, vel_z, hip_h]; yaw pinned to 0 (linear-only).
    assert parsed["target_velocity"] == pytest.approx(
        [0.0, -0.2, 0.3, DEFAULT_HIP_HEIGHT_M]
    )
    assert parsed["vr_release"] is False


def test_planner_cmd_locomotion_custom_hip_height():
    cmd = build_planner_cmd_locomotion(0.0, 0.1, hip_height_m=0.70)
    parsed = _planner_parse(json.dumps(cmd).encode("utf-8"))
    assert parsed["target_velocity"][3] == pytest.approx(0.70)


def test_planner_cmd_idle_keeps_vr_ownership_without_release():
    parsed = _planner_parse(json.dumps(build_planner_cmd_idle()).encode("utf-8"))
    assert parsed["intent"] == "idle"
    assert parsed["source"] == "vr"
    assert parsed["target_velocity"] is None
    assert parsed["vr_release"] is False


def test_planner_cmd_vr_release():
    parsed = _planner_parse(
        json.dumps(build_planner_cmd_vr_release()).encode("utf-8")
    )
    assert parsed["intent"] == "idle"
    assert parsed["source"] == "vr"
    assert parsed["vr_release"] is True


# ---------------------------------------------------------------------------
# End-to-end over real ZMQ: sink PUB-connect -> consumer SUB bind (loopback),
# arm frames decoded by the REAL pc2_kplanner_onnx.ArmTargetIngest.
# ---------------------------------------------------------------------------

def _dummy_pose_payload() -> dict:
    return {
        "joint_pos_mj": np.zeros(31, dtype=np.float32),
        "joint_pos_mj_future": np.zeros((9, 31), dtype=np.float32),
        "left_hand_joints": np.zeros(10, dtype=np.float32),
        "right_hand_joints": np.zeros(10, dtype=np.float32),
    }


def test_sink_end_to_end_zmq():
    zmq = pytest.importorskip("zmq")
    msgpack = pytest.importorskip("msgpack")
    from gear_sonic.scripts.pc2_kplanner_onnx import ArmTargetIngest

    ctx = zmq.Context.instance()
    arm_port, planner_port = _free_port(), _free_port()

    # The REAL consumer: planner-side arm ingest (owns the SUB bind).
    ingest = ArmTargetIngest(arm_port)
    planner_sub = _sub_bind_socket(ctx, planner_port, ["planner_cmd"])
    sink = SonicZmqSink(
        arm_hands_host="127.0.0.1",
        arm_hands_port=arm_port,
        planner_cmd_host="127.0.0.1",
        planner_cmd_port=planner_port,
    )

    def publish_until_recv(pub_fn, sub, timeout_s: float = 5.0) -> bytes:
        """Re-publish the same (idempotent) frame until one arrives.

        A PUB-connect -> SUB-bind handshake drops whatever is sent before
        the subscription propagates (~1 s); production streams self-heal at
        the publish rate, one-shot test frames must retry instead.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pub_fn()
            try:
                return sub.recv_multipart(copy=False)[1].bytes
            except zmq.error.Again:
                time.sleep(0.05)
        raise AssertionError("no frame arrived within timeout")

    try:
        # Arm frame: 14-joint dict in hardware naming (== model naming here).
        joint_targets = {
            name: 0.01 * (i + 1) for i, name in enumerate(MUJOCO_JOINT_NAMES[15:29])
        }

        def _pub_arms():
            sink.publish_arm_targets(joint_targets, is_engaged=True)
            sink.publish_hand_targets(left_claw=0.6, right_claw=1.0)

        # Drive the arm stream like production does (every control tick)
        # until the ingest thread caches both frames.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _pub_arms()
            time.sleep(0.02)
            with ingest._lock:
                got_arms = ingest._left is not None and ingest._left_hand is not None
            if got_arms:
                break
        else:
            pytest.fail("ArmTargetIngest never received arm/hand frames")

        payload = _dummy_pose_payload()
        assert ingest.overlay(payload) is True
        exp_left, exp_right = split_arm_targets(joint_targets)
        # First overlay frame slews from the seed (planner arms = 0) toward
        # the target by exactly ARM_SLEW_RAD_PER_FRAME (targets are all
        # positive, so sign is preserved and magnitude = min(step, target)).
        step = ArmTargetIngest.ARM_SLEW_RAD_PER_FRAME
        np.testing.assert_allclose(
            payload["joint_pos_mj"][15:22],
            np.clip(exp_left, -step, step),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            payload["joint_pos_mj"][22:29],
            np.clip(exp_right, -step, step),
            atol=1e-6,
        )
        # Future window pinned to the same pose.
        np.testing.assert_allclose(
            payload["joint_pos_mj_future"][:, 15:22],
            np.broadcast_to(np.clip(exp_left, -step, step), (9, 7)),
            atol=1e-6,
        )
        # claw 0.6 -> grasp ratio 0.4 (slewed from 0 by HAND_SLEW step).
        hand_step = ArmTargetIngest.HAND_SLEW_RAD_PER_FRAME
        exp_hand = np.clip(
            grasp_command_from_ratio("left", 0.4), -hand_step, hand_step
        )
        np.testing.assert_allclose(
            payload["left_hand_joints"], exp_hand, atol=1e-6
        )

        # Planner socket: direct-velocity locomotion, idle, release in order.
        loco = _planner_parse(
            publish_until_recv(
                lambda: sink.publish_walk(vel_x=0.0, vel_z=0.2), planner_sub
            )
        )
        assert loco["intent"] == "locomotion"
        assert loco["target_velocity"] == pytest.approx(
            [0.0, 0.0, 0.2, DEFAULT_HIP_HEIGHT_M]
        )
        idle = _planner_parse(
            publish_until_recv(
                lambda: sink.publish_walk(vel_x=0.0, vel_z=0.0), planner_sub
            )
        )
        assert idle["intent"] == "idle"
        assert idle["vr_release"] is False
        release = _planner_parse(
            publish_until_recv(sink.publish_walk_release, planner_sub)
        )
        assert release["vr_release"] is True

        # Passthrough after a live frame must clear the ingest cache so the
        # planner's own arm pose flows through again.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            sink.publish_arm_passthrough()
            time.sleep(0.05)
            with ingest._lock:
                if ingest._left is None:
                    break
        else:
            pytest.fail("passthrough did not clear the ingest cache")
    finally:
        sink.close()
        ingest.stop()
        planner_sub.close(linger=0)
