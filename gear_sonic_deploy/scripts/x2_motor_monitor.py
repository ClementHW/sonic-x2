#!/usr/bin/env python3
"""PC2-side X2 motor-state forensic monitor.

Subscribes to the four AimDK joint-state groups, writes compact snapshots and
domain-state transitions to JSONL, and publishes the same records as
``[topic, json]`` ZMQ multipart messages. It never publishes a ROS command and
therefore cannot take motor control.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


GROUP_TOPICS = {
    "leg": "/aima/hal/joint/leg/state",
    "waist": "/aima/hal/joint/waist/state",
    "arm": "/aima/hal/joint/arm/state",
    "head": "/aima/hal/joint/head/state",
}

DOMAIN_STATE_NAMES = {
    0: "NONE",
    1: "DAMPING",
    2: "POWERDOWN",
    3: "DISABLE",
    4: "DISCONNECT",
}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def summarize_joint_state(msg: Any) -> dict[str, Any]:
    """Convert an AimDK JointStateArray-like object into JSON-safe metrics."""
    joints = list(getattr(msg, "joints", ()))
    names = [str(getattr(joint, "name", "")) for joint in joints]
    velocities = [_finite(getattr(joint, "velocity", 0.0)) for joint in joints]
    efforts = [_finite(getattr(joint, "effort", 0.0)) for joint in joints]
    coil_temps = [int(getattr(joint, "coil_temp", 0)) for joint in joints]
    motor_temps = [int(getattr(joint, "motor_temp", 0)) for joint in joints]
    motor_vols = [int(getattr(joint, "motor_vol", 0)) for joint in joints]

    peak_temps = [max(coil, motor) for coil, motor in zip(coil_temps, motor_temps)]
    hottest_index = max(range(len(peak_temps)), key=peak_temps.__getitem__) if peak_temps else None
    header = getattr(msg, "header", None)
    state_value = int(getattr(getattr(msg, "state", None), "value", -1))

    return {
        "sequence": int(getattr(header, "sequence", 0)),
        "domain_state": state_value,
        "domain_state_name": DOMAIN_STATE_NAMES.get(state_value, f"UNKNOWN_{state_value}"),
        "joint_count": len(joints),
        "max_abs_velocity": max((abs(v) for v in velocities), default=0.0),
        "max_abs_effort": max((abs(v) for v in efforts), default=0.0),
        "max_coil_temp_c": max(coil_temps, default=0),
        "max_motor_temp_c": max(motor_temps, default=0),
        "hottest_joint": names[hottest_index] if hottest_index is not None else "",
        "hottest_temp_c": peak_temps[hottest_index] if hottest_index is not None else 0,
        "min_motor_vol": min(motor_vols, default=0),
        "max_motor_vol": max(motor_vols, default=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="durable local JSONL output")
    parser.add_argument("--zmq-port", type=int, default=5567)
    parser.add_argument("--zmq-topic", default="motor_monitor")
    parser.add_argument("--publish-hz", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.publish_hz <= 0:
        raise SystemExit("--publish-hz must be > 0")

    import rclpy
    import zmq
    from aimdk_msgs.msg import JointStateArray  # type: ignore
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    output = Path(args.jsonl).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("a", encoding="utf-8", buffering=1)

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 20)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind(f"tcp://*:{args.zmq_port}")
    topic_bytes = args.zmq_topic.encode("utf-8")

    class MotorMonitor(Node):
        def __init__(self) -> None:
            super().__init__("x2_motor_monitor")
            self.latest: dict[str, dict[str, Any]] = {}
            self.last_domain_state: dict[str, int] = {}
            self.last_fsync = time.monotonic()
            self.records = 0
            for group, ros_topic in GROUP_TOPICS.items():
                self.create_subscription(
                    JointStateArray,
                    ros_topic,
                    lambda msg, group_name=group: self.on_state(group_name, msg),
                    qos_profile_sensor_data,
                )
            self.create_timer(1.0 / args.publish_hz, self.emit_snapshot)

        def write_record(self, record: dict[str, Any]) -> None:
            payload = json.dumps(record, separators=(",", ":"), sort_keys=True)
            stream.write(payload + "\n")
            self.records += 1
            try:
                publisher.send_multipart(
                    [topic_bytes, payload.encode("utf-8")], flags=zmq.NOBLOCK
                )
            except zmq.Again:
                pass
            now = time.monotonic()
            if now - self.last_fsync >= 0.5:
                stream.flush()
                os.fsync(stream.fileno())
                self.last_fsync = now

        def on_state(self, group: str, msg: Any) -> None:
            summary = summarize_joint_state(msg)
            self.latest[group] = summary
            current = int(summary["domain_state"])
            previous = self.last_domain_state.get(group)
            self.last_domain_state[group] = current
            if previous is not None and previous != current:
                self.write_record(
                    {
                        "kind": "domain_state_change",
                        "ts": time.time(),
                        "monotonic_s": time.monotonic(),
                        "group": group,
                        "from": previous,
                        "from_name": DOMAIN_STATE_NAMES.get(previous, f"UNKNOWN_{previous}"),
                        "to": current,
                        "to_name": summary["domain_state_name"],
                        "state": summary,
                    }
                )

        def emit_snapshot(self) -> None:
            if not self.latest:
                return
            self.write_record(
                {
                    "kind": "snapshot",
                    "ts": time.time(),
                    "monotonic_s": time.monotonic(),
                    "groups_ready": sorted(self.latest),
                    "groups": self.latest,
                }
            )

    rclpy.init()
    node = MotorMonitor()
    node.write_record(
        {
            "kind": "start",
            "ts": time.time(),
            "monotonic_s": time.monotonic(),
            "topics": GROUP_TOPICS,
            "publish_hz": args.publish_hz,
            "zmq_port": args.zmq_port,
            "zmq_topic": args.zmq_topic,
        }
    )
    print(
        f"[motor-monitor] ROS groups={','.join(GROUP_TOPICS)} "
        f"JSONL={output} PUB=tcp://*:{args.zmq_port} topic={args.zmq_topic}",
        flush=True,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # ROS2 Humble's SIGINT handler can shut the context down before
        # Executor.spin_once() returns, which surfaces as a RuntimeError
        # instead of KeyboardInterrupt. Preserve real runtime failures.
        if rclpy.ok():
            raise
    finally:
        try:
            node.write_record(
                {
                    "kind": "stop",
                    "ts": time.time(),
                    "monotonic_s": time.monotonic(),
                    "records": node.records,
                }
            )
        finally:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            publisher.close(linger=0)
            context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
