#!/usr/bin/env python3
"""PC2-local single-input pose watchdog for X2.

The watchdog sits between the PC2 planner and C++ SONIC deploy. Fresh frames
are forwarded unchanged. If the local planner stream stalls it transitions
through HOLD, BLEND, and IDLE_CLIP while rebasing root yaw from ``x2_debug``.
This keeps the deploy input continuous without a multi-radian joint step.

Dependencies: numpy, pyzmq, stdlib, and
``gear_sonic.utils.pose_pipeline`` from the PC2 workspace.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import zmq

# The lifecycle manager adds the PC2 workspace source tree to PYTHONPATH.
# Also add the inferred repository root for direct diagnostics.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for _candidate in (_REPO_ROOT,):
    _p = str(_candidate)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gear_sonic.utils.pose_pipeline.fallback import (  # noqa: E402
    IDLE_MODE_BLEND,
    IDLE_MODE_HOLD_LAST,
    IDLE_MODE_IDLE_STAND,
    IDLE_MODES,
    IdleStandReplay,
    STATE_BLEND,
    STATE_COLD_IDLE,
    STATE_GAP,
    STATE_HOLD,
    STATE_IDLE_CLIP,
    STATE_LIVE,
    build_idle_frame_msg,
    decide_fallback_state,
    rebase_hold_msg,
)
from gear_sonic.utils.pose_pipeline.wire import (  # noqa: E402
    DEFAULT_PUB_RATE_HZ,
    decode_pose_joint_pos_mj,
    decode_pose_joint_pos_mj_future,
    decode_pose_joint_vel_mj_future,
    decode_pose_motion_token,
    decode_pose_root_quat_xyzw_future,
    decode_x2_debug_base_quat,
    load_x2m2,
    nlerp_quat_arrays_xyzw,
    rebase_quats_xyzw_by_yaw,
    yaw_from_quat_wxyz,
)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--upstream-host",
        required=True,
        help="PC2-local planner pose PUB host (normally 127.0.0.1).",
    )
    p.add_argument(
        "--upstream-port",
        type=int,
        default=5556,
        help="PC2-local planner pose PUB port (default 5556).",
    )
    p.add_argument(
        "--upstream-topic",
        default="pose",
        help="ZMQ topic prefix on upstream (default 'pose').",
    )
    p.add_argument(
        "--downstream-host",
        default="*",
        help="Local bind iface for downstream PUB (default '*' = all).",
    )
    p.add_argument(
        "--downstream-port",
        type=int,
        default=5558,
        help="Local PUB port the C++ deploy SUBs to (default 5558). "
             "MUST match the deploy's --vla-zmq-port.",
    )
    p.add_argument(
        "--downstream-topic",
        default="pose",
        help="Topic prefix on downstream (must match deploy "
             "--vla-zmq-topic).",
    )
    p.add_argument(
        "--idle-x2m2",
        type=Path,
        required=True,
        help="Path to baked idle_stand.x2m2 binary on PC2.",
    )
    p.add_argument(
        "--idle-stale-ms",
        type=int,
        default=100,
        help="Switch to idle fallback after this many ms of upstream "
             "silence (default 100).",
    )
    p.add_argument(
        "--idle-mode",
        choices=list(IDLE_MODES),
        default=IDLE_MODE_BLEND,
        help="Behaviour when upstream is silent past --idle-stale-ms. "
             "'blend' (DEFAULT, safe): hold the last forwarded "
             "upstream frame for --hold-last-secs, then lerp toward "
             "the baked idle_stand clip over --blend-secs. 'hold-"
             "last': hold the last forwarded frame indefinitely "
             "(operator owns recovery). 'idle-stand': pre-2026-06-08 "
             "behaviour; switch to the baked idle clip on the first "
             "stale tick (causes arms to slam to default on any wifi "
             "hiccup -- regression escape only).",
    )
    p.add_argument(
        "--hold-last-secs",
        type=float,
        default=5.0,
        help="How long (s) to hold the last forwarded upstream frame "
             "before transitioning toward idle (default 5.0; reduced "
             "from 10.0 on 2026-06-23 alongside the BLEND future-"
             "window continuity fix so intentional operator shutdowns "
             "reach the smooth BLEND ramp sooner). Only applies when "
             "--idle-mode=blend.",
    )
    p.add_argument(
        "--blend-secs",
        type=float,
        default=3.0,
        help="Duration (s) of the lerp from cached-upstream to baked "
             "idle_stand at the end of the hold window (default 3.0). "
             "Only applies when --idle-mode=blend.",
    )
    p.add_argument(
        "--rate-hz",
        type=float,
        default=DEFAULT_PUB_RATE_HZ,
        help="Downstream publish rate when idle (default 50; matches "
             "deploy control loop). Fresh upstream forwarding follows "
             "the PC2 planner cadence.",
    )
    p.add_argument(
        "--status-every-s",
        type=float,
        default=5.0,
        help="Periodic status print interval (default 5s).",
    )
    p.add_argument(
        "--x2-debug-host",
        default="127.0.0.1",
        help="Host of the deploy's x2_debug PUB (default 127.0.0.1; "
             "the deploy is colocated on PC2 in onbot mode).",
    )
    p.add_argument(
        "--x2-debug-port",
        type=int,
        default=5557,
        help="Port of the deploy's x2_debug PUB (default 5557).",
    )
    p.add_argument(
        "--x2-debug-topic",
        default="x2_debug",
        help="Topic prefix for the x2_debug PUB (default 'x2_debug').",
    )
    p.add_argument(
        "--x2-debug-max-age-s",
        type=float,
        default=0.5,
        help="Max age (s) of the latest x2_debug frame before we treat "
             "it as stale and fall back to the last-known-good "
             "measured yaw (default 0.5s).",
    )
    p.add_argument(
        "--no-x2-debug-yaw-track",
        action="store_true",
        help="Disable x2_debug yaw tracking. Idle-fallback frames will "
             "publish the baked clip's R_z(0) root quat verbatim, "
             "which causes the deploy to twist the body back to world "
             "+X on every IDLE entry. Regression-test escape; never "
             "use in prod.",
    )

    args = p.parse_args(raw_argv)

    if args.hold_last_secs < 0.0:
        print(
            f"[pose_watchdog] ERROR: --hold-last-secs must be >= 0, "
            f"got {args.hold_last_secs}",
            file=sys.stderr,
        )
        return 1
    if args.blend_secs < 0.0:
        print(
            f"[pose_watchdog] ERROR: --blend-secs must be >= 0, got "
            f"{args.blend_secs}",
            file=sys.stderr,
        )
        return 1

    if not args.idle_x2m2.is_file():
        print(
            f"[pose_watchdog] ERROR: idle X2M2 not found: "
            f"{args.idle_x2m2}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[pose_watchdog] loading idle X2M2 from {args.idle_x2m2}",
        flush=True,
    )
    dof, quat, fps = load_x2m2(args.idle_x2m2)
    print(
        f"[pose_watchdog] idle clip: {dof.shape[0]} frames @ {fps:g} "
        f"Hz ({dof.shape[0] / fps:.2f} s loop)",
        flush=True,
    )
    replay = IdleStandReplay(dof, quat)

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    upstream_url = f"tcp://{args.upstream_host}:{args.upstream_port}"
    sub.setsockopt(zmq.RCVHWM, 100)
    sub.connect(upstream_url)
    sub.setsockopt(zmq.SUBSCRIBE, args.upstream_topic.encode("utf-8"))

    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 100)
    bind_url = f"tcp://{args.downstream_host}:{args.downstream_port}"
    pub.bind(bind_url)

    # Optional x2_debug SUB for measured-yaw tracking on idle frames.
    yaw_sub: zmq.Socket | None = None
    yaw_track_enabled = (
        not args.no_x2_debug_yaw_track
    ) and int(args.x2_debug_port) > 0
    if yaw_track_enabled:
        yaw_sub = ctx.socket(zmq.SUB)
        yaw_sub.setsockopt(zmq.RCVHWM, 4)
        x2_debug_url = f"tcp://{args.x2_debug_host}:{args.x2_debug_port}"
        yaw_sub.connect(x2_debug_url)
        yaw_sub.setsockopt(
            zmq.SUBSCRIBE, args.x2_debug_topic.encode("utf-8")
        )

    print(
        f"[pose_watchdog] upstream SUB:   {upstream_url} "
        f"topic={args.upstream_topic!r}",
        flush=True,
    )
    print(
        f"[pose_watchdog] downstream PUB: {bind_url} "
        f"topic={args.downstream_topic!r}",
        flush=True,
    )
    if yaw_track_enabled:
        print(
            f"[pose_watchdog] yaw-track SUB:  "
            f"tcp://{args.x2_debug_host}:{args.x2_debug_port} "
            f"topic={args.x2_debug_topic!r} "
            f"(max_age={args.x2_debug_max_age_s:.3f}s)",
            flush=True,
        )
    else:
        reason = (
            "--no-x2-debug-yaw-track"
            if args.no_x2_debug_yaw_track
            else f"--x2-debug-port={args.x2_debug_port}"
        )
        print(
            f"[pose_watchdog] yaw-track SUB:  DISABLED ({reason}); "
            f"idle frames will publish baked R_z(0) -- expect snap-"
            f"back to world +X on planner-stack termination",
            flush=True,
        )
    print(
        f"[pose_watchdog] idle stale threshold: {args.idle_stale_ms} "
        f"ms (switch to idle fallback after {args.idle_stale_ms} ms "
        f"of upstream silence)",
        flush=True,
    )
    if args.idle_mode == IDLE_MODE_BLEND:
        print(
            f"[pose_watchdog] idle mode: blend "
            f"(HOLD last frame for {args.hold_last_secs:.1f}s, then "
            f"BLEND to idle_stand over {args.blend_secs:.1f}s)",
            flush=True,
        )
    elif args.idle_mode == IDLE_MODE_HOLD_LAST:
        print(
            "[pose_watchdog] idle mode: hold-last (republish last "
            "upstream frame indefinitely; operator owns recovery)",
            flush=True,
        )
    else:  # idle-stand
        print(
            "[pose_watchdog] idle mode: idle-stand (LEGACY; arms snap "
            "to default on the first stale tick -- known to slam "
            "tables during WiFi hiccups)",
            flush=True,
        )
    # HOLD yaw-rebase (2026-06-24): when x2_debug is alive, the HOLD
    # state splices R_z(measured_yaw) into the cached re-publish so
    # the body is free-yaw during HOLD (operator can rotate the
    # robot by hand without policy spring-back). Pre-fix HOLD was
    # verbatim re-publish, which pinned yaw to whatever upstream
    # last sent.
    if yaw_track_enabled:
        print(
            "[pose_watchdog] HOLD yaw-rebase:  ENABLED (cached "
            "root_quat_xyzw replaced with R_z(measured_yaw) at every "
            "HOLD tick; falls back to verbatim if x2_debug stale "
            "or splice fails)",
            flush=True,
        )
    else:
        print(
            "[pose_watchdog] HOLD yaw-rebase:  DISABLED (yaw-track "
            "off -> HOLD re-publishes cached frame verbatim; "
            "expect spring-back to last-published yaw on upstream "
            "stall)",
            flush=True,
        )

    period = 1.0 / max(args.rate_hz, 1e-6)
    stale_s = args.idle_stale_ms / 1000.0
    hold_last_secs = float(args.hold_last_secs)
    blend_secs = float(args.blend_secs)
    idle_mode = str(args.idle_mode)
    yaw_max_age_s = float(args.x2_debug_max_age_s)
    next_tick = time.monotonic()
    last_upstream_s = -1.0
    last_measured_yaw_rad = 0.0
    last_measured_yaw_s = -1.0
    yaw_decode_failures = 0
    last_upstream_msg: bytes | None = None
    last_upstream_jpos: np.ndarray | None = None
    # Cached future-window + motion_token from the most recent fresh
    # upstream frame. Used by the BLEND branch (2026-06-23 future-window
    # continuity fix) to lerp the full upstream-derived wire frame
    # toward the idle clip instead of letting the policy planning
    # horizon and intent token snap in one tick at the HOLD -> BLEND
    # boundary. Each cache stays None when the upstream is a legacy v4
    # producer (heuristic planner, mock VLA) that never set the
    # corresponding field -- the BLEND branch then falls back to
    # today's snap on that field only.
    last_upstream_jpos_future: np.ndarray | None = None
    last_upstream_quat_future: np.ndarray | None = None
    last_upstream_jvel_future: np.ndarray | None = None
    last_upstream_motion_token: np.ndarray | None = None
    cur_state = STATE_COLD_IDLE
    prev_state = STATE_COLD_IDLE
    tick = 0
    idle_tick = 0
    fwd_frames = 0
    idle_frames = 0
    idle_frames_with_rebase = 0
    hold_frames = 0
    hold_frames_with_rebase = 0
    blend_frames = 0
    gap_skips = 0
    last_status_s = time.monotonic()
    # One-shot log gates for the HOLD yaw-rebase path so a 50 Hz HOLD
    # loop never floods the journal. Mirrors the recorder's idle
    # rebase log gates (x2_dataset_recorder._idle_yaw_rebase_logged_*).
    # We log once when HOLD-with-rebase first engages, once if x2_debug
    # goes stale during HOLD, and once if rebuild_msg_with_field_overrides
    # rejects a cached frame (legacy v4 without future, corrupt header).
    hold_yaw_rebase_logged_active: bool = False
    hold_yaw_rebase_logged_fallback: bool = False
    hold_yaw_rebase_logged_splice_fail: bool = False

    print(
        "[pose_watchdog] starting (initial state: COLD_IDLE; will "
        "switch to LIVE as soon as upstream publishes anything)",
        flush=True,
    )

    try:
        while True:
            now = time.monotonic()

            # Drain x2_debug queue first so the measured-yaw cache is
            # fresh by the time we decide what to publish on this tick.
            if yaw_sub is not None:
                latest_debug = None
                while True:
                    try:
                        latest_debug = yaw_sub.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                if latest_debug is not None:
                    base_quat_wxyz = decode_x2_debug_base_quat(
                        latest_debug, args.x2_debug_topic
                    )
                    if base_quat_wxyz is not None:
                        try:
                            last_measured_yaw_rad = yaw_from_quat_wxyz(
                                base_quat_wxyz
                            )
                            last_measured_yaw_s = now
                        except (ValueError, TypeError):
                            yaw_decode_failures += 1
                    else:
                        yaw_decode_failures += 1

            # Drain upstream queue. We forward the latest frame each
            # tick, not every frame -- if the planner publishes faster
            # than we tick, intermediate frames are intentionally
            # dropped (the deploy only ever sees the freshest
            # reference anyway).
            latest = None
            while True:
                try:
                    latest = sub.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break

            if latest is not None:
                last_upstream_msg = latest
                jpos = decode_pose_joint_pos_mj(
                    latest, args.upstream_topic
                )
                if jpos is None:
                    jpos = decode_pose_joint_pos_mj(
                        latest, args.downstream_topic
                    )
                if jpos is not None:
                    last_upstream_jpos = jpos
                # Snapshot the future window + motion token so BLEND can
                # lerp against them instead of snapping to idle-clip
                # values on the first BLEND tick. We try both topic
                # candidates (upstream / downstream) the same way as
                # the joint_pos_mj decode above. Each decoder returns
                # None for absent fields (e.g. v4 heuristic planner
                # frames that never set the future window); we keep
                # whatever the previous fresh frame had cached in that
                # case rather than clearing -- a slightly stale future
                # cache is strictly better than no cache for the
                # BLEND continuity invariant.
                jpos_future = decode_pose_joint_pos_mj_future(
                    latest, args.upstream_topic
                )
                if jpos_future is None:
                    jpos_future = decode_pose_joint_pos_mj_future(
                        latest, args.downstream_topic
                    )
                if jpos_future is not None:
                    last_upstream_jpos_future = jpos_future
                quat_future = decode_pose_root_quat_xyzw_future(
                    latest, args.upstream_topic
                )
                if quat_future is None:
                    quat_future = decode_pose_root_quat_xyzw_future(
                        latest, args.downstream_topic
                    )
                if quat_future is not None:
                    last_upstream_quat_future = quat_future
                jvel_future = decode_pose_joint_vel_mj_future(
                    latest, args.upstream_topic
                )
                if jvel_future is None:
                    jvel_future = decode_pose_joint_vel_mj_future(
                        latest, args.downstream_topic
                    )
                if jvel_future is not None:
                    last_upstream_jvel_future = jvel_future
                motion_token = decode_pose_motion_token(
                    latest, args.upstream_topic
                )
                if motion_token is None:
                    motion_token = decode_pose_motion_token(
                        latest, args.downstream_topic
                    )
                if motion_token is not None:
                    last_upstream_motion_token = motion_token
                last_upstream_s = now

            if latest is not None:
                try:
                    pub.send(latest, zmq.NOBLOCK)
                    fwd_frames += 1
                except zmq.Again:
                    pass
                cur_state = STATE_LIVE
            else:
                # No upstream this tick. Decide what to fill in with
                # via the staged fallback ladder.
                age = (
                    float("inf") if last_upstream_s < 0
                    else (now - last_upstream_s)
                )
                target_state, blend_alpha = decide_fallback_state(
                    have_upstream=(last_upstream_msg is not None),
                    age_s=age,
                    stale_s=stale_s,
                    hold_last_secs=hold_last_secs,
                    blend_secs=blend_secs,
                    idle_mode=idle_mode,
                )
                cur_state = target_state

                if target_state == STATE_GAP:
                    gap_skips += 1
                elif target_state == STATE_HOLD:
                    if last_upstream_msg is None:
                        cur_state = STATE_COLD_IDLE
                    else:
                        # HOLD with optional yaw-rebase (2026-06-24).
                        # When x2_debug yaw is fresh, splice
                        # R_z(measured_yaw) into the cached message's
                        # root_quat fields before re-publishing so the
                        # SONIC policy tracks the robot's CURRENT
                        # heading instead of fighting back to the cached
                        # yaw. This makes a planner interruption
                        # leave the body free-yaw (joint pose still
                        # frozen on the cached jpos) instead of
                        # snapping back to whatever yaw upstream last
                        # published. Falls back to verbatim republish
                        # when (a) yaw-track is off, (b) x2_debug is
                        # stale, or (c) the splice fails (legacy
                        # message header without root_quat fields).
                        msg_to_send = last_upstream_msg
                        rebased = False
                        if (
                            yaw_track_enabled
                            and last_measured_yaw_s >= 0
                            and (now - last_measured_yaw_s) <= yaw_max_age_s
                        ):
                            spliced = rebase_hold_msg(
                                last_upstream_msg,
                                args.upstream_topic,
                                last_measured_yaw_rad,
                            )
                            if spliced is None:
                                # Try downstream topic as a fallback --
                                # mirrors the dual-topic decode pattern
                                # used elsewhere in the watchdog (the
                                # upstream may have already retopicked).
                                spliced = rebase_hold_msg(
                                    last_upstream_msg,
                                    args.downstream_topic,
                                    last_measured_yaw_rad,
                                )
                            if spliced is not None:
                                msg_to_send = spliced
                                rebased = True
                                if not hold_yaw_rebase_logged_active:
                                    print(
                                        f"[pose_watchdog] HOLD yaw-rebase: "
                                        f"ACTIVE -- cached frame's "
                                        f"root_quat_xyzw replaced with "
                                        f"R_z(measured_yaw="
                                        f"{math.degrees(last_measured_yaw_rad):+.2f}deg); "
                                        f"operator can freely rotate the "
                                        f"body during HOLD without policy "
                                        f"spring-back",
                                        flush=True,
                                    )
                                    hold_yaw_rebase_logged_active = True
                                    hold_yaw_rebase_logged_fallback = False
                            elif not hold_yaw_rebase_logged_splice_fail:
                                print(
                                    "[pose_watchdog] HOLD yaw-rebase: "
                                    "splice FAILED (cached message header "
                                    "missing root_quat_xyzw, or corrupt); "
                                    "falling back to verbatim re-publish "
                                    "for this HOLD cycle (pre-2026-06-24 "
                                    "behaviour)",
                                    flush=True,
                                )
                                hold_yaw_rebase_logged_splice_fail = True
                        elif (
                            hold_yaw_rebase_logged_active
                            and not hold_yaw_rebase_logged_fallback
                        ):
                            print(
                                "[pose_watchdog] HOLD yaw-rebase: "
                                "x2_debug went stale; falling back to "
                                "verbatim cached-frame re-publish "
                                "(operator may feel spring-back if the "
                                "body rotates during this HOLD cycle)",
                                flush=True,
                            )
                            hold_yaw_rebase_logged_fallback = True
                            hold_yaw_rebase_logged_active = False
                        try:
                            pub.send(msg_to_send, zmq.NOBLOCK)
                            hold_frames += 1
                            if rebased:
                                hold_frames_with_rebase += 1
                        except zmq.Again:
                            pass
                elif target_state == STATE_BLEND:
                    yaw_rebase: float | None = None
                    if (
                        yaw_track_enabled
                        and last_measured_yaw_s >= 0
                        and (now - last_measured_yaw_s) <= yaw_max_age_s
                    ):
                        yaw_rebase = last_measured_yaw_rad
                    if last_upstream_jpos is None:
                        msg = build_idle_frame_msg(
                            replay,
                            idle_tick,
                            args.downstream_topic,
                            yaw_rebase_rad=yaw_rebase,
                        )
                    else:
                        # Full-frame BLEND lerp (2026-06-23 future-window
                        # continuity fix). Pre-rebase the idle-clip
                        # quats and lerp ALL upstream-derived fields
                        # toward the idle clip with the same alpha as
                        # the current jpos -- otherwise the policy's
                        # planning horizon flips in one tick at the
                        # HOLD -> BLEND boundary even though current
                        # jpos is continuous, which is what produced
                        # the shoulder/elbow snap operators reported.
                        # Each *_future_override is only passed when
                        # the corresponding upstream cache is non-None
                        # (legacy v4 producers don't set futures; we
                        # gracefully revert to today's snap on those
                        # fields rather than fabricate values).
                        idle_jpos, _ = replay.current(idle_tick)
                        idle_jpos_future, idle_quat_future_raw, idle_jvel_future = (
                            replay.future_window(idle_tick)
                        )
                        if yaw_rebase is not None:
                            idle_quat_future = rebase_quats_xyzw_by_yaw(
                                idle_quat_future_raw, yaw_rebase
                            )
                        else:
                            idle_quat_future = idle_quat_future_raw
                        lerp = (
                            (1.0 - blend_alpha) * last_upstream_jpos
                            + blend_alpha * idle_jpos
                        ).astype(np.float32)
                        overrides: dict[str, np.ndarray] = {}
                        if last_upstream_jpos_future is not None:
                            overrides["joint_pos_mj_future_override"] = (
                                (1.0 - blend_alpha)
                                * last_upstream_jpos_future
                                + blend_alpha * idle_jpos_future
                            ).astype(np.float32)
                        if last_upstream_quat_future is not None:
                            overrides["root_quat_xyzw_future_override"] = (
                                nlerp_quat_arrays_xyzw(
                                    last_upstream_quat_future,
                                    idle_quat_future,
                                    blend_alpha,
                                ).astype(np.float32)
                            )
                        if last_upstream_jvel_future is not None:
                            overrides["joint_vel_mj_future_override"] = (
                                (1.0 - blend_alpha)
                                * last_upstream_jvel_future
                                + blend_alpha * idle_jvel_future
                            ).astype(np.float32)
                        if last_upstream_motion_token is not None:
                            overrides["motion_token_override"] = (
                                (1.0 - blend_alpha)
                                * last_upstream_motion_token
                            ).astype(np.float32)
                        msg = build_idle_frame_msg(
                            replay,
                            idle_tick,
                            args.downstream_topic,
                            yaw_rebase_rad=yaw_rebase,
                            joint_pos_mj_override=lerp,
                            **overrides,
                        )
                    try:
                        pub.send(msg, zmq.NOBLOCK)
                        blend_frames += 1
                    except zmq.Again:
                        pass
                    idle_tick += 1
                elif target_state in (STATE_COLD_IDLE, STATE_IDLE_CLIP):
                    yaw_rebase = None
                    if (
                        yaw_track_enabled
                        and last_measured_yaw_s >= 0
                        and (now - last_measured_yaw_s) <= yaw_max_age_s
                    ):
                        yaw_rebase = last_measured_yaw_rad
                    msg = build_idle_frame_msg(
                        replay,
                        idle_tick,
                        args.downstream_topic,
                        yaw_rebase_rad=yaw_rebase,
                    )
                    try:
                        pub.send(msg, zmq.NOBLOCK)
                        idle_frames += 1
                        if yaw_rebase is not None:
                            idle_frames_with_rebase += 1
                    except zmq.Again:
                        pass
                    idle_tick += 1

            # Emit a one-line transition log every time the state name
            # changes. Operators rely on this to correlate "I saw the
            # robot arm freeze for a few seconds then drift to stand"
            # with the watchdog's view of upstream availability.
            if cur_state != prev_state:
                if cur_state == STATE_LIVE:
                    if last_upstream_s < 0 or prev_state == STATE_COLD_IDLE:
                        msg_txt = (
                            f"{prev_state} -> LIVE (first upstream "
                            f"frame received)"
                        )
                    else:
                        gap_ms = (now - last_upstream_s) * 1000.0
                        msg_txt = (
                            f"{prev_state} -> LIVE (upstream pose "
                            f"frames flowing again after {gap_ms:.0f} "
                            f"ms gap)"
                        )
                elif cur_state == STATE_GAP:
                    msg_txt = (
                        f"{prev_state} -> GAP (upstream silent < "
                        f"{args.idle_stale_ms} ms; holding deploy "
                        f"cache)"
                    )
                elif cur_state == STATE_HOLD:
                    msg_txt = (
                        f"{prev_state} -> HOLD (re-publishing last "
                        f"upstream frame; will hold for "
                        f"{hold_last_secs:.1f}s)"
                    )
                elif cur_state == STATE_BLEND:
                    msg_txt = (
                        f"{prev_state} -> BLEND (lerping cached -> "
                        f"idle_stand over {blend_secs:.1f}s)"
                    )
                elif cur_state == STATE_IDLE_CLIP:
                    msg_txt = (
                        f"{prev_state} -> IDLE_CLIP (upstream silent "
                        f"past hold + blend window; tracking baked "
                        f"idle clip)"
                    )
                elif cur_state == STATE_COLD_IDLE:
                    msg_txt = f"{prev_state} -> COLD_IDLE"
                else:
                    msg_txt = f"{prev_state} -> {cur_state}"
                print(f"[pose_watchdog] state: {msg_txt}", flush=True)
                if (
                    cur_state in (
                        STATE_IDLE_CLIP, STATE_COLD_IDLE, STATE_BLEND
                    )
                    and prev_state not in (
                        STATE_IDLE_CLIP, STATE_COLD_IDLE, STATE_BLEND
                    )
                ):
                    idle_tick = 0
                prev_state = cur_state

            tick += 1

            if now - last_status_s >= args.status_every_s:
                age = (
                    float("inf") if last_upstream_s < 0
                    else (now - last_upstream_s)
                )
                age_str = (
                    "never" if last_upstream_s < 0
                    else f"{age * 1000:.0f}ms"
                )
                if cur_state == STATE_HOLD and last_upstream_s >= 0:
                    fb_age = max(0.0, age - stale_s)
                    state_str = (
                        f"HOLD t={fb_age:.1f}/{hold_last_secs:.1f}s"
                    )
                elif cur_state == STATE_BLEND and last_upstream_s >= 0:
                    fb_age = max(0.0, age - stale_s)
                    if blend_secs > 0.0:
                        alpha = (fb_age - hold_last_secs) / blend_secs
                        alpha = max(0.0, min(1.0, alpha))
                    else:
                        alpha = 1.0
                    state_str = f"BLEND alpha={alpha:.2f}"
                else:
                    state_str = cur_state
                if not yaw_track_enabled:
                    yaw_str = "off"
                elif last_measured_yaw_s < 0:
                    yaw_str = "never"
                else:
                    yaw_age_ms = (now - last_measured_yaw_s) * 1000.0
                    yaw_str = (
                        f"yaw={math.degrees(last_measured_yaw_rad):+.1f}deg "
                        f"age={yaw_age_ms:.0f}ms"
                    )
                print(
                    f"[pose_watchdog] tick={tick} state={state_str} "
                    f"mode={idle_mode} upstream_age={age_str} "
                    f"fwd={fwd_frames} hold={hold_frames} "
                    f"hold_rebased={hold_frames_with_rebase} "
                    f"blend={blend_frames} idle={idle_frames} "
                    f"idle_rebased={idle_frames_with_rebase} "
                    f"gap_skip={gap_skips} "
                    f"x2_debug=({yaw_str}) "
                    f"yaw_decode_fail={yaw_decode_failures}",
                    flush=True,
                )
                last_status_s = now

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print(
            "[pose_watchdog] SIGINT received; tearing down",
            flush=True,
        )
    finally:
        sub.close(linger=0)
        pub.close(linger=0)
        if yaw_sub is not None:
            yaw_sub.close(linger=0)
        ctx.term()

    print(
        f"[pose_watchdog] done. total_ticks={tick} fwd={fwd_frames} "
        f"hold={hold_frames} hold_rebased={hold_frames_with_rebase} "
        f"blend={blend_frames} idle={idle_frames} "
        f"idle_rebased={idle_frames_with_rebase} "
        f"gap_skip={gap_skips}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
