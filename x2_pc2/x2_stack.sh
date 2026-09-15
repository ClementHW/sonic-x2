#!/usr/bin/env bash
# PC2-local lifecycle manager for the complete X2 SONIC + Pico stack.
#
# Install this file as <PC2_PREFIX>/x2, then operate the robot after SSHing
# directly into PC2:
#
#   ./x2 preflight
#   ./x2 start              # starts every component and attaches to safety gate
#   ./x2 status
#   ./x2 logs deploy
#   ./x2 stop               # graceful RAMP_OUT -> MC handoff, then stops helpers
#
set -euo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
SELF_DIR="$(cd "$(dirname "${SELF}")" && pwd)"
PC2_PREFIX="${PC2_PREFIX:-${SELF_DIR}}"
PC2_WS="${PC2_WS:-${PC2_PREFIX}/ws}"
PC2_VENV="${PC2_VENV:-${PC2_PREFIX}/venv}"
PC2_ONNXRUNTIME="${PC2_ONNXRUNTIME:-${PC2_PREFIX}/onnxruntime}"
PC2_LOG_ROOT="${PC2_LOG_ROOT:-${PC2_PREFIX}/log}"
PC2_AIMDK_PREFIX="${PC2_AIMDK_PREFIX:-/agibot/data/home/agi/lx2501_3-v0.9.0.4/install/aimdk_msgs}"
PC2_DEPLOY_SH="${PC2_DEPLOY_SH:-${PC2_PREFIX}/gear_sonic_deploy/deploy_x2.sh}"
PC2_TELEOP_SH="${PC2_TELEOP_SH:-${PC2_PREFIX}/planner_stack/run_x2_pc2_teleop_stack.sh}"
XRT_REPO="${XRT_REPO:-/agibot/data/home/agi/XRoboToolkit-Teleop-Sample-Python}"

X2_MODEL="${X2_MODEL:-${PC2_PREFIX}/policies/x2_sonic_policy.onnx}"
X2_TUNING="${X2_TUNING:-${PC2_PREFIX}/gear_sonic_deploy/configs/real_deploy_tuning/walking_soft_kp.yaml}"
X2_WRIST_BYPASS="${X2_WRIST_BYPASS:-ik}"
X2_HEAD_BYPASS="${X2_HEAD_BYPASS:-ref}"

POSE_PORT="${POSE_PORT:-5556}"
WATCHDOG_PORT="${WATCHDOG_PORT:-5558}"
DEBUG_PORT="${DEBUG_PORT:-5557}"
RESUME_PORT="${RESUME_PORT:-5566}"
MONITOR_PORT="${MONITOR_PORT:-5567}"
POSE_STALE_MS="${POSE_STALE_MS:-300}"
POSE_HOLD_SECS="${POSE_HOLD_SECS:-5.0}"
POSE_BLEND_SECS="${POSE_BLEND_SECS:-3.0}"

DEPLOY_SESSION=x2_deploy
WATCHDOG_SESSION=x2_pose_watchdog
HAND_SESSION=x2_hand_bridge
MONITOR_SESSION=x2_motor_monitor
PLANNER_SESSION=x2_planner
BRIDGE_SESSION=x2_xrt_bridge
RUN_STAMP="${X2_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

green=$'\e[32m'; yellow=$'\e[33m'; red=$'\e[31m'; reset=$'\e[0m'
log()  { printf '%s[x2]%s %s\n' "${green}" "${reset}" "$*"; }
warn() { printf '%s[x2 WARN]%s %s\n' "${yellow}" "${reset}" "$*" >&2; }
die()  { printf '%s[x2 ERROR]%s %s\n' "${red}" "${reset}" "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
PC2-local lifecycle manager for the complete X2 SONIC + Pico stack.

Typical use after SSHing into PC2:
  ./x2 preflight
  ./x2 start
  ./x2 status
  ./x2 stop

Commands:
  preflight              verify all PC2-local dependencies without starting
  start [--no-attach]    start the complete stack; attach to deploy safety gate
  start --no-confirm     start after pose gate without the deploy y/N prompt
  gate                   show deploy safety-gate/control output
  yes                    answer the visible deploy y/N safety prompt
  status                 show processes, tmux sessions, and local ports
  logs NAME              NAME = deploy|watchdog|hand|monitor|planner|bridge
  attach NAME            attach to a tmux session (Ctrl-B d detaches)
  release-arms           clear a stale arm override once (bridge should be off)
  stop [--yes]           gracefully return control to MC, then stop all helpers
  restart                stop, then start again (interactive confirmations)
EOF
}

session_alive() { tmux has-session -t "$1" 2>/dev/null; }

proc_alive() {
    local pat="$1" p cmdline
    for p in /proc/[0-9]*; do
        [[ "${p##*/}" = "$$" || "${p##*/}" = "$PPID" ]] && continue
        [[ -e "${p}/cmdline" && -r "${p}/cmdline" ]] || continue
        cmdline="$( { tr '\0' ' ' < "${p}/cmdline"; } 2>/dev/null)" || continue
        case "${cmdline}" in *"${pat}"*) return 0 ;; esac
    done
    return 1
}

deploy_node_alive() {
    local p exe
    for p in /proc/[0-9]*; do
        [[ -L "${p}/exe" ]] || continue
        exe="$(readlink "${p}/exe" 2>/dev/null)" || continue
        [[ "${exe##*/}" = x2_deploy_onnx_ref ]] && return 0
    done
    return 1
}

port_listening() {
    ss -ltn 2>/dev/null | grep -Eq "[:.]$1[[:space:]]"
}

source_robot_env() {
    # ROS2/AimDK generated setup files legitimately probe unset variables;
    # isolate them from this manager's nounset policy, then restore it.
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    export AMENT_PREFIX_PATH="${PC2_AIMDK_PREFIX}:${AMENT_PREFIX_PATH:-}"
    export LD_LIBRARY_PATH="${PC2_AIMDK_PREFIX}/lib:${PC2_ONNXRUNTIME}/lib:${LD_LIBRARY_PATH:-}"
    # shellcheck disable=SC1091
    source "${PC2_WS}/install/setup.bash"
    export PYTHONPATH="${PC2_AIMDK_PREFIX}/local/lib/python3.10/dist-packages:${PC2_WS}/src:${PYTHONPATH:-}"
    set -u
}

required_file() {
    [[ -e "$1" ]] || { printf '  MISSING %s\n' "$1" >&2; return 1; }
    printf '  ok      %s\n' "$1"
}

preflight() {
    local failed=0
    printf 'PC2 prefix: %s\n' "${PC2_PREFIX}"
    command -v tmux >/dev/null || { echo '  MISSING command: tmux' >&2; failed=1; }
    command -v ss >/dev/null || { echo '  MISSING command: ss' >&2; failed=1; }
    command -v setsid >/dev/null || { echo '  MISSING command: setsid' >&2; failed=1; }
    for f in \
        "${PC2_VENV}/bin/python3" \
        "${PC2_AIMDK_PREFIX}/local/lib/python3.10/dist-packages/aimdk_msgs" \
        "${PC2_WS}/install/setup.bash" \
        "${PC2_PREFIX}/data/idle_stand.x2m2" \
        "${PC2_PREFIX}/gear_sonic_deploy/scripts/x2_pose_watchdog.py" \
        "${PC2_WS}/src/x2/agi_x2_deploy_onnx_ref/scripts/x2_hand_zmq_to_aimdk_bridge.py" \
        "${PC2_WS}/src/x2/agi_x2_deploy_onnx_ref/scripts/x2_motor_monitor.py" \
        "${PC2_DEPLOY_SH}" \
        "${PC2_TELEOP_SH}" \
        "${PC2_PREFIX}/planner-venv/bin/python" \
        "${PC2_PREFIX}/planner_models/kplanner_torch/vqvae/x2_kplanner_vqvae.ckpt" \
        "${PC2_PREFIX}/planner_models/kplanner_torch/pose/x2_kplanner_pose.ckpt" \
        "${PC2_PREFIX}/planner_models/kplanner_torch/root/x2_kplanner_root.ckpt" \
        "${PC2_PREFIX}/planner_models/kplanner_torch/x2_clip.ckpt" \
        "${PC2_PREFIX}/planner_stack/models/kplanner_idle_anchor_g1teleop_v3.pkl" \
        "${PC2_PREFIX}/teleop_bridge/x2_xrt_sonic_bridge.py" \
        "${XRT_REPO}/xrobotoolkit_teleop" \
        "${X2_MODEL}" \
        "${X2_TUNING}"; do
        required_file "${f}" || failed=1
    done
    if [[ -x "${PC2_VENV}/bin/python3" ]]; then
        "${PC2_VENV}/bin/python3" -c 'import zmq; print("  ok      python zmq")' || failed=1
    fi
    if [[ "${failed}" -ne 0 ]]; then
        die 'preflight failed; keep SONIC stopped until every required item is present'
    fi
    log 'preflight passed (all runtime dependencies are on PC2)'
}

start_session() {
    local session="$1" pattern="$2" worker="$3" logfile="$4"
    if proc_alive "${pattern}"; then
        log "${session}: already running"
        return 0
    fi
    if session_alive "${session}"; then
        warn "${session}: removing stale tmux pane (no matching process)"
        tmux kill-session -t "${session}"
    fi
    tmux new-session -d -s "${session}" \
        "env PC2_PREFIX='${PC2_PREFIX}' PC2_WS='${PC2_WS}' PC2_VENV='${PC2_VENV}' PC2_ONNXRUNTIME='${PC2_ONNXRUNTIME}' PC2_LOG_ROOT='${PC2_LOG_ROOT}' PC2_AIMDK_PREFIX='${PC2_AIMDK_PREFIX}' PC2_DEPLOY_SH='${PC2_DEPLOY_SH}' X2_RUN_STAMP='${RUN_STAMP}' X2_MODEL='${X2_MODEL}' X2_TUNING='${X2_TUNING}' X2_WRIST_BYPASS='${X2_WRIST_BYPASS}' X2_HEAD_BYPASS='${X2_HEAD_BYPASS}' POSE_PORT='${POSE_PORT}' WATCHDOG_PORT='${WATCHDOG_PORT}' DEBUG_PORT='${DEBUG_PORT}' RESUME_PORT='${RESUME_PORT}' MONITOR_PORT='${MONITOR_PORT}' POSE_STALE_MS='${POSE_STALE_MS}' POSE_HOLD_SECS='${POSE_HOLD_SECS}' POSE_BLEND_SECS='${POSE_BLEND_SECS}' X2_DEPLOY_NO_CONFIRM='${X2_DEPLOY_NO_CONFIRM:-0}' '${SELF}' '${worker}' 2>&1 | tee -a '${logfile}'"
    log "${session}: started"
}

wait_for_pose() {
    log "waiting for PC2-local pose stream on 127.0.0.1:${WATCHDOG_PORT} ..."
    "${PC2_VENV}/bin/python3" - "${WATCHDOG_PORT}" <<'PY'
import sys
import zmq

port = int(sys.argv[1])
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt_string(zmq.SUBSCRIBE, "pose")
sub.connect(f"tcp://127.0.0.1:{port}")
sub.RCVTIMEO = 15000
try:
    parts = sub.recv_multipart()
except zmq.Again:
    print("NO POSE STREAM within 15 seconds", file=sys.stderr)
    raise SystemExit(1)
print(f"pose stream OK ({len(parts[-1])} bytes)")
PY
}

start_all() {
    local attach=1
    X2_DEPLOY_NO_CONFIRM=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-attach) attach=0 ;;
            --no-confirm) X2_DEPLOY_NO_CONFIRM=1; attach=0 ;;
            *) die "unknown start option: $1" ;;
        esac
        shift
    done

    preflight
    mkdir -p "${PC2_LOG_ROOT}"

    log 'starting PC2 planner + Pico/XR bridge'
    PC2_PREFIX="${PC2_PREFIX}" "${PC2_TELEOP_SH}" start

    start_session "${WATCHDOG_SESSION}" x2_pose_watchdog.py _watchdog \
        "${PC2_LOG_ROOT}/pose_watchdog_${RUN_STAMP}.log"
    start_session "${HAND_SESSION}" x2_hand_zmq_to_aimdk_bridge.py _hand \
        "${PC2_LOG_ROOT}/hand_bridge_${RUN_STAMP}.log"
    start_session "${MONITOR_SESSION}" x2_motor_monitor.py _monitor \
        "${PC2_LOG_ROOT}/motor_monitor_${RUN_STAMP}.log"

    wait_for_pose || die 'pose gate failed; deploy was NOT started'
    local deploy_was_running=0
    proc_alive 'deploy_x2.sh onbot' && deploy_was_running=1
    start_session "${DEPLOY_SESSION}" 'deploy_x2.sh onbot' _deploy \
        "${PC2_LOG_ROOT}/deploy_${RUN_STAMP}.log"

    if [[ "${deploy_was_running}" -eq 1 ]]; then
        log 'complete stack is active; deploy was already running and was not touched'
        return 0
    fi

    if [[ "${X2_DEPLOY_NO_CONFIRM}" -eq 1 ]]; then
        warn 'deploy y/N prompt bypassed; the PC2-local pose gate was the ignition gate'
        return 0
    fi

    log 'deploy preflight is running; the y/N hardware takeover gate is inside tmux:x2_deploy'
    if [[ "${attach}" -eq 1 && -t 0 && -t 1 ]]; then
        log 'attaching now (type y only after the physical safety check; Ctrl-B d detaches)'
        if [[ -n "${TMUX:-}" ]]; then
            exec tmux switch-client -t "${DEPLOY_SESSION}"
        fi
        exec tmux attach-session -t "${DEPLOY_SESSION}"
    fi
    echo "Run: ${SELF} gate"
    echo "Then, only when the prompt is visible and the area is safe: ${SELF} yes"
}

gate() {
    session_alive "${DEPLOY_SESSION}" || die 'x2_deploy tmux session is not running'
    tmux capture-pane -t "${DEPLOY_SESSION}" -p -S -30 2>/dev/null \
        | grep -vE '^\s*$' | tail -20
}

confirm_deploy() {
    local pane
    session_alive "${DEPLOY_SESSION}" || die 'x2_deploy tmux session is not running'
    pane="$(tmux capture-pane -t "${DEPLOY_SESSION}" -p -S -40 2>/dev/null)"
    if ! grep -Fq 'Stop MC and launch policy?' <<<"${pane}"; then
        printf '%s\n' "${pane}" | grep -vE '^\s*$' | tail -15
        die 'the deploy safety prompt is not visible yet; wait and run x2 gate again'
    fi
    warn 'sending y: SONIC will take hardware control now'
    tmux send-keys -t "${DEPLOY_SESSION}" y Enter
}

status() {
    printf '%-22s %-9s %-9s\n' COMPONENT TMUX PROCESS
    local item session pattern tmux_state proc_state
    while IFS='|' read -r item session pattern; do
        session_alive "${session}" && tmux_state=RUNNING || tmux_state=-
        proc_alive "${pattern}" && proc_state=RUNNING || proc_state=-
        printf '%-22s %-9s %-9s\n' "${item}" "${tmux_state}" "${proc_state}"
    done <<EOF
planner|${PLANNER_SESSION}|pc2_kplanner_onnx.py
XR bridge|${BRIDGE_SESSION}|x2_xrt_sonic_bridge.py
pose watchdog|${WATCHDOG_SESSION}|x2_pose_watchdog.py
deploy|${DEPLOY_SESSION}|deploy_x2.sh onbot
hand bridge|${HAND_SESSION}|x2_hand_zmq_to_aimdk_bridge.py
motor monitor|${MONITOR_SESSION}|x2_motor_monitor.py
EOF
    printf '\nLocal ports:\n'
    local p state
    for p in 60061 5563 5564 "${POSE_PORT}" "${WATCHDOG_PORT}" "${DEBUG_PORT}" "${MONITOR_PORT}"; do
        port_listening "${p}" && state=LISTENING || state=-
        printf '  :%-5s %s\n' "${p}" "${state}"
    done
}

logs() {
    local name="${1:-deploy}" session
    case "${name}" in
        deploy) session="${DEPLOY_SESSION}" ;;
        watchdog) session="${WATCHDOG_SESSION}" ;;
        hand) session="${HAND_SESSION}" ;;
        monitor) session="${MONITOR_SESSION}" ;;
        planner) session="${PLANNER_SESSION}" ;;
        bridge) session="${BRIDGE_SESSION}" ;;
        *) die 'logs NAME: deploy|watchdog|hand|monitor|planner|bridge' ;;
    esac
    session_alive "${session}" || die "tmux session not running: ${session}"
    tmux capture-pane -t "${session}" -p -S -100 2>/dev/null \
        | grep -vE '^\s*$' | tail -80
}

attach_session() {
    local name="${1:-deploy}" session
    case "${name}" in
        deploy) session="${DEPLOY_SESSION}" ;;
        watchdog) session="${WATCHDOG_SESSION}" ;;
        hand) session="${HAND_SESSION}" ;;
        monitor) session="${MONITOR_SESSION}" ;;
        planner) session="${PLANNER_SESSION}" ;;
        bridge) session="${BRIDGE_SESSION}" ;;
        *) die 'attach NAME: deploy|watchdog|hand|monitor|planner|bridge' ;;
    esac
    session_alive "${session}" || die "tmux session not running: ${session}"
    if [[ -n "${TMUX:-}" ]]; then
        exec tmux switch-client -t "${session}"
    fi
    exec tmux attach-session -t "${session}"
}

release_arms() {
    # This mirrors the bridge's normal shutdown frame. If the live bridge is
    # still publishing, it will intentionally assert its next target again.
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    # shellcheck disable=SC1091
    source /agibot/data/home/agi/lx2501_3-v0.9.0.4/install/setup.bash
    # shellcheck disable=SC1091
    source /agibot/data/home/agi/miniconda3/etc/profile.d/conda.sh
    conda activate xr_tabletop
    set -u
    python -u "${PC2_PREFIX}/teleop_bridge/x2_xrt_sonic_bridge.py" --release-arms-once
}

stop_helper_session() {
    local session="$1"
    session_alive "${session}" || return 0
    tmux send-keys -t "${session}" C-c 2>/dev/null || true
    sleep 1
    tmux kill-session -t "${session}" 2>/dev/null || true
}

stop_all() {
    local assume_yes=0 reply
    [[ "${1:-}" = '--yes' ]] && assume_yes=1
    if [[ "${assume_yes}" -ne 1 ]]; then
        [[ -t 0 ]] || die 'stop requires an interactive terminal or: x2 stop --yes'
        printf 'Gracefully stop SONIC and return the robot to native MC? [y/N] '
        read -r reply
        [[ "${reply}" =~ ^[Yy]$ ]] || { log 'stop cancelled'; return 0; }
    fi

    if session_alive "${DEPLOY_SESSION}" || deploy_node_alive; then
        warn 'requesting deploy RAMP_OUT -> HOLD_FOR_MC -> native MC restart'
        tmux send-keys -t "${DEPLOY_SESSION}" C-c 2>/dev/null || true
        local deadline=$((SECONDS + 50))
        while deploy_node_alive || proc_alive 'deploy_x2.sh onbot'; do
            if (( SECONDS >= deadline )); then
                die 'deploy did not finish its safe handoff within 50 s; helpers were left running. Inspect: x2 attach deploy'
            fi
            sleep 1
        done
        tmux kill-session -t "${DEPLOY_SESSION}" 2>/dev/null || true
        log 'deploy exited and native MC handoff completed'
    fi

    stop_helper_session "${HAND_SESSION}"
    stop_helper_session "${MONITOR_SESSION}"
    stop_helper_session "${WATCHDOG_SESSION}"
    PC2_PREFIX="${PC2_PREFIX}" "${PC2_TELEOP_SH}" stop
    log 'complete stack stopped'
}

# Internal workers are installed with the lifecycle manager on PC2.
worker_watchdog() {
    source_robot_env
    exec "${PC2_VENV}/bin/python3" \
        "${PC2_PREFIX}/gear_sonic_deploy/scripts/x2_pose_watchdog.py" \
        --upstream-host 127.0.0.1 --upstream-port "${POSE_PORT}" --upstream-topic pose \
        --downstream-port "${WATCHDOG_PORT}" --downstream-topic pose \
        --idle-x2m2 "${PC2_PREFIX}/data/idle_stand.x2m2" \
        --idle-stale-ms "${POSE_STALE_MS}" --idle-mode blend \
        --hold-last-secs "${POSE_HOLD_SECS}" --blend-secs "${POSE_BLEND_SECS}" \
        --x2-debug-port "${DEBUG_PORT}"
}

worker_hand() {
    source_robot_env
    exec "${PC2_VENV}/bin/python3" \
        "${PC2_WS}/src/x2/agi_x2_deploy_onnx_ref/scripts/x2_hand_zmq_to_aimdk_bridge.py" \
        --zmq-host localhost --zmq-port "${WATCHDOG_PORT}" --zmq-topic pose
}

worker_monitor() {
    source_robot_env
    exec "${PC2_VENV}/bin/python3" \
        "${PC2_WS}/src/x2/agi_x2_deploy_onnx_ref/scripts/x2_motor_monitor.py" \
        --jsonl "${PC2_LOG_ROOT}/motor_monitor_${RUN_STAMP}.jsonl" \
        --zmq-port "${MONITOR_PORT}" --zmq-topic motor_monitor
}

DEPLOY_CHILD_PID=""

forward_deploy_signal() {
    local signal_name="$1"
    warn "deploy ${signal_name} received; forwarding TERM to deploy process group for RAMP_OUT -> HOLD_FOR_MC"
    # deploy_x2 runs in its own session so the tmux pane's pipeline cannot
    # accidentally deliver Ctrl-C to tee but miss the process which owns the
    # MC handoff. A negative PID addresses that complete process group.
    #
    # A background process inherits SIGINT=ignored from bash before `setsid`
    # execs it; forwarding INT would therefore be acknowledged but ignored.
    # deploy_x2 installs the same graceful cleanup trap for TERM, whose
    # default disposition is not inherited as ignored, so TERM is the
    # reliable Ctrl-C transport across this session boundary.
    if [[ -n "${DEPLOY_CHILD_PID}" ]] && kill -0 "${DEPLOY_CHILD_PID}" 2>/dev/null; then
        kill -s TERM -- "-${DEPLOY_CHILD_PID}" 2>/dev/null || true
    fi
}

worker_deploy() {
    local args=(
        onbot --no-docker --vla
        --vla-zmq-host localhost --vla-zmq-port "${WATCHDOG_PORT}" --vla-zmq-topic pose
        --vla-debug-port "${DEBUG_PORT}" --vla-debug-topic x2_debug
        --vla-resume-host 127.0.0.1 --vla-resume-port "${RESUME_PORT}" --vla-resume-topic pose_resume
        --wrist-bypass "${X2_WRIST_BYPASS}" --head-bypass "${X2_HEAD_BYPASS}"
        --model "${X2_MODEL}"
        --log-dir "${PC2_LOG_ROOT}/deploy_${RUN_STAMP}"
        --onbot-prefix "${PC2_PREFIX}" --onbot-ws "${PC2_WS}"
        --onbot-venv "${PC2_VENV}" --onbot-onnxruntime "${PC2_ONNXRUNTIME}"
        --onbot-aimdk-prefix "${PC2_AIMDK_PREFIX}"
        --deploy-extra-arg --disable-pose-ref-watchdog
        --enable-soft-shutdown
        --tuning-config "${X2_TUNING}"
        --no-require-stand-default
    )
    [[ "${X2_DEPLOY_NO_CONFIRM:-0}" -eq 1 ]] && args+=(--no-confirm)
    cd "${PC2_PREFIX}/gear_sonic_deploy"
    local rc=0 wait_rc
    # Put the deploy launcher in a dedicated session. Ctrl-C then reaches
    # this worker predictably and the trap below forwards it to exactly the
    # launcher (and its C++ deploy child) that owns the MC handoff protocol.
    #
    # Bash gives an asynchronous job /dev/null on stdin when job control is
    # disabled.  deploy_x2 has an interactive y/N safety gate, so without
    # this explicit tmux-pane TTY it reads EOF, exits, and this worker tears
    # down the planner/bridge as though the deploy had completed.  The
    # non-TTY branch keeps the internal worker usable in automated harnesses;
    # normal `x2 start` always launches this worker inside a tmux pane.
    if [[ -t 0 ]]; then
        setsid "${PC2_DEPLOY_SH}" "${args[@]}" </dev/tty &
    else
        setsid "${PC2_DEPLOY_SH}" "${args[@]}" &
    fi
    DEPLOY_CHILD_PID=$!
    trap 'forward_deploy_signal INT' INT
    trap 'forward_deploy_signal TERM' TERM
    set +e
    # A trapped signal interrupts `wait` before the child has necessarily
    # exited. Keep waiting until the forwarded graceful handoff actually
    # finishes; only then may helper sessions be stopped.
    while true; do
        wait "${DEPLOY_CHILD_PID}"
        wait_rc=$?
        if ! kill -0 "${DEPLOY_CHILD_PID}" 2>/dev/null; then
            rc=${wait_rc}
            break
        fi
    done
    set -e
    DEPLOY_CHILD_PID=""
    trap - INT TERM

    # The deploy pane owns the whole stack lifecycle. Whether it ended by
    # Ctrl-C, a normal return, or an error, stop every helper after the deploy
    # launcher has completed its MC handoff. This makes Ctrl-C in the tmux
    # window equivalent to `x2 stop` instead of leaving a half-running stack.
    stop_helper_session "${HAND_SESSION}"
    stop_helper_session "${MONITOR_SESSION}"
    stop_helper_session "${WATCHDOG_SESSION}"
    PC2_PREFIX="${PC2_PREFIX}" "${PC2_TELEOP_SH}" stop
    log "deploy ended (status=${rc}); complete stack stopped"
    return "${rc}"
}

cmd="${1:-help}"
[[ $# -gt 0 ]] && shift
case "${cmd}" in
    preflight) preflight "$@" ;;
    start) start_all "$@" ;;
    gate) gate ;;
    yes) confirm_deploy ;;
    status) status ;;
    logs) logs "$@" ;;
    attach) attach_session "$@" ;;
    release-arms) release_arms ;;
    stop) stop_all "$@" ;;
    restart) stop_all; start_all ;;
    _watchdog) worker_watchdog ;;
    _hand) worker_hand ;;
    _monitor) worker_monitor ;;
    _deploy) worker_deploy ;;
    help|-h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
