#!/usr/bin/env bash
# X2 Pico-teleop stack — PC2 runtime.
#
#   Pico -> XRoboToolkit PC Service (:60061)
#     -> x2_xrt_sonic_bridge.py   (arm_targets/hand_finger_cmd + planner_cmd)
#     -> pc2_kplanner_onnx.py     (merge planner: neural gait legs/waist/head
#                                  + teleop arm/hand overlay, torch backend)
#     -> pose :5556 -> x2_pose_watchdog :5558 -> SONIC deploy -> robot
#
# This script manages the two teleop tmux sessions (planner + bridge).
# The complete deploy stack (including these two sessions) is managed locally
# by <PC2_PREFIX>/x2. This lower-level script remains useful for focused
# planner/bridge debugging on PC2.
#
# Usage (on PC2):
#   run_x2_pc2_teleop_stack.sh start    # planner + bridge in tmux
#   run_x2_pc2_teleop_stack.sh stop     # SIGINT both, verify gone
#   run_x2_pc2_teleop_stack.sh status   # tmux/process/port summary
#   run_x2_pc2_teleop_stack.sh logs planner|bridge
#   run_x2_pc2_teleop_stack.sh planner  # planner in FOREGROUND (debug)
#   run_x2_pc2_teleop_stack.sh bridge   # bridge in FOREGROUND (debug)
#
# Env overrides:
#   PLANNER_MODE=slow_walk|walk|idle|run_proxy   (gait template; default slow_walk)
#   WALK_MAX_LIN_VEL=0.22  WALK_MAX_SIDE_MPS=0.22 (m/s, fixed speed past deadzone)
#   WALK_PROPORTIONAL=1    (restore deflection-scaled speed; default 0 = fixed)
#   WALK_DEADMAN=none      (e.g. left_trigger = hold-to-walk)
#   WALK_ZERO_CONFIRM_S=0.25 (ignore short Pico/XR [0,0] reconnect glitches)
#   PC2_PREFIX=/agibot/data/home/agi/gear-sonic
#   XRT_REPO=/agibot/data/home/agi/XRoboToolkit-Teleop-Sample-Python
set -euo pipefail

PC2_PREFIX="${PC2_PREFIX:-/agibot/data/home/agi/gear-sonic}"
STACK="${PC2_PREFIX}/planner_stack"
MODELS="${PC2_PREFIX}/planner_models/kplanner_torch"
VENV_PY="${PC2_PREFIX}/planner-venv/bin/python"
XRT_REPO="${XRT_REPO:-/agibot/data/home/agi/XRoboToolkit-Teleop-Sample-Python}"
BRIDGE="${PC2_PREFIX}/teleop_bridge/x2_xrt_sonic_bridge.py"

PLANNER_MODE="${PLANNER_MODE:-slow_walk}"
PLANNER_TORCH_THREADS="${PLANNER_TORCH_THREADS:-4}"
PLANNER_TORCH_INTEROP_THREADS="${PLANNER_TORCH_INTEROP_THREADS:-1}"
WALK_MAX_LIN_VEL="${WALK_MAX_LIN_VEL:-0.22}"
WALK_MAX_SIDE_MPS="${WALK_MAX_SIDE_MPS:-0.22}"
WALK_PROPORTIONAL="${WALK_PROPORTIONAL:-0}"
WALK_DEADMAN="${WALK_DEADMAN:-none}"
WALK_ZERO_CONFIRM_S="${WALK_ZERO_CONFIRM_S:-0.25}"
HIP_HEIGHT_M="${HIP_HEIGHT_M:-0.687}"

PLANNER_SESSION="x2_planner"
BRIDGE_SESSION="x2_xrt_bridge"
LOG_DIR="${PC2_PREFIX}/log"
XR_SERVICE_LOG="/agibot/data/home/agi/roboticsservice.log"

mkdir -p "${LOG_DIR}"

# Bridge environment: ROS2 humble + robot SDK + conda xr_tabletop
# (same SOP as the XRoboToolkit teleop samples).
read -r -d '' BRIDGE_ENV <<'EOF' || true
source /opt/ros/humble/setup.bash
source /agibot/data/home/agi/lx2501_3-v0.9.0.4/install/setup.bash
source /agibot/data/home/agi/miniconda3/etc/profile.d/conda.sh
conda activate xr_tabletop
EOF

PLANNER_ENV="PYTHONPATH=${STACK}:${STACK}/motionbricks"

log()  { printf '\e[32m[pc2-teleop %s]\e[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\e[33m[pc2-teleop %s WARN]\e[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
err()  { printf '\e[31m[pc2-teleop %s ERROR]\e[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

session_alive() { tmux has-session -t "$1" 2>/dev/null; }

port_listening() { ss -ltn 2>/dev/null | grep -q ":$1 "; }

# 0 = a process whose cmdline matches the pattern is alive. Scans /proc
# rather than pgrep -f (which matches the checking command itself).
proc_alive() {
    local pat="$1" p c
    for p in /proc/[0-9]*; do
        [[ "${p##*/}" = "$$" || "${p##*/}" = "$PPID" ]] && continue
        [[ -e "${p}/cmdline" && -r "${p}/cmdline" ]] || continue
        c="$( { tr '\0' ' ' < "${p}/cmdline"; } 2>/dev/null)" || continue
        case "$c" in *"${pat}"*) return 0 ;; esac
    done
    return 1
}

# XRoboToolkit PC Service (:60061) — the bridge's data source; idempotent.
ensure_xr_service() {
    if port_listening 60061; then return 0; fi
    log "XRoboToolkit PC Service not listening on :60061, starting ..."
    nohup /opt/apps/roboticsservice/runService.sh > "${XR_SERVICE_LOG}" 2>&1 </dev/null &
    for _ in $(seq 1 20); do
        sleep 0.5
        if port_listening 60061; then
            log "XRoboToolkit PC Service ready (:60061)"
            return 0
        fi
    done
    err "XR service did not come up in 10 s; recent log:"
    tail -15 "${XR_SERVICE_LOG}" >&2 || true
    return 1
}

planner_cmd() {
    cat <<EOF
env OMP_NUM_THREADS=${PLANNER_TORCH_THREADS} MKL_NUM_THREADS=${PLANNER_TORCH_THREADS} \
  ${PLANNER_ENV} ${VENV_PY} ${STACK}/gear_sonic/scripts/pc2_kplanner_onnx.py \
  --backend torch --device cpu \
  --torch-num-threads ${PLANNER_TORCH_THREADS} \
  --torch-num-interop-threads ${PLANNER_TORCH_INTEROP_THREADS} \
  --vqvae-ckpt ${MODELS}/vqvae/x2_kplanner_vqvae.ckpt \
  --pose-ckpt  ${MODELS}/pose/x2_kplanner_pose.ckpt \
  --root-ckpt  ${MODELS}/root/x2_kplanner_root.ckpt \
  --planner-mode ${PLANNER_MODE} \
  --cold-start-ramp-tau-s 0 \
  --warmup-qpos ${STACK}/models/kplanner_idle_anchor_g1teleop_v3.pkl \
  --no-yaw-rebase \
  --dances-dir ${STACK}/dances_x2m2 \
  --cmd-bind --cmd-port 5563 \
  --arm-port 5564 \
  --pub-port 5556 --pub-host 0.0.0.0
EOF
}

bridge_cmd() {
    local prop_arg=""
    [[ "${WALK_PROPORTIONAL}" == "1" ]] && prop_arg="--walk-proportional"
    cat <<EOF
python -u ${BRIDGE} --xrt-repo ${XRT_REPO} \
  --planner-cmd-host 127.0.0.1 --planner-cmd-port 5563 \
  --arm-hands-host 127.0.0.1 --arm-hands-port 5564 \
  --walk-max-lin-vel ${WALK_MAX_LIN_VEL} \
  --walk-max-side-mps ${WALK_MAX_SIDE_MPS} \
  --hip-height-m ${HIP_HEIGHT_M} \
  --walk-deadman ${WALK_DEADMAN} \
  --walk-zero-confirm-s ${WALK_ZERO_CONFIRM_S} ${prop_arg}
EOF
}

# tmux body: run the given command string with logging, keep the pane
# afterwards so scrollback/post-mortem survives.
tmux_run() {
    local session="$1" body="$2" logfile="$3"
    tmux new-session -d -s "${session}" \
        "bash -c '${body} 2>&1 | tee -a \"${logfile}\"; echo; echo \"[tmux] process exited (status=\$?)\"; exec bash'"
}

preflight_planner() {
    [[ -x "${VENV_PY}" ]] || { err "planner venv missing: ${VENV_PY}"; return 1; }
    for f in vqvae/x2_kplanner_vqvae.ckpt pose/x2_kplanner_pose.ckpt \
             root/x2_kplanner_root.ckpt x2_clip.ckpt; do
        [[ -f "${MODELS}/${f}" ]] || { err "missing model: ${MODELS}/${f}"; return 1; }
    done
    [[ -f "${STACK}/models/kplanner_idle_anchor_g1teleop_v3.pkl" ]] || {
        err "missing warmup pkl in ${STACK}/models"; return 1; }
}

preflight_bridge() {
    [[ -f "${BRIDGE}" ]] || { err "bridge missing: ${BRIDGE}"; return 1; }
    [[ -d "${XRT_REPO}/xrobotoolkit_teleop" ]] || { err "XRoboToolkit repo missing: ${XRT_REPO}"; return 1; }
}

cmd="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "${cmd}" in
  start)
    preflight_planner
    preflight_bridge
    ensure_xr_service
    # 幂等以"进程在跑"为准：tmux 窗格可能是进程退出后留下的 keep-alive 壳。
    if proc_alive "pc2_kplanner_onnx.py"; then
        warn "planner already running (stop first, or status)"
    else
        session_alive "${PLANNER_SESSION}" && tmux kill-session -t "${PLANNER_SESSION}" 2>/dev/null
        log "starting planner (mode=${PLANNER_MODE}) -> tmux:${PLANNER_SESSION}"
        tmux_run "${PLANNER_SESSION}" "$(planner_cmd)" "${LOG_DIR}/planner_latest.log"
        sleep 1
    fi
    if proc_alive "x2_xrt_sonic_bridge"; then
        warn "bridge already running (stop first, or status)"
    else
        session_alive "${BRIDGE_SESSION}" && tmux kill-session -t "${BRIDGE_SESSION}" 2>/dev/null
        log "starting XRoboToolkit bridge -> tmux:${BRIDGE_SESSION}"
        tmux_run "${BRIDGE_SESSION}" "${BRIDGE_ENV}; $(bridge_cmd)" "${LOG_DIR}/bridge_latest.log"
    fi
    sleep 2
    "$0" status
    ;;
  service)
    ensure_xr_service
    ss -ltnp 2>/dev/null | grep 60061 || true
    echo "日志: tail -F ${XR_SERVICE_LOG}"
    ;;
  stop)
    log "stopping bridge + planner (SIGINT) ..."
    tmux send-keys -t "${BRIDGE_SESSION}" C-c 2>/dev/null || true
    sleep 1
    tmux send-keys -t "${PLANNER_SESSION}" C-c 2>/dev/null || true
    for _ in $(seq 1 10); do
        alive=0
        proc_alive "x2_xrt_sonic_bridge" && alive=1
        proc_alive "pc2_kplanner_onnx.py" && alive=1
        [[ "${alive}" -eq 0 ]] && break
        sleep 1
    done
    pkill -INT -f x2_xrt_sonic_bridge 2>/dev/null || true
    pkill -INT -f pc2_kplanner_onnx.py 2>/dev/null || true
    sleep 1
    tmux kill-session -t "${BRIDGE_SESSION}" 2>/dev/null || true
    tmux kill-session -t "${PLANNER_SESSION}" 2>/dev/null || true
    log "stopped."
    ;;
  status)
    if proc_alive "pc2_kplanner_onnx.py"; then echo "  planner (tmux ${PLANNER_SESSION})   : RUNNING"; else echo "  planner (tmux ${PLANNER_SESSION})   : -"; fi
    if proc_alive "x2_xrt_sonic_bridge"; then echo "  bridge  (tmux ${BRIDGE_SESSION})  : RUNNING"; else echo "  bridge  (tmux ${BRIDGE_SESSION})  : -"; fi
    for s in x2_pose_watchdog x2_deploy x2_hand_bridge x2_motor_monitor; do
        if session_alive "$s"; then echo "  tmux $s            : RUNNING"; else echo "  tmux $s            : -"; fi
    done
    for p in 60061 5563 5564 5556 5558; do
        if port_listening "$p"; then echo "  port :$p          : LISTENING"; else echo "  port :$p          : -"; fi
    done
    ;;
  logs)
    what="${1:-planner}"
    case "${what}" in
      planner) tmux capture-pane -t "${PLANNER_SESSION}" -p -S -60 2>/dev/null | grep -vE '^\s*$' | tail -40 ;;
      bridge)  tmux capture-pane -t "${BRIDGE_SESSION}" -p -S -60 2>/dev/null | grep -vE '^\s*$' | tail -40 ;;
      *) err "logs planner|bridge"; exit 2 ;;
    esac
    ;;
  planner)
    preflight_planner
    log "planner in FOREGROUND (Ctrl-C exits; mode=${PLANNER_MODE})"
    cd "${STACK}"
    # shellcheck disable=SC2086
    exec env ${PLANNER_ENV} ${VENV_PY} ${STACK}/gear_sonic/scripts/pc2_kplanner_onnx.py \
      --backend torch --device cpu \
      --vqvae-ckpt "${MODELS}/vqvae/x2_kplanner_vqvae.ckpt" \
      --pose-ckpt  "${MODELS}/pose/x2_kplanner_pose.ckpt" \
      --root-ckpt  "${MODELS}/root/x2_kplanner_root.ckpt" \
      --planner-mode "${PLANNER_MODE}" \
      --warmup-qpos "${STACK}/models/kplanner_idle_anchor_g1teleop_v3.pkl" \
      --no-yaw-rebase \
      --dances-dir "${STACK}/dances_x2m2" \
      --cmd-bind --cmd-port 5563 \
      --arm-port 5564 \
      --pub-port 5556 --pub-host 0.0.0.0 "$@"
    ;;
  bridge)
    preflight_bridge
    ensure_xr_service
    log "bridge in FOREGROUND (Ctrl-C exits)"
    eval "${BRIDGE_ENV}"
    # shellcheck disable=SC2046
    exec $(bridge_cmd) "$@"
    ;;
  *)
    sed -n '2,40p' "$0" | grep -E "^# " | sed 's/^# //'
    ;;
esac
