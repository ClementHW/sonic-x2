import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.teleop.x2_hand_retarget import (
    HAND_GRASP_CLOSED_RAD_LEFT,
    HAND_GRASP_CLOSED_RAD_RIGHT,
    HAND_GRASP_OPEN_RAD_LEFT,
    HAND_GRASP_OPEN_RAD_RIGHT,
)
from gear_sonic_deploy.scripts.x2_hand_zmq_to_aimdk_bridge import (
    claw_position_from_hand_joints,
)
from gear_sonic_deploy.scripts.x2_motor_monitor import summarize_joint_state


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pc2_runtime_artifacts_exist() -> None:
    """Every Python daemon used by the PC2 lifecycle must ship in the repo."""
    for name in (
        "x2_motor_monitor.py",
        "x2_hand_zmq_to_aimdk_bridge.py",
    ):
        path = REPO_ROOT / "gear_sonic_deploy" / "scripts" / name
        assert path.is_file(), f"PC2 lifecycle references missing runtime: {path}"


@pytest.mark.parametrize(
    ("side", "open_q", "closed_q"),
    [
        ("left", HAND_GRASP_OPEN_RAD_LEFT, HAND_GRASP_CLOSED_RAD_LEFT),
        ("right", HAND_GRASP_OPEN_RAD_RIGHT, HAND_GRASP_CLOSED_RAD_RIGHT),
    ],
)
def test_claw_projection_preserves_trigger_contract(side, open_q, closed_q) -> None:
    open_q = np.asarray(open_q, dtype=np.float64)
    closed_q = np.asarray(closed_q, dtype=np.float64)
    assert claw_position_from_hand_joints(side, open_q) == pytest.approx(1.0)
    assert claw_position_from_hand_joints(side, closed_q) == pytest.approx(0.0)
    assert claw_position_from_hand_joints(
        side, 0.75 * open_q + 0.25 * closed_q
    ) == pytest.approx(0.75)


def test_pc2_lifecycle_owns_deploy_exit_cleanup() -> None:
    script = (REPO_ROOT / "x2_pc2" / "x2_stack.sh").read_text()
    assert "--enable-soft-shutdown" in script
    assert "if [[ -t 0 ]]; then" in script
    assert 'setsid "${PC2_DEPLOY_SH}" "${args[@]}" </dev/tty &' in script
    assert "forward_deploy_signal" in script
    assert 'stop_helper_session "${HAND_SESSION}"' in script
    assert 'PC2_PREFIX="${PC2_PREFIX}" "${PC2_TELEOP_SH}" stop' in script


def test_pc2_deploy_skips_mc_stop_when_preflight_detects_no_native_mc() -> None:
    """PC1 EM runtime state, not PC2 DDS discovery, controls MC stop."""
    deploy = (REPO_ROOT / "gear_sonic_deploy" / "deploy_x2.sh").read_text()

    assert 'MC_EM_APP_NAME="${MC_EM_APP_NAME:-mc}"' in deploy
    assert '"$MC_EM_URL/json/get_all_apps_info"' in deploy
    assert 'pid = (app or {}).get("pid")' in deploy
    assert 'active_statuses = {"running", "run"}' in deploy
    assert '"idel", "idle"' in deploy
    assert "NO_STOP_MC=true" in deploy
    assert "PC1 EM reports native MC inactive; skipping stop_app" in deploy
    assert "Native MC has no external arm-command publisher" not in deploy


def test_pc2_deploy_records_pc1_em_rejection_and_cleans_cancel_artifacts() -> None:
    deploy = (REPO_ROOT / "gear_sonic_deploy" / "deploy_x2.sh").read_text()

    assert "PC1 EM response:" in deploy
    assert "curl_rc=$http_rc, HTTP $http_status" in deploy
    assert "clear_handoff_runtime_artifacts()" in deploy
    assert "stop_deploy_tail()" in deploy
    assert '"${READY_SENTINEL:-}"' in deploy
    assert '"${DEPLOY_STDOUT_LOG:-}" == /tmp/x2_deploy_stdout.*.log' in deploy


def test_ctrl_c_in_deploy_worker_stops_complete_stack(tmp_path: Path) -> None:
    marker = tmp_path / "teleop-stopped"
    ready = tmp_path / "deploy-ready"
    env = os.environ.copy()
    env.update(
        PC2_PREFIX=str(REPO_ROOT),
        PC2_DEPLOY_SH=str(REPO_ROOT / "tests/fixtures/fake_pc2_deploy.sh"),
        PC2_TELEOP_SH=str(REPO_ROOT / "tests/fixtures/fake_pc2_teleop.sh"),
        X2_TEST_STOP_MARKER=str(marker),
        X2_TEST_READY_MARKER=str(ready),
        X2_TEST_HANDOFF_MARKER=str(tmp_path / "deploy-handoff"),
    )
    proc = subprocess.Popen(
        [str(REPO_ROOT / "x2_pc2/x2_stack.sh"), "_deploy"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "fake deploy did not reach the signal trap"
        os.killpg(proc.pid, signal.SIGINT)
        output, _ = proc.communicate(timeout=8)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2)
    assert proc.returncode == 130, output
    assert marker.read_text() == "stopped\n"
    assert (tmp_path / "deploy-handoff").read_text() == "complete\n"


def test_pc2_planner_disables_redundant_cold_start_replans() -> None:
    launcher = (
        REPO_ROOT / "gear_sonic" / "scripts" / "run_x2_pc2_teleop_stack.sh"
    ).read_text()
    assert "--cold-start-ramp-tau-s 0" in launcher
    assert "OMP_NUM_THREADS=${PLANNER_TORCH_THREADS}" in launcher
    assert "--torch-num-threads ${PLANNER_TORCH_THREADS}" in launcher
    assert "--torch-num-interop-threads ${PLANNER_TORCH_INTEROP_THREADS}" in launcher


def test_pc2_pico_walk_uses_slow_walk_planner_by_default() -> None:
    """Pico velocity commands must select the requested kplanner slow-walk mode."""
    launcher = (
        REPO_ROOT / "gear_sonic" / "scripts" / "run_x2_pc2_teleop_stack.sh"
    ).read_text()
    assert 'PLANNER_MODE="${PLANNER_MODE:-slow_walk}"' in launcher
    assert "--planner-mode ${PLANNER_MODE}" in launcher
    assert 'WALK_ZERO_CONFIRM_S="${WALK_ZERO_CONFIRM_S:-0.25}"' in launcher
    assert "--walk-zero-confirm-s ${WALK_ZERO_CONFIRM_S}" in launcher


def test_motor_monitor_summary_is_compact_and_json_safe() -> None:
    msg = SimpleNamespace(
        header=SimpleNamespace(sequence=42),
        state=SimpleNamespace(value=1),
        joints=[
            SimpleNamespace(
                name="left_knee_joint",
                velocity=-2.5,
                effort=4.0,
                coil_temp=53,
                motor_temp=49,
                motor_vol=48,
            ),
            SimpleNamespace(
                name="right_knee_joint",
                velocity=1.0,
                effort=-7.0,
                coil_temp=50,
                motor_temp=61,
                motor_vol=47,
            ),
        ],
    )

    summary = summarize_joint_state(msg)

    assert summary == {
        "sequence": 42,
        "domain_state": 1,
        "domain_state_name": "DAMPING",
        "joint_count": 2,
        "max_abs_velocity": 2.5,
        "max_abs_effort": 7.0,
        "max_coil_temp_c": 53,
        "max_motor_temp_c": 61,
        "hottest_joint": "right_knee_joint",
        "hottest_temp_c": 61,
        "min_motor_vol": 47,
        "max_motor_vol": 48,
    }
