"""Pure-Python regression tests for the PC2 Pico-to-SONIC bridge."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = REPO_ROOT / "gear_sonic" / "scripts" / "x2_xrt_sonic_bridge.py"


def _load_bridge_module():
    """Load the bridge without importing ROS2 or XRoboToolkit."""
    name = "x2_xrt_sonic_bridge_under_test"
    spec = importlib.util.spec_from_file_location(name, BRIDGE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_walk_zero_confirm_holds_a_single_xr_zero_glitch() -> None:
    bridge = _load_bridge_module()
    hold = bridge.WalkZeroConfirm(zero_confirm_s=0.25)

    assert hold.update(vel_x=0.0, vel_z=0.22, now_s=10.0) == (
        0.0,
        0.22,
        False,
    )
    # XRoboToolkit can briefly report [0, 0] while its Pico device stream
    # reconnects. A single zero sample must not turn a held stick into stop.
    assert hold.update(vel_x=0.0, vel_z=0.0, now_s=10.02) == (
        0.0,
        0.22,
        True,
    )
    # A real centred stick still stops promptly after the bounded window.
    assert hold.update(vel_x=0.0, vel_z=0.0, now_s=10.26) == (
        0.0,
        0.0,
        False,
    )


def test_walk_zero_confirm_never_delays_deadman_or_error_stop() -> None:
    bridge = _load_bridge_module()
    hold = bridge.WalkZeroConfirm(zero_confirm_s=0.25)
    hold.update(vel_x=-0.22, vel_z=0.22, now_s=20.0)

    hold.reset()
    assert hold.update(vel_x=0.0, vel_z=0.0, now_s=20.01) == (
        0.0,
        0.0,
        False,
    )


@pytest.mark.parametrize("duration", [-0.1, float("nan")])
def test_walk_zero_confirm_rejects_invalid_duration(duration: float) -> None:
    bridge = _load_bridge_module()
    with pytest.raises(ValueError):
        bridge.WalkZeroConfirm(zero_confirm_s=duration)
