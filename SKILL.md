# X2 SONIC PC2 runtime

Use this repository's real-robot runtime only from X2 PC2. The canonical
installation prefix is `/agibot/data/home/agi/gear-sonic`, and `x2` is the
only operator lifecycle entrypoint.

## Operator flow

```bash
ssh agi@192.168.31.150
cd /agibot/data/home/agi/gear-sonic
./x2 preflight
./x2 start
```

At the deploy safety gate, verify the robot is supported, the motion area is
clear, and the e-stop is reachable before entering `y`.

```bash
./x2 status
./x2 logs planner
./x2 logs bridge
./x2 logs hand
./x2 logs deploy
./x2 stop
```

Use `./x2 stop --yes` for non-interactive shutdown. In the deploy tmux pane,
`Ctrl+C` performs the same graceful shutdown: SONIC ramps out, waits for the
native motion controller, then all helper sessions stop. `Ctrl+B d` only
detaches from tmux.

## Pico controls

- Left stick: forward/backward and lateral walking.
- Left/right grip: enable the corresponding arm target.
- Left/right front trigger: matching gripper; released is open, pressed is
  closed.
- Hold the right menu button for 0.5 seconds: software e-stop.

## PC2 data path

```text
Pico -> XRoboToolkit :60061
  -> x2_xrt_sonic_bridge.py
       -> planner_cmd :5563
       -> arm_targets / hand_finger_cmd :5564
  -> pc2_kplanner_onnx.py -> pose :5556
  -> x2_pose_watchdog.py -> pose :5558
  -> SONIC deploy -> X2
```

The hand bridge must preserve the `HandType` reported by AimDK. X2 `CLAW`
hardware accepts one scalar per side (`1=open`, `0=closed`); `NIMBLE`
hardware accepts ten named joints.

## Safety

- Do not kill the deploy binary directly. Use `./x2 stop` or `Ctrl+C` in the
  managed deploy pane so `RAMP_OUT -> HOLD_FOR_MC` completes.
- Do not start deploy unless the local pose stream passes the lifecycle
  manager's gate.
- After shutdown, verify the native motion controller has resumed before
  approaching or re-engaging the robot.
- Keep the e-stop reachable and monitor motor temperatures during extended
  runs.

## Offline verification

```bash
python -m pytest tests/test_pc2_runtime_artifacts.py \
  gear_sonic/tests/test_x2_xrt_sonic_bridge.py -q
```
