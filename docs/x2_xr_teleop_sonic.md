# X2 Pico 遥操与 SONIC（PC2）

该运行时由 PC2 本地的 `x2` 生命周期管理器统一启动和停止。

## 数据流

```text
Pico
  -> XRoboToolkit :60061
  -> x2_xrt_sonic_bridge.py
       -> planner_cmd :5563
       -> arm_targets / hand_finger_cmd :5564
  -> pc2_kplanner_onnx.py
  -> pose :5556
  -> x2_pose_watchdog :5558
  -> SONIC deploy
```

规划器拥有 `:5563` 和 `:5564` 的 SUB bind，XR 桥通过回环地址连接。pose、手臂、
手部和行走命令都不会离开 PC2。

## 输入契约

| Pico 输入 | PC2 命令 |
|---|---|
| 左摇杆 Y | 前后速度，满偏默认 0.35 m/s |
| 左摇杆 X | 侧向速度，满偏默认 0.30 m/s |
| 左/右 grip | 对应手臂 IK 激活 |
| 左/右前扳机 | 对应夹爪，松开=1（张开），按下=0（闭合） |
| 右菜单键长按 | 软件急停 |

XR 桥先把扳机标量变换为规划器的十关节手部姿态。手部硬件桥读取 AimDK 上报的
`HandType`：X2 的 `CLAW` 手使用单关节 0..1 命令，`NIMBLE` 手使用十关节命令。
因此消息类型、关节数量和真实硬件保持一致。

行走使用直接速度意图 `[yaw=0, lateral, forward, hip_height]`。PC2 启动配置关闭
规划器内部的冷启动速度渐变；该渐变会把一次推杆拆成多次 CPU 重规划，导致几秒钟
内仍只输出站立缓冲。现在首次非零输入只触发一次目标速度重规划。

## 生命周期

```bash
./x2 preflight
./x2 start
./x2 status
./x2 stop
```

deploy 使用软停止触发器。`./x2 start` 附着窗口中的 `Ctrl+C` 会先完成
`RAMP_OUT -> HOLD_FOR_MC -> 原生 MC 接管`；deploy 返回后，同一 worker 自动关闭
手部桥、监控、watchdog、XR 桥与规划器，避免半停状态。

离线测试：

```bash
python -m pytest tests/test_pc2_runtime_artifacts.py \
  gear_sonic/tests/test_x2_xrt_sonic_bridge.py -q
```
