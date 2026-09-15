# X2 SONIC 真机操作指南（仅 PC2）

整套 SONIC + Pico 遥操栈只在机器人 PC2 上运行。登录 PC2 后，所有启动、
安全确认、状态检查、日志查看与停止操作都通过同一个入口完成：

```bash
ssh agi@192.168.31.150
cd /agibot/data/home/agi/gear-sonic
./x2 preflight
./x2 start
```

`start` 会依次启动 XRoboToolkit 服务、规划器、XR 桥、pose watchdog、手部桥、
电机监控和 SONIC deploy。出现 `Stop MC and launch policy? [y/N]` 后，确认机器人
稳固、急停在手边且运动范围无人，再输入 `y`。

## Pico 控制

- 左摇杆：前后行走和左右侧移；松杆停止。
- 左/右 grip：按住时控制对应手臂，松开后平滑回默认位。
- 左/右前扳机：对应夹爪；松开张开，按下闭合。
- 长按右手柄菜单键 0.5 秒：软件急停。

规划器在 PC2 CPU 上运行。启动器禁用了会反复触发推理的冷启动速度渐变，并限制
Torch 线程数，第一次推杆只触发一次完整速度重规划。

## 运行中

```bash
./x2 status
./x2 gate
./x2 logs deploy
./x2 logs planner
./x2 logs bridge
./x2 logs hand
./x2 attach deploy
```

`./x2 start` 附着的 deploy 窗口中按 `Ctrl+C`，以及执行 `./x2 stop`，都会走同一
条收尾路径：SONIC `RAMP_OUT -> HOLD_FOR_MC -> 原生 MC 接管`，随后停止全部辅助
会话。`Ctrl-B d` 只分离 tmux，不停止项目。

非附着模式：

```bash
./x2 start --no-attach
./x2 gate
./x2 yes
```

## 停止

```bash
./x2 stop
```

非交互调用使用 `./x2 stop --yes`。如果安全交接未在超时内完成，脚本会报错并保留
仍参与控制的进程，供现场排查。

## PC2 本地数据链

```text
Pico -> XRoboToolkit PC Service (:60061)
  -> x2_xrt_sonic_bridge.py
       |-- planner_cmd -> 127.0.0.1:5563
       `-- arm_targets/hand_finger_cmd -> 127.0.0.1:5564
  -> pc2_kplanner_onnx.py -> pose :5556
  -> x2_pose_watchdog -> pose :5558
  -> SONIC deploy -> X2
```

日志位于 `/agibot/data/home/agi/gear-sonic/log/`。
