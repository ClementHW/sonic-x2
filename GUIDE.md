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

## Git LFS 与仓库同步

仓库中的大二进制不直接写入普通 Git 对象，而是由 Git LFS 存储。当前受 LFS
管理的内容包括模型权重与 checkpoint、ONNX 策略、USD 机器人资产、共享库（SO）、
视频（MP4）和 Linux 安装包（DEB）。首次完整拉取前，先安装并启用 Git LFS：

```bash
git lfs install
git clone https://github.com/ClementHW/sonic-x2.git
cd sonic-x2
git lfs pull
```

仓库默认不自动拉取 MotionBricks 的大 checkpoint。如需运行 MotionBricks，再显式
下载这些权重：

```bash
git lfs pull --include="motionbricks/out/**" --exclude=""
```

可用 `git lfs ls-files` 检查本地已登记的 LFS 文件。新增大型二进制文件时，先使用
`git lfs track '<pattern>'` 更新 `.gitattributes`，再执行 `git add`；不要把权重、
模型、视频或安装包直接作为普通 Git blob 提交。

## main 分支保护

`main` 是受保护分支。所有变更必须通过 Pull Request，并由代码所有者
`@ClementHW` 审核批准后才能合并。规则同样约束管理员：不允许直接推送、强制推送、
删除分支或绕过审核。
