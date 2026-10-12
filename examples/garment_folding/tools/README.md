# 叠衣服工具

在 `genesis-world` 根目录使用现有 Python 环境，输出请使用新路径。各工具按需调用。

| 工具 | 用途 | 输入 → 输出 |
|---|---|---|
| [compile_plan](#compile_plan动作编译) | 将动作段编译成平滑路径 | 计划 JSON → TCP 路径、检查报告 |
| [edit_trajectory](#edit_trajectory路径修正) | 局部或持续修正路径 | 路径目录、修改 JSON → 新路径、报告 |
| [points_to_plan](#points_to_plan点位转换) | 把抓放点写入动作 | 计划、选点、绑定 JSON → 新计划 |
| [workbench](#workbench工作台) | 看布态、选点、拖节点、预览夹爪 | 起点目录 → 网页、保存的选点或路径 |
| [ik_preflight](#ik_preflight机器人检查) | 求关节运动并检查限位、碰撞 | 实际起点、TCP 路径 → 关节轨迹、IK 报告 |
| [export_trajectory](#export_trajectory轨迹导出) | 导出给叠衣服入口执行 | 起点、通过 IK 的路径 → 轨迹、运行脚本 |

## compile_plan：动作编译

输入动作计划 JSON，输出路径目录。示意例子：右手一秒抬高 50 mm，生成 `tcp-path.npz` 和 `path-report.json`。

```powershell
python -m examples.garment_folding.tools.compile_plan --plan examples/garment_folding/tools/samples/lift.json --output-dir out/lift-path
```

### 计划字段

| 字段 | 含义 |
|---|---|
| `start.left/right` | 实际起始 TCP；机器人执行时应从准备的模板保留。 |
| `actions[].id` | 唯一动作名，例如接近、闭爪、抬升、放手。 |
| `duration_frames` | 60 Hz 时间区间数，60 表示一秒。 |
| `left/right.pos` | 世界 XYZ，单位米。保持某只手时填上一段终点。 |
| `quat` | WXYZ 四元数，省略则继承上一段姿态。 |
| `opening` | 每根夹指开度，米，范围 0–0.044，省略则继承。 |
| `arc_height_m` | 向上弧高；共同数值或左右手分别指定。 |
| `boundaries` | 需要标记的动作终点名称，可为 `[]`；与动作 id 对应。 |
| `position_end_velocity_mps` | 左右手 XYZ 端点速度。 |
| `rotation_profile` | 世界 / 工具 X 轴旋转，终点与姿态一致。 |
| `gripper_events` | 按绝对源帧指定开闭时序，同手事件不重叠。 |

机器人 TCP 是夹爪工具参考点，不等于布面抓点。抓取顺序通常为接近、闭爪、抬升、移动、释放、撤离；高度、姿态、开度和时长由使用者设计。

`gripper_events` 可为 JSON 列表或 `t=120 close both duration=18` 文本。非空事件接管双手开度；没有事件的手保持初始开度。空列表保留动作端点的开度插值。

在 `compile_plan` 命令中将 `--plan` 换为 `--draft-from 原计划.json --pull-axis x --pull-sign -1`，生成已抓取后的抬升、拉动、保持草稿，保留起始姿态和开度。

默认速度上限 0.75 m/s、加速度上限 20 m/s²。拒绝时查看 `path-report.json`，调整位移或时长。可加 `--workspace PATH --ik` 同时检查机器人。

## edit_trajectory：路径修正

输入已编译路径目录和修改 JSON，输出新路径。示意例子：在第 30 源帧抬高 5 mm。新路径在 `tcp-path.npz`，保留的 `plan.json` 是基础计划。

```powershell
python -m examples.garment_folding.tools.edit_trajectory --compiled out/lift-path --edits examples/garment_folding/tools/samples/edits.json --output-dir out/lift-corrected
```

- 节点修改的 `frame` 是绝对源帧，`delta_m` 是 XYZ 米制位移，`support_radius` 是前后影响帧数。不能修改起始 TCP。

### 修改参数

将 `--edits` 换成 `samples/edits_advanced.json` 可试持续偏移和语义锚点。

| 修改 JSON 字段 | 用法 |
|---|---|
| 节点 `mode: local` | 局部修改，影响范围结束后回到基础路径；默认模式。 |
| 节点 `mode: hold_after` | 在节点前 `support_radius` 帧平滑增加偏移，节点后保持偏移。 |
| `semantic_keyframes` | 每项填 `hand/frame/delta_m`；锚点之间平滑插值，首尾外保持最近锚点偏移。 |
| `stage_range: [开始, 结束]` | 只在这段源帧应用修正；节点须在范围内，语义锚点可在范围外定义过渡。 |
| `limits` | 可设 `max_step_mm/max_speed_mps/max_accel_mps2`。 |

三类修正可以组合；起始 TCP 须保持不变。报告的 `required_duration_scale` 表示建议放慢的倍数，需自己修改时长并重新编译。

默认检查单帧位移 6.5 mm、速度 0.4 m/s、内部二阶差分加速度 15 m/s²。拒绝时查看 `path-report.json`；可加 `--workspace PATH --ik` 一起检查机器人。

## points_to_plan：点位转换

输入计划、选点与绑定 JSON，输出新计划和来源记录。示意例子：将示意抓点写入绑定指定的动作。实际使用时替换选点和绑定文件。

```powershell
python -m examples.garment_folding.tools.points_to_plan --plan examples/garment_folding/tools/samples/lift.json --points examples/garment_folding/tools/samples/points.json --bindings examples/garment_folding/tools/samples/bindings.json --output out/point-plan.json
```

- 选点须保存于计划起始帧。`anchor_action` 对齐抓放点，`translate_actions` 指定一起平移的动作段。
- `tcp_offset_world_mm` 是布面到 TCP 的世界坐标毫米偏移，需实际标定；`kind: placement` 使用世界落点。
- 换网格须重新选点；参考七阶段计划必须匹配起始状态、帧号及抓放位置。

## workbench：工作台

输入 `prepare_workbench` 准备的起点目录，查看布态并保存选点或路径。

```powershell
python -m examples.garment_folding.tools.workbench --workspace out/cloth-view
```

打开终端显示的网址；`Ctrl+C` 退出。

| 操作 | 网页使用方式 |
|---|---|
| 抓点 | `Hand` → `Grasp point` → 点击布料 → 选候选层 → `Confirm layer` → `Save selections` |
| 落点 | `Placement point` → 点击或填毫米 XYZ → `Set placement` → 保存 |
| 改路径 | 编译计划或加载参考路径 → `Drag path node` → 拖动或填偏移 → `Add correction` → `Save corrected path` |
| 姿态 | `Gripper pose` → `Load measured pose` → 调局部旋转 → 选动作 → `Apply pose` → 重新编译 |
| IK | `Inverse kinematics` → 设置 → `Check current path` |
| 材料与遮挡 | `Color` 选材料标签，`View` 选透视，按需勾选 `Show robot` |

只保存选点不会改动作；用 `points_to_plan` 写入计划。节点在视图平面拖动，透视拖动保持世界 Z。

- `--replay PATH` 加载当前样例的匹配回放；`Frame` 切换帧，`Plan start` 回到动作起点。回放由原入口 `--dump-replay` 保存。
- 网页 `Save + compile TCP path` 将结果保存到新的 `actions/时间/`，不覆盖工作目录计划。对这个目录调用独立 IK 即可。

保持服务终端运行；默认自动选空闲端口，也可加 `--port` 指定。

## ik_preflight：机器人检查

输入实际起点和已编译路径，计划起点须对应实际状态。路径换成自己的：

```powershell
python -m examples.garment_folding.tools.ik_preflight --workspace out/cloth-view --compiled out/my-path
```

输出 `ik-report.json`；通过生成 `robot_q.npz`，拒绝生成 `rejected-robot_q.npz`。失败查看 `failed_checks`，起点已越限不能靠增加时长解决。

### 检查设置

在 `ik_preflight` 命令后加 `--options examples/garment_folding/tools/samples/ik_options.json`，并按需要修改该文件。

| 设置 | 用法 |
|---|---|
| `hand` | `both/left/right`。编辑导入回放时，单手模式保留另一手的实测轨迹；从末帧设计新段或没有回放基线时，另一手保持 seed。 |
| `orientation_mode` | `full` 保持完整姿态；`length_x/closing_y/tool_z` 分别约束工具 X/Y/Z 方向，允许其他方向旋转。 |
| `orientation_assist.is_enabled` | 开启后，按位置修正总长度绕工具局部 Y 轴辅助旋转；需通过路径修正生成的来源记录。 |
| `gain_deg_per_mm/max_abs_angle_deg` | 默认 -0.181379499 °/mm、最大 15°。辅助开启时使用完整姿态约束。 |
| `table_z_m` | 桌面世界高度，米。 |
| `min_tool_table_clearance_m/min_clearance_delta_m` | 绝对间距下限、相对初始间距的变化下限，米；默认均为 -0.0035。 |

放松姿态有助于可达性，也会改变夹爪朝向。报告保存实际设置；抓取效果需运行物理验证。

默认检查位置误差 5 mm、姿态误差 2°、关节单帧变化 0.05 rad、限位、自碰撞和离桌间距。当前双 X5 的 TCP 局部偏移为 `(0.155, 0.001786, 0.014)` 米，参考 link 为 `link16/link26`。物理状态可能轻微越过夹指限位，这类 seed 仍会被拒绝。

## export_trajectory：轨迹导出

输入 IK 使用的起点目录和通过检查的路径目录，输出 `trajectory.npz`、运行清单及 `run.ps1`。

```powershell
python -m examples.garment_folding.tools.export_trajectory --workspace out/cloth-view --compiled out/my-path --output-dir out/my-run
```

执行 `& ./out/my-run/run.ps1` 才启动仿真，需要 CUDA / pyuipc，结果在 `out/my-run/physics/`。路径和 IK 接受不代表抓取、折叠成功。续跑导出核对 checkpoint、seed、原指令前缀和输入哈希；新结果重新准备时指定新轨迹。

### 帧号与恢复

一个动作区间对应两个物理步。第 120 动作步状态执行了源帧 0–59，下一段锚点为 59；checkpoint 应在偶数动作步。

初态轨迹保留起始行，控制器先保持两步，再执行计划；初始下落另计。恢复会重置场景时钟，累计动作进度由控制器恢复，跨阶段使用源帧、动作步和 `trajectory_time_s` 对照。

## prepare_workbench：状态准备（辅助）

辅助工具 `prepare_workbench` 准备衣服快照、机器人 seed 和保持动作模板 `plan.json`。初始快照尚未经过衣服下落；已有状态可加 `--checkpoint 实际路径.pt`，自定义轨迹状态还需 `--trajectory`，换网格加 `--mesh`。这些输入须来自同一次仿真。

```powershell
python -m examples.garment_folding.tools.prepare_workbench --output-dir out/cloth-view
```

要查看落桌布态，可使用原入口 `--settle-only` 生成的 `settled-scene-state.pt`。准备失败时核对 checkpoint、网格、原轨迹与偶数动作步。

`history/8k` 与 `history/13k` 各含源帧 770–829 的 60 帧第一折观察数据、机器人实测关节/IPC 变换、TCP 和材料 atlas。8k/13k 指网格面数等级。

**打开历史样本：**在仓库根目录执行，第二条命令后打开终端显示的网址。换成 `13k` 即可查看另一套网格。

```powershell
python -m examples.garment_folding.tools.prepare_workbench --import-config examples/garment_folding/tools/history/8k/workbench.json --start-frame 770 --output-dir out/history-8k
python -m examples.garment_folding.tools.workbench --workspace out/history-8k
```

点 `Load imported reference path` 可编辑历史 TCP。起点设为 770 便于检查整段；省略 `--start-frame` 则使用原配置的默认帧 829。

**导入其他上游配置：**使用同一条 `--import-config` 命令；配置内相对路径可用 `--source-root 原仓库目录` 指定根目录。上游运行包用 `--run-manifest` 读取其配置。

**接着设计原生运行结果：**`prepare_workbench --run-manifest out/my-run/run-manifest.json --checkpoint out/my-run/physics/garment-folding-checkpoint-000004.pt --output-dir out/next-start`。这里的步号替换成实际保存的 checkpoint。

历史观察包支持查看、选点、路径编辑和运动学检查。物理续跑需要匹配的当前 Genesis `.pt`、网格和命令轨迹；旧 `.pkl`/IPC 存档保留在原环境读取。

## build_atlas：材料标签（辅助）

工作台自动显示衣宽、衣长和材料表面标签。需要单独文件时用辅助入口：

```powershell
python -m examples.garment_folding.tools.build_atlas --mesh examples/garment_folding/assets/cloth/short-shirt-8000f.obj --output-dir out/shirt-atlas
```

输出 `atlas.npz` 标签数组与 `atlas.json` 下摆、领口、袖端的两表面材料点。

衣服穿着者左右侧需人工标定：可加 `--positive-x-side wearer_left` 或 `wearer_right`；默认未知。标签基于衣服原始网格坐标，折叠后仍表示同一材料部位，当前上下层通过布态视图判断。

## 文件与可选数据

- `core/` 是入口共享的算法与读写实现；`ui/` 是工作台界面代码。
- `samples/` 是小型格式示例；`plans/` 和 `bindings/` 是上游七阶段动作及点位绑定参考，使用前须匹配起点。
- `history/8k`、`13k` 是可选历史观察包，普通工具调用不依赖它们，部分测试使用；没有可续跑的物理 checkpoint。
- 原仓库组织、核对版本与适配范围见 [来源](SOURCES.md)。

所有工具使用新输出路径；输出已存在时换路径，保留旧结果。示意坐标用于学习格式，实际执行须使用匹配的机器人起点。
