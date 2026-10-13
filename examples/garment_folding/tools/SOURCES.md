# 来源与适配

来源：[happy1041/genesis-ipc-shirt-demo](https://github.com/happy1041/genesis-ipc-shirt-demo)，核对提交 `c88813fab6ebc3678a0976d338e9e00f8b199a10`（2026-10-10）。参考文件路径与哈希见 `source_manifest.json`。

## 原仓库如何组织

原仓库在 `tools/` 放命令行入口，`tools/trajectory_workbench/` 放网页和共享算法，`configs/workbench/` 放配置，`reproduction/` 放仿真入口、资产和历史阶段数据。独立脚本通过文件交换结果，按需调用。

| 原仓库功能 | 本地对应 |
|---|---|
| `action_plan.py`：动作段插值 | `core/action_plan.py`、`compile_plan.py` |
| `trajectory_compiler.py`：已有路径的节点修正 | `core/trajectory_edit.py`、`edit_trajectory.py` |
| `points_to_plan.py`：抓放点绑定 | `core/points.py`、`points_to_plan.py` |
| `action_plan_ik.py`、`ik_preflight.py`：不同输入的 IK，共享检查 | `core/kinematics.py`、`core/preflight.py`、`ik_preflight.py` |
| `server.py`、网页、CPU 回放与姿态预览 | `core/browser.py`、`core/preview.py`、`ui/`、`workbench.py` |
| 准备、导入及运行导出 | `prepare_workbench.py`、`export_trajectory.py` |

模块分开是为了让命令行和网页共用计算。编译计划和修正已有路径是两种输入方式；TCP 路径描述夹爪怎么走，IK 输出机器人关节怎么转，衣服运动由物理仿真产生。准备和导出负责实际状态及续跑的衔接。

## 参考数据与适配范围

- `plans/` 七份计划来自 `reproduction/scene527_55k_73s/source_records/`；`bindings/third_right.json` 参考原点位绑定。`samples/` 是本地示意输入。
- 8k/13k 指网格面数等级：分别 4,090 顶点 / 8,000 面与 7,021 顶点 / 13,767 面。观察包各含第一折源帧 770–829 的 60 帧布态、机器人关节、TCP 和材料标签，没有物理 checkpoint。保留原数据及模板，旁边的 `workbench.json` 使用本地相对路径。
- 本地按当前 Genesis API 重写，物理执行使用原生 `.pt` 和 `joint_q/joint_names` 轨迹；旧 `.pkl`/IPC 状态在原环境读取，观察配置可导入。Isaac 与外部 SIM1 工具按需求排除。
- 默认插值、节点修正、点位绑定及新增姿态辅助等已与上述来源对照。37 项工具检查通过，并验证历史导入、网页操作、实际机器人 CPU IK 与导出；未新增完整折叠物理实验。

来源 [NOTICE.md](https://github.com/happy1041/genesis-ipc-shirt-demo/blob/c88813fab6ebc3678a0976d338e9e00f8b199a10/NOTICE.md) 声明用于研究审阅，尚未选择开源许可证；参考计划与资产再发布遵循来源条款。

原说明：[动作计划、IK、导出及网页使用](https://github.com/happy1041/genesis-ipc-shirt-demo/blob/c88813fab6ebc3678a0976d338e9e00f8b199a10/docs/PLAN_TO_TRAJECTORY_CN.md) · [8k/13k 工作台样本](https://github.com/happy1041/genesis-ipc-shirt-demo/blob/c88813fab6ebc3678a0976d338e9e00f8b199a10/docs/MESH_WORKBENCH_EXAMPLES_CN.md)。
