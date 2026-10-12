# Scene527 叠衣服示例

这个示例让双 X5 机器人回放一段已有的关节轨迹，衣服通过 FEM/IPC 仿真与桌面、夹爪及自身接触。随包提供衣服网格、机器人模型和完整轨迹，默认使用 55k 网格。

运行需要 NVIDIA CUDA、CUDA 版 PyTorch 和 `pyuipc`；本机验证的 `pyuipc` 版本为 0.0.28。当前仓库按项目说明安装，以下命令均从仓库根目录执行。

## 快速运行

```powershell
python -m examples.garment_folding.run --steps 120 --record
```

这条命令先让衣服下落 60 个物理步，再执行 120 个动作物理步，并将视频、逐步数据和可恢复状态写入 `out/garment_folding/`。每步为 1/120 秒，所以覆盖 0.5 秒初始下落和 1 秒动作；动作视频只录制后面的 1 秒。

要跑完整轨迹，去掉 `--steps 120`；要显示实时窗口，追加 `--vis`；要用粗网格，追加 `--mesh 13k` 或 `--mesh 8k`。完整运行共有 8,746 个动作物理步，加初始下落约 73.38 秒仿真时间，本机 55k 全程曾耗时约五小时。

## 参数怎么用

参数由 [run.py 的 main()](run.py) 解析。

| 参数 | 默认值 | 用途和限制 |
| --- | --- | --- |
| `--steps N` | 执行剩余完整轨迹 | 本次执行多少个**动作物理步**。120 步是 1 秒；不计前面的 60 步初始下落。恢复状态后，N 仍表示本次追加的步数。 |
| `--vis` | 关闭 | 打开实时窗口。 |
| `--record` | 关闭 | 将动作过程录成 MP4；每个保存区间一个视频文件。 |
| `--output-dir PATH` | `out/garment_folding` | 保存视频、CSV 和状态文件。每次新实验应选新目录：动作 CSV 会追加，同名视频和状态文件会被覆盖。 |
| `--mesh 8k/13k/55k` | `55k` | 选择衣服网格，同时选择对应的顶点数和面数校验。 |
| `--settle-only` | 关闭 | 只观察衣服下落：机器人保持轨迹第一帧姿态，仿真结束后退出。自动保存下落视频、CSV、摘要和状态文件。 |
| `--settle-steps N` | 60 | 只与 `--settle-only` 一起使用，指定观察下落的物理步数，例如 120 步为 1 秒。步数必须大于零。 |
| `--checkpoint PATH` | 从初态开始 | 从本示例的 `.pt` 状态文件继续。加载后跳过初始下落；动作状态还会恢复轨迹进度。恢复时使用相同网格和场景配置。 |
| `--segment-steps N` | 120 | 每执行 N 个动作物理步保存一次状态；开启录像时也在这里结束一个视频片段。最后不足 N 步的区间照常保存。必须大于零。 |
| `--asset-root PATH` | 本示例目录 | 从另一个目录读取配套网格、机器人和轨迹，目录结构见下文。 |
| `--trajectory PATH` | 配套 4,373 帧轨迹 | 读取自定义长度的 `joint_q/joint_names` NPZ，支持动作工具导出的轨迹；续跑身份按导出信息校验。 |
| `--dump-replay` | 关闭 | 每个输出区间保存衣服位置/速度、机器人关节和帧号，按两个物理步采样。便于 CPU 逐帧检查，增加内存与磁盘开销。 |
| `--disable-robot-ipc-proxies` | 关闭 | 将机器人 link 从 IPC 接触中排除。用于检查机器人接触的影响；开启后衣服无法通过这些碰撞代理与夹爪正常交互。 |
| `--progress-every N` | 10 | 每执行 N 个动作物理步打印进度；0 关闭周期进度日志。下落观察模式会逐步打印自己的日志。 |

`--settle-only` 与 `--steps`、`--checkpoint` 不能同时使用。普通折叠运行的初始下落长度由 `config.py` 中的 `GarmentFoldingTaskConfig.settle_steps` 控制，当前是固定 60 步。它不会自动等待衣服完全静止。

**参数名是 `--asset-root`，入口没有 `--asset-robot`。** 在 [scene.py 的 build_scene()](scene.py) 中，`asset_root` 分别与 `assets.garment_mesh`、`assets.robot`、`assets.trajectory` 拼接。因此它替换的是整套文件的根目录。以默认 55k 为例，目录内需要：

```text
PATH/
  assets/cloth/short-shirt-55068f.obj
  assets/robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf
  assets/robots/dual_x5_2025_ipc_v1/meshes/...
  controls/trajectory.npz
```

选择 8k 或 13k 时，衣服文件名按下表替换。机器人和轨迹仍需符合本例的关节名称、接触代理及轨迹格式。若通过 Python 配置更换机器人，修改的是 `GarmentFoldingAssets.robot`，并同步核对这些条件。

## 衣服网格

8k、13k、55k 表示三角面数量等级。切换时只需在运行命令中加 `--mesh` 参数。

| 选择 | 顶点数 | 三角面数 | 文件 |
| --- | ---: | ---: | --- |
| `--mesh 8k` | 4,090 | 8,000 | [short-shirt-8000f.obj](assets/cloth/short-shirt-8000f.obj) |
| `--mesh 13k` | 7,021 | 13,767 | [short-shirt-13767f.obj](assets/cloth/short-shirt-13767f.obj) |
| `--mesh 55k` | 27,811 | 55,068 | [short-shirt-55068f.obj](assets/cloth/short-shirt-55068f.obj) |

选择逻辑在 [config.py 的 create_scene527_config()](config.py)：它修改 `garment_mesh`、`expected_garment_vertices` 和 `expected_garment_faces`。材料、轨迹和时间步继续使用同一组设置。粗网格可以较快排查运行流程，但接触和褶皱会随网格改变，抓取效果需要重新检查。

**单位的依据：** OBJ 本身只保存数值，没有长度单位字段。上游提供的 [mesh_variants.json](assets/cloth/mesh_variants.json) 明确记录 `units: millimeter` 和 `runner_scale: 0.001`。本例在 `GarmentFoldingAssets.garment_scale` 中设置 `0.001`，`scene.py` 将它传给 `gs.morphs.Mesh(scale=...)`，把原始坐标转换成场景使用的米。例如 OBJ 中的坐标 443.138 经缩放成为约 0.443138 米，随后再施加旋转和平移。

三个 OBJ 的文件哈希均与上游一致，记录见 [source_manifest.json](assets/source_manifest.json)。8k/13k 已检查坐标、面索引和数量，并通过配置测试；完整 GPU 折叠效果尚未验证。恢复状态时必须使用原来那种网格：不同网格的顶点数、顺序和连接关系不同，已有 55k 状态无法用于 8k/13k。

## 配置与代码对应

| 文件 | 负责什么 | 常用修改位置 |
| --- | --- | --- |
| [config.py](config.py) | 场景参数和网格选择 | `GarmentFoldingAssets`：网格、URDF、摆放和相机；`GarmentFoldingMaterialConfig`：材料和摩擦；`GarmentFoldingSolverConfig`：重力、步长和求解设置；`GarmentFoldingTaskConfig`：下落步数和动作频率。 |
| [scene.py](scene.py) | 读取文件、创建衣服/桌面/机器人和相机 | `build_scene()` 将配置传给 Genesis，并检查网格数量、轨迹形状和关节名称。 |
| [task.py](task.py) | 控制机器人运动 | `step_settling()` 保持第一帧姿态；`GarmentFoldingController` 插值关节目标并推进轨迹。 |
| [run.py](run.py) | 运行、录像、记录数据和保存/恢复状态 | `main()` 执行动作；`run_settle_only()` 观察初始下落。 |

轨迹文件为 `controls/trajectory.npz`，包含 4,373 帧机器人关节目标。`task.py` 每个目标插值为两个物理步：动作频率 60 Hz，物理频率 120 Hz。这是已有动作回放；更换机器人或改轨迹速度时，需要同时检查关节名称、初始姿态、抓取位置和时间设置。

**坐标方向：** Genesis 的默认重力是 `(0, 0, -9.81)`，默认相机上方向是 `(0, 0, 1)`，通常按 Z 向上使用；这些设置可以修改，所以不能把它理解成所有自定义场景都强制使用同一方向。本例在 `GarmentFoldingSolverConfig.gravity` 明确设置负 Z 重力，并将桌面按 Z 高度摆放，因此这里使用 Z 向上。世界坐标与网格原始坐标是两回事：衣服还会按 `garment_euler=(-90, 0, 0)` 旋转。场景位置、桌面尺寸和厚度用米。

## 输出文件怎么看

“沉降”在本例中指**衣服在重力下落到桌面的观察过程**。普通运行先执行这段初始化，再开始机器人动作；`--settle-only` 则只观察这段过程。下面分别说明两种输出。

### 机器人动作输出

| 文件 | 内容与用途 |
| --- | --- |
| `garment-folding-segment-000000.mp4` 等 | 开启 `--record` 后生成的动作视频，30 帧/秒。文件名数字是片段开始时的动作物理步，`000000` 表示从动作起点开始。默认每 120 步一段，约 1 秒视频。 |
| `garment-folding-metrics.csv` | 每个动作物理步一行，记录轨迹进度、仿真/实际耗时、衣服离桌高度和速度，用于查找异常出现的时刻。 |
| `garment-folding-checkpoint-000120.pt` 等 | 在动作第 120 步等区间末尾保存的可恢复状态，包含场景状态、原生 IPC 状态和动作进度。传给 `--checkpoint` 可以接着执行；它不是视频或单纯的网格文件。即使关闭录像，也会保存状态。 |

入口按区间输出视频。已经完成的全程实验另经后处理拼接成 `out/scene527-full-run/scene527-full.mp4`；普通运行入口不会自动生成这个拼接文件。

### 只观察下落的输出

以下文件仅由 `--settle-only` 生成，即使没有 `--record` 也会录像。

| 文件 | 内容与用途 |
| --- | --- |
| `garment-folding-settle.mp4` | 衣服从初始位置下落的视频，机器人保持第一帧姿态。用于观察翘角、袖口褶皱和落桌过程。 |
| `settle-metrics.csv` | 每个下落物理步一行。除高度、速度和耗时外，还记录衣服中心位置及包围范围，便于判断是否还在移动。 |
| `settle-summary.json` | 本次配置、观察步数、总耗时、首尾指标及上述文件名，便于回查实验条件。 |
| `settled-scene-state.pt` | 观察结束时的场景状态。恢复它后，机器人动作从轨迹开头开始；文件名中的 `settled` 只表示下落观察结束，程序没有自动检查衣服是否已完全静止。 |

例如想看更长的下落过程，使用 `--settle-only --settle-steps 120`，并选择新的输出目录。

### CSV 中的高度和速度

`run.py` 将桌面高度算为 `table_pos[2] + table_size[2] / 2`，默认是 0.8 米；每个顶点离桌高度为 `(顶点 Z - 桌面 Z) * 1000`，因此高度列的单位是毫米。速度列来自顶点速度向量的长度，单位是米/秒。

| 列 | 读法 |
| --- | --- |
| `step` | 动作 CSV 中是累计动作物理步；下落 CSV 中是本次观察的下落步。 |
| `trajectory_time_s` | 动作已经执行的仿真秒数，不含初始下落。 |
| `scene_cur_t_s` / `simulated_time_s` | 当前场景时钟；从初态运行时包含初始下落。`scene.restore()` 后时钟从 0 重新累计，累计动作时间由 `trajectory_time_s` 表示。 |
| `step_wall_seconds` / `elapsed_wall_seconds` | 计算单步/本段过程实际用了多少秒，区别于仿真时间。 |
| `height_median_mm` | 一半顶点低于这个离桌高度，用来看衣服整体是否贴近桌面。 |
| `height_p90_mm` | 90% 顶点低于这个高度，用来看较大区域是否仍被抬起。 |
| `height_max_mm` | 最高顶点的离桌高度，适合配合图像查翘角；单凭该数值无法定位顶点。 |
| `speed_rms_m_per_s` | 全部顶点速度的均方根，用来看整体是否仍明显运动。 |
| `speed_max_m_per_s` | 运动最快顶点的速度，用来看局部是否仍在抖动。 |

速度变小表示运动减弱，仍需检查衣角和袖口的形状，不能据此认定衣服已经铺平。

## 动作设计与布态查看

独立工具目录 [tools/](tools/README.md) 按需提供动作编译、局部路径修正、抓放点转换、CPU IK、浏览器布态查看与轨迹导出。动作编译和路径修正可直接处理文件；只有查看实际状态、做机器人检查和续跑时才需要准备当前样例 `.pt`。各工具的输入、输出和操作说明见目录中的功能索引；[原仓库组织与来源](tools/SOURCES.md) 解释模块关系。

主入口新增 `--trajectory PATH`，读取自定义长度的 `joint_q/joint_names` NPZ；默认轨迹仍为 4,373 帧。`--dump-replay` 按两个物理步采样衣服位置/速度和机器人关节，每个保存区间生成 `garment-folding-replay-起始步.npz`，可用于工具中的逐帧查看。启用回放采集会增加内存和磁盘开销。

原仓库 2026-10-10 的后续更新已核对到 `c88813f`，新增动作编辑链、点位绑定和低面数观察样本。本地适配及验证范围见 [tools/SOURCES.md](tools/SOURCES.md)。

## 与原仓库的比较

参考仓库为 [genesis-ipc-shirt-demo 的 Scene527 交接分支](https://github.com/happy1041/genesis-ipc-shirt-demo/tree/handoff/scene527-ipc-demo-20261005)。原结果取自 `05b1d31`；2026-10-10 核对到 `c88813f`。`f7bbdaf` 的 8k/13k 网格已加入，后续动作工具在独立 `tools/` 目录按当前 Genesis API 适配。

### 保留了哪些内容

55k 衣服网格和完整关节轨迹的文件哈希与原仓库一致，机器人网格也保留原文件；URDF 中的文件路径改为本地相对路径。布料材料、摩擦、初始摆放、60 步下落和动作时间设置按源 Scene527 长例子保留。机器人依靠真实接触和摩擦带动衣服。

### 目前有哪些差异

| 项目 | 对使用和结果的影响 |
| --- | --- |
| 运行代码 | 原仓库将场景、轨迹、诊断和录制放在大型运行器中；这里分到上面的四个 Python 文件，通过 Genesis 接口构建场景。 |
| 引擎与环境 | 原结果使用旧 Genesis 加补丁、Linux / Python 3.12 / pyuipc 0.0.25；本机使用当前 Genesis、Windows / Python 3.11 / pyuipc 0.0.28。相同输入仍可能产生不同布料状态。 |
| 布料系数 | 新旧 pyuipc 对拉伸和弯曲系数的定义不同。当前引擎按厚度和网格三角形面积换算，目的是保持原来的能量尺度；这还不能证明两个版本求解出的每一步完全相同。 |
| 摩擦与碰撞 | 保留布料自摩擦和机器人内部 IPC 接触规则。原仓库有额外的指定接触对摩擦接口；本例布与桌的材料摩擦均为 1，正常组合即可得到源值 1。当前还关闭 Genesis 刚体碰撞，衣服、桌面和机器人接触由 IPC 处理，这与源运行路径有差异。 |
| 全程运行方式 | 原参考结果通过七阶段保存状态续跑；当前已验证的全程从初态连续运行。期间保存文件不会中断或重新加载仿真。原工具也支持连续运行。 |
| 恢复状态格式 | 原仓库保存求解器字段、IPC 位置/速度及阶段元数据；当前保存 `SimState`、原生 IPC 求解历史和动作进度。原仓库的 checkpoint 不能直接传给当前 `--checkpoint`。 |
| 显示和诊断 | 当前提供单相机视频、整体高度/速度、可选布态回放和三方向浏览器投影选点。实际 IPC 碰撞代理显示、语义区域分析和原始接触导出仍在来源工具中。 |
| 阶段工具 | 独立 tools 目录提供七份参考计划、本地状态准备、TCP 插值、CPU IK、指令前缀校验和新轨迹导出。完整七阶段新动作效果需要逐段物理验收。 |

原 checkpoint 压缩包包含 `checkpoint.pkl`（Genesis 求解器字段）、`checkpoint.ipc_state.npz`（布料位置/速度、机器人变换矩阵/速度及诊断初始位置）和 `checkpoint.meta.json`（阶段、时钟、前一关节目标、配置与动作签名）。它们依赖原字段格式，IPC 伴随文件也没有完整原生求解历史。已核验 `02_first_pull` 包的三个成员哈希；其余阶段需要按需下载和检查。

七个源 checkpoint 对应动作物理步 1480、2272、3880、5230、6586、7186、8746，都在机器人动作之后，且只适用于 55k。若要导入当前示例，需要适配字段、拓扑、机器人顺序和轨迹进度，再验证续跑结果。

相关 engine 改动由其他 IPC 示例共享，包括布料系数换算、同一刚体实体内部全部耦合 link 的 IPC 自接触关闭、FEM 状态同步、状态保存恢复和非微分运行的历史内存管理。`scene.get_state()` 会额外保存原生 IPC 状态，带来磁盘和内存开销；逐步采集衣服数据使用的是衣服实体的状态查询。现有测试覆盖了壳系数、接触和单/双环境恢复，其他 IPC 示例尚未完成全量回归。

### 初始下落的对照结论

原参考视频第 0 帧采于 **60 步下落加 2 步动作之后**，对应总第 62 步。当前初始下落结束是第 60 步，因此此前两张图的时刻并不完全一致。

| 离桌高度（mm） | 当前第 60 步 | 原参考第 62 步 |
| --- | ---: | ---: |
| 中位数 | 3.284 | 3.280 |
| 90% 分位数 | 12.008 | 11.167 |
| 最大值 | 24.892 | 25.621 |

两组总体形状接近，但对应顶点平均相差 3.417 mm，最大相差 35.497 mm。局部翘角和袖口褶皱的原因仍需检查，不能把这些差异全部归因于数值误差。

另做过一次延长下落观察：机器人始终保持第一帧姿态，共运行到第 944 步，约 7.87 秒，衣服整体运动已明显减弱。这是独立诊断实验；正式完整折叠仍从初态下落 60 步开始，未加载这个状态。低速度也没有消除初始形状问题。证据在 `out/scene527-settle-comparison/` 和 `out/scene527-settle-continuation/`。

后续调试优先补齐同一时刻（60/62 步）的完整位置、速度和机器人状态，再分别检查袖口、衣角、下摆的高度和接触。需要多视角及实际 IPC 碰撞代理来定位原因。原仓库部分评价关键帧仍来自早期短轨迹，移入时要重新对应当前 4,373 帧长例子。

### 当前验证到什么程度

55k 连续全程已于 2026-10-10 完成：60 个下落步加 8,746 个动作步，最终原生 IPC 帧为 8,806。逐步数据完整且有限，最终状态的位置和速度与 CSV 一致。完整视频有 2,186 帧；相机每四个物理步采样，最后两步保留在状态和 CSV 中。

结果位于 `out/scene527-full-run/`：`scene527-full.mp4` 看全程，`final.png` 看末态，`trajectory-contact-sheet.jpg` 看关键过程，`verification.json` 记录校验。最终收拢形状已被本任务接受；尚未解决的是动作开始前的翘角和袖口不平整，与原仓库完整阶段结果的等价性也仍待验证。

三种网格的资源与配置、轨迹插值和动作进度恢复共 4 项 CPU 检查通过，日志在 `out/scene527-mesh-checks.log`。壳模型和 IPC 接触/状态恢复测试位于 [tests/core/test_garment_folding.py](../../tests/core/test_garment_folding.py)、[tests/ipc/test_shell_conventions.py](../../tests/ipc/test_shell_conventions.py)、[tests/ipc/test_deformable.py](../../tests/ipc/test_deformable.py)。
