# 独立运行：最新版切黄油与涂抹

本仓库同时包含 Genesis 物理模拟、Isaac 表面重建和渲染代码，以及运行本案例所需的
米制网格、USD、纹理和场景配置。无需另一个 checkout、原作者的 `out/` 目录或 Blender。
从 Git 拉取时需保留 `assets/scene.zip`（约 61 MB），不要只复制 Python 文件。

## 安装

Genesis 与 Isaac 使用两个独立 Python 环境。使用 Python 3.11，先在 Genesis 环境安装
适配 GPU 的 PyTorch，再在仓库根目录安装项目：

```shell
python -m pip install -e ".[dev]"
```

Isaac 环境单独安装，不要向其中安装 Genesis 或 Genesis 的依赖：

```shell
ISAAC_PYTHON -m pip install -r examples/butter_cut_spread/requirements-isaac.txt
```

`ISAAC_PYTHON` 表示该环境的 Python 可执行文件路径。Isaac 需要 NVIDIA RTX GPU 和
兼容驱动；首次启动按 NVIDIA 的流程接受许可。此示例不会自动替用户接受许可。
在 Windows PowerShell 中，从仓库根目录运行，并确保测试加载当前 checkout：

```powershell
$env:PYTHONPATH = $PWD.Path
```

## 完整运行

在 Genesis 环境执行，替换下列 Isaac Python 路径：

```powershell
python -m examples.butter_cut_spread.asset_pipeline --run out/firm-smooth --isaac-python "C:/path/to/isaac-env/python.exe"
```

入口依次完成资产校验/解包、目标机器上的粒子与 SDF 准备、完整物理模拟、粒子视频、
物理检查、全部帧的表面重建、Isaac 视频与最终一致性检查。首次准备和编译需要额外时间。
默认使用 0.6 mm 黄油粒子间距、0.8 mm 面包间距、50 μs 时间步、24 fps、128 次渲染采样。
完整过程为 8.4 s、168,000 步、203 帧。`--workers` 默认 3；增加进程数需要更多内存。
完整轨迹、表面和渲染会占用较多磁盘空间，请保留数十 GB 可用空间。

默认材质为 `configs/asset_firm_smooth.json`：G=80 kPa、K=150 kPa、屈服应力 600 Pa、
稠度 25、流动指数 0.5、密度 900 kg/m³，整个过程固定，不做分阶段软化或参数渐变。
材料参数与 `51c83edc` 完全一致；刀面法向黏附为 1500 Pa，厚层黏附上限为 60 Pa。
完整原轨迹与加速刀路的对照结果见 [参数验证](PARAMETER_ROBUSTNESS.md)。
这是演示校准参数；有限变形、涂抹边缘缺口和细小三维通道仍可能出现。

双面刀片使用一致的接触与沉积规则。薄层保留原粒子 ID，黏在刀面时暂时退出
体积求解，释放后重新参与模拟；薄层颜色按实际厚度标定，保持淡色、透出金属。
实现与完整对照见 [黏附说明](HYBRID_ADHESION.md) 和 [沉积薄层验证](COHERENT_FILM_VALIDATION.md)。

只检查五个渲染时刻可加 `--preview`，物理过程仍完整。只需物理结果可加 `--physics-only`，
此时无需安装 Isaac。`--duration 0.01 --physics-only` 用于短运行检查，不代表完整效果验收。

每次新模拟使用新 `--run` 目录。默认准备目录为 `RUN-prepared`；若已存在，请复用分阶段
命令或换一个新路径，不要重复执行默认的 `all` 阶段。

## 分阶段复用

```shell
python -m examples.butter_cut_spread.asset_pipeline --stage prepare --run out/firm-smooth
python -m examples.butter_cut_spread.asset_pipeline --stage simulate --run out/firm-smooth
python -m examples.butter_cut_spread.asset_pipeline --stage render --run out/firm-smooth --isaac-python ISAAC_PYTHON --preview
```

最后一条去掉 `--preview` 即重建并渲染全部帧。已有准备目录可通过 `--prepared DIRECTORY`
复用。改变粒子间距、时间步、资产、引擎或接触/运动代码后，需要重新准备。
`--particle-size` 和 `--dt` 只在准备阶段生效。自定义 `--material` 时，准备与模拟阶段均传入
同一份文件；记录的 `material.json` 是该次运行实际采用的常量参数。

直接渲染已有完整仿真，也可以使用：

```shell
python -m examples.butter_cut_spread.asset_prepare --run RUN_DIRECTORY --preview --workers 3 --coating-voxel 0.0001
ISAAC_PYTHON -m examples.butter_cut_spread.asset_render --run RUN_DIRECTORY --preview --samples 128
```

两个阶段的 `--preview` 应一致；完整渲染会拒绝仅有预览帧的表面清单。
刀面薄层会随重建生成在 `blade-film/`，渲染同时读取实体网格和双面薄层。
黏附、残留约束和薄层分配规则见 [HYBRID_ADHESION.md](HYBRID_ADHESION.md)。

主要输出为 `aligned-cut-spread-particles.mp4`、`aligned-cut-spread-isaac.mp4` 和
`final-validation.json`。五帧预览使用 `aligned-cut-spread-preview.*` 文件名，不能作为完整视频。

## 独立性与效果边界

- 资产压缩包只含输入数据，不含旧机器缓存、初始化粒子或录制轨迹。首次准备从网格生成这些数据。
- 渲染模块和光学材质位于本例内部；共享物理模块来自同仓库的 `examples/butter_spreading`。
- SDF 缓存与机器/引擎有关，换机器后重新执行准备；无需复制原机器缓存目录。
- 小孔修复仅调整显示密度，保留密度积分；模拟粒子、质量和材质不变。网格体积可有小幅变化。
- 50 μs 时间步超过引擎保守建议，会打印稳定性提示；它已针对本配置验证，其他材质需重新验证。

资产来源与校验见 [assets/NOTICE.md](assets/NOTICE.md)，性能和视觉取舍见
[PERFORMANCE.md](PERFORMANCE.md) 与主 README 的薄层孔洞说明。
此前干净副本、空缓存和完整物理运行的验证范围见
[独立运行验证记录](STANDALONE_VALIDATION.md)。本次双面黏附与薄层的完整对照见
[双面黏附验证记录](HYBRID_VALIDATION.md)。
