# 标注资产切割—涂抹的 Isaac 回放

该入口读取 `asset_demo` 生成的完整运行目录。目录内应包含 `state.npz`、
`config.json`、`assets.json`、实际网格，以及 `annotated` 中的 ZIP 原始 USD 和纹理。
`assets.json` 的 `is_cut_and_spread` 为真时，启用真实砧板、带把手刀及变形面包。

模拟与渲染代码均在本仓库，首次运行见 [独立运行说明](STANDALONE.md)。
在仓库根目录执行，`RUN_DIRECTORY` 是仿真的输出路径：

```powershell
$env:PYTHONPATH = $PWD.Path
GENESIS_PYTHON -m examples.butter_cut_spread.asset_prepare --run RUN_DIRECTORY --workers 6 --coating-voxel 0.0001
ISAAC_PYTHON -m examples.butter_cut_spread.asset_render --run RUN_DIRECTORY --preview --samples 64
ISAAC_PYTHON -m examples.butter_cut_spread.asset_render --run RUN_DIRECTORY --samples 128
GENESIS_PYTHON -m examples.butter_cut_spread.aligned_validate --run RUN_DIRECTORY
```

预处理需要 NumPy、SciPy、trimesh、manifold3d、scikit-image、numba。
Isaac 环境需要 Isaac Sim、NumPy、SciPy、trimesh、Pillow 和 imageio-ffmpeg。
首次启动 Isaac 时需接受其许可协议。Genesis 和 Isaac 使用各自的 Python 进程。
`--workers` 默认取 CPU 逻辑核心数与 6 的较小值。多个进程共享只读粒子轨迹并独立重建不同帧，可缩短处理时间，
但需要额外的工作内存；不改变核函数、布尔裁剪或时间上的处理。
内存紧张时可显式设为 1 或 3。密度核复用各轴权重并减少临时数组，仍保留逐粒子的累加顺序。

完整视频为 `aligned-cut-spread-isaac.mp4`；`--preview` 仅输出五个采样时刻，
不能作为完整轨迹。画面为 1280 × 720，帧率和时间顺序取自仿真记录。
无窗口渲染关闭后台视口更新，只保留输出相机。场景初始化完成后重新设置采样数，并逐帧验证实际采样数与请求一致；
报告中的 `effective_samples` 是读回的有效值。首次完成的有效图像直接用于编码，空图像继续重试同一时刻。
三角面索引使用批量数组上传，避免逐行拼接百万面网格产生临时对象；顶点、法线、面顺序和显示精度保持不变。
逐帧报告区分 `geometry_seconds` 与 `capture_seconds`，便于区分几何更新和图像采样的开销。

厚黄油由原粒子重建；刀面薄残留通过 `blade_film.py` 分配到双面透明薄层，实体与薄层使用互补的粒子体积权重。
薄层使用随仓库提供的 `materials/blade_film.mdl`，覆盖率控制整层材质（包括反光）的可见性。
每帧剔除无覆盖三角面并更新朝外法线，避免透明矩形轮廓与刀具移动时的反光伪影。
该 MDL 文件也是独立运行所需文件；渲染报告记录它的哈希，最终检查核对实际使用的版本。
薄层随真实刀具变换，透明度由局部粒子体积驱动。`blade-film/` 的几何及逐帧文件哈希写入表面清单，渲染时逐一验证。
完整规则见 [HYBRID_ADHESION.md](HYBRID_ADHESION.md)。靠近面包的薄涂层继续使用保持粒子静止体积的各向异性核。
`--coating-voxel 0.0001` 将已经转移到面包区域的独立黄油部分以 0.1 mm 网格重建，
同时降低薄涂层法向核宽的下限；瓷盘中的主体仍使用 0.3 mm 网格。默认值 0.0003
保留原重建设置。该选项只影响表面分辨率，不改变粒子、材质颜色或透明度。
局部细网格依赖粒子分布。`film.py` 对距邻近粒子不超过两个粒子间距的封闭薄层缺口
重新分配显示密度；密度积分保持不变，但等值面体积可有小幅变化。开放边缘和较大缺口保留。
显示核与实际刀具、瓷盘的重叠通过布尔裁剪清除；裁剪只影响显示表面，不改粒子轨迹。
若资产准备阶段提供了验证通过的 `dish_collision_file`，托盘裁剪复用该简化碰撞网格，
避免逐帧对高分辨率托盘做布尔计算；显示托盘仍使用原始网格。裁剪网格记录在 `surface-report.json`。
砧板、刀具和面包使用 ZIP 中的原始材质与纹理。面包网格通过固定的初始材料点邻域
插值位移，保留原始 UV；不得每帧重选邻域。

黄油使用本例内的 `materials/approved_yellow/appearance.json`。
中性基础照明、相机和该材质由 `asset_scene.py` 配置，未对单帧颜色或刀具位置做修饰。
三个 ZIP 自带的预览灯和相机在引用时禁用，避免环境灯互相竞争或随刀具运动。
渲染只允许场景自身的四盏固定灯，灯光路径写入渲染报告。
`asset-display-check.json` 记录砧板、刀具和面包显示网格与初始化网格的最大误差，超过 1 μm 即停止。
`surface-report.json` 与渲染报告保存同一 `state.npz` 哈希，防止混用不同仿真的表面和运动。
`aligned_validate` 还需先完成物理端的 `asset_audit` 和 `aligned_audit`；它核对两份视频的
完整帧数、帧率、轨迹哈希及物理验收结果，并输出 `final-validation.json` 和关键帧拼图。
当前验收还要求逐帧记录有效采样数；只记录请求值的旧渲染报告不能证明实际采样预算。
回放导出的 `.usdc` 保存最终时刻场景；完整运动由轨迹文件和回放脚本重现。

## 完整刀具的取景

为让抬起后的完整刀柄留在画面内，可保持相机位置与场景灯光，改用以下整段统一取景：

```shell
ISAAC_PYTHON -m examples.butter_cut_spread.asset_render --run RUN_DIRECTORY --samples 128 --camera-target 0.11 -0.025 0.07 --focal-length 30
```

`--camera-target` 为世界坐标米制注视点，`--focal-length` 为毫米制焦距；省略时保留原注视点和焦距。
相机水平/垂直光圈明确设置为 36/20.25 mm，与 1280×720 的画幅一致。取景参数写入渲染报告；
这些选项不调整黄油材质、透明度或场景灯光。

刀面近景可使用 `--frame-indices`、`--camera-eye`、`--camera-target` 和 `--output-name` 单独输出；
指定帧的检查视频会如实标为非完整回放，原完整视频保留。
