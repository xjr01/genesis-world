# 黄油涂抹的 Isaac 离线渲染

本目录提供从 Genesis 粒子轨迹到 Isaac 视频的完整参考流程：黄油表面、随粒子变形的面包表面、刀具、支撑板、桌面、材质、灯光、相机和视频编码。场景几何均由代码生成，不需要其他项目、外部纹理、购买资产或本地私有文件。

Genesis 完成物理仿真并导出轨迹；Isaac Sim 只回放几何和渲染，不再积分物理，也不修改接触、本构、粒子或刀具轨迹。此功能可选，原有 `demo` 和 `pipeline` 无需安装或导入 Isaac。

## 环境

使用两个独立 Python 环境，所有命令从仓库根目录运行：

| 环境 | 用途 | 安装 |
| --- | --- | --- |
| Genesis | 仿真、审计、黄油重建、面包表面准备和视频验收 | 按 [README](README.md) 安装本仓库；视频验收使用随项目安装的 `av` |
| Isaac | 场景、材质、路径追踪和 MP4 编码 | Python 3.11、Isaac Sim 5.1.0、兼容的 NVIDIA RTX GPU 和驱动 |

渲染参考环境为 Windows、Isaac Sim 5.1.0、NumPy 1.26.0、Pillow 11.3.0、imageio-ffmpeg 0.6.0。版本固定用于复现此参考，并非推荐使用的最新版本；其他版本需要重新验证。Isaac 官方提供 [Python 安装说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)。

在独立 Python 3.11 环境中安装：

```shell
python -m pip install --upgrade pip
python -m pip install -r examples/butter_spreading/requirements-isaac.txt
```

已有 Isaac 安装时，也可使用其 `python.bat`（Windows）或 `python.sh`（Linux），安装清单中的 Pillow 和 imageio-ffmpeg 后运行。不要在该环境中安装 Genesis 的整套依赖。`carb`、`omni`、`pxr` 和 `OmniSurface.mdl` 由 Isaac 提供，必须在 `SimulationApp` 启动后使用；不要另外安装同名包。首次启动可能需要下载扩展和编译着色器。

使用者需自行接受 [NVIDIA Omniverse 许可协议](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)。入口不自动接受协议；已接受者可按官方说明为渲染进程设置 `OMNI_KIT_ACCEPT_EULA=YES`，例如 PowerShell 中使用 `$env:OMNI_KIT_ACCEPT_EULA='YES'`。

## 完整运行

下面 `python` 指当前步骤对应环境的解释器。每次实验使用新的输出目录。

**1. Genesis 环境：生成并验收四秒轨迹。** 已有合格的完整流水线结果可跳过此步。

```shell
python -m examples.butter_spreading.pipeline --config examples/butter_spreading/configs/dataset.json --output-dir out/spreading
```

**2. Genesis 环境：准备可独立传输的渲染输入。**

```shell
python -m examples.butter_spreading.prepare_isaac --source out/spreading --output-dir out/spreading-isaac
```

该步骤检查流水线完成状态、物理审计和轨迹／黄油表面哈希，复制输入，并从面包粒子生成逐帧形变网格。目标目录必须为空。不要只复制 `mpm-state.npz`：准备后的整个目录才是一份完整渲染输入，可发给另一台具有本仓库和 Isaac 环境的机器。

**3. Isaac 环境：渲染完整正面和背面视频。**

```shell
python -m examples.butter_spreading.render_isaac --run out/spreading-isaac
python -m examples.butter_spreading.render_isaac --run out/spreading-isaac --back-view
```

如需先检查单帧外观：

```shell
python -m examples.butter_spreading.render_isaac --run out/spreading-isaac --frozen-frame 48
```

预览重复同一源帧八次，不代表完整轨迹。完整渲染默认使用 128 samples、1280×720、24 fps，导出 97 帧，包括 4.00001 s 的离散终点。入口拒绝覆盖同名 MP4、报告或场景文件；更改参数做对照时，重新准备一个输出目录。

**4. Genesis 环境：核验视频和几何回放。**

```shell
python -m examples.butter_spreading.validate_isaac --run out/spreading-isaac
```

验收要求完整正面视频；若存在背面报告，也会验收背面视频，并要求两侧材质和光照设置一致。检查输入哈希、源帧顺序、源时间、USD 几何坐标读回误差、材质配置、视频帧数／尺寸／帧率及连续编码时间。它不能替代画面质量人工检查，也不是重新运行物理审计。可通过 `--reference path/to/render-report.json` 与另一份完整渲染报告比较材质、灯光和采样参数。

## 文件和坐标

| 文件 | 内容 |
| --- | --- |
| `mpm-state.npz`、`mpm-state.json` | 原始粒子、面包粒子、刀具世界坐标、时间和采样间距 |
| `mpm-surface.npz`、`mpm-surface.json` | 原样复制的黄油重建表面 |
| `bread-surface.npz` | 各帧面包顶点、法线、固定三角形和顶点颜色 |
| `isaac-inputs.json` | 输入哈希、几何参数、接触检查和准备耗时 |
| `isaac-spreading.mp4`、`isaac-spreading-back.mp4` | 完整正面／背面视频 |
| `isaac-spreading*.json` | 材质、灯光、相机、输入哈希、逐帧误差和耗时 |
| `isaac-spreading*.usdc` | 最后一帧静态场景，供检查材质和构图；不是动画缓存 |
| `isaac-spreading*-NNN.png` | 代表帧截图，实际命名由渲染入口生成 |
| `isaac-validation.json` | 验收结果和渲染源码哈希 |

单位为米，Z 向上。`knife_positions` 是世界坐标中的工具参考位置；实际刀具中心还要加 `knife_collider_xy_offset_m`，不能重复叠加面包高度。刀具尺寸来自轨迹文件，不为画面效果缩放。该参考适配本示例的轴对齐、平移刀具；接入旋转刀具时需扩展轨迹格式及变换，不能仅替换位置。

面包初始粒子必须是完整的规则三维格点。表面通过三线性位移插值和半格外推跟随粒子，形变增益为 1，刚体运动和均匀压缩均保持一致。面包颜色固定在材料坐标中；细小色差为程序纹理，没有额外制造孔隙几何。支撑板沿用本示例固定尺寸 0.18×0.14×0.02 m、顶高 0.01 m；改动物理场景支撑几何时需同步修改准备入口。

## 外观与调节位置

| 位置 | 作用及取舍 |
| --- | --- |
| `materials/appearance.json` | 黄油 OmniSurface 参数：暖黄色、粗糙度 0.4、折射率 1.46、次表面权重 0.2、散射尺度 0.0001 m；增大散射会改变薄处透光和整体观感 |
| `isaac_materials.py` | 金属／板／桌面材质、面积光、环境光和路径追踪设置 |
| `isaac_scene.py` | 面包材质、相机位置、焦距、画面尺寸与几何回放 |
| `--samples` | 默认 128；提高可减小噪声，但增加耗时 |
| `--background-fill off/on` | 默认 off 仍有基础环境光和主光；on 仅把环境光 650 提高至 750、前补光 900 提高至 1050；其他参数相同 |
| `--back-view` | 使用另一侧固定相机，材质和灯光不变 |

外观 JSON 中的颜色是视觉拟合参数，不能直接当作实测黄油反照率。散射、透光和粗糙度是明确标注的近似。材质数据按线性 shader 输入使用，不要把显示器截图中的 sRGB 数值直接替换进去。光照、色调映射和相机响应共同影响最终画面；对照实验一次只改变一组参数。

此示例的合格基准轨迹没有需要独立处理的刀面薄膜粒子，使用黄油体积表面渲染。准备入口检测刀具内部的重建顶点，以及距离两侧宽面小于一粒子间距的候选粒子；若出现这类未适配的薄膜候选，会报错，避免伪造附着层。它不包含粒子薄膜 hack、切割场景或薄膜混合渲染。如果外部轨迹包含刀面薄膜，需要另行定义捕获粒子编号、刀面局部坐标和厚薄分区。

## 维护验证

在 Genesis 环境中运行面包表面回归测试：

```shell
python -m pytest -n 0 tests/particles/test_butter_spreading_render.py
```

测试验证静止尺寸、刚体平移、剪切／压缩、法线变换和粒子编号重排下的几何一致性；不启动 Isaac。修改材质、相机或 Isaac 接口后，还需运行上面的完整渲染和验收流程。渲染报告分别记录启动、场景准备、逐帧绘制及总体耗时，不与 Genesis 物理步进耗时混算。
