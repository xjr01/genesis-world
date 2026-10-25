# 标注资产的连续切割与涂抹

该入口将盘中的黄油端部切片带到右侧砧板上的面包，从左向右涂抹。
黄油和面包在同一个 Genesis 场景中连续积分；黄油保留原粒子编号和材料状态。
刀具是带把手、偏置刀颈的真实钢片网格。CPIC 仅选择刀具；瓷盘和砧板参与普通接触。

## 资产与坐标

- `board-资产.zip`：对应 `砧板_第二块.blend` 中的 `Cutting_Board_02`。
  两份砧板 Blend 的成品几何相同；ZIP 提供第二块砧板的纹理和标注碰撞网格。
- `bread-资产.zip`：约 118.17 × 118.46 × 13.74 mm 的面包。将原本竖立的 ZIP 坐标
  转为平放，并依据砧板标注碰撞网格的真实顶面高度放置。
- `knife-资产.zip`：与 `把手刀.blend` 的五个成品部件对应；坐标变换通过对应钢片顶点
  的刚体拟合求得。拟合残差超过 1 μm 时拒绝初始化。
- 黄油和瓷盘使用 `asset_init` 生成的米制网格、支撑高度场和尺寸。

世界坐标采用米、Z 向上。ZIP 中的 assembly anchor 是原场景位置记录；本示例采用新的
并排工作区布局，几何尺寸保持不变。`assets.json` 保存全部显示/物理坐标变换与 ZIP 哈希。
刀具局部负 Z 面承载切片，水平涂抹时刀柄位于刀面上方。

## 初始化与仿真

先按主 README 的资产流程导出黄油、瓷盘及刀具，生成 `out/initialized` 和
`out/assets/knife`。初始化阶段另需 USD 读取依赖；实际 Genesis 仿真读取导出的网格。
资产预处理与校验入口使用 Python 3.11–3.13。输入 ZIP、Blend、生成的缓存和视频不随源码分发，
需要提供自己的资产并在目标机器执行离线准备；已有的本机缓存路径不能直接搬到另一台机器使用。

```shell
python -m pip install -r examples/butter_cut_spread/requirements-assets.txt
blender --background --python examples/butter_cut_spread/triangulate_assets.py -- --archives ASSET_DIRECTORY --output-dir out/triangulated
python -m examples.butter_cut_spread.aligned_init --base out/initialized --archives ASSET_DIRECTORY --knife-export out/assets/knife --triangulated out/triangulated --output-dir out/aligned-initialized
python -m examples.butter_cut_spread.prepare_runtime --assets out/aligned-initialized --output out/aligned-prepared --blender PATH_TO_BLENDER
python -m examples.butter_cut_spread.asset_demo --assets out/aligned-prepared --run out/aligned-cut-spread
python -m examples.butter_cut_spread.asset_particles --run out/aligned-cut-spread
python -m examples.butter_cut_spread.asset_audit --run out/aligned-cut-spread
python -m examples.butter_cut_spread.aligned_audit --run out/aligned-cut-spread
```

Blender 用于离线三角化和托盘碰撞代理减面。资产导出的三角化保留凹多边形边界和原始顶点；
托盘另生成经过误差验证的碰撞代理。输出的三角网格与 ZIP 哈希绑定。
渲染读取 ZIP 原始 USD、UV、纹理和材料；碰撞计算读取显式导出的网格。
砧板碰撞使用 ZIP 的 `/Asset/collision_0`。面包使用原网格采样为可变形 MPM 材料。
刀具碰撞保留实际钢片厚度和刀颈形状。

默认切片厚度为 10 mm，定义为刀具外侧面到黄油端面的距离，可用
`aligned_init --slice-thickness 0.010` 设置。面包直边按 XY 轴对正，原始网格、纹理和真实尺寸保留。
刀具采用 15° 横刀姿态，先完整下压，再沿短边的 Y 负方向单向推进约 24.7 mm，随后连续抬起。
路径覆盖同时计入上部下压切口和底部抽切，因此刀面高度小于黄油高度时也不必增加回程。
切割和抬刀过程中都没有前后折返或晃动。实际刀刃轮廓决定切入位置和单向行程。
抬刀阶段 Y 坐标不变，前 0.2 s 同时沿 X 向外单向离缝 3 mm，之后继续上升；没有停顿或反向移动。
离缝只发生在完成切割之后，不改变初始 10 mm 切片厚度。切割阶段仍有覆盖曲线刀刃所需的短行程。

## 离线预处理与运行时加载

`prepare_runtime` 只需针对一组资产及轨迹执行一次：

1. 用 Blender 离线生成约 2,000 面的托盘碰撞网格。验证黄油支撑区最大误差低于 0.05 mm、
   双向顶点到表面距离低于 0.2 mm，原始托盘网格继续用于显示。
2. 托盘 SDF 使用 1 mm 目标间距、最大分辨率 256；砧板使用 1 mm、384。
   薄刀具保留原有精细 SDF。瓷盘和砧板仍只参与普通接触。
3. 生成 `butter-particles.npy`、`bread-particles.npy`、`motion.npz`，并生成 Genesis 原生 SDF 缓存。
4. 用与正式运行相同的显式粒子入口构建求解器，执行一个丢弃的编译预热步。
   保存的初始化粒子是在预热之前生成的，正式仿真从这些初始粒子重新开始。

正式运行直接加载显式粒子和轨迹，不重新读取黄油、面包网格进行采样，也不在线简化托盘。
运行前检查资产哈希和 SDF 缓存；资产变化或缓存缺失会停止并提示重新预处理，避免悄悄执行昂贵预处理。
`scene.add_entity` / `scene.build` 仍需要在运行时创建求解器实例和 GPU 状态，这部分不等于重新制作资产。
首次换机器、引擎版本或缓存目录时，需要在目标机器预处理一次。

仅修改同一组资产的轨迹配置时，可直接更新预处理目录的配置并重新预热，无需再做碰撞减面：

```shell
python -m examples.butter_cut_spread.asset_demo --assets out/aligned-prepared --run out/aligned-prepared --prepare-only
```

`runtime-preparation.json` 保存预处理耗时、缓存命中情况和 SDF 格点数；`simulation.json` 的
`startup_seconds` 分开记录文件校验、载入、粒子实例化、构建和接触缓冲区耗时。

批量生成可复用同一预处理目录，每次给 `--run` 指定新的独立输出目录。修改资产、轨迹或
生成配置后重新准备，再运行对应案例。当前示例使用单环境接触与记录；批量案例通过独立
运行组织，不能仅增大 `scene.build` 的环境数来替代这些入口。

默认沿用实际尺寸黄油的校准参数：屈服应力 160 Pa、稠度 15 Pa·s^0.5、密度 900 kg/m³，
剪切/体积模量分别 20/150 kPa。该参数是演示校准值。时间步 25 μs，黄油/面包粒子间距
0.4/0.8 mm。完整过程 8.4 s，336,000 步，按 24 fps 记录，并额外保留最终时刻。

瓷盘采用原生普通接触及 0.35 摩擦系数，`board_normal_stress` 和 `board_shear_stress` 均为零，
不会继承木砧板示例的额外黏附。面包与黄油的接触由独立的面包接触模块处理。

| 阶段 | 结束时刻（s） |
| --- | ---: |
| 静置 | 0.6 |
| 下压 | 1.5 |
| 完成短边切割 | 2.9 |
| 抬刀前段（连续上升，无停顿） | 3.05 |
| 抬刀 | 3.7 |
| 转运并将载料面朝下 | 5.2 |
| 接近面包 | 6.0 |
| 向右涂抹 | 7.5 |
| 退刀 | 8.1 |
| 观察沉积结果 | 8.4 |

改变面包或刀具尺寸时，应重新检查有效涂抹区域、刀柄离地间隙、切割覆盖范围和 MPM 边界。
`aligned_motion.py` 控制转运与涂抹轨迹；`asset_motion` 控制端部切割。
刀面携带接触采用 2,000 Pa 法向应力上限、1,500 Pa 切向应力上限，捕获间隙为 0.8 mm、
脱离间隙为 1.6 mm。间隙从粒子半径外侧到刀面计算。这些是针对网格接触尺度的拟合参数，
不是黄油材料的实测黏附数据；仍是有范围、有上限的接触力，不直接锁定粒子位置。
`adhesion_capture_range=None` 保留一个粒子间距的捕获默认值，不影响其他入口的原设置。
`cut_end_time=2.9` 与切割阶段同步，切割过程中保留较低的切向作用上限，避免拖拽黄油主体。
`cutting_direction=-1` 表示沿 Y 负方向完成一次切割；取 `1` 可反转整次行程，两种设置都不会往返。
切割阶段的附加切向应力上限由 `cutting_shear_stress` 控制，默认 50 Pa。
真实刀面的接触范围由 `blade.obj` 的中面轮廓离线采样为 `blade-face.npz`，网格间距 0.1 mm。
该范围同时用于黏附和跨刀面检查，包含原 35 mm 矩形之外的圆弧刀尖；柄部弯曲连接区不计入平面黏附。
正式运行只载入这个二维距离场，不重新求刀面轮廓。三维钢片碰撞仍使用原刀具距离场。
`spread_transition_time` 控制从携带黏附向可滑动涂抹接触的转换，应随轨迹阶段一起调整。
转运过程中将刀具朝向转过 180°，使切片的悬出部分朝向涂抹前方。
涂抹姿态倾斜 10°，前缘较高、后缘较低，随向右运动逐渐压薄黄油。
最低刀缘相对面包原始中央顶面间隙由 0.6 mm 降至 0 mm；面包本身仍可受压变形。
涂抹阶段切向松弛时间为 50 μs，切向应力上限为 450 Pa。较短松弛时间增加刀面带动，
但也增加面包受剪和黄油拖尾；应力上限限制每步冲量。涂抹阶段法向黏附应力为 40 Pa，
允许黄油在退刀时更易留在面包上。这些是接触拟合值，不是黄油本构参数。

## 检查与输出

`simulation.json` 分别记录初始化、各阶段步进和轨迹写出后的总耗时。
首次 SDF 预处理和编译在离线准备阶段完成；运行时只加载准备好的资产和缓存。
`checkpoint-*.npz` 是各阶段的可视化诊断快照，完整粒子轨迹保存在 `state.npz`。

`aligned-cut-spread-particles.mp4` 显示中央截面与俯视粒子抽样。橙色跟踪最初切片中的
固定粒子编号；截面显示范围以外的粒子仍参与仿真。
`aligned-audit.json` 检查完成状态、有限数值、粒子存活、仅刀具参与 CPIC、切片带起比例及沉积比例。
此外要求实际留在面包附近的切片粒子沿涂抹方向的 5%—95% 分位跨度相对接近阶段增加至少
10 mm，避免只落料而未铺展。前后比较使用最终沉积的同一批粒子编号；留在瓷盘或抬起刀具上的
粒子不计入涂层跨度。
并要求至少 15% 的初始切片粒子最终沉积到初始接近位置 90% 分位前缘再向右 5 mm 以外，
避免少量远处残留将跨度指标夸大。
`asset-audit.json` 检查保存帧中的钢片穿透和瓷盘支撑情况。
保存帧的侧别与穿透检查不能证明两帧之间的连续时间接触完全无误，应结合完整视频检查。

面包接触使用每个初始 XY 粒子列的顶部粒子，覆盖真实起伏表面。
旋转刀具的防穿透约束分别使用前、后时刻的接触坐标系，并计入角速度产生的表面速度。
面包显示网格使用固定材料点邻域插值位移，保持原始 UV 和纹理随表面移动。

当前横刀版本的结果见 [横刀验证记录](HORIZONTAL_VALIDATION.md)。
碰撞优化对照见 [资产优化记录](ASSET_OPTIMIZATION_VALIDATION.md)。
上一版 5 mm 切片的历史结果见 [早期验证记录](ALIGNED_VALIDATION.md)。
