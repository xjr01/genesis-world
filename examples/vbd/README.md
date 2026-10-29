# Genesis VBD Solver

Profiling: [original bottleneck analysis](results/profile/README.md) and
[graph scheduling and cooperative vertex results](results/scheduling/README.md).

CUDA runs replay each color sweep through a Quadrants graph and use eight cooperating threads per vertex.
Coloring is computed during scene construction. CPU runs use one thread per vertex.

Local energy line search is disabled by default. Add `--line-search` to `vbd_elastic.py`, `vbd_shell.py`, or
`vbd_profile.py` to enable backtracking for difficult deformations at extra computational cost. The corresponding
solver option is `VBDOptions(is_line_search_enabled=True)`; `line_search_iterations` limits its backtracking attempts.
Contact displacement limits and invalid-state checks remain active in both modes. This flag applies to the VBD solver.

## 运行示例

在仓库根目录、安装了当前 checkout 的 Python 环境中运行：

```powershell
python examples/vbd/vbd_shell.py -g -v
python examples/vbd/vbd_shell.py -g -v --scenario hang
python examples/vbd/vbd_shell.py -g -v --solver ipc --scenario hang
python examples/vbd/vbd_shell.py -g -v --solver ipc --scenario shirt
python examples/vbd/vbd_shell.py -g --scenario hang -r -o out/vbd/shirt
python examples/vbd/vbd_elastic.py -g -v
python examples/vbd/vbd_elastic.py -g -v --material-model log_neohookean
python examples/vbd/vbd_elastic.py -g -v --solver ipc
python examples/vbd/vbd_shell.py -g --scenario fold --plastic
python examples/vbd/vbd_elastic.py -g --scenario load_unload --plastic --yield-strain 1
```

`-g` selects the GPU, `-v` opens the viewer, and `-s` sets the number of frames (shell: 180; elastic: 200). Both scripts run
independently. The shell example defaults to dropping `genesis/assets/meshes/shirt.glb`; `--scenario hang` holds its
shoulders, and `--scenario fold` folds a rectangular sheet. `-r` records a video under `--output-dir`; recording and
benchmarking run separately. Videos and benchmark reports use `<solver>_shell_<scenario>` filenames.

`vbd_shell.py --solver ipc` selects `FEM.Cloth` with `IPCCouplerOptions` and requires `--gpu`. It supports shirt drop
and static shoulder pins; folding with moving/released pins and plasticity uses `--solver vbd` (the default).
FEM vertex constraints can be set or removed before `scene.build()` for every environment and hold the initial positions.
Each shell frame advances 1/60 s through twenty 1/1200 s physics steps for either solver. `--dt` and `--substeps` control
these intervals. Reports distinguish these frame
settings (`dt`, `substeps`) from actual scene settings (`scene_dt`, `scene_substeps`).

Both modes use the specified Young's modulus, Poisson ratio, density, full mass thickness, self-friction and activation gap.
The IPC material receives half the full thickness, since its shell thickness parameter is a radius and mass integrates
over twice that radius. IPC uses its shell constitutive law, barrier contacts and Newton convergence settings.
`--ipc-bending-stiffness` controls the FEM.Cloth bending coefficient and defaults to 40, following the garment example's
calibration. VBD uses `--bending-rigidity` in N*m; the two bending coefficients have different definitions.
`--no-bending` disables bending in either mode. VBD iteration, search, contact stiffness and plasticity options are
accepted only with `--solver vbd`; benchmark entries for inactive parameters are `null`.

The shirt keeps its texture and connected inner/outer surfaces: 8,412 simulation vertices and 16,832 triangles.
The example rotates the asset from Y-up to Z-up and places its hem 0.35 m above the ground. Hanging selects mesh
vertices nearest the two shoulder locations `(+-0.2, 0, 1.09)` m, then fixes each surrounding 0.025 m-radius patch
with `entity.set_vertex_constraints(fixed_indices)` before `scene.build()`. This fixes 12 vertices in total at their
initial positions; the remaining vertices move under gravity. Orange markers show the fixed patches, with a visual
support above them. Setting the constraints before building also makes `scene.reset()` restore them.

The shell example uses material parameters from the [Gaia multilayer gravity-cloth preset](https://github.com/AnkaChan/Gaia/blob/c229692045465a76233f9fba9197fb22bbfb3694/Simulator/VBDCloth/ParameterGen/S03_MultiLayerClothOnCollider.py),
interpreted in centimetres, grams and seconds: the 100-unit cloth spans 1 m, and its area density `0.02` becomes
`0.2 kg/m^2`. The two membrane coefficients `mu=lambda=10000` become `10 N/m` each. At the example's full thickness
`h=0.00025 m`, `rho=800 kg/m^3`, `E=320000/3 Pa`, `nu=1/3` and `shear_stiffness=0` reproduce that area density and
membrane energy through `mu=E*h/(2*(1+nu))+shear_stiffness` and `lambda=E*h*nu/(1-nu^2)-shear_stiffness`.

The example uses `QuadraticBending`: each interior edge contributes a cotangent quadratic curvature energy with
`bending_rigidity=1e-6 N*m` (reference coefficient `10`). Its equilibrium favors flat cloth while the rest triangles
define membrane lengths and cotangent weights. `DiscreteShellBending` instead restores imported rest dihedral angles;
`--plastic` selects that model for angular plasticity. `--bending-model` selects the elastic bending model explicitly.
The two coefficients represent distinct energies and should be calibrated for the selected model.

`membrane_damping=bending_damping=2e-6 s` applies each local stiffness block to the vertex displacement divided by the
substep duration. This follows the reference's local damping, including resistance to common translation. Both damping
times have command-line controls. The membrane solve uses the full Saint Venant-Kirchhoff Hessian.

| Setting | Shell example value |
| --- | ---: |
| Frame duration / substeps / iterations per substep | 1/60 s / 20 / 15 |
| Gravity magnitude | 10 m/s^2 |
| Initial iterate / correction scale | Bounded inertial prediction / 1 |
| Self-contact stiffness / activation gap / search radius | 100 N/m / 2 mm / 4.5 mm |
| Full contact thickness, independent of mass thickness | 1 micrometre |
| Conservative clearance fraction | 0.4 |
| Self-friction / regularization speed | 0.1 / 0.001 m/s |
| Ground stiffness / activation gap | 10 N/m / 0 |
| Ground friction / regularization speed | 0.3 / 0.0001 m/s |

Ground distances are measured from the simulated vertices; self-contact subtracts the two surface radii. A finite ground
penalty permits load-dependent penetration. `--thickness` controls mass and membrane stiffness, while
`--contact-thickness` controls collision spacing independently. Larger search radii cost more candidate pairs and allow
more motion before rebuilding. `--contact-distance`, `--search-radius` and `--contact-stiffness` expose these tradeoffs.
The solver options `shell_ground_*` control ground response independently.
The initial iterate follows velocity and gravity within collision clearance, improving free motion at a finite iteration
budget. The 2 mm activation gap is comparable to the shirt's inner/outer layer separation. The reference scene instead
uses a 3 mm gap and previous-position initialization; the continuation measurements below document their effect here.

Normal self-contact follows the reference's quadratic/logarithmic branches, including their junction discontinuity.
The logarithmic cutoff is `1e-7 m`; its additive energy offset uses the converted 1 cm reference length. The edge
parallelism cutoff is `1e-9 m^2`. Friction uses the reference edge-based tangent basis and lagged tangent Hessian.
The actual contact-enabled reference path takes bounded full Newton corrections and skips local line search even though
its JSON enables `useLineSearch`; the example therefore leaves `--line-search` off. It rebuilds candidates when clearance
is exhausted and at each substep.

These settings align the selected material/contact formulas and iteration budget. The shirt mesh, contact gap, initial
iterate, shoulder pins and ground location are specific to this example. GPU reduction order, graph coloring and
bounding-volume hierarchy (BVH) traversal differ from
the reference CPU implementation, so trajectories can diverge. IPC retains its own bending law and barrier contacts;
the VBD stiffness damping and separate collision thickness are VBD-only controls.

The current defaults complete a six-second hanging run from the original asset pose on CUDA/float32. Both cuffs drape
beside the torso. Their mean heights descend 205.3 mm and 202.7 mm; the final vertex root mean square (RMS) speed is
0.001755 m/s and the fastest vertex moves at 0.013272 m/s. Throughout the run, the same 12 shoulder vertices remain fixed,
with zero fixed cuff vertices and a maximum pin error of `3.2e-8 m`. The existing vertex-control/state tests pass for both
single and batched environments. See the [current-default summary](results/gaia_alignment/cuff_fix_summary.json) and
[frame measurements](results/gaia_alignment/cuff_fix_samples.json).

Reference-preset validation on the RTX 4090, float32, uses the same 8,412-vertex shirt and 12 shoulder pins, with a 3 mm
contact gap and previous-position initialization. All changes are applied together, including the smaller substep and
changed contact parameters. The fixed-point deviation from the double-precision input coordinates is below `3.2e-8 m`.

| Hanging run | Simulated time | Vertex root mean square (RMS) speed | Fastest vertex |
| --- | ---: | ---: | ---: |
| Previous defaults | 3 s | 0.207038 m/s | 0.790432 m/s |
| Reference preset | 3 s | 0.022235 m/s | 0.771104 m/s |
| Reference preset | 6 s | 0.006929 m/s | 0.109770 m/s |
| Aligned physics, previous step/iteration budget | 3 s | 0.016862 m/s | 0.156599 m/s |

Overall motion decreases substantially; isolated speed spikes persist. The last second averages `0.010091 m/s` RMS.
The cuffs descend only 40-43 mm over these six seconds and remain extended sideways. Lower speed therefore accompanies
arrested draping in this asset and is insufficient evidence of improved hanging behavior.
The preset raises the iteration budget from 8,000 to 18,000 sweeps per simulated second, so this comparison measures
the combined alignment and increased solve work. See the [summary](results/gaia_alignment/hang_summary.json),
[frame measurements](results/gaia_alignment/hang_samples.json) and [speed plot](results/gaia_alignment/speed.png).
The connected inner and outer shirt surfaces have a median corresponding-vertex separation of 2.006 mm in the rest
mesh, within the 3 mm self-contact activation gap. The reference scene spaces separate cloth layers 3 mm apart;
the shirt geometry therefore activates contact under different initial conditions.

Controlled continuations start from the same six-second state and advance another 0.5 s. Every run retains the same
12 shoulder pins and zero cuff pins. Cuffs are selected by rest coordinates `abs(x) > 0.61 m` (159 left, 157 right vertices).

| Single change from reference preset | Left cuff descent | Right cuff descent |
| --- | ---: | ---: |
| Baseline | 1.01 mm | 1.05 mm |
| Membrane and bending damping both zero | 1.01 mm | 1.05 mm |
| 100 iterations per substep | 6.40 mm | 5.97 mm |
| Self-friction zero | 78.73 mm | 66.24 mm |
| Self-contact activation gap 0.5 mm | 158.46 mm | 162.14 mm |
| Bounded inertial prediction as initial iterate | 175.43 mm | 188.94 mm |

The last row retains the 3 mm gap, friction, damping, pins and iteration budget. Its large response to the initial
iterate, together with the iteration-count comparison, exposes substantial convergence sensitivity in the arrested
state. Lowering friction or the gap changes both the physical model and solve difficulty; those controls alone cannot
establish that the arrested shape is a physical equilibrium. See the
[cuff measurements](results/gaia_alignment/cuff_diagnosis.json).

Releasing all pins from that state, zeroing initial velocity and disabling material damping lets the shirt fall as a
whole. After 1/60 s its mass-weighted downward speed is 0.155234 m/s with the previous-position initial iterate,
0.163391 m/s with friction zero, and 0.169372 m/s with bounded inertial prediction. Gravity alone predicts 0.166667 m/s.
Self-contact forces are internal; fixed shoulders supply external support in the hanging scene. These release results
show integration error, while the pinned continuations expose a much larger sensitivity of the draping motion.

The last row of the speed table keeps `--dt 0.005 --substeps 2 --iterations 20` with the aligned material, contact and solver
formulas. Reduced speed therefore remains at the previous iteration budget. Coarser implicit integration also
increases numerical dissipation, so lower speed alone does not establish more accurate dynamics.
[Previous-budget samples](results/gaia_alignment/old_timestep_samples.json) record
this separate comparison.

The reference-preset 6-second drop reaches a minimum sampled height of `-0.001918 m` with the `10 N/m` ground penalty;
its final RMS speed is `0.022766 m/s`. This finite penetration follows the selected soft ground response. Increasing
`shell_ground_stiffness` reduces it at the cost of a harder contact solve. [Drop samples](results/gaia_alignment/drop_samples.json)
retain the full frame measurements.

The existing material-response test includes an analytical free-corner quadratic bending solve with and without damping,
membrane damping under load, and a ground-penalty solve with an independent zero activation gap. Removing either damping
term or replacing quadratic bending with rest-angle bending makes the corresponding physical assertion fail.
Final validation passes all 18 VBD tests on CUDA/float32 and all 12 contact/material cases on CPU/float64, covering
single and batched environments and both line-search settings. Existing support-force tolerances are unchanged.

The elastic drop, load/unload and elastic FEM/IPC baseline use `genesis/assets/meshes/jelly/remeshed.obj`. That asset uses
metres and Z-up; these examples raise its bottom to 0.015 m.

`SimOptions` controls the time step and gravity. The elastic example uses `dt=0.005`, two substeps and 20 iterations per
substep; shell settings are listed above.
`--iterations` and `--precision 32|64` support convergence and precision comparisons. `--tolerance` defaults to zero,
running every iteration without a convergence reduction; a positive distance enables early stopping.

The elastic example uses `E=10000 Pa` for both drop and load/unload scenarios; the `VBD.Elastic` material defaults to
`100000 Pa`. Override this with `--youngs-modulus`. Higher stiffness reduces deformation and requires more work to converge.
`--substeps` increases temporal resolution and reduces implicit integration damping at the cost of more work per frame.

体积加载卸载场景固定底部顶点，在前半段向顶部施加总计 `--load-force` N 的水平力（默认 20 N），后半段清除。上面的塑性演示显式降低屈服阈值以产生残余形变；可省略 `--plastic` 对照弹性恢复。JSON 记录卸载前位移、末态残余位移和累积塑性应变。

## 共享仿真资产

`remeshed.obj` 从原始 `jelly.glb` 的 `Jelly_Fluted_12Lobes` 节点生成，采用 2 mm 目标边长进行各向同性重网格化。原始 GLB 保留为来源。新表面包含 12,740 个顶点和 25,476 个三角形；最小三角形角从 0.819 度提高到 17.297 度，最小角低于 5 度的三角形从 2,160 个降到 0。双向顶点及面心采样的最大表面偏差为 0.180 mm，体积变化为 +0.00422%。采样偏差不是严格 Hausdorff 距离上界。

The jelly asset passed closedness, orientation, self-intersection and initial tetrahedron-volume checks. The elastic
example tetrahedralizes this surface with `--tet-max-volume` controlling the volume limit. Archived jelly shell
measurements use its surface triangles. Benchmark JSON files record the asset path and SHA-256. Mesh quality and
surface comparisons are documented in the [asset notes](../../genesis/assets/meshes/jelly/README.md).

重新生成资产及质量报告：

```powershell
python examples/vbd/vbd_remesh.py
```

重网格化后的下落验收在 RTX 4090 上使用 GPU fp32、每帧 5 ms 和两个物理子步，每组运行 200 帧。每帧读取公开顶点，以 float64 计算相对于初始网格的有向体积比：

| 网格 / 求解器 | 刚度 Pa | 全程采样最小体积比 | 采样中最多负体积单元 |
| --- | ---: | ---: | ---: |
| 原始网格 / IPC | 10,000 | -37.0624 | 294 |
| 重网格 / IPC | 10,000 | 0.6510 | 0 |
| 重网格 / VBD | 10,000 | 0.5373 | 0 |
| 重网格 / IPC | 100,000 | 0.9530 | 0 |
| 重网格 / VBD | 100,000 | 0.5505 | 0 |

原始网格 IPC 对照末态仍有 293 个负体积单元，并呈现顶部尖刺；重网格化后该次对照中的尖刺消失。此验证检查采样时刻的体积，覆盖当前下落设置。原网格较差的单元形状是这次问题的重要因素；这些数据并不能把所有视觉伪影都归为四面体翻转。结果及逐帧 CSV 见 [重网格验收](results/remesh_validation.json)，图像见 [IPC 新旧网格对比](results/remesh_ipc_comparison.png)。

同一新资产还完成了 200 帧 [薄壳下落](results/remesh/vbd_shell_jelly.json) 和 [20 N 塑性加载卸载](results/remesh/vbd_elastic_load_unload.json)。薄壳末态最低点为 0.170 mm，高于 0.125 mm 半厚度；加载卸载末态最小体积比为 0.0203，记录到正的累积塑性应变。加载卸载的位移同时包含重力和残余运动。

## 接口与参数映射

`gs.materials.VBD.Cloth` 创建三角形薄壳，`gs.materials.VBD.Elastic` 创建四面体弹性体。两者都返回 `VBDEntity`。`gs.morphs.TetrahedralMesh` 可直接提供体积网格；封闭表面网格也可复用 Genesis 的四面体化流程。薄壳保留表面三角形拓扑。

| 需求参数 | Genesis 接口 |
| --- | --- |
| `youngs_modulus_pa`, `poisson_ratio`, `density_kg_m3` | 材料 `E`, `nu`, `rho` |
| `self_friction` | 材料 `friction_mu`，同时用于地面和自接触 |
| `shear_stiffness_n_m`, `thickness_m` | Cloth 的 `shear_stiffness`, `thickness` |
| `bending_model`, `bending_rigidity_nm` | Cloth 的 `bending_model`, `bending_rigidity` |
| `bending_plasticity` | Cloth 的 `is_bending_plasticity_enabled` |
| `bending_yield_angle_deg`, `bending_plastic_hardening` | Cloth 的同名字段 |
| `volume_plasticity` | Elastic 的 `is_volume_plasticity_enabled` |
| `volume_yield_strain_percent`, `volume_plastic_hardening` | Elastic 的同名字段 |
| `contact_d_hat_m` | `gs.options.VBDOptions(contact_d_hat=...)` |
| `dt`, `gravity` | `gs.options.SimOptions(dt=..., gravity=...)` |
| `tet_max_volume_m3` | 弹性体示例的 `--tet-max-volume`，实际网格规模记录在输出中 |

```python
entity.set_vertex_constraints([0, 1])
entity.set_vertex_forces([[1, 0, 0], [2, 0, 0]], [2, 2])
entity.clear_vertex_forces()
state = entity.get_state()
scene.reset()
```

点力单位为 N，采用世界坐标。每次 `set_vertex_forces` 替换所选环境中该实体的完整力集合，同一顶点的重复项相加，其余顶点清零。力持续到下一次替换或清除，更新力保留位置、速度和塑性历史。控制接口支持 `envs_idx`；顶点索引均为实体局部索引。`get_state().pos`、`vel` 和 `forces` 始终为 `[B, N, 3]`。

构建前设置的控制进入初始快照。`scene.reset()` 恢复初始位置、速度、固定点、力和塑性历史；也可通过 `scene.get_state()` 保存快照，再传给 `scene.reset(state)`。实体的 `surface_triangles` 和 `vgeoms[*].sim_verts_idx` 提供表面映射。

## 材料与接触约定

The solver minimizes implicit Euler inertia, material and contact energies. Graph coloring permits parallel local
gradient and 3-by-3 Hessian evaluation, followed by updates in color order. Local energy backtracking is optional;
`--line-search` enables it. A zero convergence tolerance runs the configured iteration count.

`vbd_elastic.py --material-model stable_neohookean|log_neohookean` selects the VBD constitutive law and defaults to
`stable_neohookean`. The material API is `gs.materials.VBD.Elastic(model=...)`. Benchmark output records the selected
model. The FEM/IPC baseline supports `stable_neohookean`.

Both solid models use `mu=E/(2*(1+nu))` and `lambda=E*nu/((1+nu)*(1-2*nu))`.
The log Neo-Hookean energy is `mu/2*(||F||^2-3)-mu*log(J)+lambda/2*log(J)^2` and requires `J>0`.
It increasingly resists volume collapse; zero or negative volume halts the simulation. Local line search can reject
inverted trial updates. Stable Neo-Hookean permits recovery through zero and negative volume:
The energy per reference volume is `mu/2*(||F||^2-3)-mu*(J-1)+(lambda+mu)/2*(J-1)^2`, where `J=det(F)`.
The volumetric coefficient preserves the small-strain interpretation of `E` and `nu`. Energy and restoring forces
remain defined at zero and negative signed volume, permitting inversion recovery subject to constraints, loading and
convergence. Plasticity uses logarithmic deviatoric strain return mapping, with yield threshold `log(1+percent/100)`
and accumulated-strain hardening. Plastic flow preserves volume and updates only after positive orientation returns.

Volume containment skips tetrahedra with `abs(J) < VBDOptions.contact_min_volume_ratio`, which defaults to `0.01`.
The ratio uses each tetrahedron's original reference volume, so uniform scaling preserves the decision. Increasing
the threshold suppresses unreliable contacts near collapse at the cost of missed contacts in compressed regions;
zero retains every nonzero-volume tetrahedron. Material forces remain active for skipped tetrahedra.
The surface search processes up to 500 candidates and discards the current match when this budget is exceeded.

薄壳使用平面应力膜能量；独立剪切刚度添加到 Green 应变的偏量能量中。完整厚度 `t` 决定面密度 `rho*t`、膜刚度和两侧各 `t/2` 的接触半径。离散弯曲使用静止二面角和 `3*D*edge_length^2/(area_left+area_right)` 的铰链系数。折痕通过更新参考二面角产生，硬化按累积塑性转角增加屈服阈值。`bending_model=None` 使弯曲能量及其历史更新全部关闭。

Solid contacts use containment and swept vertex-face queries; shell contacts use vertex-face and edge-edge candidates.
Static planes use analytic contact. Penalty forces include regularized friction, with `contact_d_hat` setting an
activation gap outside the physical surface independently of shell thickness. A shell candidate limits each participating
vertex's motion from the latest query position to `0.45*max(distance-thickness, 0)`, further bounded by the search radius.
Vertices take the smallest candidate bound. Exhausting a bound requests another search. Zero clearance produces a zero
bound, which can lock motion because the bound also limits separation and tangential motion. For self-contact, the
initial guess retains predicted displacement within these bounds. External shell contacts retain the preceding
configuration to limit transient contact impulses. Common translation is handled separately for entities with free vertices
throughout and whose retained shell candidates lie within the same entity.

Prediction limits the initial deformation change under concentrated loads while retaining the complete inertial target
in the objective. Local solid updates can cross zero volume during recovery. A vertex's incident tetrahedra are excluded
from its self-contact candidates, allowing recovery through an incident tetrahedron's opposite face.

The integration supports contact among VBD entities and fixed rigid Plane entities at their build-time poses. The viewer
uses Rasterizer. Unsupported differentiation, two-way contact with other solvers and other renderers raise errors.
NaN, infinity and invalid internal state halt the step and require resetting every environment.
Optional `--contact-diagnostics` reports interpenetration warnings; it is disabled by default.

## 性能复现

```powershell
python examples/vbd/vbd_shell.py -g --benchmark
python examples/vbd/vbd_shell.py -g --scenario hang --benchmark
python examples/vbd/vbd_elastic.py -g --youngs-modulus 100000 --benchmark
python examples/vbd/vbd_elastic.py -g --solver ipc --youngs-modulus 100000 --benchmark
```

`--benchmark` 预热 20 个采样帧，然后重置并重复 5 次。默认每次推进 1 秒仿真时间（200 帧），每帧同步后计时，输出平均耗时、P95、总墙钟时间及独立的初始化时间。初始化从 `gs.init()` 返回后开始，包含网格预处理、场景构建和首次内核编译；总墙钟时间包含五轮推进和轮间重置，预热单独排除。报告写入 `out/vbd/`，可用 `-o` 更改目录。

弹性体对照采用相同资产提取、四面体化输入、材料参数、物理子步长度和接触距离。每个采样帧是 5 ms、两个 2.5 ms 物理子步。当前 IPC 接入每次调用使用 `SimOptions.dt`，因此脚本为 IPC 构建 `dt=0.0025, substeps=1` 的场景，并在一帧内调用两次 `scene.step()`；VBD 使用 `dt=0.005, substeps=2` 并调用一次。JSON 同时记录 `dt/substeps`（采样帧）和 `scene_dt/scene_substeps`（实际场景设置）。

VBD defaults to stable Neo-Hookean and also supports log Neo-Hookean. FEM/IPC uses stable Neo-Hookean, with contact laws
and convergence rules specific to each integration.
JSON `precision` describes Genesis state precision; the IPC native exchange interface uses float64. Comparisons measure
runtime and deformation over equal physical durations. Archived results retain their recorded material variants and
settings. The FEM/IPC baseline covers elastic drop scenes; plasticity and loading/unloading are validated separately.

IPC 使用 libuipc 0.0.25 的原生默认求解配置：Newton 最多 1024 次、速度容差 0.05、线搜索最多 8 次、`fused_pcg` 线性求解容差 0.001。软件版本、完整采样设置及相关默认值见 [environment.json](results/environment.json)。

以下为 2026-10-23 **原始 GLB 网格、回弹修复前的初版性能记录**，用于保留初版对照；当前求解器及仿真网格均已变化，需用上述命令重新计时。设备为 RTX 4090、Ryzen 9 7950X、Windows，关闭查看器。每轮 200 帧、5 轮，共推进 5 秒物理时间；表中耗时对应一个 5 ms 采样帧，含两个物理子步。弹性体使用 `E=100000 Pa`。

| 场景 | 网格规模 | 平均 ms/帧 | P95 ms/帧 | 五轮墙钟 s | 初始化 s |
| --- | --- | ---: | ---: | ---: | ---: |
| [VBD jelly 薄壳](results/vbd_shell_jelly.json) | 13,442 顶点 / 26,880 三角形 | 357.13 | 420.18 | 357.14 | 14.49 |
| [VBD jelly 弹性体](results/vbd_elastic_drop.json) | 22,212 顶点 / 96,745 四面体 | 406.61 | 438.59 | 406.63 | 18.83 |
| [FEM/IPC jelly 弹性体](results/ipc_elastic_drop.json) | 同上 | 110.35 | 193.71 | 110.49 | 12.89 |

初版 VBD 弹性体在这组设置下耗时约为 FEM/IPC 的 3.68 倍。该次 VBD 弹性体末态最小体积比为 0.426，最低点为 1.14e-6 m；薄壳末态最低点为 0.164 mm，高于半厚度 0.125 mm。所有计时运行均启用自接触、地面接触和摩擦。

原始网格的单次 1 秒 IPC 下落诊断发现 232 / 96,745 个四面体的有向体积为负；直接检查原生 float64 顶点得到相同数量，排除了公开 fp32 状态转换作为原因。最小体积比约为 -22.11，原生与公开顶点最大差约为 3.73e-9 m，详见 [ipc_quality.json](results/ipc_quality.json)。该诊断独立于五轮计时运行，属于原始网格的历史记录。

## 自动验收

```powershell
python -m pytest -n 0 tests/deformable/test_vbd.py --backend cpu
python -m pytest -n 0 tests/deformable/test_vbd.py --backend gpu
```

测试覆盖单环境和两个环境：自由落体、固定点、重复力求和、运行中力替换与持续作用、分环境重置、地面与自接触、弹性回弹、密集表面的静止形状、独立实体的运动、摩擦、材料参数响应、两类塑性卸载、屈服与硬化、关闭弯曲，以及失败后重置。接触测试采用 4 个子步以分辨小弹性体的回振。CPU 使用双精度，GPU 使用单精度。渲染可用上述 `-v` 命令检查。

重网格化前的求解器验证（2026-10-23）：CPU fp64 与 GPU fp32 各 8 项通过。本地相机检查确认原始 jelly 的 VBD 网格随仿真更新，原 FEM 网格的更新也通过。已有 FEM 渲染快照测试因远程参考图片下载失败而未完成。

原始网格在回弹修复后另运行 200 帧，采用下落示例默认的 `E=10000 Pa`、2 个子步和 20 次迭代。首次接触约在 0.055 s，首次压缩极值在 0.075 s，随后质心回升 2.11 mm，上升速度峰值为 0.105 m/s；最低顶点全程高于地面 2.44e-6 m，末态最小体积比为 0.00387。数据按顶点质量加权，见 [回弹验收](results/vbd_elastic_rebound.json) 和 [逐帧运动记录](results/vbd_elastic_rebound.csv)。这次运行用于原始网格的形变与回弹验收，前面的五轮性能表保留初版计时。

原始网格在回弹修复后的 [20 N 加载卸载](results/vbd_elastic_load_unload.json) 已复跑 200 帧，刚度为 `100000 Pa`，屈服应变设为 1%，末态最小体积比为 0.104。顶部平均位移在卸载前为 48.2 mm，末态为 53.3 mm；该位移包含重力和未消散的运动，塑性残留由上面的无重力单元测试单独验证。[薄片折叠](results/vbd_shell_fold.json) 也复跑 200 帧，含塑性折痕和自接触，末态最小高度 0.432 mm，大于半厚度 0.125 mm。

算法参考：[Gaia](https://github.com/AnkaChan/Gaia)。性能对照使用当前 Genesis FEM/IPC。

## 实现与验收要求

在 Genesis 中实现 GPU VBD solver，支持本仓库的薄壳与体积软体，以及下列材料参数。目标是在保留功能和合理变形效果的前提下提高仿真速度。

只做 solver 及必要的 Genesis 接入，不涉及前端、HTTP/WebSocket、文件导入导出或渲染。参数需要实际生效、调节方向合理；不要求精确标定真实材料，也不要求数值复现现有 IPC。材料模型和参数转换方式可自行选择，但应简要说明映射关系。

## 1. 材料参数

以下字段和默认值保留需求文档中的参数语义；Genesis 对应接口见前面的映射表。

| 参数 | 适用类型 | 默认值 | 要求 |
| --- | --- | --- | --- |
| `youngs_modulus_pa` | 两者 | 100000 | 弹性刚度，Pa；越大越难变形。 |
| `poisson_ratio` | 两者 | 0.25 | 参与材料响应调节；体积模式控制横向收缩和可压缩性，薄壳可沿用系数换算语义。 |
| `density_kg_m3` | 两者 | 360 | 密度，kg/m³；决定质量和惯性，薄壳结合厚度计算质量。 |
| `self_friction` | 两者 | 0.4 | 自接触与地面接触的摩擦系数；0 表示无摩擦。 |
| `shear_stiffness_n_m` | 薄壳 | 8 | 面内剪切刚度，N/m；独立于杨氏模量、泊松比和厚度调整。 |
| `thickness_m` | 薄壳 | 0.00025 | 完整厚度，m；影响质量、面内拉伸刚度和接触厚度，不能误作半厚度。 |
| `bending_model` | 薄壳 | `DiscreteShellBending` | 支持离散壳弯曲；`null` 表示关闭弯曲能量。 |
| `bending_rigidity_nm` | 薄壳 | 5e-8 | 独立弯曲刚度，N·m；越大越难弯曲，允许为 0。 |
| `bending_plasticity` | 薄壳 | false | 开启后支持永久折痕；不要求面内永久拉伸或剪切。 |
| `bending_yield_angle_deg` | 薄壳 | 10 | 参考屈服折角，度；越小越容易留下折痕。 |
| `bending_plastic_hardening` | 薄壳 | 0 | 折痕积累后提高后续屈服门槛；0 表示不硬化。 |
| `volume_plasticity` | 体积 | false | 开启后支持永久变形；塑性保持体积，不要求永久体积压实。 |
| `volume_yield_strain_percent` | 体积 | 10 | 参考屈服形变百分比；越小越容易产生永久变形，不是残留变形上限。 |
| `volume_plastic_hardening` | 体积 | 0 | 塑性积累后提高后续屈服门槛；0 表示不硬化。 |

数值必须有限。杨氏模量、密度、厚度和剪切刚度为正；泊松比范围为 `(0, 0.5)`；屈服折角范围为 `[0, 180]`。摩擦、弯曲刚度、屈服形变和两种硬化参数允许为 0，不能为负。

## 2. Solver 必须支持的行为

- **几何与变形：**三角形薄壳、四面体体积软体；支持大转动、拉伸、压缩、剪切及薄壳弯曲。网格生成可复用 Genesis，`tet_max_volume_m3` 属于预处理参数。
- **推进：**三维重力 `gravity`，默认 `(0, 0, -9.81)`；可调时间步 `dt`，默认 0.005 s。允许内部子步，一次推进的总时间仍为 `dt`。
- **固定点：**支持指定顶点保持位置不变。变更固定点允许重建并重置，不要求无重置切换。
- **点力：**按顶点施加世界坐标三维恒定力，单位 N；同一点多个力相加。支持运行中替换或清除力，不重置形状、速度和塑性历史。固定点不因外力移动。
- **接触：**支持静态地面、自碰撞和摩擦，覆盖薄壳折叠接触与体积软体大变形。支持可调接触作用距离，对应 `contact_d_hat_m`，默认 0.0005 m；说明其在新算法中的含义，与材料厚度区分。无需照搬 IPC，但不能靠关闭接触提速，也不能忽略明显穿透。
- **塑性：**关闭时保持弹性；开启后卸载仍有残留形变。塑性弯曲不能重复叠加两份弯曲刚度；`bending_model=null` 时全部弯曲及折痕参数不参与。历史不能随每次 VBD 迭代重复累计，重置时清空。
- **状态与稳定性：**可读取变形顶点，保留拓扑和表面顶点映射；支持重置初始位置、速度与塑性历史。无效状态应报错并停止，失败步不能作为成功结果发布；允许要求重置，不强制实现整步回滚。
- **实现与性能：**实际使用 VBD 求解，接入 Genesis 的场景、材料和状态接口。主要推进在 GPU 上完成，避免逐顶点 Python 循环和不必要的 CPU/GPU 同步。数值精度、迭代数和加速方式由实现选择并记录。

## 3. 简单验收与交付

1. 薄壳、体积两种模式都能构建、推进、读取状态和重置；不依赖本仓库前端。
2. 对材料参数做有针对性的对照，确认软硬、剪切、弯曲、质量和摩擦调节实际生效；塑性用加载—卸载检查残留形变、屈服和硬化效果。
3. 检查固定点、动态增删点力、地面接触、薄壳折叠自接触；检查关闭塑性、关闭弯曲和重置后的行为。
4. 提供可复现性能结果：记录网格规模、材料、时间步、子步、迭代数、精度、接触设置，以及预热后的平均耗时和 P95。初始化单独计时；按相同仿真时长和相近效果比较，不能只比较迭代次数或不同时间步的单步耗时。暂不设未经确认的 FPS 或加速倍数门槛。
5. 交付 Genesis 内的实现、最小运行示例、必要测试和简短的使用/性能说明。两类塑性、自碰撞等完整要求不能仅以 TODO 代替。

The target performance device is an RTX 4090. Shell examples use `genesis/assets/meshes/shirt.glb`; the archived
shell optimization measurements and volume examples use `genesis/assets/meshes/jelly/remeshed.obj` (12,740 surface
vertices, 25,476 triangles). Default tetrahedralization retains those surface vertices and adds interior vertices;
JSON reports record the resulting size. Folding uses a separate rectangular sheet.
