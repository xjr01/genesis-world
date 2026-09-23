# P2 设计文档：有外表面的杯壁"壳模型"边界（multiflow PBF+ST teapot effect）

> 来源：agent-43 explore 调研（2026-09-01），调研依据：multiflow 副本 `boundaries.py` / `pbd_solver.py` / `options/solvers.py` / `mf12_pbf_st_tilt_pour.py`；同事 PBSTF `surface tension/genesis-world-main/genesis/engine/boundaries/static_colliders.py`、`pbstf_solver.py`、`examples/teapot/`。

## 0. 结论速览

1. **路线：解析壳模型**。把 `TiltedCylinderBoundary` 升级为**新类** `TiltedCylinderShellBoundary`（旋转体精确 SDF，截面 L 形=两矩形并集），不移植 mesh SDF。新 option `boundary_pitcher_shell` 与旧类并存，零回归。
2. 壳的数学是**精确 SDF**：旋转体 3D SDF 严格等于 (r, s) 半平面上的 2D 截面 SDF；唇口锐边天然正确，唇口圆角用可选偏移量 ρ 免费获得。
3. 大杯**保持现有内腔 clamp 不动**；桌面 adhesion 用 P1 的 PlaneBoundary，仲裁=各边界查 |C| 取最小。
4. 壳厚维持视觉网格的 0.016（=2 粒子直径=support radius）；防隧穿依赖 P1 的迭代内 impose。
5. mf12 场景需加 **Phase E"停场挂流段"**才能让水落到桌面（现有几何下外壁滑下的水仍落入大杯）。
6. 最大风险：薄壁隧穿与跨墙 ST/密度互拉（support radius 恰等于壳厚，临界）。

## 1. 路线选型理由（选 a 解析壳，不选 b mesh SDF）

- demo 目的匹配：要"越唇→沿外壁→滴落"物理叙事，不要几何保真；小杯是程序生成的圆柱壳（`multiflow/assets/gen_cup.py`，t-wall 0.016），解析模型与视觉网格完全同形。
- 动画免费：继承现有 `set_pose` qd field（boundaries.py:190-201）+ `solver.set_pitcher_pose` 每帧驱动（pbd_solver.py:169-172，mf12 :406）。mesh 路线需整套移植 colliders pos/quat 场与局部帧变换。
- 薄壁分辨率是 mesh 路线硬伤：壳厚 0.016，SDF 需 res≥150-200³ 才能 3-4 cell 跨壁；三线性插值梯度噪声会让挂流膜抖动。解析法线精确。
- 与 P1 架构同构：P1 的"迭代内 impose_pos + 桌面 PlaneBoundary + 遍历多边界 adhesion 取最近"建立在解析边界对象 + `adhesion_query(pos)->(ok,C,n)` 接口上。
- 帧耗时：解析壳 ~30 flop；mesh 查询=坐标变换+27 场读取插值+梯度。
- 移植量：~150 行 vs 600+ 行。

## 2. 壳模型数学细节

### 2.1 参数化

- `origin` O=内腔底面圆心，`axis` a（单位向量，底→口），沿用现有 qd field pose；
- `R_in`（=现 radius 0.16）、`t_wall`（壁厚，R_out=R_in+t_wall）、`L`（内底→唇口沿轴距离 0.45）、`t_bottom`（底厚 0.016）；
- 可选 `lip_round` ρ（默认 0，**建议启用 0.008**=0.5·t_wall）。

局部坐标：`rel = p − O; s = rel·a; rad = rel − s·a; r = |rad|; e_r = rad / max(r, EPS)`

### 2.2 精确 SDF：(r,s) 截面 L 形 = 两矩形并集

```
B_wall  = [R_in, R_out] × [-t_bottom, L]      （侧壁带）
B_floor = [0,    R_out] × [-t_bottom, 0]      （底整圆盘）
sd_box(p2, c, h)：q = |p−c| − h；= |max(q,0)|₂ + min(max(q.x,q.y),0)
d2(r, s) = min(sd_box((r,s), B_wall), sd_box((r,s), B_floor)) − lip_round
```

3D signed distance 严格等于截面 2D SDF（r>0 处处精确）；唇口凸角的圆弧等距面自动正确。`lip_round>0` 时整体外扩 ρ、凸唇角变圆角、法线连续。

```
(r*, s*) = 达到 min 的 box 的最近点；closest = O + s*·a + r*·e_r
n = ∇d = n_r·e_r + n_s·a
```
- r<EPS（轴线上）时 e_r 取固定 fallback（参考 `_cone_radial_direction` :206-212 写法）。
- min 平局（壁中线）取**靠内一侧**（减少穿壁一半粒子被吐到外壁的误判）。

### 2.3 `impose_pos_vel` 新语义（关键区别）

```
d = d2(r,s) − lip_round     （d<0 ⇔ 粒子在壳固体内部）
if d < 0:
    n = 外指梯度
    if v·n < 0: v −= (1+restitution)·(v·n)·n
    p = closest             （沿梯度推到最近表面：内/外/唇/底自动）
else:
    不约束                   （壳外是自由空间——核心语义变化）
```
- 腔内冲内壁→穿进浅层→最近面仍是内壁→推回腔内（与旧行为一致）；
- 穿越超壁厚一半→最近面变外壁→被推到外壁=**泄漏（隧穿）**；
- 壳外粒子（贴外壁滑行、唇上爬行）**完全不被 impose 触碰**——挂流膜位置约束全交 adhesion；
- P1 的迭代内 `impose_pos` = 同上去掉速度部分；旧底部平面特判（:220-225）删除（被 B_floor 覆盖）。

### 2.4 `adhesion_query` 双侧版（约定切换）

切换为 **PBSTF 固体中心约定**（与 pbstf_solver.py:1009-1013 一致）：
```
ok = |d| ≤ particle_radius    （附着壳层内外两侧对称，唇口上表面/外壁/杯底下全 ok=True）
C  = d                        （固体外为正，表面为 0）
n  = 外指梯度
```
约束公式不动（pbd_solver.py:1397-1405）：`dpos += −C/denominator/mass·n`。**接口对齐点**：大杯 `CylinderBoundary.adhesion_query` 也要统一到同一约定，P1 的 min|C| 仲裁才有意义（与 P1 联调敲定）。删除旧版"s≤L 才 ok"的人工裁剪。

### 2.5 `in_keep_region` / boundary_group 兼容

现 keep 判据上限 R_in+band+margin=0.180，贴外壁粒子 r∈[0.176,0.180] 恰在边界上必误转移。改：
```
keep ⇔ s ≤ L+margin 且 r ≤ R_out+band+margin 且 s ≥ −t_bottom−margin
```
加豁免：`s > L+margin` 的粒子若 |d|≤r_p（正贴唇口/外壁爬行）**不转移**。（若 P1 adhesion 是全局遍历不按 group，豁免可省；推荐 adhesion 全局最近查询、group 只管 impose/clamp 所有权。）
膜滑到杯底沿以下→转 group 0→大杯 above-rim clamp 对 r≈0.5 的粒子不受力→自由落体到桌面，行为正确无需处理。

## 3. 大杯与桌面配合

- 大杯**不需要壳**，保持内腔 clamp（剧情无外壁戏份；加壳只增唇口 adhesion 交叠复杂度）。
- 桌面 PlaneBoundary 与壳 adhesion 竞争：取 |C| 最小者；近平局（<0.1·r_p）取法线更竖直者（桌面）。impose 侧无交叠，顺序执行幂等。

## 4. 参数预期与场景几何

### 4.1 壳厚与附着层
- t_wall=0.016（=2·ps，与视觉网格同厚；不加厚——物理壳外凸会肉眼可见悬空）；
- 附着层=粒子半径 r_p=0.004 内外对称；挂流膜厚=1 层粒子；
- 临界：support radius=0.016=t_wall，跨墙核交互在"刚好没有"的临界点（见 R2）。

### 4.2 挂流成立条件
- `wall_adhesion_compliance`：**5~10** 起扫（比现 20 更粘）；
- `wall_friction`：**0.05~0.2**（太大膜冻在唇口，太小膜加速脱离）；
- 挂流期流速要小（残余奶小角度缓慢渗出；大流量只会冲成自由射流不贴壁）；
- 外壁近竖直（小倾角 15-30°）膜沿壁长距离下滑；近水平时膜在唇附近直接滴落。

### 4.3 场景几何必须调整（关键发现）
- 直立位唇外缘 x=0.104 < 大杯内半径 0.20——唇口正下方是大杯内部；
- 最大倾斜时唇口附近就是外表面最低点（x≈0.10），膜滴落 r=0.10<0.20 **仍落进大杯**；
- **叙事补丁：Phase E 停场挂流段**。小杯 phase D 已平移 +0.30 到 x≈0.58（完全在大杯外）；加 15~25° 小角度二次倾斜（3-4s），残余奶缓慢越唇、沿近竖直外壁滑下、滴落桌面。主倾倒（进杯）与挂流（落桌）解耦，三路指标干净。无需改 PX/PZ/PIVOT。
- 可选：停场高度 z≈1.33 滴落 1.3m 着桌 ~5m/s 偏快；要"挂滴"美感可降停场高度到 ~0.6。

## 5. 风险清单

- **R1 薄壁隧穿**（高关注中概率）：单 substep 穿越>t_wall/2=0.008（v>3.84m/s）时 impose 把粒子推到外壁（泄漏）。缓解：① P1 迭代内 impose；② 监控 n_leak；③ 兜底：上一步在腔内+当前最近面是外壁→强制投内壁（~10 行，泄漏率>0.1% 才上）。
- **R2 跨墙 ST/密度互拉**（中概率临界）：贴内壁与贴外壁粒子间距=t_wall=support radius 临界；float 抖动+附着层可缩到 ~0.012<support。PBSTF 对策是 separates 检查（两侧粒子距面≤r_p 且法线点积<0⇒隔离）。**不预防性实现**；若 n_outer_wall 膜粒子密度持续高于 rho_rest 5% 或出现穿墙拉拽，再在密度回调与 ST 筛选加解析版 separates（各 ~5 行）。
- **R3 group 转移瞬态**（低）：keep 区改造+豁免后，膜过杯底沿整片翻 group 只改 impose 所有权，adhesion 全局查询则无感；抽帧确认翻转前后速度连续。
- **R4 唇口法线不连续**（低）：lip_round=0.008 直接启用，消除翻唇抖动。
- **R5 性能**（很低）：壳查询 ~30 flop 零增量。
- **R6 旧行为回归**：新类与旧类并存，新 option `boundary_pitcher_shell`（10-11 元组），mf12 基线不动。

## 6. 验证方案

### 6.1 单元级（mf14_shell_unit.py 小场景）
- **U1 静态壳挂流**：壳固定倾角 20°，唇口上方 2cm 每秒投 ~500 粒子奶团。判据：① n_leak=0；② 投放后 1s 内 n_outer_wall>200 且连续 30 帧不归零；③ 附着粒子 |C| p95 ≤1.2·r_p；④ 膜切向速度向下 |v_t|∈[0.05,1.0]m/s；⑤ 膜过杯底沿后 n_outer_wall 指数衰减、脱离粒子弹道 ±20%。
- **U2 内腔零回归**：直立小杯装水静置 5s，壳开/关 n_in_pitcher/KE/泄漏曲线差 <1%。
- **U3 唇口圆角消融**：lip_round∈{0,0.008} 各跑 U1，比翻唇瞬间附着粒子速度跳变量。

### 6.2 叙事级（mf14 = mf12 + 壳 + Phase E）
CSV 新列：`n_outer_wall`（|d|≤r_p 且最近特征为外壁/唇侧）、`n_table`（z≤2·ps 且在大杯足印外）、`n_milk_cup`（牛奶粒子在大杯腔内数）、`n_leak`（d<−0.1·ps 计数）、保留 n_adh。
通过判据：
1. mf12 原判据全维持 PASS；
2. 挂流段（二次倾斜+2s 内）n_outer_wall 峰值 ≥300 且持续 ≥1s；
3. 挂流段结束后离开小杯的残余奶中 n_table/(n_table+n_milk_cup) ≥60%；主倾倒期 n_milk_cup 占比 ≥90%；
4. n_leak 全程=0；nan=0；挂流段 KE 在二次倾斜结束后 5s 内衰减到峰值 <10%；
5. 帧耗时增幅 <10%。
抽帧位：① 首个越唇粒子接触外壁；② 膜滑到外壁中段（膜-壁视觉无缝隙）；③ 膜脱离杯底沿；④ 首滴着桌；⑤ 结束后桌面奶滩分布。

## 附：实现触点预估

| 改动 | 位置 | 量级 |
|---|---|---|
| 新类 `TiltedCylinderShellBoundary`（2D box-SDF 并集、impose×2、adhesion_query、in_keep_region） | `boundaries.py` 新增 ~150 行 | 核心 |
| 新 option `boundary_pitcher_shell`（10-11 元组）+ 校验 | `options/solvers.py` PBDOptions | ~20 行 |
| `setup_boundary` 分支建壳；`set_pitcher_pose` 透传（不变） | `pbd_solver.py:157-172` | ~15 行 |
| adhesion/friction/stats dispatch 符号约定统一（配合 P1） | `pbd_solver.py:1146-1153, 1174-1181, 1389-1396` | 接口对齐 |
| group 转移 s>L 附着豁免 | `pbd_solver.py:1112-1115` | ~5 行 |
| mf14 脚本：shell option、Phase E、新 CSV 列与判据 | 复制 mf12 改造 | ~80 行 |
| （条件触发）解析版 separates | 密度回调+ST 筛选 | 各 ~5 行 |

**与 P1 联调点**：① adhesion_query 全局统一固体中心 signed 约定（大杯也改）与 min|C| 仲裁；② adhesion 查询按 group 分发还是全局遍历（推荐全局最近查询）；③ 迭代内 impose_pos 对壳语义的调用点确认。
