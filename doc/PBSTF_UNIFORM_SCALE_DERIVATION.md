# PBSTF 整场景等比例缩放：公式推导

迭代数变化的非线性关系与dt联合公式已补充在 `PBSTF_ITERATION_SCALING_DERIVATION.md`；本文件的严格齐次相似结论默认各约束执行次数不变。

2026-09-22。仅讨论用户明确的整场景等比例缩放：几何、粒子直径、support、初始粒子位置、碰撞几何一起缩放，粒子数量及对应关系不变。分析对象为当前 multiflow 中已接入的同事 PBSTF。本文不修改引擎或默认参数，也不把该结论套用于 IPBF 的 alpha。

## 1. 定义与结论

设长度缩放比 s=L_new/L_old=ps_new/ps_old，时间缩放比 tau=T_new/T_old，密度 rho0 保持不变。要求轨迹满足：

    x_new(t_new) = s * x_old(t_new/tau)
    v_new = (s/tau) * v_old
    a_new = (s/tau^2) * a_old

这里 x 可相对任意固定缩放中心定义。时间变换同时作用于子步、运动关键帧、重力预测、初始速度、运动 collider 及外部驱动。

完整相似配方：

| 参数/量 | 新值/旧值 |
|---|---:|
| 所有长度、ps、support、hash cell、SDF体素长度、接触偏移 | s |
| rho0、表面目标系数.7 | 1 |
| 单粒质量、默认质量 | s^3 |
| 子步dt、帧dt（substeps不变）、动作时刻、总时长 | tau |
| 线速度 | s/tau |
| 重力及其他加速度 | s/tau^2 |
| 姿态角 | 1 |
| 角速度 | 1/tau |
| density_compliance | s^-2 |
| surface_tension_compliance（面积约束） | s^2 |
| surface/interior_distance_compliance | 1 |
| collider_adhesion_compliance（法向距离约束） | 1 |
| PBSTF surface/interior_viscosity（XSPH每步滤波系数） | 1 |
| 每步速度保留系数q、接触切向速度衰减比例 | 1 |
| 每子步迭代数、topology interval、偶数轮距离调度 | 1 |
| 角度阈值、协方差特征值比例阈值、邻域容量 | 1 |
| 原始色场法线长度阈值 | s^-1 |

上述是匹配离散步数的相似变换；在固定拓扑、阈值正确缩放、相同运算顺序、无新外部机制条件下，各步位置修正也严格缩放为s倍。实际f32、重排和阈值分支使仿真只能近似一致。

固定重力时 tau=sqrt(s)。固定播放时长时 tau=1，必须同时令 g_new=s*g_old。固定物理张力系数的毛细主导相似过程则 tau=s^(3/2)；三种要求不同。

## 2. 从实际约束求解公式推导 compliance

当前 `pbstf_solver.py` 密度、面积、距离和壁附着均使用以下结构（以单个约束表示）：

    lambda = -C / (c/m0 + sum_j |grad_j C|^2/m_j)
    delta x_j = lambda * grad_j C / m_j

其中 c 是代码参数，m0 为标定默认质量。面积约束分母没有显式dt^-2，也没有XPBD累计lambda。不能将该c直接视作物理柔度。

若 C(s*x)=s^p*C(x)，则 grad C 缩放为 s^(p-1)。质量标定给出 m_new=s^3*m，因此梯度分母项缩放为 s^(2p-5)。令 c/m0 同比例变换：

    c_new/(s^3*m0) = s^(2p-5) * c/m0
    => c_new = s^(2p-2) * c

此时 lambda_new=s^(5-p)*lambda，而 delta x_new=s*delta x，完成逐约束证明。

- 密度 C=rho/rho_target-1 无量纲，p=0，c_density_new=c_density/s^2。
- 面积 C=sum triangle_area，p=2，c_area_new=s^2*c_area。
- 距离 C=|x_i-x_j|-d0，p=1，c_distance 不变。
- 壁附着 C=(x-anchor) dot normal，同样 p=1，c_adhesion 不变。

这里没有额外的tau因子：时间缩放已经通过初始速度、预测加速度和dt共同实现，使预测位置本身缩放为s倍。再给这些c乘dt^-2会重复补偿，破坏相似性。

同事的W=h^-3*w(r/h)，所以W_new=s^-3*W、gradW_new=s^-4*gradW；build的m=rho0/max(sum W)自动给出m_new=s^3*m。密度和m/rho*W权重不变。质量标定应重跑，不硬编码旧质量。

## 3. dt为何是平方根或3/2次方

预测式 y=x+dt*v+dt^2*g。要求 y_new=s*y，且dt_new=tau*dt、v_new=s/tau*v，重力项要求：

    tau^2 * g_new = s * g

固定g，得到dt_new=sqrt(s)*dt。全部倒杯动作时刻也乘sqrt(s)，否则流体正确缩放而杯子运动失配。

从物理时间尺度也能得到：

    T_gravity ~ sqrt(L/g)
    T_capillary ~ sqrt(rho*L^3/sigma)
    T_viscous ~ L^2/nu

rho固定。固定g对应tau=s^0.5；固定sigma对应tau=s^1.5；固定运动粘度nu对应tau=s^2。这里是分别保持相应主导机制的相似时间；要让其他机制也相似，必须同步调整它们的物性。

例如固定g且保持所有机制相似，需要sigma_new=s^2*sigma，nu_new=s^1.5*nu。若坚持相同水的g、sigma、nu全部不变，则通常不存在任何dt能让缩放前后动态完全相似：Bond数rho*g*L^2/sigma已经改变为s^2倍。

毛细s^1.5时间尺度可与[Basilisk张力实现](https://basilisk.fr/src/tension.h)交叉验证；该代码的显式稳定步长公式不应直接当作本PBSTF的稳定性定理。PBSTF仍须分别测稳定与精度。

## 4. 物理粘度与代码XSPH不是同一参数

同事速度滤波：

    delta v_i = epsilon * sum_j (m_j/rho_j)*W_ij*(v_j-v_i)

整体缩放后(m/rho)*W不变，速度差乘s/tau，因此epsilon保持不变就让delta v乘s/tau。surface/interior分别保留自己的epsilon。

光滑体积流动下，其等效运动粘度量级为nu_eff ~ K*epsilon*support^2/dt，K依赖核、邻域和分类，非精确物性标定。所以epsilon不变意味着nu_eff_new/nu_eff=s^2/tau。固定g时为s^1.5，恰好符合动力相似需求。

若用户另要求物理nu不变，则小滤波近似下epsilon_new/epsilon=(dt_new/dt)/s^2。这是另一个目标，不能与“相似动画epsilon不变”混用。若同一s/tau目标下擅自选dt_new=eta*dt，则近似epsilon_new/epsilon=eta/tau；强滤波、多模态情况下单一系数不能保证完全等效。

每步速度保留q在匹配步数的相似变换中不变；连续衰减率gamma=-ln(q)/dt则缩放为gamma/tau。若dt_new=eta*dt但目标时间比仍为tau，q_new=q^(eta/tau)。壁摩擦是逐步滤波时也有类似频率依赖，不能仅因其系数无量纲就忽略执行频率。

物理浓度扩散系数D同nu，D_new=s^2/tau*D。若扩散实现是无量纲核权重的每步Jacobi滤波，滤波系数不变；若迁移中直接使用有量纲cubic W而无体积归一化，则必须按该实际公式重新推导，不按变量名猜测。

## 5. 对应物理量与迭代数

在上述轨迹相似变换下，rho固定时：

    pressure_new/pressure = bulk_modulus_new/bulk_modulus = s^2/tau^2
    sigma_new/sigma = s^3/tau^2
    nu_new/nu = mu_new/mu = s^2/tau
    force_new/force = s^4/tau^2

sigma公式由Laplace压力sigma/L与惯性压力rho*(L/T)^2配平得到。代码c_area是面积平方约束的分母参数，不直接等于sigma或1/sigma；这里只说明其整体相似变换对应的物理量比例，不宣称已标定出PBSTF实际sigma。

每子步迭代数应保持不变。局部线性、等质量、独立约束近似下，设G=sum|grad C|^2，则一次投影后：

    C_after ~ [c/(c+G)] * C_before
    C_after_N ~ [c/(c+G)]^N * C_initial

迭代效果通常非线性；实际还存在约束耦合、变化的拓扑和距离开关。因此20轮不等于5轮的简单4倍张力。[XPBD原文](https://matthias-research.github.io/pages/publications/XPBD.pdf)也讨论了传统PBD的时间步和迭代依赖；当前PBSTF并非其带累计lambda的时间归一化形式。

只有在compliance主导分母、每轮弱修正、几何几乎不变时，修正量可近似按N/c累加。若偏离标准时间缩放（实际dt比eta，目标时间比tau），弱修正估算才有：

    c_new ~ s^(2p-2) * (N_new/N_old) * (tau/eta)^2 * c_old

这是基于delta x/dt^2比较等效加速度的局部启发式，不是相似定律；硬约束饱和、薄片、接触和强面积收缩场景不保证适用。首选保持N不变、eta=tau的严格方案。

## 6. 容易破坏缩放的隐藏常数

`_kernel_compute_normals`使用raw_length<=1.0。raw_normal=sum(m/rho)*gradW按s^-1变化，因此阈值应按s^-1变化，或改为比较support*|raw_normal|与固定无量纲阈值（需要单独实现验证）。

同一`gs.EPS`目前用于不同量纲，不能整体只乘一次s。应逐处区分：长度阈值乘s，面积/协方差trace阈值乘s^2，无量纲角度/比例阈值不变；分母阈值则密度乘s^-5、面积乘s^-1、距离乘s^-3，或者改用各自无量纲/相对判断。大范围缩放时浮点精度、网格退化判断、collider的固定padding都需要核查。

包括杯壁厚度与接触锚点在内的全部碰撞长度须同比缩放；SDF保持相同网格分辨率时其体素长度随物体缩放。hash/domain比例、相对接触间隙保持不变。相机位置、near/far、重建网格间距也应相应调整，避免显示差异被当成物理差异。

## 7. 数值例子：整场景缩到0.3倍、重力保持不变

以同事teapot参数作为基准，但必须真的缩放全部几何，而不是只改--scale：

| 项目 | 原值 | 新值 |
|---|---:|---:|
| ps | .0066667 | .002 |
| support | .02 | .006 |
| 单粒质量 | m | .027m |
| 子步dt | .01 | .00547723 |
| density compliance | 33750 | 375000 |
| area compliance | 3/225=.0133333 | .0012 |
| 表面/内部距离compliance | 40/180 | 40/180 |
| 表面/内部XSPH | .2/.05 | .2/.05 |
| 子步迭代数 | 5 | 5 |
| 10s动作 | 10s | 5.47723s |
| 线速度 | v | .5477226v |
| 原始法线阈值 | 1 | 3.333333 |

对应物理sigma比=.09、nu比=.1643168。因此是相似动作，不是相同物理水的缩小实验。当前h=1/960、20轮与该同事基准不是这套严格映射，不能因两个compliance匹配便称整体等效。

## 8. 本轮验证与后续验证建议

本轮用非对称梯度与不等质量构造，对p=0/1/2、s=.25/.3/50逐项检查实际分母与位置修正齐次式；双精度相对误差最大约3e-16。该计算验证代数，不是整引擎仿真通过。

下一步若实施，保留当前已验证场景，复制生成s=.5/1/2三组；统一相似参数与阈值，按相同子步编号比较x_new/s、v_new*tau/s、rho/rho0、表面分类、归一化面积/体积、质心与浓度。先零重力方块，再重力下落，再运动杯；视频按t/tau对齐。接触、拓扑切换附近允许浮点分歧，但应单独报告差异，不能用改迭代数掩盖。
