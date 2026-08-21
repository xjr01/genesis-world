# Implicit Position-Based Surface-Tension Fluid (IPBSTF) 的变分能量与局部求解

流体系统由 $N$ 个粒子组成。在 position-based 框架中，未知量是粒子位置
$\boldsymbol x_i$，记 $\boldsymbol x\in\mathbb R^{3N}$ 为所有位置的堆叠向量。

## Density Constraints

使用 Smoothed Particle Hydrodynamics（SPH）的 cubic spline kernel 估算粒子 $i$ 的密度：

$$
\rho_i(\boldsymbol x)=\sum_{j\in\mathcal N_i}m_jW(\Vert\boldsymbol x_j-\boldsymbol x_i\Vert),
$$

其中 $\mathcal N_i$ 是与粒子 $i$ 的距离小于核半径 $R$ 的粒子集合。密度约束为

$$
C_i^\rho(\boldsymbol x)=\max\left(\frac{\rho_i(\boldsymbol x)}{\rho_i^0}-1,0\right).
$$

这是单边约束，只抵抗体积压缩。边界（solid）粒子同样携带密度约束 $C_i^\rho$，其密度由邻域内流体与
边界粒子的质量共同累加。边界粒子的位置固定，其约束作为单向源进入邻近流体粒子的局部密度能量，使靠近
壁面的流体粒子获得与内部粒子一致的约束曲率。

## Local Surface Mesh

表面张力系数 $\gamma>0$ 时，每次 Newton iteration 都根据当前位置重建局部表面几何。哈希表在时间步开始时
构建；候选对必须在该参考构型中位于核半径内，并且在当前构型中仍位于核半径内。因此，拓扑与三角形几何会
跟随 iteration 更新，同时不需要在 iteration 内重新构建全局哈希表。

表面分类、法向和 one-ring mesh 的构建过程如下：

1. 对每个可移动流体粒子，用邻居在 $18\times36$ 球面方向网格上遮挡的面积判断其是否为表面粒子；加权未遮挡
   比例至少为 $1/9$ 时进入表面候选集合。
2. 用 SPH 核梯度估算法向。当表面粒子没有内部流体邻居时，对表面邻居的协方差矩阵做 Principal Component
   Analysis（PCA），用最小特征值对应的方向稳定法向。
3. 将法向兼容的表面邻居投影到当前粒子的切平面，按极角和半径排序，再按局部空圆角度条件删除被包围的顶点，
   得到有序的 one-ring。至少包含三个顶点且投影面积非零的环才是有效拓扑。

在上述表面重建中，固定粒子只参与第 1 步的球面遮挡，使壁面附近的自由表面分类能够感知固体占据的方向。固定
粒子不作为表面中心，不参与法向估计，也不成为 local mesh 顶点。这一角色划分是后续比较和调整边界接触线行为
的独立轴；当前能量定义以上述划分为准。

设粒子 $i$ 的有序 one-ring 为
$\mathcal M_i=(j_0,\ldots,j_{K_i-1})$，并令 $j_{K_i}=j_0$。当前粒子的局部表面积为

$$
A_i(\boldsymbol x)=\frac12\sum_{k=0}^{K_i-1}
\left\|(\boldsymbol x_{j_k}-\boldsymbol x_i)\times
(\boldsymbol x_{j_{k+1}}-\boldsymbol x_i)\right\|.
$$

这组以 $i$ 为中心的三角形近似自由表面在粒子 $i$ 附近的几何。求解器不维护全局 surface mesh，而是把全局
表面能 $\alpha\gamma A(\boldsymbol x)$ 在每个局部系统中近似为
$\alpha\gamma A_i(\boldsymbol x)$。

对单个三角形，记

$$
\boldsymbol a=\boldsymbol x_i,\qquad
\boldsymbol b=\boldsymbol x_{j_k},\qquad
\boldsymbol c=\boldsymbol x_{j_{k+1}},
$$

$$
\boldsymbol u=(\boldsymbol b-\boldsymbol a)\times(\boldsymbol c-\boldsymbol a),\qquad
\boldsymbol n=\frac{\boldsymbol u}{\Vert\boldsymbol u\Vert},\qquad
\boldsymbol e=\boldsymbol c-\boldsymbol b.
$$

非退化三角形的面积、关于中心顶点 $\boldsymbol a$ 的梯度和精确 Hessian 为

$$
A_\triangle=\frac12\Vert\boldsymbol u\Vert,
$$

$$
\nabla_{\boldsymbol a}A_\triangle=\frac12\boldsymbol n\times\boldsymbol e,
$$

$$
\nabla_{\boldsymbol a}^2A_\triangle=
\frac{\Vert\boldsymbol e\Vert^2}{2\Vert\boldsymbol u\Vert}\boldsymbol n\boldsymbol n^\mathsf T.
$$

对 one-ring 中所有三角形求和即可得到 $\nabla_iA_i$ 和 $\nabla_i^2A_i$。

## Local Optimization

记可移动流体粒子的共享质量为 $m$。每个时间步的连续目标为

$$
\boldsymbol x^{n+1}=\arg\min_{\boldsymbol x}
\alpha\frac m{2h^2}\Vert\boldsymbol x-\boldsymbol y\Vert^2
+\frac12\sum_{j=1}^N(C_j^\rho(\boldsymbol x))^2+\alpha\gamma A(\boldsymbol x),
$$

其中
$\boldsymbol y=\boldsymbol x^n+h\boldsymbol v^n+h^2\boldsymbol a^\star$，$\boldsymbol a^\star$ 是已知外力提供的
加速度。求解器使用 Vertex Block Descent（VBD），并行求解每个粒子坐标对应的局部问题。记
$\mathcal C_i$ 为所有依赖 $\boldsymbol x_i$ 的密度约束中心，则实际离散局部目标是

$$
E_i(\boldsymbol x)=\alpha\frac m{2h^2}\Vert\boldsymbol x_i-\boldsymbol y_i\Vert^2
+\frac12\sum_{j\in\mathcal C_i}(C_j^\rho(\boldsymbol x))^2
+\alpha\gamma A_i(\boldsymbol x).
$$

$\alpha$ 同时缩放惯性能量和表面积能量，因此 $\gamma$ 控制表面项相对于惯性项的强度，而密度能量保持原权重。

表面项只包含当前粒子 $i$ 自己构建的 $A_i$。即使 $i$ 是邻居粒子 $j$ 的 local mesh 顶点，$A_j$ 也不进入
粒子 $i$ 的局部目标。这个取舍避免了为每个粒子反向收集所有包含它的邻居 local mesh，并保持局部系统的构建
开销有界。

在一个 iteration 中求解

$$
\boldsymbol H_i\Delta\boldsymbol x_i=\boldsymbol f_i,
$$

其中局部目标的负梯度为

$$
\boldsymbol f_i=-\alpha\frac m{h^2}(\boldsymbol x_i-\boldsymbol y_i)
-\sum_{j\in\mathcal C_i}C_j^\rho(\boldsymbol x)
\frac{\partial C_j^\rho(\boldsymbol x)}{\partial\boldsymbol x_i}
-\alpha\gamma\nabla_iA_i(\boldsymbol x).
$$

Newton 矩阵为

$$
\begin{aligned}
\boldsymbol H_i={}&\alpha\frac m{h^2}\mathbf I\\
&+\sum_{j\in\mathcal C_i}\left[
\frac{\partial C_j^\rho}{\partial\boldsymbol x_i}
\left(\frac{\partial C_j^\rho}{\partial\boldsymbol x_i}\right)^\mathsf T
+C_j^\rho\,\mathcal D\left(\frac{\partial^2C_j^\rho}{\partial\boldsymbol x_i^2}\right)
\right]
+\alpha\gamma\mathcal D\left(\nabla_i^2A_i\right),
\end{aligned}
$$

其中 $\mathcal D$ 将矩阵每一列的二范数放到对应对角元素：

$$
\mathcal D(\boldsymbol Q)=\operatorname{diag}
\left(\Vert\boldsymbol Q_{:,0}\Vert_2,
\Vert\boldsymbol Q_{:,1}\Vert_2,
\Vert\boldsymbol Q_{:,2}\Vert_2\right).
$$

面积项先累加当前 one-ring 的精确三角形 Hessian，再进行一次 $\mathcal D$ 对角近似。面积梯度和 Hessian 都只对
中心位置 $\boldsymbol x_i$ 求导，环上顶点在该局部系统中保持不变。

当 $\boldsymbol H_i$ 接近奇异时，直接求逆会放大浮点舍入误差。用
`IPBSTFOptions.hessian_determinant_epsilon` 配置的阈值 $\varepsilon$ 保护局部求解：

$$
\Delta\boldsymbol x_i=
\begin{cases}
\boldsymbol H_i^{-1}\boldsymbol f_i, & \det(\boldsymbol H_i)\geq\varepsilon,\\
\boldsymbol 0, & \det(\boldsymbol H_i)<\varepsilon.
\end{cases}
$$

随后并行更新
$\boldsymbol x_i^{(k+1)}\gets\boldsymbol x_i^{(k)}+\frac12\Delta\boldsymbol x_i$，并在下一 iteration 根据更新后的
位置重算密度、表面分类、法向和 local mesh，直到达到 `max_solver_iterations`。

`surface_tension=0` 时不分配表面重建缓冲区，也不执行相关 kernel。局部候选和最终 one-ring 分别由
`max_surface_neighbors` 与 `max_localmesh_neighbors` 限制；容量溢出会使对应拓扑失效并通过 solver errno 停止仿真，
避免使用截断的表面积。
