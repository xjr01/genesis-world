# 将 Position-Based Constraints 写成变分能量形式求解

流体系统由 $N$ 个质量为 $m$ 的粒子组成，在 position-based 框架里，我们只关注它们的位置 $\boldsymbol x_i$ 怎么求解，记 $\boldsymbol x\in\mathbb R^{3N}$ 为所有 $\boldsymbol x_i$ 的堆叠向量。

## Density Constraints

直接用现有 PBF（SPH）的方法估算粒子 $i$ 的密度：

$$
\rho_i(\boldsymbol x)=m\sum_{j\in\mathcal N_i}W(\Vert\boldsymbol x_j-\boldsymbol x_i\Vert)
$$

其中 $W(\cdot)$ 为 cubic spline kernel，与 `genesis/engine/solvers/pbstf_solver.py` 中使用的核函数一致（见其中的 `PBSTFSolver.cubic_kernel()`），$\mathcal N_i$ 为距粒子 $i$ 小于核半径 $R$ 的粒子集合。则 density constraint 定义为：

$$
C_i^\rho(\boldsymbol x)=\max\left(\frac{\rho_i(\boldsymbol x)}{\rho}-1,0\right)
$$

注意这里用了单边约束，只抵抗体积压缩，不抵抗体积膨胀，与 `PBSTFSolver` 是不同的。

## 求解器

每个时间步需要求解这样一个优化问题：

$$
\boldsymbol x^{n+1}=\arg\min_{\boldsymbol x}\alpha\frac m{2h^2}\Vert\boldsymbol x-\boldsymbol y\Vert^2+E(\boldsymbol x)
$$

其中 $\boldsymbol y=\boldsymbol x^n+h\boldsymbol v^n+h^2\boldsymbol a^\star$，$\boldsymbol v^n$ 为粒子的当前速度，$\boldsymbol a^\star$ 为已知外力提供的加速度（暂时只需要考虑重力）；目前只考虑 density constraint，因此

$$
E(\boldsymbol x)=\frac 12\sum_{i=1}^n(C_i^\rho(\boldsymbol x))^2
$$

我们采用 VBD solver，即 block coordinate descent 的优化方法进行求解。将整个问题拆解成只关于单个粒子坐标的**局部优化问题**，对于粒子 $i$：

$$
\boldsymbol x_i^{n+1}=\arg\min_{\boldsymbol x_i}\alpha\frac m{2h^2}\Vert\boldsymbol x_i-\boldsymbol y_i\Vert^2+E_i(\boldsymbol x)
$$

其中 $E_i(\boldsymbol x)$ 为 $E(\boldsymbol x)$ 中只包含 $\boldsymbol x_i$ 的项：

$$
E_i(\boldsymbol x)=\frac 12\sum_{j\in\mathcal N_i}(C_j^\rho(\boldsymbol x))^2
$$

对于这个局部优化问题，采用牛顿迭代法进行求解，即在单个 iteration 内求解 $\Delta\boldsymbol x_i$ 满足

$$
\boldsymbol H_i\Delta\boldsymbol x_i=\boldsymbol f_i
$$

并将其作为 $\boldsymbol x_i$ 的更新方向（具体地，$\boldsymbol x_i^{(k+1)}\gets\boldsymbol x_i^{(k)}+\frac 12\Delta\boldsymbol x_i$，上标带括号表示单个时间步内的迭代序号，不带括号表示时间步序号），其中 $\boldsymbol f_i$ 是局部目标函数的负梯度：

$$
\boldsymbol f_i=-\alpha\frac m{h^2}(\boldsymbol x_i-\boldsymbol y_i)-\sum_{j\in\mathcal N_i}C_j^\rho(\boldsymbol x)\frac{\partial C_j^\rho(\boldsymbol x)}{\partial\boldsymbol x_i}
$$

$\boldsymbol H_i$ 是局部目标函数的海瑟矩阵：

$$
\boldsymbol H_i=\alpha\frac m{h^2}\mathbf I+\sum_{j\in\mathcal N_i}\left[\left(\frac{\partial C_j^\rho(\boldsymbol x)}{\partial\boldsymbol x_i}\right)^\top\frac{\partial C_j^\rho(\boldsymbol x)}{\partial\boldsymbol x_i}+C_j^\rho(\boldsymbol x)\frac{\partial^2C_j^\rho(\boldsymbol x)}{\partial\boldsymbol x_i^2}\right]
$$

其中关于约束的一阶梯度和二阶海瑟矩阵都可以分别通过核函数的一阶导和二阶导求得，但对于约束的海瑟矩阵我们会采用一种对角近似：将矩阵每一列的二范数作为该列矩阵的对角元素；经过这样的近似后，求得一个近似的 $\boldsymbol H_i$ 再代入上述求解 $\Delta\boldsymbol x_i$ 的线性系统。

当密度约束已经满足或接近满足，且 $\alpha$ 较小时，$\boldsymbol H_i$ 可能接近奇异。直接计算其逆矩阵会放大浮点舍入误差，因此用 `IPBSTFOptions.hessian_determinant_epsilon` 配置的非零阈值 $\varepsilon$ 保护局部求解，其默认值为 $10^{-7}$：

$$
\Delta\boldsymbol x_i=
\begin{cases}
\boldsymbol H_i^{-1}\boldsymbol f_i, & \det(\boldsymbol H_i)\geq\varepsilon,\\
\boldsymbol 0, & \det(\boldsymbol H_i)<\varepsilon.
\end{cases}
$$

第二种情况下仅跳过粒子 $i$ 在当前 iteration 的局部更新；下一个 iteration 会根据更新后的邻域状态重新组装并判断 $\boldsymbol H_i$。

### 强调：一些细节

局部优化问题的每一步牛顿迭代都只需要求解一个 $3\times 3$ 的小线性系统，因此直接用解析的方法计算 $\boldsymbol H_i^{-1}$ 即可。

所有的局部优化问题完全并行求解。单个 iteration 内，使用 $\boldsymbol x_i^{(k)}$ 的值并行计算所有 $\boldsymbol f_i$ 和 $\boldsymbol H_i$，进而并行计算所有的 $\Delta\boldsymbol x_i$ 然后并行地更新 $\boldsymbol x_i^{(k+1)}\gets\boldsymbol x_i^{(k)}+\frac 12\Delta\boldsymbol x_i$，随后进入下个 iteration 直到达到 `max_iter`。
