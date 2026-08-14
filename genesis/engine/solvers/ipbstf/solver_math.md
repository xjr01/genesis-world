# Implicit Position-Based Fluid Density Solver

The solver follows *Implicit Position-Based Fluids* for its density energy, relaxed parallel local Newton solve, and
final-iteration artificial damping. A density-aware trust region bounds each parallel position update so the local
Newton models remain useful when neighboring particles move at the same time. Global backtracking accepts only
strict decreases of the full variational energy.

## Density energy

For particle positions $\boldsymbol x_i$, masses $m_i$, rest density $\rho_0$, and cubic-spline support radius $r_s$,
the smoothed-particle hydrodynamics (SPH) density and unilateral density constraint are

$$
\rho_i(\boldsymbol x)=\sum_j m_j W(\lVert\boldsymbol x_i-\boldsymbol x_j\rVert,r_s),
\qquad
C_i(\boldsymbol x)=\max\left(\frac{\rho_i(\boldsymbol x)}{\rho_0}-1,0\right).
$$

The unilateral constraint penalizes compression while allowing a free surface to expand. Its energy is

$$
E(\boldsymbol x)=\frac12\sum_i C_i(\boldsymbol x)^2.
$$

Given the unconstrained prediction

$$
\boldsymbol y_i=\boldsymbol x_i^n+\Delta t\boldsymbol v_i^n+\Delta t^2\boldsymbol a_i^\star,
$$

one time step minimizes

$$
\alpha\sum_i\frac{m_i}{2\Delta t^2}
\lVert\boldsymbol x_i-\boldsymbol y_i\rVert^2+E(\boldsymbol x).
$$

The default $\alpha=0$ gives the paper's stiff density solve. Positive values retain more of the unconstrained
prediction and permit more compression.

The unconstrained prediction is projected into the collision-feasible domain before its initial density and energy
are evaluated. Candidate iterates use the same projection, so every energy comparison refers to one feasible domain.

## Parallel local Newton solve

Each Jacobi iteration assembles one $3\times3$ local system per particle:

$$
\boldsymbol H_i\Delta\boldsymbol x_i=\boldsymbol f_i,
$$

$$
\boldsymbol f_i=-\alpha\frac{m_i}{\Delta t^2}(\boldsymbol x_i-\boldsymbol y_i)
-\sum_{j:i\in\mathcal N_j}C_j\nabla_i C_j,
$$

$$
\boldsymbol H_i=\alpha\frac{m_i}{\Delta t^2}\boldsymbol I
+\sum_{j:i\in\mathcal N_j}\left(\nabla_i C_j\nabla_i C_j^T+C_j\nabla_i^2 C_j\right).
$$

The constraint-Hessian term uses the paper's positive diagonal approximation: each diagonal entry is the Euclidean
norm of the corresponding exact Hessian column. All local systems are assembled from the same positions, solved in
parallel, and followed by the paper's relaxation factor:

$$
\boldsymbol d_i=\frac12\Delta\boldsymbol x_i.
$$

The determinant test is scale normalized. With

$$
s_i=\max_a\lvert(\boldsymbol H_i)_{aa}\rvert,
\qquad
\overline{\boldsymbol H}_i=\boldsymbol H_i/s_i,
$$

the inverse is evaluated when $\det(\overline{\boldsymbol H}_i)$ is finite and at least
`hessian_determinant_epsilon`. The resulting update is
$\overline{\boldsymbol H}_i^{-1}(\boldsymbol f_i/s_i)$, which is algebraically equal to
$\boldsymbol H_i^{-1}\boldsymbol f_i$. Normalization makes the singularity decision independent of particle and
kernel units. A rank-deficient local system uses the scaled descent direction $\boldsymbol f_i/s_i$; global
backtracking below decides whether that fallback is useful.

## Density-aware parallel trust region

A relaxed local Newton step can still overshoot because every neighbor applies its independently computed update at
the same time. The solver assigns each particle a share of its linearized compression reduction. Let

$$
q_i=\nabla C_i^T\boldsymbol d_i.
$$

When $C_i>0$ and $q_i<0$, the update limit is

$$
r_i=\max\left(
\mathtt{surface\_update\_scale}\ r_s,
\mathtt{density\_update\_fraction}\frac{C_i}{-q_i}\lVert\boldsymbol d_i\rVert
\right).
$$

Otherwise, $r_i=\mathtt{surface\_update\_scale}\ r_s$. The applied update is

$$
\widehat{\boldsymbol d}_i=
\min\left(1,\frac{r_i}{\lVert\boldsymbol d_i\rVert}\right)\boldsymbol d_i.
$$

`density_update_fraction` trades faster volume recovery for greater parallel overshoot risk.
`surface_update_scale` bounds corrections that are driven only by neighboring constraints, trading fine interface
motion for suppression of isolated pressure-driven spray. Collision projection is applied after this trust region.

## Global energy backtracking

The relaxed Jacobi direction is evaluated against the complete variational energy because simultaneous local Newton
steps need not form a global descent step. Starting with step size $s=1$, the candidate positions are

$$
\boldsymbol x_i(s)=\boldsymbol x_i+s\widehat{\boldsymbol d}_i.
$$

Let $D=\sum_i\max(0,\boldsymbol f_i^T\widehat{\boldsymbol d}_i)$. The step is accepted only when

$$
\overline\Psi(\boldsymbol x(s))<\overline\Psi(\boldsymbol x)-10^{-4}sD.
$$

Otherwise $s$ is halved, for up to 24 trials. If no trial satisfies this sufficient-decrease test, the iteration
retains its input positions. Candidate motion is bounded to one quarter of the support radius, so densities can be
evaluated from the iteration-start hash using a two-cell stencil. This keeps backtracking on the device and preserves
complete neighbor queries.

The unilateral active set uses a dimensionless tolerance of `max(1e-8, 512 * EPS)`. This leaves accumulated floating
point residuals inactive during a rigid translation while remaining far below visually meaningful density errors.
An exact zero-energy state remains at zero; every positive-energy accepted iteration strictly decreases energy.

## Final-iteration artificial damping

On the last solver iteration, the paper's artificial damping performs one extra local solve from the same starting
positions with `damping_alpha`. After applying the same trust region and collision projection, it produces a reference
position $\boldsymbol x_i^\star$. Define

$$
\boldsymbol v_i=\frac{\boldsymbol x_i-\boldsymbol x_i^n}{\Delta t},
\qquad
\boldsymbol v_i^\star=\frac{\boldsymbol x_i^\star-\boldsymbol x_i^n}{\Delta t}.
$$

When $\lVert\boldsymbol v_i^\star\rVert<\lVert\boldsymbol v_i\rVert$ and
$\lVert\boldsymbol x_i-\boldsymbol x_i^\star\rVert<\beta r_s$, let

$$
w_i=1-\frac{\lVert\boldsymbol x_i-\boldsymbol x_i^\star\rVert}{\beta r_s}.
$$

The final velocity keeps its direction and scales its speed to

$$
\boldsymbol v_i\leftarrow\boldsymbol v_i
\sqrt{\max\left(0,
1-w_i\frac{\lVert\boldsymbol v_i\rVert^2-\lVert\boldsymbol v_i^\star\rVert^2}
{\lVert\boldsymbol v_i\rVert^2}
\right)}.
$$

The paper settings `damping_alpha=1e-3` and `damping_beta=60` are the defaults. Damping is enabled only when the
mass-weighted root-mean-square displacement of the fluid is below `damping_velocity_scale * support_radius`. The
default scale is $10^{-3}$, which confines the extra dissipation to motion near rest.

An optional XSPH velocity blend runs after the pressure velocity update. It averages only liquid-liquid relative
velocities, so uniform translation remains unchanged. The material's `viscosity` coefficient trades coherent bulk
flow for dissipation of particle-scale relative kinetic energy and defaults to zero.

The material's `kinetic_smoothing` coefficient uses the same neighborhood blend and then applies one global affine
velocity transform per environment. The transform restores the original mean velocity and, when the filtered
covariance is well-conditioned, the covariance of relative velocities. A scalar trace correction handles degenerate
covariances. Both paths preserve linear momentum and total kinetic energy while retaining the spatial coherence
created by the blend. The default coefficient is $0.2$; zero leaves the pressure velocity field unchanged.

## Length-scale interpretation

`particle_size` is the particle diameter and nominal nearest-neighbor spacing. The default
`support_radius = 2 * particle_size` matches the paper's particle diameter of 0.5 and kernel support of 1.0. In terms
of geometric particle radius, the support spans four radii. With `support_radius = particle_size`, nominal neighbors
lie at normalized kernel distance one, where the cubic-spline value and derivatives vanish; only closer contacts
remain coupled and the motion becomes granular.

The sampling and hash-grid bounds may include fixed boundary particles outside the fluid container. Separate collision
bounds project liquid particles at the intended container walls.

Reference: [Implicit Position-Based Fluids](https://graphics.cs.utah.edu/research/projects/ipbf/ipbf.pdf).
