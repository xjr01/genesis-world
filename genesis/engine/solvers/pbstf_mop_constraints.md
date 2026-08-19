# PBSTF Porous Mop Formulations

This document describes the Position-Based Surface-Tension Flow (PBSTF) formulations that are active in `CASE_MOP` in
`examples/teapot/pbstf_surface_tension.py`. It follows the current implementation in
`pbstf_solver.py` and `pbstf_porous.py`.

## 1. Solver classification

`CASE_MOP` runs the liquid and the porous elastic entity in one position-based solver loop. The loop predicts both
phases, accumulates fluid and porous position corrections, applies all accumulated corrections together, and rebuilds
velocity from the corrected positions.

`PBSTFSolver` is independent of the engine's generic `PBDSolver`; PBD here names the numerical projection family rather
than the owning solver class.

The implementation is a mixture of several position-based formulations:

- Fluid density, surface area, particle distance, and collider adhesion use the non-time-scaled compliant
  position-based dynamics (PBD) convention inherited from PBSTF.
- Porous strain and pore-collapse constraints use `compliance / (dt^2 V0)` in their denominators.
- Capillary attraction is a compliant pair-distance projection.
- Fluid-solid drag is a dissipative relative-displacement projection.
- Static contact and domain bounds use direct geometric projection.
- Gravity, extended smoothed-particle hydrodynamics (XSPH) viscosity, and collider friction are integration or velocity
  updates.

Every local multiplier is recomputed as `lambda = -C / D` in every solver iteration. Multipliers are not accumulated
across iterations, and the update has no `compliance * lambda_old` term. Consequently, this is compliant position-based
dynamics (PBD) with some extended position-based dynamics (XPBD)-style compliance scaling, rather than canonical XPBD.

The porous state also contains derived fields such as density, porosity, saturation, rotation, and strain. These fields
drive constraints but are not independent constraints.

## 2. Active `CASE_MOP` configuration

The default case uses:

| Quantity | Value |
| --- | ---: |
| Particle diameter `s` | `0.1` |
| Particle radius `r_p` | `0.05` |
| Kernel support radius `h` | `0.3 = 3s` |
| Time step `dt` | `0.01` |
| Gravity | `(0, -9.8, 0)` |
| Solver iterations | `10` |
| Topology rebuild interval | `10` |
| Fluid rest density | `1000` |
| Fluid density compliance | `150` |
| Surface-area compliance | `1` |
| Surface distance compliance | `40` |
| Interior distance compliance | `180` |
| Surface and interior XSPH coefficients | `0.5` |
| Table adhesion compliance | `20` |
| Table tangential friction | `0.5` |
| Porous matrix density | `1000` |
| Dry porosity `phi0` | `0.8` |
| Deviatoric compliance | `1e-6` |
| Volumetric compliance | `1e-6` |
| Pore-collapse compliance | `0` |
| Capillary compliance | `10` |
| Capillary saturation falloff | `1` |
| Fluid-solid drag | `10` |
| Fully wet compliance scales | `1.5`, `1.5` |
| Fully wet target volume strain | `0.05` |

The mop is a porous elastic box, not a collider. Only its highest particle layer is fixed and moved kinematically. After
a `1` second settling interval, that layer moves from `x=-3.5` to `x=3.5` over `5` seconds. The only static PBSTF
collider in this case is the table. Liquid particles may therefore overlap the porous particles and occupy the pore
volume.

## 3. Notation and cubic kernel

Fluid indices are `i,j`, porous indices are `s,t`, and a porous reference-neighbor index is also written `t` when the
meaning is clear. Current positions are `x`, reference porous positions are `X`, particle masses are `m`, and inverse
masses are

$$
w_a =
\begin{cases}
1/m_a, & \text{active and movable},\\
0, & \text{inactive or kinematically fixed}.
\end{cases}
$$

Let `r = ||d||`, `q = r/h`, and

$$
W(r,h) = \frac{8}{\pi h^3}
\begin{cases}
6q^2(q-1)+1, & 0 \le q < 1/2,\\
2(1-q)^3, & 1/2 \le q < 1,\\
0, & q \ge 1.
\end{cases}
$$

For `d_ij = x_i-x_j`, the code defines

$$
G_{ij} = \operatorname{cubic\_gradient}(d_{ij},h)
       = -W'(r,h)\frac{d_{ij}}{r}
       = \nabla_{x_j}W(||x_i-x_j||,h).
$$

This sign convention is important: `G_ij` is the gradient with respect to the second point.

All liquid particles use the common build-time calibrated mass

$$
m_0=\frac{\rho_f^0}{
\max_i\left[W(0,h)+\sum_{j\in N_i}W(||x_i-x_j||,h)\right]}.
$$

## 4. Common local PBD update

For a scalar constraint `C_k(x)` with gradients `g_ka = grad_xa C_k`, most constraints use

$$
D_k = \widehat\alpha_k + \sum_a w_a ||g_{ka}||^2,
\qquad
\lambda_k = -\frac{C_k}{D_k},
\qquad
\Delta x_a = w_a\lambda_k g_{ka}.
$$

The exact definition of `alpha_hat` differs by constraint family and is stated below. Local corrections accumulate in
phase-local `dpos` arrays, with atomic writes where constraints share destination particles. A solver iteration
evaluates them at the same input positions and applies their sum once, so the main projection loop is Jacobi rather
than Gauss-Seidel.

## 5. Derived porous fields

These quantities are recalculated from the current particle configuration before the porous constraints in every
solver iteration.

### 5.1 Porous mass and reference density

For dry porosity `phi_s^0`, matrix density `rho_matrix`, and reference sample volume `V_s^0`,

$$
m_s = (1-\phi_s^0)\rho_{matrix}V_s^0.
$$

The entity-isolated reference density is

$$
\rho_s^{ref} = m_s W(0,h) + \sum_{t\in N_s^0}m_t W(||X_s-X_t||,h).
$$

`N_s^0` is a fixed material-local reference neighborhood. It defines the elastic connectivity even when current
spatial neighbors change.

### 5.2 Current porous density and porosity

The current solid density uses current same-material neighbors:

$$
\rho_s = m_s W(0,h) + \sum_{t\in N_s}m_tW(||x_s-x_t||,h).
$$

The reported and coupled porosity is

$$
\phi_s = \operatorname{clamp}\left(
1-(1-\phi_s^0)\frac{\rho_s}{\rho_s^{ref}},
0,1
\right).
$$

Compression raises solid density and reduces pore volume; expansion has the opposite effect.

### 5.3 Saturation

The dimensionless kernel-weighted fluid-volume occupancy is

$$
\theta_s^f = \sum_i \frac{m_i}{\rho_i^0}W(||x_s-x_i||,h),
$$

where `rho_i^0` is the fluid rest density. The solid Shepard normalization is

$$
Q_s = \frac{m_s}{\max(\rho_s,\epsilon)}W(0,h)
    + \sum_{t\in N_s}\frac{m_t}{\max(\rho_t,\epsilon)}W(||x_s-x_t||,h).
$$

Saturation is the clamped pore occupancy

$$
S_s =
\begin{cases}
\operatorname{clamp}\left(\theta_s^f/(\phi_sQ_s),0,1\right), & \phi_sQ_s>\epsilon,\\
0, & \text{otherwise}.
\end{cases}
$$

Saturation is recomputed from overlapping liquid particles. No fluid mass is transferred into a separately transported
wetness variable. The public absorbed-volume diagnostic is

$$
V_{abs} = \sum_s S_s\phi_sV_s^0
\frac{\rho_s^{ref}}{\max(\rho_s,\epsilon)}.
$$

### 5.4 Corrected meshless kinematics

For reference offset `r_st^0 = X_t-X_s` and raw reference kernel gradient `g_st^raw`, define

$$
M_s = \sum_{t\in N_s^0}V_t^0r_{st}^0(g_{st}^{raw})^T,
\qquad
g_{st} = (M_s^+)^Tg_{st}^{raw}.
$$

The corrected gradients reproduce affine deformation. Build fails if a reference neighborhood is rank deficient or
ill-conditioned.

The current deformation gradient and its proper polar rotation are

$$
F_s = \sum_{t\in N_s^0}V_t^0(x_t-x_s)g_{st}^T,
\qquad
R_s = \operatorname{polar}(F_s),
\qquad
\det R_s = 1.
$$

The corotated displacement gradient and symmetric strain are

$$
A_s = I + \sum_{t\in N_s^0}V_t^0
\left[(x_t-x_s)-R_s(X_t-X_s)\right](R_sg_{st})^T,
$$

$$
E_s = \frac{1}{2}(A_s+A_s^T)-I.
$$

`rho_s`, `phi_s`, `S_s`, `R_s`, and `E_s` are derived fields. The constraints below consume them.

## 6. Fluid density and porous-capacity constraint

### 6.1 Formulation

Let the target density of fluid particle `i` be `rho_i^star`. It normally equals the material rest density. An exposed
free-surface particle uses the empirical target

$$
\rho_i^\star = 0.7\rho_i^0.
$$

A particle classified inside a porous region uses the full target `rho_i^0`.

The constant solid-matrix volume carried by porous sample `s` is

$$
V_s^{solid} = (1-\phi_s^0)V_s^0.
$$

The capacity-augmented density is

$$
\rho_i^{mix} = m_iW(0,h)
+ \sum_{j\in N_i^f}m_jW(||x_i-x_j||,h)
+ \rho_i^\star\sum_{s\in N_i^p}V_s^{solid}W(||x_i-x_s||,h).
$$

Fluid pairs separated by a static collider are omitted. The constraint is

$$
C_i^{density} = \frac{\rho_i^{mix}}{\rho_i^\star}-1.
$$

Its neighbor gradients are

$$
g_{ij} = \frac{m_j}{\rho_i^\star}G_{ij},
\qquad
g_{is} = V_s^{solid}G_{is},
\qquad
g_{ii} = -\sum_jg_{ij}-\sum_sg_{is}.
$$

With fluid default mass `m_0` and liquid density compliance `alpha_rho`,

$$
D_i = \frac{\alpha_\rho}{m_0}
+ \frac{||g_{ii}||^2}{m_i}
+ \sum_j\frac{||g_{ij}||^2}{m_j}
+ \sum_sw_s||g_{is}||^2,
$$

$$
\lambda_i=-\frac{C_i^{density}}{D_i},
$$

$$
\Delta x_i += \frac{\lambda_i}{m_i}g_{ii},
\qquad
\Delta x_j += \frac{\lambda_i}{m_j}g_{ij},
\qquad
\Delta x_s += w_s\lambda_i g_{is}.
$$

### 6.2 Activation and physical role

Let `Sigma_i` be the covariance of neighboring free-surface offsets, with eigenvalue sum `tau_i` and largest eigenvalue
`ell_i`. Ordinary fluid density projection is enabled when

$$
\tau_i\le\epsilon
\quad\text{or}\quad
\frac{\ell_i}{\tau_i}\le 0.8.
$$

This suppresses density projection for highly anisotropic free-surface neighborhoods. Inside the porous entity the
classifier forces density projection on, clears the free-surface flag, and invalidates the local surface mesh.

Outside porous material, the density constraint is bilateral. Inside porous material it is unilateral:

$$
C_i^{density}>0.
$$

An under-filled pore region is therefore not expanded to artificial full saturation. An over-filled region pushes
fluid and movable solid apart.

This one formulation produces two related phenomena:

- Standard liquid incompressibility outside the mop.
- Finite pore capacity and pressure-driven squeeze-out inside the mop.

When the skeleton is compressed, porous samples move closer, the kernel estimate of solid occupancy rises, and
`C_i^density` becomes positive. The resulting projection expels liquid toward available pore or free space. Fluid
particle mass and count remain constant throughout absorption and expulsion.

## 7. Fluid free-surface area constraint

For a free-surface particle `i`, the topology builder constructs an ordered one-ring
`j_0,...,j_(n-1)`. With cyclic indexing,

$$
C_i^{area}=A_i=
\sum_{k=0}^{n-1}\frac{1}{2}
||(x_{j_k}-x_i)\times(x_{j_{k+1}}-x_i)||.
$$

For a triangle `(a,b,c)`, the implemented gradient with respect to `x_a` is

$$
\nabla_{x_a}A_{abc}
= \frac{1}{2}\widehat{(x_b-x_a)\times(x_c-x_a)}\times(x_c-x_b).
$$

After accumulating the center and ring-vertex gradients,

$$
D_i = \frac{\alpha_A}{m_0}
+ \frac{||g_i||^2}{m_i}
+ \sum_k\frac{||g_{j_k}||^2}{m_{j_k}},
\qquad
\lambda_i=-\frac{A_i}{D_i},
$$

$$
\Delta x_a += \frac{\lambda_i}{m_a}g_a.
$$

The zero-area target minimizes exposed liquid area and produces surface tension. Density and distance constraints resist
complete collapse. Fluid classified inside the mop has no free-surface mesh, so pore-scale liquid does not shrink into
artificial droplets through this constraint.

## 8. Fluid minimum-distance constraint

For a pair with the same surface classification that is not separated by a static collider, define

$$
d_{ij}=||x_i-x_j||,
$$

$$
d_{ij}^0 = \frac{s}{2}\left[
\left(\frac{m_i}{m_0}\right)^{2/3}
+ \left(\frac{m_j}{m_0}\right)^{2/3}
\right].
$$

Equal-mass `CASE_MOP` particles have `d_ij^0=s`. The unilateral constraint is active for
`0 < d_ij < d_ij^0`:

$$
C_{ij}^{distance}=d_{ij}-d_{ij}^0<0,
\qquad
g_i=\frac{x_i-x_j}{d_{ij}},
\qquad
g_j=-g_i.
$$

Using surface or interior compliance `alpha_d`,

$$
D_{ij}=\frac{\alpha_d}{m_0}+\frac{1}{m_i}+\frac{1}{m_j},
\qquad
\lambda_{ij}=-\frac{C_{ij}^{distance}}{D_{ij}},
$$

$$
\Delta x_i += \frac{\lambda_{ij}}{m_i}g_i,
\qquad
\Delta x_j -= \frac{\lambda_{ij}}{m_j}g_i.
$$

This is an anti-overlap and particle-spacing regularizer. It runs only on iterations `0,2,4,6,8` in `CASE_MOP`.

## 9. Porous deviatoric strain constraints

The implementation uses five Frobenius-orthonormal traceless symmetric bases:

$$
B_0=\frac{1}{\sqrt{2}}\operatorname{diag}(1,-1,0),
$$

$$
B_1=\frac{1}{\sqrt{6}}\operatorname{diag}(1,1,-2),
$$

$$
B_2=\frac{e_0e_1^T+e_1e_0^T}{\sqrt{2}},\quad
B_3=\frac{e_0e_2^T+e_2e_0^T}{\sqrt{2}},\quad
B_4=\frac{e_1e_2^T+e_2e_1^T}{\sqrt{2}}.
$$

For `a=0,...,4`,

$$
C_{s,a}^{dev}=B_a:E_s.
$$

The full-saturation interpolation of deviatoric compliance is

$$
\alpha_{s,dev}^{eff}
= \alpha_{s,dev}\left[(1-S_s)+S_s k_{s,dev}^{wet}\right].
$$

For a reference neighbor `t`,

$$
g_{st,a}=V_t^0B_a(R_sg_{st}),
\qquad
g_{ss,a}=-\sum_tg_{st,a}.
$$

The denominator and local multiplier are

$$
D_{s,a}=\frac{\alpha_{s,dev}^{eff}}{dt^2\max(V_s^0,\epsilon)}
+w_s||g_{ss,a}||^2
+\sum_tw_t||g_{st,a}||^2,
\qquad
\lambda_{s,a}=-\frac{C_{s,a}^{dev}}{D_{s,a}}.
$$

The code applies an overlap relaxation

$$
\omega_s=\frac{1}{6(|N_s^0|+1)},
$$

then scatters

$$
\Delta x_s += \omega_sw_s\lambda_{s,a}g_{ss,a},
\qquad
\Delta x_t += \omega_sw_t\lambda_{s,a}g_{st,a}.
$$

These five constraints resist shear and shape distortion of the porous skeleton. In `CASE_MOP`, saturation changes the
compliance term from `1e-6` when dry to `1.5e-6` when fully wet, making the fully wet constraint response softer.

## 10. Porous volumetric strain and swelling constraint

The sixth strain basis is `B_5=I`. Its constraint is

$$
C_s^{volume}=\operatorname{tr}(E_s)-S_sb_s,
$$

where `b_s` is `bloating_volume_strain`. The effective compliance is

$$
\alpha_{s,vol}^{eff}
=\alpha_{s,vol}\left[(1-S_s)+S_sk_{s,vol}^{wet}\right].
$$

Its gradients, denominator, relaxation, and scatter are the same as the deviatoric formulation with `B_a=I` and
`alpha_dev_eff` replaced by `alpha_vol_eff`.

This constraint supplies the ordinary bulk response of the porous skeleton. Its saturation-dependent target also
produces swelling: in `CASE_MOP`, a fully saturated sample targets `trace(E_s)=0.05`. The wet compliance changes from
`1e-6` to `1.5e-6` at full saturation.

## 11. Pore-collapse constraint

Define the unclamped porosity estimate

$$
\phi_s^{raw}=1-(1-\phi_s^0)\frac{\rho_s}{\rho_s^{ref}}.
$$

The unilateral pore constraint is

$$
C_s^{pore}=(1-\phi_s^0)\frac{\rho_s}{\rho_s^{ref}}-1
=-\phi_s^{raw}\le 0.
$$

It is projected only when `C_s^pore>0`. For a current same-material neighbor `t`,

$$
g_{st}^{pore}
=(1-\phi_s^0)\frac{m_t}{\rho_s^{ref}}G_{st},
\qquad
g_{ss}^{pore}=-\sum_tg_{st}^{pore}.
$$

With pore compliance `alpha_pore`,

$$
D_s^{pore}
=\frac{\alpha_{pore}}{dt^2\max(V_s^0,\epsilon)}
+w_s||g_{ss}^{pore}||^2
+\sum_tw_t||g_{st}^{pore}||^2,
$$

$$
\lambda_s^{pore}=-\frac{C_s^{pore}}{D_s^{pore}},
$$

followed by the ordinary `w lambda g` scatter to the center and current solid neighbors.

This constraint prevents compression beyond zero pore volume. If `J_s` is approximated by
`rho_s^ref/rho_s`, the limit is

$$
J_s\ge 1-\phi_s^0.
$$

The `CASE_MOP` value `alpha_pore=0` makes this a hard local inequality up to finite-iteration error. The volumetric
strain constraint governs normal bulk deformation; the pore constraint only activates at the physical collapse limit.

## 12. Fluid-solid capillary constraint

The coupling-degree sums include every fluid-solid pair with `d_is<h`, including coincident pairs. Define

$$
k_{is}=\frac{W(d_{is},h)}{W(0,h)},
\qquad
A_i=\sum_sk_{is},
\qquad
B_s=\sum_ik_{is},
$$

$$
p_{is}=\frac{k_{is}}{\max(A_i,B_s,\epsilon)}.
$$

Both capillary and drag corrections require `epsilon<d_is<h`; a coincident pair contributes to the degree sums but
receives no pair correction.

The saturation-dependent strength is

$$
\eta_s=\max(0,1-\beta_sS_s),
$$

where `beta_s` is `capillary_saturation_falloff`. The pair uses a zero-rest-length distance constraint

$$
C_{is}^{cap}=d_{is}=||x_i-x_s||,
\qquad
n_{is}=\frac{x_i-x_s}{d_{is}}.
$$

With capillary compliance `alpha_cap`,

$$
D_{is}^{cap}=w_i+w_s
+\frac{\alpha_{cap}}{\max(m_0k_{is}\eta_s,\epsilon)},
\qquad
\lambda_{is}^{cap}=-\frac{d_{is}}{D_{is}^{cap}},
$$

$$
\Delta x_i += p_{is}w_i\lambda_{is}^{cap}n_{is},
\qquad
\Delta x_s -= p_{is}w_s\lambda_{is}^{cap}n_{is}.
$$

This constraint attracts nearby water into and toward the porous matrix. With the `CASE_MOP` falloff `beta_s=1`, the
strength factor decreases linearly toward zero as the local pore reaches full saturation. The normalized `p_is` limits
the aggregate correction when either particle has many cross-phase neighbors.

This is a local PBD capillary surrogate. It represents retention and uptake through a kernel-weighted attraction rather
than solving a separate capillary-pressure field.

## 13. Fluid-solid drag projection

Let the substep displacements before the current correction be

$$
u_i=x_i-x_i^{old},
\qquad
u_s=x_s-x_s^{old}.
$$

The three-component relative-displacement constraint is

$$
C_{is}^{drag}=u_i-u_s.
$$

With drag coefficient `gamma_s`,

$$
D_{is}^{drag}=w_i+w_s+\frac{1}{dt\gamma_sk_{is}},
\qquad
c_{is}=-\frac{C_{is}^{drag}}{D_{is}^{drag}},
$$

$$
\Delta x_i += p_{is}w_ic_{is},
\qquad
\Delta x_s -= p_{is}w_sc_{is}.
$$

This dissipative vector projection reduces fluid-solid relative velocity. It makes absorbed liquid follow a moving mop
and controls the rate at which liquid redistributes or drains through the skeleton. It acts on every active correction
pair in the kernel support and is not gated by saturation or the inside-porous classification.

## 14. Static-collider adhesion constraint

This constraint applies only to liquid particles marked as free surface and only to static colliders. In `CASE_MOP` the
only such collider is the table.

Let `a` be the radius-offset contact anchor returned by the collider query and `n` its outward normal. For a particle in
the adhesion band,

$$
C_i^{adhesion}=(x_i-a)\mathbin{\cdot}n,
$$

$$
D_i^{adhesion}=\frac{\alpha_{adh}}{m_0}+\frac{1}{m_i},
$$

$$
\Delta x_i += -\frac{C_i^{adhesion}}{D_i^{adhesion}m_i}n.
$$

The query is accepted when the raw surface distance is at most `2r_p` and `||x_i-a||<=r_p`. This keeps exposed water
attached to the table. It does not implement water-mop adhesion; porous uptake is supplied by capillary attraction and
drag.

## 15. Hard domain and static-contact projection

After all `dpos` contributions in an iteration, each movable particle is updated by

$$
x_a \leftarrow P_{static}\left(P_{domain}(x_a+\Delta x_a)\right).
$$

`P_domain` clamps the particle center to the solver bounds. `P_static` pushes a particle center out of each static
collider to a radius-offset surface point when it is inside the collider or closer than `r_p`.

This is a hard geometric PBD-style contact projection with no stored multiplier, mass weighting, or compliance. It
prevents liquid and movable porous particles from penetrating the table. The static table receives no reaction.

A static-collider separation test also removes liquid pairs lying on opposite sides of a collider from density,
free-surface topology, and minimum-distance interactions.

## 16. Kinematic mop anchors

The highest layer of porous particles has zero inverse mass. Before each simulation step, the example prescribes their
position and velocity along the wiping trajectory. A fixed porous particle uses

$$
x_s^{old}=x_s-dt\,v_s,
\qquad
x_s^{pred}=x_s,
\qquad
w_s=0.
$$

It participates in every neighboring constraint but receives no correction. Elastic constraints transmit its motion
through the skeleton; drag transmits the prescribed displacement to nearby liquid. This is a kinematic boundary
condition rather than an anchor-distance constraint.

## 17. Position-based loop support operations

The following operations are part of the current simulation but are not scalar physical constraints.

### 17.1 Gravity and prediction

For movable fluid and porous particles,

$$
v_a^\star=v_a^n+dt\,g,
\qquad
x_a^{old}=x_a^n,
\qquad
x_a^{pred}=x_a^n+dt\,v_a^\star.
$$

This is semi-implicit Euler prediction. Gravity is an external acceleration.

### 17.2 Velocity reconstruction

After the ten position iterations,

$$
v_a^{n+1}=\frac{x_a^{final}-x_a^{old}}{dt}.
$$

### 17.3 XSPH liquid viscosity

After a fresh fluid neighbor search and density evaluation, XSPH applies

$$
\Delta v_i^{XSPH}
=\sum_{j:\,surface_j=surface_i}
\frac{m_j}{\rho_j}(v_j-v_i)W(||x_i-x_j||,h),
$$

$$
v_i\leftarrow v_i+c_i\Delta v_i^{XSPH}.
$$

Both surface and interior coefficients are `c_i=0.5` in `CASE_MOP`. This velocity filter smooths liquid velocity and
dissipates relative motion. It is not a position constraint.

### 17.4 Table friction

For a free-surface liquid particle within one radius of the table, decompose

$$
v_n=(v\mathbin{\cdot}n)n,
\qquad
v_t=v-v_n,
$$

then apply

$$
v\leftarrow v_n+(1-\mu)v_t.
$$

`CASE_MOP` uses `mu=0.5`. This is tangential velocity damping rather than a Coulomb or PBD friction constraint. Porous
particles receive hard table contact projection but no analogous table-friction update.

### 17.5 Porous classification and free-surface suppression

A fluid particle is classified inside porous material when an active porous particle whose stored porosity is positive
at topology rebuild lies within `h`. The classifier then:

- clears its free-surface state;
- invalidates its local surface mesh;
- forces its density constraint to be evaluated;
- changes the density projection to the unilateral over-capacity form.

This classification prevents fluid inside the mop from being treated as an exposed liquid surface. Capillary and drag
still operate across the support-radius transition layer, which lets adjacent water enter the mop.

## 18. Exact substep order

For the default ten iterations, a substep performs:

1. The example updates the fixed top-layer target position and velocity.
2. Fluid and porous particles are reordered, predicted under gravity, and reordered at predicted positions.
3. Iteration `0` rebuilds fluid density, surface classification, normals, density gating, porous classification, and the
   local free-surface meshes. With rebuild interval `10`, this happens once per substep.
4. Every iteration recomputes porous density, porosity, saturation, rotation, and strain.
5. Fluid and porous `dpos` arrays are cleared.
6. The solver accumulates density plus capacity corrections.
7. It accumulates free-surface area corrections.
8. It accumulates five deviatoric and one volumetric porous strain correction per porous center.
9. It accumulates the unilateral pore-collapse correction.
10. It computes cross-phase pair weights, then accumulates capillary and drag corrections.
11. On even iterations, it accumulates liquid minimum-distance corrections.
12. It accumulates table-adhesion corrections.
13. It applies the summed Jacobi correction and hard domain/static-contact projection.
14. After all iterations, it reconstructs fluid and porous velocities from position differences.
15. It performs a fresh fluid neighbor search, applies XSPH viscosity and table friction, rebuilds fluid position from
    the filtered velocity, and projects contact again.
16. It refreshes density and porous derived fields and halts through the solver errno mechanism if a non-finite state
    is detected.

## 19. Phenomenon-to-formulation map

| Physical or numerical phenomenon | Formulation |
| --- | --- |
| Bulk liquid incompressibility | Fluid density constraint |
| Finite water capacity in the mop | Solid-volume term in the fluid density constraint |
| Water expelled by compression | Capacity violation plus symmetric fluid-solid density corrections |
| Exposed liquid surface tension | Local free-surface area constraint |
| Fluid particle anti-overlap | Same-class minimum-distance inequality |
| Mop shear and shape stiffness | Five deviatoric strain constraints |
| Mop bulk stiffness | One volumetric strain constraint |
| Wet softening | Saturation-scaled deviatoric and volumetric compliance |
| Wet swelling | Saturation-dependent volumetric strain target |
| Nonnegative remaining pore volume | Pore-collapse inequality |
| Water uptake and retention | Saturation-weakened capillary pair constraint |
| Water carried by a moving mop | Fluid-solid drag projection |
| Exposed water wetting the table | Static-collider adhesion constraint |
| No table penetration | Hard static-contact projection |
| Mop actuation | Zero-inverse-mass kinematic top layer |
| Falling motion | Gravity predictor |
| Liquid viscous smoothing | XSPH velocity filter |
| Tangential water-table damping | Collider-friction velocity update |

## 20. Implementation index

| Formulation or field | Implementation |
| --- | --- |
| Cubic kernel and gradient | `pbstf_porous.cubic_kernel`, `pbstf_porous.cubic_gradient` |
| Porous reference topology | `pbstf_porous.build_porous_rest_topology` |
| Porous density and porosity | `pbstf_porous.kernel_compute_porous_density` |
| Saturation | `pbstf_porous.kernel_compute_porous_saturation` |
| Rotation and strain | `pbstf_porous.kernel_compute_porous_kinematics` |
| Fluid density and capacity | `PBSTFSolver._kernel_prepare_density_constraints`, `accumulate_porous_capacity` |
| Fluid density scatter | `PBSTFSolver._kernel_apply_density_constraints`, `apply_porous_capacity` |
| Free-surface area | `PBSTFSolver._kernel_apply_surface_constraints` |
| Fluid minimum distance | `PBSTFSolver._kernel_apply_distance_constraints` |
| Porous strain | `pbstf_porous.kernel_apply_porous_elastic_constraints` |
| Pore collapse | `pbstf_porous.kernel_apply_porous_pore_constraints` |
| Capillary and drag | `pbstf_porous.kernel_apply_porous_capillary_drag` |
| Table adhesion | `PBSTFSolver._kernel_apply_static_collider_adhesion` |
| Hard contact projection | `PBSTFSolver._kernel_apply_position_delta` |
| Gravity prediction | `PBSTFSolver._kernel_predict_positions`, `kernel_predict_porous_positions` |
| XSPH and table friction | `PBSTFSolver._kernel_compute_viscosity`, `PBSTFSolver._kernel_apply_viscosity` |
| Full solve order | `PBSTFSolver.substep_pre_coupling` |

## 21. Modeling implications

- Absorption means spatial overlap of mass-conserving liquid particles with a porous skeleton. The mop does not delete
  liquid particles or convert their mass into a wetness scalar.
- The capacity constraint uses constant matrix volume `(1-phi0)V0`; compression changes its kernel occupancy by moving
  porous samples closer together.
- Saturation is a bounded local diagnostic and constitutive input. It is not an independently conserved state.
- Internal free-surface area projection is suppressed, while capillary and drag remain active, so pore water is retained
  without being contracted into artificial free droplets.
- Finite Jacobi iterations, local relaxation, stale within-substep topology, XSPH viscosity, drag, and friction make the
  method dissipative and approximate.
- Kinematic anchors and the static table exchange momentum with the simulated phases without solving their reaction
  motion.
- The porous entity resolves to `visual` mode, where its box surface uses opacity `0.3` and skinning driven by porous
  particles. The liquid resolves to particle rendering. Opacity has no effect on any formulation above; the visible
  liquid particles are the simulated fluid particles.
