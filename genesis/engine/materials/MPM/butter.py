"""Material Point Method (MPM) models for yield-stress spreads and porous solids."""

from typing import Annotated, Any

from pydantic import Field

import genesis as gs
from genesis.engine.materials.MPM.base import Base
from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

qd = gs.qd


class HerschelBulkleyButter(Base):
    """Yield-stress Herschel-Bulkley material with an implicit local return."""

    shear_modulus: PositiveFloat = 20000.0
    bulk_modulus: PositiveFloat = 150000.0
    yield_stress: PositiveFloat = 80.0
    consistency: NonNegativeFloat = 25.0
    flow_exponent: Annotated[ValidFloat, Field(ge=0.25, le=1.0)] = 0.5
    dt: PositiveFloat = 5e-5

    def model_post_init(self, context: Any) -> None:
        G, K = self.shear_modulus, self.bulk_modulus
        self.mu, self.lam = G, K - 2.0 * G / 3.0
        self.E = 9.0 * K * G / (3.0 * K + G)
        self.nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
        super().model_post_init(context)
        self.update_F_S_Jp = self._plastic_return
        self.update_stress = self._stress

    @qd.func
    def _plastic_return(self, J, F_tmp, U, S, V, Jp):
        volume_scale = qd.pow(J, 1.0 / 3.0)
        b = qd.Vector([S[0, 0] ** 2, S[1, 1] ** 2, S[2, 2] ** 2]) / (volume_scale**2)
        mean_b = b.sum() / 3.0
        dev_b = b - mean_b
        trial_norm = self.shear_modulus * dev_b.norm()
        threshold = qd.sqrt(2.0) * self.yield_stress
        S_new = S
        F_new = F_tmp
        if trial_norm > threshold:
            excess = trial_norm - threshold
            a = 2.0 * self.shear_modulus * mean_b * self.dt
            eta = 2.0 ** ((self.flow_exponent + 1.0) / 2.0) * self.consistency
            remaining = 0.0
            if qd.static(self.consistency > 0.0):
                if qd.static(self.flow_exponent == 1.0):
                    remaining = excess / (1.0 + a / eta)
                elif qd.static(self.flow_exponent == 0.5):
                    # Stable positive root of x + a*(x/eta)^2 = excess.
                    remaining = 2.0 * excess / (1.0 + qd.sqrt(1.0 + 4.0 * a * excess / (eta * eta)))
                else:
                    low, high = 0.0, excess
                    for _ in range(24):
                        middle = 0.5 * (low + high)
                        residual = middle + a * (middle / eta) ** (1.0 / self.flow_exponent) - excess
                        if residual > 0.0:
                            high = middle
                        else:
                            low = middle
                    remaining = 0.5 * (low + high)
            corrected_b = mean_b + dev_b * ((threshold + remaining) / trial_norm)
            corrected_b *= qd.pow(corrected_b[0] * corrected_b[1] * corrected_b[2], -1.0 / 3.0)
            S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
            for d in qd.static(range(3)):
                S_new[d, d] = volume_scale * qd.sqrt(corrected_b[d])
            F_new = U @ S_new @ V.transpose()
        return F_new, S_new, Jp

    @qd.func
    def _stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        identity = qd.Matrix.identity(gs.qd_float, 3)
        b = (F_new @ F_new.transpose()) / qd.pow(J, 2.0 / 3.0)
        return (
            self.shear_modulus * (b - b.trace() / 3.0 * identity) + 0.5 * self.bulk_modulus * (J * J - 1.0) * identity
        )


class PorousBread(Base):
    """Pressure-only pore collapse with compaction and densification hardening.

    ``Jp`` is the plastic volume ratio.  It starts at one and is constrained to
    be non-increasing, so tension and pure shear cannot create pore volume.
    Compression is returned to a pressure cap

        p_cap(xi) = p0 + H*xi + Hd*max(xi-xi_d, 0)^2,
        xi = -log(Jp),

    using a scalar implicit bisection. Deviatoric Hencky stress is projected
    independently to a shear cap without changing ``Jp``.
    """

    compaction_yield_pressure: PositiveFloat = 650.0
    compaction_hardening: NonNegativeFloat = 5200.0
    densification_strain: Annotated[ValidFloat, Field(gt=0.05, lt=0.8)] = 0.32
    densification_hardening: NonNegativeFloat = 45000.0
    min_plastic_volume_ratio: Annotated[ValidFloat, Field(gt=0.1, lt=1.0)] = 0.35
    shear_yield_stress: PositiveFloat = 700.0
    shear_hardening: NonNegativeFloat = 7000.0

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.update_F_S_Jp = self._plastic_return
        self.update_stress = self._stress

    @qd.func
    def _cap_pressure(self, plastic_volume_ratio):
        jp = qd.min(1.0, qd.max(self.min_plastic_volume_ratio, plastic_volume_ratio))
        compaction = -qd.log(jp)
        densification_log_strain = -qd.log(1.0 - self.densification_strain)
        beyond_densification = qd.max(0.0, compaction - densification_log_strain)
        return (
            self.compaction_yield_pressure
            + self.compaction_hardening * compaction
            + self.densification_hardening * beyond_densification**2
        )

    @qd.func
    def _plastic_return(self, J, F_tmp, U, S, V, Jp):
        safe = qd.Vector(
            [
                qd.max(S[0, 0], 0.05),
                qd.max(S[1, 1], 0.05),
                qd.max(S[2, 2], 0.05),
            ]
        )
        strain = qd.Vector([qd.log(safe[0]), qd.log(safe[1]), qd.log(safe[2])])
        trial_trace = strain.sum()
        mean = trial_trace / 3.0
        deviatoric = strain - mean
        bulk_modulus = self.lam + 2.0 * self.mu / 3.0

        # Positive p means compression.  Only compression may reduce Jp.
        trial_pressure = -bulk_modulus * trial_trace
        corrected_trace = trial_trace
        jp_new = qd.min(1.0, qd.max(self.min_plastic_volume_ratio, Jp))
        if trial_pressure > self._cap_pressure(jp_new):
            # delta transfers trial elastic volume strain into plastic pore
            # collapse: Jp_new = Jp*exp(-delta).  Solve p(delta)=p_cap(Jp).
            low = 0.0
            high = qd.min(
                qd.max(0.0, -trial_trace),
                qd.max(0.0, qd.log(jp_new / self.min_plastic_volume_ratio)),
            )
            for _ in qd.static(range(28)):
                middle = 0.5 * (low + high)
                candidate_jp = jp_new * qd.exp(-middle)
                candidate_pressure = -bulk_modulus * (trial_trace + middle)
                if candidate_pressure > self._cap_pressure(candidate_jp):
                    low = middle
                else:
                    high = middle
            delta = 0.5 * (low + high)
            corrected_trace = trial_trace + delta
            jp_new = qd.max(self.min_plastic_volume_ratio, jp_new * qd.exp(-delta))

        # Isochoric shear yielding supplies permanent indentation/shear without
        # changing pore volume.  q = sqrt(3/2)||tau_dev||.
        trial_q = qd.sqrt(6.0) * self.mu * deviatoric.norm(gs.EPS)
        shear_cap = self.shear_yield_stress + self.shear_hardening * (-qd.log(jp_new))
        if trial_q > shear_cap:
            deviatoric *= shear_cap / trial_q

        corrected_mean = corrected_trace / 3.0
        corrected = qd.Matrix.zero(gs.qd_float, 3, 3)
        for d in qd.static(range(3)):
            corrected[d, d] = qd.exp(corrected_mean + deviatoric[d])
        return U @ corrected @ V.transpose(), corrected, jp_new

    @qd.func
    def _stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        strain = qd.Vector(
            [
                qd.log(qd.max(S[0, 0], 0.05)),
                qd.log(qd.max(S[1, 1], 0.05)),
                qd.log(qd.max(S[2, 2], 0.05)),
            ]
        )
        trace = strain.sum()
        deviatoric = strain - trace / 3.0
        bulk_modulus = self.lam + 2.0 * self.mu / 3.0
        principal = 2.0 * self.mu * deviatoric + bulk_modulus * trace
        diagonal = qd.Matrix.zero(gs.qd_float, 3, 3)
        for d in qd.static(range(3)):
            diagonal[d, d] = principal[d]
        return U @ diagonal @ U.transpose()
