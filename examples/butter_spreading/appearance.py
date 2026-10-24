"""Serializable optical settings for finite-volume butter."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ButterAppearance:
    """OmniSurface controls; numerical values alone do not imply calibration."""

    diffuse_reflection_color: tuple[float, float, float] = (0.94, 0.76, 0.32)
    diffuse_reflection_weight: float = 1.0
    specular_reflection_roughness: float = 0.30
    specular_reflection_ior: float = 1.46
    enable_diffuse_transmission: bool = True
    subsurface_weight: float = 1.0
    subsurface_transmission_color: tuple[float, float, float] = (0.96, 0.80, 0.32)
    subsurface_scattering_color: tuple[float, float, float] = (1.0, 0.90, 0.80)
    subsurface_scale: float = 0.00020
    subsurface_anisotropy: float = 0.15
    thin_walled: bool = False
    enable_specular_transmission: bool = False
    geometry_opacity: float = 1.0

    def __post_init__(self):
        for color in (
            self.diffuse_reflection_color,
            self.subsurface_transmission_color,
            self.subsurface_scattering_color,
        ):
            if len(color) != 3 or not np.all(np.isfinite(color)) or min(color) <= 0 or max(color) > 1:
                raise ValueError("Material colors require three finite values in (0, 1].")
        if not (
            0 <= self.diffuse_reflection_weight <= 1
            and 0 <= self.specular_reflection_roughness <= 1
            and 1 < self.specular_reflection_ior <= 3
            and 0 < self.subsurface_scale <= 0.1
            and -1 < self.subsurface_anisotropy < 1
            and 0 < self.subsurface_weight <= 1
            and self.enable_diffuse_transmission
            and not self.thin_walled
            and not self.enable_specular_transmission
            and self.geometry_opacity == 1
        ):
            raise ValueError("Invalid finite-volume butter material.")
