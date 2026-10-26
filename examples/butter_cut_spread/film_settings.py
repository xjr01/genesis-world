"""Dependency-free blade-film settings shared by preparation and Isaac rendering."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FilmSettings:
    pixel: float = 0.0002
    kernel: float = 0.0009
    capture_distance: float = 0.0012
    column_distance: float = 0.012
    thin_layers: float = 1.5
    thick_layers: float = 3.0
    clearance: float = 0.000008
    optical_depth_per_layer: float = 0.28
    color: tuple[float, float, float] = (0.96, 0.84, 0.56)
    roughness: float = 0.38
