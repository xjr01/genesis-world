from dataclasses import dataclass

import torch


@dataclass
class VBDState:
    """A detached vertex block descent (VBD) snapshot with a leading environment dimension.

    Positions, velocities and forces have shape [B, vertices, 3]. Plastic inverse gradients and accumulated strains
    belong to elements; bending reference angles and accumulated plastic angles belong to interior shell edges.
    Restoring this state restores controls and material history along with the deformation.
    """

    pos: torch.Tensor
    vel: torch.Tensor
    forces: torch.Tensor
    is_fixed: torch.Tensor
    fixed_pos: torch.Tensor
    plastic_inverse: torch.Tensor
    plastic_strain: torch.Tensor
    bending_rest: torch.Tensor
    bending_strain: torch.Tensor

    def serializable(self):
        return self
