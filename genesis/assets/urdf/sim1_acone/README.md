# Sim1 Acone dual-arm robot

Source: [InternRobotics/Sim1_Assets](https://huggingface.co/InternRobotics/Sim1_Assets),
`assets/acone`, revision `a3e97c5619720f68ef5f630ce13522e5bb2c3e36`.
The upstream model card declares the MIT license; its copy is `upstream-README.md`.
Upstream citation: [SIM1](https://arxiv.org/abs/2604.08544).

`acone.urdf` and `meshes/` contain the original downloads. `SHA256SUMS` records
their SHA-256 checksums. Downloading these assets again requires a Hugging Face
account with access to the source repository.

`acone_collision.urdf` is the stationary-base variant used by the coffee example.
Its wheel joints are fixed. Its collision meshes follow the visual link meshes,
replacing the upstream placeholder cubes. `collision_meshes/` contains quadratic
error mesh simplifications with a target of 1,500 faces per link. The wheel
visuals also use these meshes to stay within the URDF importer's triangle limit.
Joint origins, axes, arm limits, gripper geometry scale, and link inertias retain
the upstream values.

The original robot uses Z up; the example rotates it into its Y-up table frame.
