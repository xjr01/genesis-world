# Bundled authored scene

`scene.zip` contains evaluated geometry and original visual USD/textures derived
from the project-provided butter, porcelain tray, board, bread and handled knife
assets. It contains 49 files; `manifest.json` records the archive checksum.
The original board/bread/knife ZIP filenames and SHA-256 hashes are retained in
the bundled `assets.json`. No additional asset license is granted by this notice.

The bundle includes the validated simplified tray collision mesh, the original
display geometry, scene transforms and support data. It does not include machine
cache paths, simulated trajectories, initial particle arrays or Blender files.
`asset_pipeline --stage prepare` generates particles, trajectories and native
SDF caches on the destination machine. Blender and the original project folder
are not needed to run this bundled scene.

To change authored meshes, use the source export workflow in `ALIGNED_ASSETS.md`
with the corresponding source assets and Blender, then regenerate the bundle and
its checksum. The included configuration uses 0.6 mm butter sampling, 0.8 mm bread
sampling and a 50 microsecond integration step. Physical material parameters live
in `configs/asset_firm_smooth.json`, outside the archive.
