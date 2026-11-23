# Rendering assets

`bread-raw.npz` is a triangle-mesh extraction of the bread level set from
[Penn Graphics Research / Ziran 2019](https://github.com/penn-graphics-research/ziran2019/blob/master/Data/LevelSets/breadxxx.vdb.zip).
The upstream MIT license is included as [BREAD_LICENSE.txt](BREAD_LICENSE.txt).
The extraction used Blender's Volume to Mesh modifier at the source grid resolution,
threshold 0 and adaptivity 0.015. This is a display mesh only; the example does not run
the upstream CD-MPM fracture solver. `prepare_bread.py` normalizes the raw geometry,
adds microscopic surface relief and assigns crust faces.

The PNG files are the example's procedural/generated appearance textures: bread crumb,
bread micro-normal, brushed knife normal, ceramic wave pattern, stone color/normal and
the decorative tub label. They contain no photographed background scene. The crumb
color texture was generated as an orthographic macro texture of white sandwich-bread
interior, without crust, butter, plates or other scene content. Its apparent pore pattern
is an appearance aid; actual surface pores come from the bread mesh.

`manifest.json` records SHA-256 hashes of the binary assets used by preparation.
No private model download, local project directory or Blender installation is needed
to prepare or render this example. The normalized mesh and all simulation/video outputs
are generated outside the tracked assets directory.
