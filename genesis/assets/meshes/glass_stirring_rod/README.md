# Glass stirring rod

Derived from [Basic Chemistry Lab Equipments](https://sketchfab.com/3d-models/basic-chemistry-lab-equipments-5236a6461b554ab1ba18399a1b101774)
by [miqdadnaduthodi](https://sketchfab.com/miqdadnaduthodi), licensed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

`chemistry_lab.glb` is the source scene downloaded from the
[Objaverse mirror](https://huggingface.co/datasets/allenai/objaverse/blob/main/glbs/000-124/5236a6461b554ab1ba18399a1b101774.glb).
`glass_rod.obj` extracts scene node `Cylinder.008_Material.003_0`, applies its
scene transform, welds coincident texture and normal seam vertices, aligns the
long axis with +Y, and scales the rod to 0.18 m long and 0.006 m in diameter.
The lower tip is at Y = 0. The mesh is closed (64 vertices, 124 triangles).
The example supplies the glass appearance. These changes are made locally;
the source author does not endorse this adaptation.

`SHA256SUMS` records the source and adapted mesh checksums. `source.json`
records the source platform's attribution and license metadata.
