# 12 oz drinking glass

Model by **Andrew Kator & Jennifer Legaz**, distributed by
[Sweet Home 3D](https://www.sweethome3d.com/free-3d-models/) under
[Creative Commons Attribution 3.0 United States](https://creativecommons.org/licenses/by/3.0/us/).
The original license notice is preserved in `License.txt`.

[Original OBJ/MTL archive](https://www.sweethome3d.com/wp-content/themes/sh3d/assets/models/katorlegaz/12-oz-glass.zip)
(retrieved 2026-09-28), SHA-256:
`725b5fbf971f0538286c7d42612de889621085b673f521427015c9e17a5f3643`.

## Coordinates and physical size

`12-oz-glass.obj` retains the source topology and material, with coordinates converted to meters and the exterior
base centered at the origin. Positive y points through the opening. The transformation from source vertices is
`(vertex - (0, -0.364254, 0)) * 0.1440772088610643`.

The source archive supplies unitless coordinates. The uniform scale sets the measured cavity volume to its named
12 US fluid ounces, or 354.88235475 mL. This is a capacity-based calibration for the example.

| Dimension | Meters |
| --- | --- |
| Exterior height | 0.104291584329 |
| Exterior diameter | 0.075900449937 |
| Straight inner diameter | 0.070522047730 |
| Straight wall thickness | 0.002689201103 |
| Center floor height above exterior base | 0.012614391867 |
| Interior depth | 0.091677192462 |
| Half-fill height above exterior base | 0.058452988098 |

The geometric volume at half the cavity height is 176.89 mL.

`12-oz-glass-cavity.obj` describes the enclosed fluid volume for half-fill height and retention measurements.
It contains the original inner faces with reversed winding and a triangle fan across the highest rim ring.
It is used only for measurements. Rendering and collisions both use the open glass shell in `12-oz-glass.obj`.
Both meshes are watertight, with 1728 glass triangles and 896 cavity triangles.
