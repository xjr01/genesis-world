import genesis as gs
from genesis.repr_base import RBC


class DeformableVisGeom(RBC):
    """A visual geom of a deformable entity, the deformable counterpart of `RigidVisGeom`.

    It carries the render mesh drawn by the visualizer, decoupled from the simulation mesh, and 'sim_verts_idx',
    the map from each render-mesh vertex to the simulated vertex standing for it (vertices co-located across visual
    geoms or duplicated by texture seams share a single simulated vertex).
    """

    def __init__(self, entity, vvert_start, vface_start, vmesh, sim_verts_idx):
        self._uid = gs.UID()
        self._entity = entity
        self._vvert_start = vvert_start
        self._vface_start = vface_start
        self._vmesh = vmesh
        self._sim_verts_idx = sim_verts_idx

    def get_trimesh(self):
        """The underlying `trimesh.Trimesh` of the render mesh."""
        return self._vmesh.trimesh

    @property
    def uid(self):
        """Unique identifier of the visual geom."""
        return self._uid

    @property
    def entity(self):
        """The deformable entity the vgeom belongs to."""
        return self._entity

    @property
    def vmesh(self):
        """The render mesh."""
        return self._vmesh

    @property
    def sim_verts_idx(self):
        """Map from render-mesh vertex index to the entity's simulated vertex index."""
        return self._sim_verts_idx

    @property
    def surface(self):
        """Rendering surface of the visual geom."""
        return self._vmesh.surface

    @property
    def uvs(self):
        """Texture coordinates of the visual geom."""
        return self._vmesh.uvs

    @property
    def metadata(self):
        """Metadata of the render mesh."""
        return self._vmesh.metadata

    @property
    def n_vverts(self):
        """Number of render vertices of the vgeom."""
        return len(self._vmesh.verts)

    @property
    def n_vfaces(self):
        """Number of render faces of the vgeom."""
        return len(self._vmesh.faces)

    @property
    def vvert_start(self):
        """Starting index of the vgeom's render vertices in the deformable solver."""
        return self._vvert_start

    @property
    def vface_start(self):
        """Starting index of the vgeom's render faces in the deformable solver."""
        return self._vface_start

    @property
    def vvert_end(self):
        """Ending index of the vgeom's render vertices in the deformable solver."""
        return self._vvert_start + self.n_vverts

    @property
    def vface_end(self):
        """Ending index of the vgeom's render faces in the deformable solver."""
        return self._vface_start + self.n_vfaces
