from ..base import EntityT, Material


class Base(Material[EntityT]):
    """
    The base class of FLIP materials.

    Note
    ----
    This class should *not* be instantiated directly.
    """
