from typing import ClassVar

from .base import Base


class Solid(Base):
    """Fixed particles that provide solid boundary support to neighboring liquid particles."""

    is_fixed: ClassVar[bool] = True
