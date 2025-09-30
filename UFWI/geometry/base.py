# UFWI/geometry/base.py

import copy


class Geometry:
    """
    Base class for geometry definitions. Carries shape and extent information.

    Parameters
    ----------
    shape : tuple of int
        Number of points along each dimension.
    extent : tuple of float, optional
        Spatial extent as (xmin, xmax, ymin, ymax, ...).
    """

    def __init__(self, shape, extent=None):
        self.shape = tuple(shape)
        self.extent = extent

    def copy(self):
        """
        Create a deep copy of this Geometry.
        """
        return copy.deepcopy(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, extent={self.extent})"
