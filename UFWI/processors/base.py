"""
UFWI.processors.base
===========================

Contains the abstract base class from which every processors must inherit.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..data import AcquisitionData


class BaseProcessor(ABC):
    """
    Abstract base-class for all preprocessing operators.

    Sub-classes must implement :meth:`__call__` and are encouraged to keep a
    stateless, functional style.  They can read or extend ``data.ctx`` for
    cross-stage communication, but should avoid storing large arrays for
    longer than absolutely necessary.
    """

    @abstractmethod
    def __call__(self, data: AcquisitionData) -> None:
        """
        Transform ``data`` in place and return ``None``.

        Notes
        -----
        * Returning the object is deliberately avoided to enforce the
          convention that every stage works on the same instance.
        * The method must not replace ``data`` with a different object.
        """
        raise NotImplementedError  # pragma: no cover
