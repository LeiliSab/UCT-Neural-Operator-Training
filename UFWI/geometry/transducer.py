# File: UFWI/geometry/transducer.py
import numpy as np
from UFWI.geometry.base import Geometry


class Transducer(Geometry):
    """
    Single transducer element with optional TX/RX roles.

    Parameters
    ----------
    position : array_like, shape (2,)
        (x, y) coordinate of the transducer.
    *, is_tx : bool
        Whether this element acts as a transmitter.
    is_rx : bool
        Whether this element acts as a receiver.
    id : any, optional
        Optional identifier.
    directivity : any, optional
        Optional directivity metadata.
    """

    def __init__(
            self,
            position: tuple[float, float] | list[float] | np.ndarray,
            *,
            is_tx: bool = False,
            is_rx: bool = False,
            id: object = None,
    ):
        # ensure a flat length‐2 array
        pos = np.asarray(position, float).ravel()
        if pos.size != 2:
            raise ValueError("position must be length‐2 [x, y]")
        self.position = pos

        # role flags
        self.is_tx = bool(is_tx)
        self.is_rx = bool(is_rx)

        # optional metadata
        self.id = id

        # call base: one element
        super().__init__(shape=(1,), extent=None)

    def __repr__(self):
        roles = []
        if self.is_tx:
            roles.append("TX")
        if self.is_rx:
            roles.append("RX")
        role_str = "|".join(roles) or "NONE"
        return (
            f"{self.__class__.__name__}("
            f"pos=({self.position[0]:.3f},{self.position[1]:.3f}), "
            f"roles={role_str}, "
            f"id={self.id!r})"
        )
