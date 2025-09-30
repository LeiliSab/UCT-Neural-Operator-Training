"""
UFWI.processors.acceptance_mask
======================================

Create and store a boolean acceptance mask excluding receivers that violate a
±Δ element index around each transmitter.
"""
from __future__ import annotations

import numpy as np

from .base import BaseProcessor
from ..data import AcquisitionData


class AcceptanceMask(BaseProcessor):
    """
    Compute and store a per-transmitter boolean mask that excludes nearby receivers.

    For each transmitter index `tx` (0 ≤ tx < N), receivers whose circular index
    distance to `tx` is ≤ `delta` are marked False; all others are True.

    Parameters
    ----------
    delta : int
        Radius of exclusion around each transmitter, in element‐index units.
        Receivers with circular distance ≤ delta will be masked out.
        Must be non‐negative. (Default: 63)

    After calling, the following entry is added to `data.ctx`:

    - `data.ctx["elem_mask"]` : ndarray of bool, shape (N, N)
        `elem_mask[tx, rx] == True` indicates receiver `rx` is kept for transmitter `tx`.
        `False` indicates it is excluded.

    Notes
    -----
    - The “circular index distance” between `tx` and `rx` is computed modulo N:

      distance = min((rx−tx) mod N, (tx−rx) mod N)

    - This mask can be used to zero out or skip channels too close to the active element.
    """

    def __init__(self, delta: int = 63):
        if delta < 0:
            raise ValueError("`delta` must be non-negative.")
        self._delta = int(delta)

    def __call__(self, data: AcquisitionData) -> None:
        """
        Build the acceptance mask and store it in data.ctx.

        Parameters
        ----------
        data : AcquisitionData
            Container whose geometry defines N = number of elements.

        Modifications
        -------------
        data.ctx["elem_mask"] : ndarray of bool, shape (N, N)
            The mask matrix as described in the class docstring.
        """
        # Number of elements (transmitters = receivers)
        n_tx = data.tx_array.n_elements

        # Initialize all True, then exclude nearby rx indices for each tx
        keep = np.ones((n_tx, n_tx), dtype=bool)
        rng = np.arange(-self._delta, self._delta + 1)

        for tx in range(n_tx):
            idx = (tx + rng) % n_tx
            keep[tx, idx] = False

        data.ctx["elem_mask"] = keep

        return data