"""
UFWI.processors.downsample
=================================

Select every *k-th* transmitter to reduce computational load.
"""
from __future__ import annotations

import numpy as np

from .base import BaseProcessor
from ..data import AcquisitionData


class DownSample(BaseProcessor):
    """
    Reduce the number of transmitters by keeping every `step`-th one.

    This processor subsamples the first axis (Tx) of the acquisition tensor
    `data.array`, which may be in time domain (shape `(Tx, Rx, T)`) or
    frequency domain (shape `(Tx, Rx, F)`).

    Parameters
    ----------
    step : int, optional
        Subsampling factor along the Tx axis.  Every `step`-th transmitter
        is retained (default 1 → no downsampling).

    After calling, the following changes occur:
    - `data.array` is replaced by `data.array[0:Tx:step, :, :]`, reducing
      the transmitter count from `Tx` to `ceil(Tx/step)`.
    - `data.ctx["tx_keep"]` stores the 1D integer array of retained Tx indices.
    - Any NaNs in the subsampled array are replaced with zero via `nan_to_num`.

    Examples
    --------
    >>> # Suppose data.array has shape (10, 4, 200)
    >>> ds = DownSample(step=2)
    >>> ds(data)
    >>> data.array.shape
    (5, 4, 200)
    >>> data.ctx["tx_keep"]
    array([0, 2, 4, 6, 8])
    """

    def __init__(self, step: int = 1) -> None:
        if step < 1:
            raise ValueError("`step` must be >= 1.")
        self._step = int(step)

    def __call__(self, data: AcquisitionData) -> None:
        """
        Apply transmitter downsampling in place.

        1. Compute indices to keep:
             tx_keep = [0, step, 2*step, ...] < Tx
        2. Subsample data.array:
             data.array = data.array[tx_keep, :, :]
        3. Store tx_keep in data.ctx for downstream use.
        4. Replace any NaNs with zero.

        Parameters
        ----------
        data : AcquisitionData
            Container whose `array` of shape (Tx, Rx, *) will be subsampled.

        Returns
        -------
        None
        """
        # choose Tx indices
        tx_keep = np.arange(0, data.array.shape[0], self._step)

        # slice in-place
        data.array = data.array[tx_keep]

        # store for solver / adjoint sources
        data.ctx["tx_keep"] = tx_keep

        # eliminate NaNs just like the original script
        np.nan_to_num(data.array, copy=False)

        return data