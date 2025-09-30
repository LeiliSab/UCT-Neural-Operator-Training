"""
UFWI.processors.outlier_removal
======================================

Zeroes the largest magnitude samples in each (Tx,Rx) slice, restricted to the
*acceptance mask* produced by :class:`~UFWI.processors.acceptance_mask.AcceptanceMask`.
"""
from __future__ import annotations

import numpy as np

from .base import BaseProcessor
from ..data import AcquisitionData


class MagnitudeOutlierFilter(BaseProcessor):
    """
    Zero out the largest‐magnitude samples in each (Tx,Rx) frequency slice,
    limited to the valid channels defined by an acceptance mask.

    This filter helps suppress spurious high‐amplitude outliers (e.g. noise
    spikes) before further processing.  It uses a precomputed boolean mask
    `data.ctx["elem_mask"]` (from `AcceptanceMask`) and optional Tx‐downsampling
    indices `data.ctx["tx_keep"]` to restrict which channels are considered.

    Parameters
    ----------
    threshold : float, optional
        Fraction of the smallest magnitudes to *keep* in each frequency slice.
        Must satisfy 0 < threshold ≤ 1.0.  For example:
        - `threshold=0.99` keeps the bottom 99% by magnitude and zeros out
          the top 1% largest values.
        - `threshold=1.0` (default) keeps everything (no removal).

    Usage
    -----
    >>> filt = MagnitudeOutlierFilter(threshold=0.95)
    >>> # data.array shape: (Tx, Rx, F)
    >>> # data.ctx["elem_mask"]: bool mask of shape (Tx_full, Rx)
    >>> # data.ctx["tx_keep"]: 1D int array of kept Tx indices (after downsampling)
    >>> filt(data)
    >>> # Now data.array has its top 5% largest‐magnitude entries zeroed
    >>> # in each frequency slice, but only where elem_mask is True.

    Effects
    -------
    - Reads `full_mask = data.ctx["elem_mask"]` if present,
      otherwise assumes all channels valid.
    - Applies `tx_keep = data.ctx["tx_keep"]` if present to slice the mask
      to the current (Tx, Rx) shape.
    - For each frequency index f:
        1. Compute absolute values `M = abs(data.array[:,:,f])`.
        2. Apply mask: `M_masked = M * mask`.
        3. Determine cutoff `thresh` as the Nth largest value where
           N = ceil((1-threshold) * (Tx*Rx)).
        4. Zero out `data.array[:,:,f]` at all positions where
           `M >= thresh` and `mask == True`.
    - If `threshold == 1.0` or computed N == 0, no values are dropped.
    - Operates in‐place on `data.array`.

    Raises
    ------
    ValueError
        - If `threshold` is not in (0, 1].
        - If the sliced mask shape does not match `data.array.shape[:2]`.
    """
    def __init__(self, threshold: float = 0.99):
        if not (0. < threshold <= 1.):
            raise ValueError("`threshold` must be in (0,1].")
        self._keep = threshold

    def __call__(self, data: AcquisitionData) -> None:
        # Retrieve full acceptance mask or default to all‐True
        full_mask = data.ctx.get(
            "elem_mask",
            np.ones(data.array.shape[:2], dtype=bool)
        )

        # Apply transmitter downsample mask if present
        tx_keep = data.ctx.get("tx_keep", np.arange(data.array.shape[0]))
        mask = full_mask[tx_keep, :]
        if mask.shape != data.array.shape[:2]:
            raise ValueError("Mask shape mismatch after down-sampling.")

        # Compute how many outliers to drop per slice
        n_tx, n_rx, n_f = data.array.shape
        n_total = n_tx * n_rx
        n_drop  = int(np.ceil((1 - self._keep) * n_total))
        if n_drop == 0:
            return

        # Process each frequency slice independently
        for f in range(n_f):
            slice_abs = np.abs(data.array[:, :, f])   # (Tx, Rx)
            mags = (slice_abs * mask).ravel()         # only masked entries

            # skip if all zero
            if np.all(mags == 0):
                continue

            # find magnitude cutoff (Nth largest)
            thresh = np.partition(mags, -n_drop)[-n_drop]

            # zero out all >= thresh within valid mask
            drop = (slice_abs >= thresh) & mask
            data.array[:, :, f][drop] = 0.0

        return data