"""
UFWI.processors.phase_screen
===================================

`PhaseScreenCorrection` multiplies the acquisition tensor (Tx, Rx, F) by a
complex phase/gain factor that removes the geometric phase error produced
when every transducer element is “snapped” onto the nearest reconstruction
grid node.

Mathematically, the phase‐screen factor for transmitter s and receiver r at
frequency f is

    PS_sr(f) = exp( j * sigma * 2*pi * f * (TOF_disc_sr - TOF_true_sr) )

where:
- sigma ∈ {+1, -1} is the sign convention,
- TOF_true_sr is the exact time‐of‐flight from element s to r,
- TOF_disc_sr is the time‐of‐flight after snapping element positions to the grid.


Store metadata in data.ctx:
data.ctx["x_idx"] = array of transmitter→grid‐x indices (ix)
data.ctx["y_idx"] = array of transmitter→grid‐y indices (iy)
data.ctx["grid_lin_idx"] = flattened linear index of each snapped transmitter in the grid
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .base import BaseProcessor
from ..data import AcquisitionData
from ..geometry.image_grid_2D import ImageGrid2D


class PhaseScreenCorrection(BaseProcessor):
    """
    Parameters
    ----------
    image_geometry
        An :class:`ImageGeometry`
        instance that defines the reconstruction grid.
    sign : {+1, −1}, optional
        Sign convention used by the Helmholtz solver
        (default −1 → `e^{-jωt}`).

    Outputs
    -------
    data.array updated with phase‐error removed
    data.ctx enriched with nearest‐neighbor and grid‐index information
    """

    def __init__(self,
                 image_geometry: ImageGrid2D,
                 sign: int = -1,
                 c0: float | None = None) -> None:
        if sign not in (-1, 1):
            raise ValueError("`sign` must be ±1.")
        if not isinstance(image_geometry, ImageGrid2D):
            raise TypeError(
                "`image_geometry` must be an ImageGeometry instance"
            )

        self._sign = sign
        self._xi: np.ndarray = image_geometry.xi.astype(float, copy=False)
        self._yi: np.ndarray = image_geometry.yi.astype(float, copy=False)

        # k-d trees for nearest-neighbour search
        self._xi_tree = cKDTree(self._xi[:, None])
        self._yi_tree = cKDTree(self._yi[:, None])

        self.c0 = None if c0 is None else float(c0)

    # ------------------------------------------------------------------ #
    def __call__(self, data: AcquisitionData) -> None:
        """
        Apply the pre-computed phase screen in place to
        :pyattr:`AcquisitionData.array`.

        Expected tensor shape: ``(Tx, Rx, F)``.
        """
        # gather acquisition geometry
        geom = data.tx_array  # TransducerArray2D
        x_t, y_t = geom.positions  # (N,), (N,)
        c0 = self.c0 if self.c0 is not None else data.c0
        print(f'c0 = {c0:.3f} m/s')
        freqs = data.freqs  # (F,)

        # nearest grid indices (done once per call – O(N log N))
        ix = self._xi_tree.query(x_t[:, None])[1]  # (Tx,)
        iy = self._yi_tree.query(y_t[:, None])[1]  # (Tx,)

        # TOF error matrix Δτ  (Tx, Rx)
        dx_disc = self._xi[ix][:, None] - self._xi[ix][None, :]
        dy_disc = self._yi[iy][:, None] - self._yi[iy][None, :]
        tof_disc = np.sqrt(dx_disc ** 2 + dy_disc ** 2) / c0

        dx_true = x_t[:, None] - x_t[None, :]
        dy_true = y_t[:, None] - y_t[None, :]
        tof_true = np.sqrt(dx_true ** 2 + dy_true ** 2) / c0

        delta_tof = tof_disc - tof_true  # (Tx,Rx)

        # build phase tensor and apply
        phase = np.exp(
            1j * self._sign * 2 * np.pi
            * delta_tof[:, :, None]  # broadcast to (Tx,Rx,F)
            * freqs[None, None, :]
        ).astype(np.complex128, copy=False)

        data.array *= phase  # in-place

        # after ix / iy are computed:
        data.ctx["x_idx"] = ix
        data.ctx["y_idx"] = iy
        data.ctx["grid_lin_idx"] = np.ravel_multi_index(
            (iy, ix),
            (self._yi.size, self._xi.size), order='F'
        )

        return data
