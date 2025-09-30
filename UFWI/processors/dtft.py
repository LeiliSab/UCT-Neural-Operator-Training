"""
UFWI.processors.dtft
=========================================

Implements :class:`DTFT`, which performs the discrete-time
Fourier transform (DTFT) at user-specified frequencies.

Because the preceding :class:`~UFWI.processors.TimeWindow`
has already tapered the traces, the transform can be applied directly.
"""
from __future__ import annotations

import numpy as np

from .base import BaseProcessor
from ..data import AcquisitionData


class DTFT(BaseProcessor):
    """
    Convert time-domain traces into frequency-domain spectra at specified frequencies.

            X(r,t; f_k) = sum_n x(r,t; t_n) * exp(-1j*2π*f_k*t_n) * Δt

    Parameters
    ----------
    freqs
        One-dimensional array with the target frequencies in Hz.
    """

    def __init__(self, freqs: np.ndarray):
        if freqs.ndim != 1:
            raise ValueError("`freqs` must be 1-D.")
        self._freqs = freqs.astype(float, copy=False)

    # ------------------------------------------------------------------ #
    def __call__(self, data: AcquisitionData) -> None:
        """
        Apply the discrete-time Fourier transform.
        Replaces data.array (shape: T × Rx× Tx) with the result
        (shape: Tx × Rx × F) and stores freqs in data.freqs.
        """
        # time axis
        t = data.time  # (T,)
        dt = float(np.diff(t).mean())  # time step

        # build DTFT kernel of shape (F, T)
        kernel = (
                np.exp(-1j * 2 * np.pi * np.outer(self._freqs, t)) * dt
        ).astype(np.complex128, copy=False)

        # rearrange data.array to (T, Rx*Tx)
        Tx, Rx, T = data.array.shape
        # bring time axis first: (T, Rx, Tx)
        traces = data.array.transpose(2, 1, 0)
        ts_matrix = traces.reshape(T, Rx * Tx, order="F")

        # compute DTFT: (F, T) @ (T, Rx*Tx) → (F, Rx*Tx)
        spec = kernel @ ts_matrix

        # reshape to (F, Rx, Tx) then back to (Tx, Rx, F)
        spec = spec.reshape(len(self._freqs), Rx, Tx, order="F")
        data.array = np.transpose(spec, (2, 1, 0))  # (Tx, Rx, F)

        # store frequency vector
        data.freqs = self._freqs

        return data
