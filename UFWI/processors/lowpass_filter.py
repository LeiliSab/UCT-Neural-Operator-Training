"""
UFWI.processors.lowpass
==============================

A zero-phase Tukey-window low-pass filter implemented as a preprocessing
operator. It can
  1) filter AcquisitionData **in place**, and
  2) wrap a Pulse to produce a *filtered* Pulse lazily.

Examples
--------
>>> # (A) Data-side: low-pass using the upper −6 dB edge of a 0.3 MHz, frac_bw=0.75 pulse
>>> lp = LowpassFilter(f0=0.3e6, frac_bw=0.75, roll=0.2, verbose=True)
>>> lp(acq)  # acq.array is now low-passed

>>> # (B) Pulse-side: low-pass a high-frequency pulse before passing to the operator
>>> pulse_hi = GaussianModulatedPulse(f0=0.5e6, frac_bw=0.75, amp=1.0)
>>> pulse_lp = lp.apply_to_pulse(pulse_hi, remove_dc=True, renorm="l2")
>>> op = WaveOperator(acq, {"sound_speed": 1500.}, record_time, pulse=pulse_lp)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import BaseProcessor
from ..data import AcquisitionData
from ..signals import Pulse

__all__ = ["LowpassFilter"]


# ----------------------------- internal ----------------------------- #
def _tukey_lowpass_nd(traces: np.ndarray, dt: float,
                      f_cut: float, roll: float) -> np.ndarray:
    """
    Vectorized zero-phase low-pass filter (cosine-taper transition).
    `traces` shape (..., T), real-valued.
    """
    n = int(traces.shape[-1])
    freqs = np.fft.rfftfreq(n, dt)

    f_pass = float(f_cut) * max(0.0, 1.0 - float(roll))
    f_stop = float(f_cut)

    H = np.ones_like(freqs, dtype=np.float64)
    m_pass = freqs <= f_pass
    m_stop = freqs >= f_stop
    m_tap = (~m_pass) & (~m_stop)

    H[m_stop] = 0.0
    if np.any(m_tap):
        x = (freqs[m_tap] - f_pass) / (f_stop - f_pass)  # 0..1
        H[m_tap] = 0.5 * (1.0 + np.cos(np.pi * x))

    D = np.fft.rfft(traces, axis=-1)
    D *= H[(None,) * (traces.ndim - 1) + (slice(None),)]
    out = np.fft.irfft(D, n=n, axis=-1)
    return out.astype(traces.dtype, copy=False)


@dataclass
class _FilteredPulse(Pulse):
    """
    Lazy low-pass Pulse wrapper: samples `base` and low-passes in the
    frequency domain with zero phase.
    """
    base: Pulse
    f_cut: float
    roll: float = 0.2
    remove_dc: bool = True
    renorm: Optional[str] = None  # None | "peak" | "l2"
    verbose: bool = False
    _printed: bool = False  # print once on the first sample

    def sample(self, dt: float, nt: int) -> np.ndarray:
        y0 = self.base.sample(dt, nt).astype(np.float32, copy=False)

        # Nyquist clipping
        f_nyq = 0.5 / dt
        f_cut_eff = min(float(self.f_cut), 0.95 * f_nyq)

        if self.verbose and not self._printed:
            cf = None
            try:
                cf = self.base.center_frequency()
            except Exception:
                cf = None
            cf_txt = f"{cf/1e6:.3f} MHz" if isinstance(cf, (int, float)) and cf else "n/a"
            print(
                f"[LP][pulse] base={self.base.__class__.__name__}, cf≈{cf_txt}, "
                f"requested f_cut={self.f_cut/1e6:.3f} MHz, "
                f"Nyquist={f_nyq/1e6:.3f} MHz → effective f_cut={f_cut_eff/1e6:.3f} MHz, "
                f"roll={self.roll:.2f}, remove_dc={self.remove_dc}, renorm={self.renorm}, "
                f"dt={dt*1e6:.3f} µs, nt={nt}"
            )
            self._printed = True

        y = _tukey_lowpass_nd(y0[None, :], dt, f_cut_eff, self.roll)[0]

        if self.remove_dc:
            m = float(y.mean(dtype=np.float64))
            y -= np.float32(m)

        if self.renorm:
            # Normalize relative to the original waveform to avoid changing energy/peak too much
            if self.renorm.lower() == "peak":
                p0 = float(np.max(np.abs(y0))) if np.any(y0) else 0.0
                p1 = float(np.max(np.abs(y))) if np.any(y) else 0.0
                if p1 > 0 and p0 > 0:
                    y *= (p0 / p1)
            elif self.renorm.lower() == "l2":
                e0 = float(np.linalg.norm(y0))
                e1 = float(np.linalg.norm(y))
                if e1 > 0 and e0 > 0:
                    y *= (e0 / e1)

        return y

    def center_frequency(self) -> Optional[float]:
        # After low-pass, the center frequency is no longer reliable; return base as a reference
        try:
            return self.base.center_frequency()
        except Exception:
            return None


# ------------------------------ public ------------------------------ #
class LowpassFilter(BaseProcessor):
    """
    Zero-phase low-pass filter with a cosine-taper (Tukey) transition.

    You can either:
    1) provide an absolute cutoff frequency `f_cut`; or
    2) provide `(f0, frac_bw)` in the spirit of `gausspulse`:
       use the upper −6 dB edge `f2 = f0 * (1 + frac_bw/2)` as `f_cut`.

    Parameters
    ----------
    f0 : float, optional
        Center frequency (Hz). Used with `frac_bw` to derive `f_cut`.
    frac_bw : float, optional
        Fractional bandwidth (defined at −6 dB by default):
        `(f2 - f1) / f0`. The upper edge `f2` is used as `f_cut`.
    roll : float, default 0.2
        Transition ratio; passband edge is defined via
        `f_pass = f_cut * (1 - roll)`.
    f_cut : float, optional
        Absolute cutoff frequency (Hz). If provided, it **takes precedence**
        and `f0/frac_bw` are ignored.
    verbose : bool, default False
        Print chosen parameters and the effective (Nyquist-clipped) cutoff.
    """

    def __init__(self, *,
                 f0: float | None = None,
                 frac_bw: float | None = None,
                 roll: float = 0.2,
                 f_cut: float | None = None,
                 verbose: bool = True) -> None:
        self.f0 = None if f0 is None else float(f0)
        self.frac_bw = None if frac_bw is None else float(frac_bw)
        self.roll = float(roll)
        self._f_cut_abs = None if f_cut is None else float(f_cut)
        self.verbose = bool(verbose)

    # ---- public: resolve f_cut (optionally Nyquist-clipped by dt) ---- #
    def resolve_fcut(self, dt: float | None = None) -> float:
        if self._f_cut_abs is not None:
            f_cut = self._f_cut_abs
        else:
            if self.f0 is None or self.frac_bw is None:
                raise ValueError("LowpassFilter: need either `f_cut` or (`f0` and `frac_bw`).")
            # Upper −6 dB frequency (consistent with scipy.signal.gausspulse)
            f_cut = self.f0 * (1.0 + self.frac_bw / 2.0)

        if dt is not None:
            f_nyq = 0.5 / float(dt)
            f_cut = min(f_cut, 0.95 * f_nyq)
        return float(f_cut)

    # ---- data-side API ---- #
    def __call__(self, data: AcquisitionData) -> None:  # noqa: D401
        """Low-pass ``data.array`` in place."""
        if data.array is None:
            raise ValueError("AcquisitionData.array is empty")
        if data.time is None:
            raise ValueError("AcquisitionData.time must be provided")

        dt = float(np.mean(np.diff(data.time)))
        f_nyq = 0.5 / dt
        f_cut_req = (self._f_cut_abs if self._f_cut_abs is not None
                     else (self.f0 * (1.0 + self.frac_bw)))
        f_cut = self.resolve_fcut(dt)

        if self.verbose:
            print(
                f"[LP][data] requested f_cut={f_cut_req/1e6:.3f} MHz, "
                f"Nyquist={f_nyq/1e6:.3f} MHz → effective f_cut={f_cut/1e6:.3f} MHz, "
                f"roll={self.roll:.2f}, dt={dt*1e6:.3f} µs, array_shape={tuple(data.array.shape)}"
            )

        data.array[:] = _tukey_lowpass_nd(data.array, dt, f_cut, self.roll)

        # Record processing history
        data.ctx["lowpass"] = dict(f_cut=f_cut, f_cut_requested=f_cut_req,
                                   roll=self.roll, dt=dt)

        return data

    # ---- pulse-side API ---- #
    def apply_to_pulse(self, pulse: Pulse, *,
                       remove_dc: bool = True,
                       renorm: Optional[str] = None) -> Pulse:
        """
        Return a new *lazy* Pulse that will be low-passed at sample time using
        the same rules as for the data side.

        Parameters
        ----------
        pulse : Pulse
            Original (possibly high-frequency) pulse.
        remove_dc : bool, default True
            Remove DC bias after low-pass.
        renorm : {"l2","peak",None}, default "l2"
            Post-filter normalization. "l2" approximately preserves energy,
            "peak" preserves peak amplitude, and None disables rescaling.
        """
        f_cut_req = self.resolve_fcut(dt=None)  # will be Nyquist-clipped at sample time
        if self.verbose:
            cf = None
            try:
                cf = pulse.center_frequency()
            except Exception:
                cf = None
            cf_txt = f"{cf/1e6:.3f} MHz" if isinstance(cf, (int, float)) and cf else "n/a"
            print(
                f"[LP][wrap] base={pulse.__class__.__name__}, cf≈{cf_txt}, "
                f"requested f_cut={f_cut_req/1e6:.3f} MHz (will be clipped at sample time), "
                f"roll={self.roll:.2f}, remove_dc={remove_dc}, renorm={renorm}"
            )
        return _FilteredPulse(pulse, f_cut=f_cut_req, roll=self.roll,
                              remove_dc=remove_dc, renorm=renorm,
                              verbose=self.verbose)