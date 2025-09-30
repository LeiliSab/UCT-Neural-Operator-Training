# UFWI/signals/pulse.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    # Optional dependency; only required by GaussianModulatedPulse
    import scipy.signal as sg
except Exception:
    sg = None


class Pulse(ABC):
    """Abstract interface for continuous-time pulses.

    Operators call `sample(dt, nt)` to obtain a discretized waveform once
    the time step `dt` and number of samples `nt` are known.
    """

    @abstractmethod
    def sample(self, dt: float, nt: int) -> np.ndarray:
        """Return a 1-D float32 waveform of shape (nt,)."""
        raise NotImplementedError

    # Convenience hook for logging/querying (optional)
    def center_frequency(self) -> Optional[float]:
        return None


# ---------------------- 1) Gaussian-modulated pulse (with carrier) ---------------------- #
@dataclass
class GaussianModulatedPulse(Pulse):
    """
    Gaussian-modulated sinusoidal pulse (matches `scipy.signal.gausspulse` semantics):

    - f0: carrier frequency (Hz)
    - frac_bw: fractional bandwidth (default defined at −6 dB)
    - amp: amplitude
    - bwr: bandwidth reference level in dB (negative), default −6 dB
    - tpr: time-band truncation level in dB (negative), used to pick the time window
    """
    f0: float = 1.0e6
    frac_bw: float = 0.75
    amp: float = 1.0
    bwr: float = -6.0
    tpr: float = -80.0

    def sample(self, dt: float, nt: int) -> np.ndarray:
        if sg is None:
            raise RuntimeError("GaussianModulatedPulse requires scipy.signal.")

        # Use gausspulse's cutoff to estimate a symmetric time window length
        tc = sg.gausspulse('cutoff', fc=self.f0, bw=self.frac_bw, bwr=self.bwr, tpr=self.tpr)
        n_half = int(np.ceil(tc / dt))
        t = (np.arange(-n_half, n_half + 1, dtype=np.float32)) * dt
        w = self.amp * sg.gausspulse(t, fc=self.f0, bw=self.frac_bw, bwr=self.bwr).astype(np.float32)

        # Write into a length-nt array (place the pulse at the beginning so energy appears early)
        y = np.zeros(nt, dtype=np.float32)
        n = min(nt, w.size)
        y[:n] = w[:n]
        return y

    def center_frequency(self) -> float:
        return self.f0


# ---------------------- 2) Baseband Gaussian pulse (no carrier) ---------------------- #
@dataclass
class GaussianPulse(Pulse):
    """
    Pure Gaussian envelope (no carrier):

    - sigma: time-domain standard deviation (s)
    - or fwhm: full-width at half-maximum (s); provide exactly one of these.
      If neither is given, defaults to a narrow pulse tied to the step size.
    - amp: amplitude
    """
    sigma: Optional[float] = None
    fwhm: Optional[float] = None
    amp: float = 1.0

    def _sigma(self, dt: float) -> float:
        if self.sigma is not None:
            return float(self.sigma)
        if self.fwhm is not None:
            # FWHM = 2*sqrt(2*ln2)*sigma
            return float(self.fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        # If unspecified, choose a narrow pulse relative to the step
        return 6.0 * dt / 6.0  # equivalent to sigma ≈ dt

    def sample(self, dt: float, nt: int) -> np.ndarray:
        sig = self._sigma(dt)
        # Truncate to ±3σ
        T = 3.0 * sig
        t = np.arange(-T, T + dt, dt, dtype=np.float32)
        w = (self.amp * np.exp(-0.5 * (t / sig) ** 2)).astype(np.float32)

        y = np.zeros(nt, dtype=np.float32)
        n = min(nt, w.size)
        y[:n] = w[:n]
        return y