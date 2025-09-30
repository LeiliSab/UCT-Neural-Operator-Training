from __future__ import annotations
import numpy as np
from typing import Callable, Optional
from .base import GradientEvaluator
from ..operator.helmholtz import HelmholtzOperator


class HelmholtzAdjointGrad(GradientEvaluator):
    """
    Adjoint‐state gradient evaluator for single‐frequency Helmholtz FWI.

    Parameters
    ----------
    op : HelmholtzOperator
    kernel_fn : Callable[[m, op], K], optional
        Sensitivity kernel K(x) = ∂H/∂s at each grid point.
        Defaults to 8π²f²·PML/V.
    """

    def __init__(
            self,
            op: HelmholtzOperator,
            *,
            deriv_fn: Callable[[np.ndarray, HelmholtzOperator], np.ndarray] | None = None,
    ):
        # No deriv_fn passed into base — override evaluate() directly
        super().__init__(deriv_fn=None)
        self.attach_operator(op)
        self._op = op
        self._deriv_fn = deriv_fn or self._default_deriv_fn

    def evaluate(
            self,
            model: np.ndarray,
            data: np.ndarray,
            kind: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute the real‐valued gradient image from adjoint source `data`.

        `data` is the q (e.g. residual).
        """
        # rebuild cache if needed
        if (self._op.get_field("model") is None
                or not np.array_equal(model, self._op.get_field("model"))):
            self._op._build_cache(model)

        WF = self._op.get_field("WF")  # (ny, nx, Tx)
        K = self._deriv_fn(model, self._op)  # (ny, nx)
        VS = K[..., None] * WF  # (ny, nx, Tx)

        ny, nx, n_tx = VS.shape
        ADJ_SRC = np.zeros_like(VS)
        scaling = np.zeros((n_tx,), np.complex128)

        for s in range(n_tx):
            idx = np.where(self._op._mask[s])[0]
            if idx.size:
                sim = WF[:, :, s].ravel(order="F")[self._op._gid[idx]]
                scaling[s] = np.vdot(sim, data[s, idx]) \
                             / (np.vdot(sim, sim) + 1e-30)
                ys, xs = np.unravel_index(self._op._gid[idx], (ny, nx), order="F")
                ADJ_SRC[ys, xs, s] = scaling[s] * sim - data[s, idx]

        # cache scaling for diagnostics
        self._op._cache.scaling = scaling  # type: ignore
        ADJ_WV, _ = self._op.adjoint(ADJ_SRC)

        # correlation to form gradient
        grad = np.empty((ny, nx, n_tx), np.float64)
        sgn = np.sign(self._op._sign)
        for s in range(n_tx):
            vsrc = VS[:, :, s] * scaling[s]
            if self._op._atten_phase:
                vsrc = 1j * sgn * vsrc
            grad[:, :, s] = -np.real(np.conj(vsrc) * ADJ_WV[:, :, s])

        return grad.sum(axis=2)

    @staticmethod
    def _default_deriv_fn(
            model: np.ndarray,
            op: HelmholtzOperator
    ) -> np.ndarray:
        """Default K(x) = 8π² f² · PML / V."""
        freq = op.get_field("freq")
        return (8 * np.pi ** 2 * freq ** 2) * (op.get_field("PML") / op.get_field("V"))

    def set_residual_callback(self, fn: Callable[[np.ndarray], float]) -> None:
        """Optional hook used by some Function wrappers; not needed here."""
        self._res_cb = fn  # store for compatibility; not used in this class
