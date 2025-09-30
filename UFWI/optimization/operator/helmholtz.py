"""
UFWI.optimization.operator.helmholtz
==========================================

Single-frequency operator that wraps an internal
:pyclass:`functions.HelmholtzSolver`.

External code should **only** interact with the wave solver via the following interfaces:

* ``forward(m)`` – Produces simulated complex Tx×Rx data F(m)
* ``solve(src, adjoint=False)`` – Solves forward or adjoint wave equation for arbitrary source
* ``get_field(key)`` – Read-only access to cache:
  ``WF`` ``VSRC`` ``scaling`` ``obs_data``
  ``PML`` ``V`` ``freq``

Deprecated interfaces ``residual`` / ``backproj`` have been removed;
their functionality is now part of
:pyclass:`NonlinearLS` and :pyclass:`HelmholtzAdjointGrad`.
"""
from __future__ import annotations
import numpy as np
from types import SimpleNamespace
from ...geometry import ImageGrid2D
from ...data import AcquisitionData
from .base import Operator
from .functions.HelmholtzSolver import HelmholtzSolver  # internal only


class HelmholtzOperator(Operator):
    """Forward/adjoint Helmholtz operator for one discrete frequency."""

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        data: AcquisitionData,
        f_idx: int,
        *,
        sign_conv: int,
        pml_alpha: float,
        pml_size: float,
        use_gpu: bool = False,
    ):
        self._freq = float(data.freqs[f_idx])
        self._sign = int(sign_conv)
        self._a0 = float(pml_alpha)
        self._L_PML = float(pml_size)

        # --- pick Tx subset & measured data --------------------------- #
        self._tx_keep = data.ctx.get("tx_keep", np.arange(data.array.shape[0]))
        self._mask = data.ctx["elem_mask"][self._tx_keep]  # (Tx,Rx)
        self._REC_f = data.array[self._tx_keep, :, f_idx]  # (Tx,Rx)

        # --- geometry & indexing -------------------------------------- #
        self._x_idx = data.ctx["x_idx"][self._tx_keep]
        self._y_idx = data.ctx["y_idx"][self._tx_keep]
        self._gid = np.asarray(data.ctx["grid_lin_idx"], np.int64)  # (Rx,)

        img_grid = data.grid
        self.ny, self.nx = img_grid.shape
        self._xi, self._yi = img_grid.xi, img_grid.yi
        self.n_tx, self.n_rx = self._REC_f.shape

        # runtime cache
        self._cache: SimpleNamespace | None = None
        self._atten_phase = False  # imag-stage flag, set by CG_Time

        self.canUseGPU = use_gpu

    # ------------------------------------------------------------------ #
    # public helpers
    # ------------------------------------------------------------------ #
    def get_field(self, name: str):
        """
        Read-only access to cached tensors/scalars.

        =============== =============================
        key             description
        --------------- -----------------------------
        WF              forward wavefield (ny,nx,Tx)
        VSRC            virtual source basis
        scaling         scaling coefficients to simData
        obs_data        measured data d (Tx,Rx)
        PML, V, freq    solver internals
        =============== =============================
        """
        if self._cache is None:
            raise RuntimeError("forward() must be called first")
        try:
            return getattr(self._cache, name)
        except AttributeError as exc:
            raise KeyError(name) from exc

    def _solve(self, src: np.ndarray, *, adjoint: bool = False):
        """Delegates to the private :pyclass:`HelmholtzSolver`."""
        if self._cache is None:
            raise RuntimeError("forward() must be called first")
        return self._cache.HS.solve(src, adjoint=adjoint)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # forward modelling
    # ------------------------------------------------------------------ #
    def forward(self, m: np.ndarray, kind=None) -> np.ndarray:
        """Return simulated Tx×Rx frequency-domain data F(m)."""
        if self._cache is None or not np.array_equal(m, self._cache.model):
            self._build_cache(m)
        return self._extract_data(self._cache.WF)

    def adjoint(self, src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the adjoint Helmholtz equation H^H w = src for arbitrary `src`.

        Parameters
        ----------
        src : np.ndarray
            Shape (ny, nx, K). Arbitrary adjoint source(s).

        Returns
        -------
        ADJ_WF : np.ndarray
            Adjoint wavefield(s), shape (ny, nx, K).
        VSRC   : np.ndarray
            Virtual-source factor applied to ADJ_WF (same shape).
        """
        if self._cache is None:
            raise RuntimeError("forward() must be called first")
        ADJ_WF, VSRC = self._solve(src, adjoint=True)
        # cache for inspection/diagnostics if desired
        self._cache.ADJ_WF = ADJ_WF  # type: ignore[attr-defined]
        return ADJ_WF, VSRC


    # ------------------------------------------------------------------ #
    # internal
    # ------------------------------------------------------------------ #
    def _build_cache(self, m: np.ndarray) -> None:
        vel = 1.0 / np.real(m)
        atten = np.sign(self._sign) * np.imag(m) * 2 * np.pi
        HS = HelmholtzSolver(
            self._xi, self._yi, vel, atten,
            self._freq, self._sign, self._a0, self._L_PML,
            canUseGPU=self.canUseGPU
        )

        SRC = np.zeros((self.ny, self.nx, self.n_tx), np.complex128)
        for s, (ix, iy) in enumerate(zip(self._x_idx, self._y_idx)):
            SRC[iy, ix, s] = 1.0
        WF, VSRC = HS.solve(SRC, adjoint=False)

        self._cache = SimpleNamespace(
            model=m.copy(), HS=HS, WF=WF, VSRC=VSRC,
            scaling=None, obs_data=self._REC_f,
            PML=HS.PML, V=HS.V, freq=HS.f,
        )

    def _extract_data(self, WF: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n_tx, self.n_rx), np.complex128)
        for s in range(self.n_tx):
            idx = self._mask[s]
            out[s, idx] = WF[:, :, s].ravel(order="F")[self._gid[idx]]
        return out
