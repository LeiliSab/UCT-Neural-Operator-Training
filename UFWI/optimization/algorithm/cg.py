"""
UFWI.optimization.algorithm.cg
====================================

Non-linear Conjugate Gradient (CG_Time) optimizer for Helmholtz full-waveform inversion (FWI).

Key Features
------------
- **Polak–Ribiere update** with Fletcher–Reeves upper bound:

      beta_k = min( max( (g_k · (g_k - g_{k-1})) / (g_{k-1} · g_{k-1}), 0 ),
                   (g_k · g_k) / (g_{k-1} · g_{k-1}) )

- **Closed-form step size alpha_k** using first-order Taylor expansion
  of the data-space misfit:

      alpha = - (g_filtered · d) / ||dREC||^2

- **Mode switching** to control which part of the model is updated:

      mode = 'real'   → updates Re{m}
      mode = 'imag'   → updates Im{m}, respecting sign convention
      mode = 'full'   → updates both parts of m

- **Online Recorder**
  At each iteration k, the following tensors are recorded:

      recorder["vel"]    [:, :, k] → velocity (1 / Re{m})
      recorder["atten"]  [:, :, k] → attenuation (2π * Im{m} * sign)
      recorder["grad"]   [:, :, k] → filtered gradient g_k
      recorder["search"][:, :, k] → search direction d_k

  Use `.get_record()` to retrieve them after optimization.

- **Visualizer Hook**
  If a `viz` object with `update(slow, grad, search_dir)` is passed,
  it will be updated after each model update to remain synchronized with terminal output.

Dependencies
------------
- `HelmholtzOperator` — for geometry and access to the forward/adjoint solver
- `NonlinearLS`       — for misfit and gradient evaluation
- `ringingRemovalFilt` — optional high-pass filter for smoother gradients

This class does **not** directly access or import `HelmholtzSolver`,
respecting modular encapsulation.

API Summary
-----------
CG_Time(mode='real').solve(fun, m0, n_iter=8, viz=viewer)

get_record() → returns a dictionary with 4 arrays (Ny, Nx, N_iter)

reset() — clears all CG_Time state and recorder buffers

Notes
-----
- Per-iteration cost: ≈ one forward + one adjoint + one perturbed solve
- Memory usage is dominated by the recorder; disabling it is not yet implemented
  but could be added easily.
"""
from __future__ import annotations
import time
from typing import Literal, Optional, Dict, List
import numpy as np

from UFWI.data.image_data import ImageData
from UFWI.optimization.operator.functions.ringingRemovalFilt import ringingRemovalFilt
from ..operator.helmholtz import HelmholtzOperator
from ..function.least_squares import NonlinearLS
from .base import Optimizer


class CG(Optimizer):
    """
    Non-linear conjugate-gradient optimiser with built-in recorder.

    Parameters
    ----------
    mode : {'full','real','imag'}, default 'full'
        Which part of the complex model to update.
    c1 : float, default 1e-4
        Armijo parameter (unused in closed-form α but kept for API parity).
    shrink : float, default 0.5
        Back-tracking factor (unused here).
    max_ls : int, default 20
        Maximum line-search iterations (unused: closed-form α).

    Recorder keys
    -------------
    vel, atten, grad, search – each of shape (ny,nx,n_iter)
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        mode: Literal["full", "real", "imag"] = "full",
        *,
        c1: float = 1e-4,
        shrink: float = 0.5,
        max_ls: int = 20,
    ):
        if mode not in ("full", "real", "imag"):
            raise ValueError("mode must be 'full', 'real' or 'imag'")
        self._mode = mode
        self._c1, self._sh, self._max = float(c1), float(shrink), int(max_ls)
        self._d = self._g_prev = None  # CG_Time state

        # recorder & visualiser buffers
        self._rec: Dict[str, List[np.ndarray]] = {k: [] for k in ("vel", "atten", "grad", "search")}
        self._vis_grad: np.ndarray | None = None
        self._vis_search: np.ndarray | None = None

    # ---------------- helper fns ---------------- #
    @staticmethod
    def _proj(tag: str, arr: np.ndarray) -> np.ndarray:
        """Project complex gradient to requested mode."""
        if tag == "full":
            return arr.copy()
        out = np.zeros_like(arr)
        out.real = arr.real
        return out

    @staticmethod
    def _beta(g: np.ndarray, g_prev: np.ndarray) -> float:
        """Polak-Ribière β with FR upper bound."""
        pr = np.vdot(g, g - g_prev).real
        denom = np.vdot(g_prev, g_prev).real + 1e-30
        fr = np.vdot(g, g).real / denom
        return min(max(pr / denom, 0.0), fr)

    # ---------------- closed-form α ---------------- #
    def _closed_alpha(self, d: np.ndarray, g_filt: np.ndarray, op: HelmholtzOperator) -> float:
        ny, nx = op.ny, op.nx
        VSRC     = op.get_field("VSRC")              # (ny,nx,Tx)
        scaling  = op.get_field("scaling")           # (Tx,)
        pert_src = VSRC * d.reshape(ny, nx, 1)

        PERT_WV, _ = op._solve(pert_src, adjoint=False)

        N_tx, N_rx = op.n_tx, op.n_rx
        dREC = np.zeros((N_tx, N_rx), np.complex128)
        for s in range(N_tx):
            idx = op._mask[s]
            if idx.any():
                vals = PERT_WV[:, :, s].ravel(order="F")[op._gid[idx]]
                dREC[s, idx] = -scaling[s] * vals

        num = -np.dot(g_filt.ravel(order="F"), d.ravel(order="F"))
        den = np.sum(np.abs(dREC)**2, dtype=np.float64) + 1e-30
        return num / den

    # ------------------------------------------------------------------ #
    def step(self, g_raw: np.ndarray, m: ImageData, fun: NonlinearLS) -> np.ndarray:
        """Single CG_Time iteration; returns raw gradient for next step."""
        op: HelmholtzOperator = fun._op
        xi, yi = op._xi, op._yi
        vel = 1.0 / np.real(m.array)

        # 1) filter gradient
        g_filt = ringingRemovalFilt(xi, yi, g_raw, np.mean(vel), op._freq, 0.75, np.inf)
        g_proj = self._proj(self._mode, g_filt)

        # 2) direction update
        d = -g_proj if self._d is None else -g_proj + self._beta(g_proj, self._g_prev) * self._d
        self._d, self._g_prev = d, g_proj.copy()

        # 3) step length
        alpha = self._closed_alpha(d, g_filt, op)

        # 4) model update
        if self._mode == "imag":
            m.array[:] += 1j * np.sign(op._sign) * alpha * d
        else:
            m.array[:] += alpha * d

        # 5) recorder & visualiser
        atten = 2 * np.pi * np.imag(m.array) * np.sign(op._sign)
        self._rec["vel"].append(vel.copy())
        self._rec["atten"].append(atten.copy())
        self._rec["grad"].append(g_filt.copy())
        self._rec["search"].append(d.copy())
        self._vis_grad, self._vis_search = g_filt, d

        # 6) next raw gradient
        return fun.gradient(m.array)

    # ------------------------------------------------------------------ #
    def solve(
        self,
        fun: NonlinearLS,
        m0: ImageData,
        *,
        n_iter: int = 10,
        mode: Literal["full", "real", "imag"] = "full",
        viz: Optional[any] = None,
        do_print_time: bool = False,
    ) -> ImageData:
        """
        Run CG_Time for `n_iter` iterations.

        Parameters
        ----------
        fun  : NonlinearLS
        m0   : ImageData
        n_iter : int
        mode    : see __init__
        viz     : object with ``update(slow, grad, search)``, optional
        do_print_time : bool
            Print wall-clock time per iteration.
        """
        if mode not in ("full", "real", "imag"):
            raise ValueError("invalid mode")
        self._mode = mode
        self._d = self._g_prev = None

        fun._op._atten_phase = (mode == "imag")
        g = fun.gradient(m0.array)                          # g₀

        op = fun._op

        for k in range(1, n_iter + 1):
            t0 = time.time()
            g = self.step(g, m0, fun)
            t1 = time.time()

            if do_print_time:
                print(f"[Mode={mode}] Iter {k}/{n_iter}   {t1 - t0:6.2f}s")


            # if viz is not None:
            #     viz.update(m0.current, self._vis_grad, self._vis_search)

            if viz is not None:
                vel = 1.0 / np.maximum(np.real(m0.array), 1e-12)

                if self._mode == "imag":
                    atten_native = 2 * np.pi * np.imag(m0.array) * np.sign(op._sign)
                    self._rec["atten"].append(atten_native.copy())

                    viz.update(
                        vel_est=vel,
                        atten_est=atten_native,
                        grad=self._vis_grad,
                        search_dir=self._vis_search,
                        title=f"f={fun._op._freq / 1e6:.3f} MHz, iter {k}"
                    )
                else:
                    viz.update(
                        vel_est=vel,
                        grad=self._vis_grad,
                        search_dir=self._vis_search,
                        title=f"f={fun._op._freq / 1e6:.3f} MHz, iter {k}"
                    )

        return m0

    # ------------------------------------------------------------------ #
    def get_record(self) -> dict[str, np.ndarray]:
        """Return recorder dict with stacked snapshots."""
        return {k: np.stack(v, axis=2) for k, v in self._rec.items()}

    def reset(self):
        """Clear CG_Time state and recorder (e.g. when switching frequency)."""
        self._d = self._g_prev = None
        self._vis_grad = self._vis_search = None
        for v in self._rec.values():
            v.clear()
