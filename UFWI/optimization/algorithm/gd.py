from __future__ import annotations
import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from ...utils.visulizer_multi_mode import Visualizer
from ..function.least_squares import NonlinearLS
from UFWI.data.image_data import ImageData

_VEL_MIN, _VEL_MAX = 800.0, 2500.0
_ALPHA_FLOOR, _ALPHA_CEIL = 1.0e-7, 20.0


class GD:
    """
    Plain real-valued Gradient Descent with optional backtracking line search.

    API mirrors the CG_Time optimizer:
        gd.solve(fun, m0, n_iter=..., kind="c")       # optimize velocity
        gd.solve(fun, m0, n_iter=..., kind="alpha")   # optimize attenuation

    The optimizer updates the model **in place**, records per-iteration snapshots
    (velocity/attenuation, gradient, search direction, misfit before/after), and
    can stream updates to an optional `Visualizer`.

    Parameter kinds
    --------------
    - 'c'     : velocity (clipped to [_VEL_MIN, _VEL_MAX])
    - 'alpha' : attenuation (clipped to [_ALPHA_FLOOR, _ALPHA_CEIL])

    Notes
    -----
    - Gradients must be real-valued.
    - When `backtrack=True`, a simple non-increasing acceptance rule is used
      (Armijo-like), with geometric halving of the step size.
    """

    def __init__(
        self,
        *,
        lr: float = 1e6,
        lipschitz_factor: float = 1.0,
        backtrack: bool = True,
        max_bt: int = 10,
        schedule_fn: Callable[[int, float], float] | None = None,
        viz: Optional[Visualizer] = None,
    ) -> None:
        """
        Parameters
        ----------
        lr : float, optional
            Initial learning rate (used on the first step and as input to `schedule_fn`).
            Default 1e6.
        lipschitz_factor : float, optional
            Scaling factor used to form a simple step-size proxy γ / ||g|| when no
            schedule is provided and k > 0. Default 1.0.
        backtrack : bool, optional
            If True, use backtracking to enforce non-increasing objective. Default True.
        max_bt : int, optional
            Maximum number of backtracking halvings. Default 10.
        schedule_fn : Callable[[int, float], float] | None, optional
            Optional callback for custom step-size scheduling. Called as
            `schedule_fn(iter_index, lr0)` and should return a scalar step size.
        viz : Visualizer | None, optional
            Optional visualizer updated each iteration with current estimates, gradient,
            and search direction.

        Attributes
        ----------
        _lr0 : float
            Initial learning rate.
        _gamma : float
            Lipschitz-like scaling factor for the fallback step rule.
        _backtrack : bool
            Whether backtracking is enabled.
        _max_bt : int
            Maximum backtracking steps.
        _schedule : Callable | None
            User-provided step scheduler.
        _viz : Visualizer | None
            Optional visualizer.
        _rec : dict[str, list[np.ndarray]]
            Per-iteration recorder: "vel", "atten", "grad", "search", "misfit".
        """
        self._lr0 = float(lr)
        self._gamma = float(lipschitz_factor)
        self._backtrack = bool(backtrack)
        self._max_bt = int(max_bt)
        self._schedule = schedule_fn
        self._viz = viz
        # record
        self._rec: Dict[str, List[np.ndarray]] = {
            "vel": [], "atten": [], "grad": [], "search": [], "misfit": []
        }

    @staticmethod
    def _clip_inplace(arr: np.ndarray, *, kind: str) -> None:
        """
        Clip the model array in place according to the parameter kind.

        Parameters
        ----------
        arr : np.ndarray
            Model array to be clipped (modified in place).
        kind : {"c","alpha"}
            'c' clips to [_VEL_MIN, _VEL_MAX]; 'alpha' clips to
            [_ALPHA_FLOOR, _ALPHA_CEIL].
        """
        if kind == "c":
            np.clip(arr, _VEL_MIN, _VEL_MAX, out=arr)
        else:
            np.clip(arr, _ALPHA_FLOOR, _ALPHA_CEIL, out=arr)

    def _step_size(self, k: int, g: np.ndarray) -> float:
        """
        Compute the step size for iteration k.

        Parameters
        ----------
        k : int
            Iteration index (0-based).
        g : np.ndarray
            Current gradient.

        Returns
        -------
        float
            Step size to use this iteration.

        Notes
        -----
        - If `schedule_fn` is provided, returns `schedule_fn(k, _lr0)`.
        - Otherwise returns:
            - `_lr0` for k == 0;
            - `_gamma / (||g||_2 + 1e-12)` for k > 0 (simple 1/L proxy).
        """
        if self._schedule:
            return float(self._schedule(k, self._lr0))
        if k == 0:
            return self._lr0
        return self._gamma / (np.linalg.norm(g) + 1e-12)

    def _one_update(
        self,
        grad_in: np.ndarray,
        m: ImageData,
        fun: NonlinearLS,
        *,
        k_iter: int,
        verbose: bool,
        phi_before: float | None,
        kind: str,
    ) -> tuple[np.ndarray, float | None]:
        """
        Perform a single GD update step.
        """
        if np.iscomplexobj(grad_in):
            raise TypeError("GD expects real-valued gradients, got complex.")

        g = grad_in.astype(np.float64, copy=True)
        g_inf = float(np.max(np.abs(g)))
        g_l2 = float(np.linalg.norm(g))
        alpha = float(self._step_size(k_iter, g))
        upd_inf = alpha * g_inf

        m_prev = m.array.copy()
        misfit_before = float(fun.last_misfit)

        bt_cnt = None
        if not self._backtrack:
            # direct update
            m.array -= alpha * g
            self._clip_inplace(m.array, kind=kind)
            np.nan_to_num(m.array, copy=False)
            g_new = fun.gradient(m.array, kind=kind)
            misfit_after = float(fun.last_misfit)
            phi_after = None
        else:
            # backtracking line search
            if phi_before is None:
                phi_before = float(fun.value(m_prev, kind=kind))
            bt_cnt = 0
            while True:
                m.array[:] = m_prev - alpha * g
                self._clip_inplace(m.array, kind=kind)
                np.nan_to_num(m.array, copy=False)

                misfit_try = float(fun.value(m.array, kind=kind))
                # Armijo-like acceptance rule:
                if (misfit_try <= phi_before) or (bt_cnt >= self._max_bt):
                    g_new = fun.gradient(m.array, kind=kind)
                    misfit_after = float(fun.last_misfit)
                    phi_after = misfit_try
                    break
                alpha *= 0.5
                bt_cnt += 1

        if verbose:
            msg = (
                f"    iter {k_iter:03d} | "
                f"misfit_before={misfit_before:.3e} -> misfit_after={misfit_after:.3e} | "
                f"|g|∞={g_inf:.3e}  α={alpha:.3e}  α|g|∞={upd_inf:.3e}  |g|₂={g_l2:.3e}"
            )
            if self._backtrack:
                msg += f"  (bt used: {bt_cnt})"
            print(msg)

        # record the iteration
        if kind == "c":
            self._rec["vel"].append(m.array.copy())
            self._rec["atten"].append(np.zeros_like(m.array))
        else:
            self._rec["vel"].append(np.zeros_like(m.array))
            self._rec["atten"].append(m.array.copy())
        self._rec["grad"].append(g.copy())
        self._rec["search"].append((-g).copy())
        self._rec["misfit"].append([misfit_before, misfit_after])

        return g_new, phi_after

    def solve(self, fun: NonlinearLS, m0: ImageData, *, n_iter: int = 50, kind: str, verbose: bool = True):
        """
        Run Gradient Descent for a fixed number of iterations.

        Parameters
        ----------
        fun : NonlinearLS
            Objective providing `.value` / `.gradient` and `last_misfit`.
        m0 : ImageData
            Initial model; its `.array` is updated **in place**.
        n_iter : int, optional
            Number of iterations to perform. Default 50.
        kind : {"c","alpha"}
            Which parameter to optimize: 'c' (velocity) or 'alpha' (attenuation).
        verbose : bool, optional
            If True, print iteration logs and timing. Default True.

        Returns
        -------
        ImageData
            The same `m0` instance, updated in place.

        Raises
        ------
        ValueError
            If `kind` is not one of {"c","alpha"}.
        TypeError
            If the objective returns a complex gradient.
        """
        if kind not in ("c", "alpha"):
            raise ValueError("kind must be 'c' or 'alpha'")

        hdr = (
            f"PlainGD  kind={kind}, α₀={self._lr0:.1e}, "
            f"γ={self._gamma}, backtrack={self._backtrack}"
        )
        print("=" * len(hdr))
        print(hdr)

        t0 = time.time()

        # initialize model
        phi = float(fun.value(m0.array, kind=kind)) if self._backtrack else None
        grad = fun.gradient(m0.array, kind=kind)
        if np.iscomplexobj(grad):
            raise TypeError("GD expects real-valued gradients, got complex")

        if verbose:
            g0_inf = float(np.max(np.abs(grad)))
            print(f"[init] |g0|∞={g0_inf:.3e}  η0(lr)={self._lr0:.3e}  misfit0={float(fun.last_misfit):.3e}")

        for k in range(n_iter):
            tic = time.time()

            # search direction is the negative gradient
            search_dir = -grad

            grad, phi = self._one_update(
                grad_in=grad,
                m=m0,
                fun=fun,
                k_iter=k,
                verbose=verbose,
                phi_before=phi,
                kind=kind,
            )

            # update the visualizer if provided
            if self._viz:
                if kind == "c":
                    vel_est = m0.array
                    atten_est = np.zeros_like(m0.array)
                else:
                    vel_est = np.zeros_like(m0.array)
                    atten_est = m0.array
                title = f"misfit: {self._rec['misfit'][-1][0]:.3e} -> {self._rec['misfit'][-1][1]:.3e}"
                self._viz.update(
                    vel_est=vel_est,
                    atten_est=atten_est,
                    grad=search_dir * (-1.0),
                    search_dir=search_dir,
                    title=title,
                )

            dt = time.time() - tic
            if verbose:
                print(
                    f">>>>> iter {k + 1:3d}/{n_iter} done "
                    f"({dt:4.1f}s | total {time.time() - t0:4.1f}s)"
                )

        return m0

    def get_record(self):
        if self._rec["misfit"]:
            return {k: np.stack(v, axis=-1) for k, v in self._rec.items()}
        else:
            return {k: np.empty((0,)) for k in self._rec}

    def reset(self):
        for lst in self._rec.values():
            lst.clear()

    def save_record(self, filename: str) -> None:
        rec = self.get_record()
        np.savez(filename, **rec)
        print(f"Record saved to {filename}")
