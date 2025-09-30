from __future__ import annotations
import time
from typing import Literal, Optional, Dict, List
import numpy as np

from UFWI.data.image_data import ImageData
from ..function.least_squares import NonlinearLS
from .base import Optimizer
from ...utils.visulizer_multi_mode import Visualizer  # import the Visualizer

# ---------------- Constants ---------------- #
_VEL_MIN, _VEL_MAX = 800.0, 2500.0
_ALPHA_FLOOR, _ALPHA_CEIL = 1.0e-7, 20.0  # global step bounds
_ALPHA_FALLBACK = 10.0  # fallback initial step


class CG_Time(Optimizer):
    """
    Minimal non-linear Conjugate Gradient optimizer with Polak–Ribière direction,
    Barzilai–Borwein (BB-1/BB-2) step initialization, and Armijo backtracking.

    This optimizer supports two parameter "kinds":
    - 'c'     : sound speed (velocity), clipped to [_VEL_MIN, _VEL_MAX]
    - 'alpha' : attenuation, clipped to [_ALPHA_FLOOR, _ALPHA_CEIL]

    It records per-iteration state (velocity/attenuation estimates, gradients,
    search directions, and misfit before/after) and can stream updates to an
    optional `Visualizer`.

    Usage
    -----
    >>> cg = CG_Time()
    >>> cg.solve(fun, m0, n_iter=20, kind="c")       # optimize velocity
    >>> cg.solve(fun, m0, n_iter=20, kind="alpha")   # optimize attenuation

    Notes
    -----
    - The model `m0` is updated **in place**.
    - The objective `fun` must implement the `NonlinearLS` interface used here:
      `value(m, kind=...)`, `gradient(m, kind=...)`, and `last_misfit` cache.
    """
    def __init__(
            self,
            *,
            c1: float = 1e-4,
            shrink: float = 0.5,
            max_ls: int = 10,
            viz: Optional[Visualizer] = None,
    ):
        """
        Parameters
        ----------
        c1 : float, optional
            Armijo sufficient-decrease parameter (0 < c1 < 1). Default 1e-4.
        shrink : float, optional
            Multiplicative backtracking factor (0 < shrink < 1). Default 0.5.
        max_ls : int, optional
            Maximum Armijo backtracking steps. Default 10.
        viz : Visualizer | None, optional
            Optional visualizer that will be updated at each step with the current
            model, gradient, and search direction.

        Attributes
        ----------
        _d : np.ndarray | None
            Current CG_Time search direction.
        _g_prev : np.ndarray | None
            Previous gradient.
        _m_prev : np.ndarray | None
            Previous model (array copy).
        _last_alpha : float
            Last accepted step size (used as fallback initialization).
        _rec : dict[str, list]
            Recorder for velocity/attenuation estimates, gradients, directions, misfit.
        _viz : Visualizer | None
            Optional visualizer.
        """
        self._c1, self._sh, self._mls = float(c1), float(shrink), int(max_ls)

        # state
        self._d = None
        self._g_prev = None
        self._m_prev = None
        self._last_alpha = _ALPHA_FALLBACK
        self._rec = {"vel": [], "atten": [], "grad": [], "search": [], "misfit": []}

        # visualizer
        self._viz = viz

    @staticmethod
    def _stat(arr: np.ndarray) -> str:
        """Return a string with RMS and max absolute value of the array."""
        a = np.abs(arr.ravel())
        return f"RMS={np.sqrt((a * a).mean()):.3e} |max|={a.max():.3e}"

    @staticmethod
    def _beta(g: np.ndarray, g_prev: np.ndarray) -> float:
        """Compute Polak–Ribière or Fletcher–Reeves β."""
        pr = float(np.vdot(g, g - g_prev))
        denom = float(np.vdot(g_prev, g_prev)) + np.finfo(np.float64).eps
        fr = float(np.vdot(g, g)) / denom
        return max(0.0, min(pr / denom, fr))

    @staticmethod
    def _value(fun: NonlinearLS, m: np.ndarray, *, kind: str) -> float:
        """Compute and return the value of the objective function."""
        return float(fun.value(m, kind=kind))

    @staticmethod
    def _clip_inplace(arr: np.ndarray, *, kind: str) -> None:
        """Clip the array in-place based on the kind."""
        if kind == "c":
            np.clip(arr, _VEL_MIN, _VEL_MAX, out=arr)
        else:  # alpha
            np.clip(arr, _ALPHA_FLOOR, _ALPHA_CEIL, out=arr)

    def _armijo(self, fun: NonlinearLS, m: np.ndarray, d: np.ndarray, g: np.ndarray,
                f0: float, a0: float, *, kind: str, verb: bool = False):
        """
        Perform Armijo backtracking line search.
        """
        gd0 = float(np.vdot(g, d))
        a = a0
        abs_t = 1e-5 * max(1.0, abs(f0))
        if verb:
            print(f"    → Armijo start: f0={f0:.3e}, gd0={gd0:.3e}, a0={a0:.3e}")
        for i in range(self._mls):
            m_try = m + a * d
            m_try = m_try.copy()
            self._clip_inplace(m_try, kind=kind)

            f_try = self._value(fun, m_try, kind=kind)
            if verb:
                print(f"      [{i + 1}] try a={a:.3e}, Φ={f_try:.3e}")
            if not (np.isnan(f_try) or np.isinf(f_try)) and f_try <= f0 + self._c1 * a * gd0 + abs_t:
                if verb:
                    print(f"        accepted a={a:.3e}, stop backtracking")
                return a, f_try

            a *= self._sh
            if a < _ALPHA_FLOOR:
                if verb:
                    print("        a below floor, break")
                break

        if verb:
            print(f"    Armijo failed, returning last trial a={a:.3e}, Φ={f_try:.3e}")
        return a, f_try

    def _step(self, g_raw: np.ndarray, m: ImageData, fun: NonlinearLS, *, kind: str, verb: bool = False):
        """
        Perform a single step of the CG_Time algorithm.
        """
        if verb:
            print("  -- CG_Time step begin --")
            print("    raw grad :", self._stat(g_raw))

        if np.iscomplexobj(g_raw):
            raise TypeError("CG_Time expects a real gradient. Did the gradient evaluator return complex?")

        g = g_raw.astype(np.float64, copy=True)
        if verb:
            print("    chosen grad:", self._stat(g), f"(kind={kind})")

        misfit_before = fun.last_misfit

        # search direction
        if self._d is None:
            beta, d = 0.0, -g
            if verb:
                print("    first step → steepest descent")
        else:
            beta = self._beta(g, self._g_prev)
            d = -g + beta * self._d
            if float(np.vdot(g, d)) >= 1e-12:
                beta, d = 0.0, -g
                if verb:
                    print("    lost descent → reset dir")
        if verb:
            print(f"    β = {beta:.3e}, dir {self._stat(d)}")

        # BB-1 / BB-2 initial step size
        if self._m_prev is not None:
            s = (m.array - self._m_prev).ravel("F")
            y = (g - self._g_prev).ravel("F")
            num1, den1 = float(np.dot(s, s)), float(np.dot(s, y))
            if den1 > 1e-20 and num1 > 0.0:
                a0 = num1 / den1
            else:
                num2, den2 = float(np.dot(s, y)), float(np.dot(y, y))
                if den2 > 1e-20 and num2 > 0.0:
                    a0 = num2 / den2
                else:
                    a0 = self._last_alpha
        else:
            a0 = self._last_alpha
        if verb:
            print(f"    α0 = {a0:.3e}")

        # Armijo
        f0 = fun.last_misfit
        alpha, f_new = self._armijo(fun, m.array, d, g, f0, a0, kind=kind, verb=verb)
        self._last_alpha = alpha
        if verb:
            print(f"    chosen α = {alpha:.3e}, Φ = {f_new:.3e}")

        # update model + clip
        self._m_prev = m.array.copy()
        self._g_prev = g.copy()
        m.array += alpha * d
        self._clip_inplace(m.array, kind=kind)

        # compute new gradient
        fun._cache = None
        g_new = fun.gradient(m.array, kind=kind)
        if np.iscomplexobj(g_new):
            raise TypeError("CG_Time expects a real gradient from fun.gradient(..., kind=...).")

        misfit_after = fun.last_misfit

        # record results
        self._d = d
        if kind == "c":
            self._rec["vel"].append(m.array.copy())
            self._rec["atten"].append(np.zeros_like(m.array))
        else:
            self._rec["vel"].append(np.zeros_like(m.array))
            self._rec["atten"].append(m.array.copy())
        self._rec["grad"].append(g.copy())
        self._rec["search"].append(d.copy())
        self._rec["misfit"].append([misfit_before, misfit_after])

        title = f'misfit: {misfit_before:.3e} → {misfit_after:.3e}'

        if self._viz is not None:
            if kind == "c":
                vel_est = m.array
                atten_est = np.zeros_like(m.array)
            else:
                vel_est = np.zeros_like(m.array)
                atten_est = m.array
            self._viz.update(
                vel_est=vel_est,
                atten_est=atten_est,
                grad=g,
                search_dir=d,
                title=title
            )

        if verb:
            print("    Δ|m|_inf =", np.abs(alpha * d).max())
            print(f"    misfit: {misfit_before:.3e} → {misfit_after:.3e}")
            print("  -- CG_Time step end --\n")

        return g_new

    def solve(self, fun: NonlinearLS, m0: ImageData, *, n_iter: int = 10, kind: str, verbose: bool = True):
        """
        Run CG_Time for a fixed number of iterations on a given objective and model.

        Parameters
        ----------
        fun : NonlinearLS
            Objective providing `.gradient(m, kind=...)`, `.value(...)`, and a
            cached `.last_misfit`.
        m0 : ImageData
            Initial model; its `.array` is updated **in place**.
        n_iter : int, optional
            Number of CG_Time iterations to perform. Default 10.
        kind : {"c","alpha"}
            Which parameter to optimize: 'c' (velocity) or 'alpha' (attenuation).
        verbose : bool, optional
            If True, print per-iteration logs. Default True.

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
        self.reset()

        # initial gradient and misfit
        g = fun.gradient(m0.array, kind=kind)
        if np.iscomplexobj(g):
            raise TypeError("CG_Time expects a real gradient from fun.gradient(..., kind=...).")
        misfit0 = fun.last_misfit

        print("=" * 72)
        print(f"CG_Time (PR+BB1/BB-2 + Armijo)  iter={n_iter}  kind={kind}\n misfit={misfit0:.3e}")
        for k in range(1, n_iter + 1):
            if verbose:
                print(f"==== Iter {k}/{n_iter} ====")
            g = self._step(g, m0, fun, kind=kind, verb=verbose)
        return m0

    def get_record(self):
        rec_out: Dict[str, np.ndarray] = {}
        if self._rec["misfit"]:
            for k, v in self._rec.items():
                if k == "misfit":
                    rec_out[k] = np.stack(v, axis=1)
                else:
                    rec_out[k] = np.stack(v, axis=2)
        else:
            rec_out["misfit"] = np.empty((2, 0))
            for k in ("vel", "atten", "grad", "search"):
                rec_out[k] = np.empty((0,))
        return rec_out

    def save_record(self, filename: str) -> None:
        rec = self.get_record()
        np.savez(filename, **rec)
        print(f"Record saved to {filename}")

    def reset(self):
        """Reset internal CG_Time state and clear recorder."""
        self._d = None
        self._g_prev = None
        self._m_prev = None
        self._last_alpha = _ALPHA_FALLBACK
        for k in self._rec:
            self._rec[k].clear()
