from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Union, List, Tuple

from .base import GradientEvaluator
from ..operator.wave_operator import WaveOperator


"""
Time-domain adjoint-state gradient (real-valued).

This evaluator computes ∇Φ(m) for a time-domain wave equation using the
adjoint-state method. The parameter to differentiate is selected via
`kind` at call time:

- kind='c'     → returns ∂Φ/∂c     (float64, same shape as c(x))
- kind='alpha' → returns ∂Φ/∂alpha (float64, same shape as c(x))

Forward/adjoint solves pass `kind` through to the operator. Optional source
encoding is supported via a K-loop over randomized encodings.

Notes
-----
- The operator must expose time step `dt`, `forward()`, `adjoint()` (or
  `adjoint_one_tx()`), `get_forward_fields()`, and model accessors
  `model_c` / `model_a`. When encoding is enabled, it should also provide
  fields/methods such as `use_encoding`, `n_tx`, `tau_step`, `enc_weights`,
  `enc_delays`, and `renew_encoded_obs()`.
- Gradient mode is controlled by `use_first_deriv_product`:
  * True  : uses ∫ (λ_t * u_t * K) dt (central differences)
  * False : uses ∫ (λ * u_tt * K) dt  (central differences)
"""


class AdjointStateGrad(GradientEvaluator):

    def __init__(
            self,
            op: WaveOperator,
            *,
            K: int | None = None,
            weight: Union[int, float, np.ndarray] = 1.0,
            seed: int | None = None,
            kernel_c_fn: Callable[[np.ndarray, WaveOperator], np.ndarray] | None = None,
            kernel_a_fn: Callable[[np.ndarray, WaveOperator], np.ndarray] | None = None,
            use_first_deriv_product: bool = True,
    ) -> None:
        """
        Initialize the adjoint-state gradient evaluator.

        Parameters
        ----------
        op : WaveOperator
            Time-domain forward/adjoint operator with cached forward fields and
            optional source-encoding support.
        K : int | None, optional
            Number of encoding realizations. Effective only if > 1 and
            `op.use_encoding` is True. If None, tries `op.K_default` (>1) else
            disables K-loop.
        weight : int | float | np.ndarray, optional
            Scalar or array broadcastable to residuals; applied as r → w·r before
            forming the adjoint source. Default 1.0.
        seed : int | None, optional
            RNG seed for randomized encodings.
        kernel_c_fn : Callable[[np.ndarray, TimeDomainOperator], np.ndarray] | None
            Sensitivity kernel K(c) for velocity updates. Defaults to
            `2 / c^3` (float64).
        kernel_a_fn : Callable[[np.ndarray, TimeDomainOperator], np.ndarray] | None
            Sensitivity kernel for attenuation updates. Defaults to constant `2`.
        use_first_deriv_product : bool, optional
            If True, accumulate with (λ_t * u_t * K); otherwise with (λ * u_tt * K).

        Attributes
        ----------
        dt : float
            Time step copied from the operator.
        _K : int | None
            Active number of encodings when > 1; otherwise None.
        _w : scalar or ndarray
            Weight used to scale residuals.
        _last_encodings : list[tuple[np.ndarray, np.ndarray]] | None
            The (weights, delays) pairs used in the most recent K-loop.
        _residual_callback : Callable[[np.ndarray], None] | None
            Optional callback invoked with residual(s) after forward evaluations.
        """
        super().__init__(deriv_fn=None)
        self.attach_operator(op)
        self._op = op

        # Determine K value (effective if >1)
        K_auto: Optional[int] = None
        if K is not None and K > 1:
            K_auto = int(K)
        elif getattr(op, "K_default", None):
            kd = int(getattr(op, "K_default"))
            if kd > 1:
                K_auto = kd
        self._K = K_auto

        self._rng = np.random.default_rng(seed)
        self._kc = kernel_c_fn or self._default_kernel_c
        self._ka = kernel_a_fn or self._default_kernel_a
        self._w = weight
        self._use_first_deriv_product = bool(use_first_deriv_product)
        self.dt = op.dt

        # cache for encoding realizations
        self._last_encodings: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

        mode = "lam_t*u_t" if self._use_first_deriv_product else "lam*u_tt"
        print(
            "[AdjointStateGrad] Init → "
            f"use_encoding={getattr(op, 'use_encoding', False)}, "
            f"K={self._K}, weight_type={'scalar' if np.isscalar(weight) else 'array'}, "
            f"grad_mode={mode}"
        )

        self._residual_callback: Optional[Callable[[np.ndarray], None]] = None

    # ---------------------- public small API ------------------------- #
    def get_last_encodings(self) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Return the list of (enc_weights, enc_delays) used in the last K-loop, if any.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]] | None
            Encodings used during the most recent multi-encoding evaluation, or None.
        """
        return self._last_encodings

    def clear_last_encodings(self) -> None:
        """
        Clear any cached encodings from the most recent K-loop.
        """
        self._last_encodings = None

    # ------------------------------------------------------------------ #
    def evaluate(self, m: np.ndarray, q: Optional[np.ndarray] = None, *, kind: str) -> np.ndarray:
        """
        Evaluate the gradient for the selected parameter kind.

        Parameters
        ----------
        m : np.ndarray
            Current model array.
        q : np.ndarray | None, optional
            Weighted residual input (q = w·r). Must be None when K>1 with encoding,
            since residuals are formed internally in that case.
        kind : {'c', 'alpha'}
            Which parameter to differentiate: 'c' (velocity) or 'alpha' (attenuation).

        Returns
        -------
        np.ndarray
            Real-valued gradient with the same spatial shape as `m`'s parameter field.

        Raises
        ------
        ValueError
            If `kind` is not one of {'c','alpha'}, or if `q` is provided while
            encoding with K>1 is active.

        Notes
        -----
        - Alias: `gradient = evaluate`.
        - Prints brief diagnostics (encoding usage, K, and gradient magnitude range).
        """
        if kind not in ("c", "alpha"):
            raise ValueError("kind must be 'c' or 'alpha'")
        print(
            "[AdjointStateGrad] evaluate() — "
            f"use_encoding={getattr(self._op, 'use_encoding', False)}, "
            f"K={self._K}, residual={'provided' if q is not None else 'None'}, kind={kind}"
        )
        g_out = self._core_grad(m, q, kind)
        g_max, g_min = float(np.max(np.abs(g_out))), float(np.min(np.abs(g_out)))
        print(f"[AdjointStateGrad] evaluate() done : |g|_max={g_max:.3e}, |g|_min={g_min:.3e}\n")
        return g_out

    gradient = evaluate

    # ------------------------------------------------------------------ #
    def _core_grad(self, m: np.ndarray, q: Optional[np.ndarray], kind: str) -> np.ndarray:
        """
        Core gradient routine (encoding/single-shot selection and residual handling).

        Parameters
        ----------
        m : np.ndarray
            Current model array.
        q : np.ndarray | None
            Weighted residual(s) (q = w·r). Must be None when K>1 with encoding; may
            be None for single-shot with encoding (a fresh encoding will be drawn).
        kind : {'c', 'alpha'}
            Parameter to differentiate.

        Returns
        -------
        np.ndarray
            Real-valued gradient (float64), shape matching the spatial model.

        Raises
        ------
        ValueError
            If `q` is not None while encoding with K>1 is enabled.
            If `q` is None and encoding is disabled (residual must be supplied).

        Side Effects
        ------------
        - When K>1 and encoding is enabled:
          * Draws encodings, renews encoded observations on the operator,
            and sums gradients over K trials (no averaging).
          * Stores the list of encodings in `_last_encodings`.
          * Calls `_residual_callback` with a stack of raw residuals (shape (K, ...))
            if the callback is set.
        - For single-shot paths, calls `_residual_callback` with the raw residual.
        """
        op = self._op

        # ---------- K-loop averaging (encoding) ---------------------- #
        if self._K and getattr(op, "use_encoding", False):
            if q is not None:
                raise ValueError("q must be None when K>1 and encoding enabled")

            g_sum = np.zeros_like(m, np.float64)
            print(f"[TD-Grad] K-loop averaging start  (K={self._K}, kind={kind})")

            # reset encodings cache
            self._last_encodings = []
            res_list = []

            for k in range(self._K):
                # Random ±1 weights & delays
                op.enc_weights = self._rng.choice([-1, 1], op.n_tx).astype(np.float32)
                tau = getattr(op, "tau_step", 0)
                if tau > 0:
                    op.enc_delays = self._rng.integers(0, tau + 1, op.n_tx, dtype=np.int32)
                else:
                    op.enc_delays = np.zeros(op.n_tx, np.int32)
                op.renew_encoded_obs()

                # call forward to generate residuals
                self._last_encodings.append((op.enc_weights.copy(), op.enc_delays.copy()))

                # compute residuals
                res_raw = (op.forward(m, kind=kind) - op.get_field("obs_data"))
                res = res_raw * self._w
                res_list.append(res_raw)

                g_k = self._single_grad(m, res, kind)  # float64
                g_sum += g_k

                print(f"   ➤ encoding #{k + 1}/{self._K}  |g_k|_max={np.max(np.abs(g_k)):.3e}")

            R = np.stack(res_list, axis=0)
            if self._residual_callback is not None:
                self._residual_callback(R)

            return g_sum

        # ---------- single-shot path (no encoding or K<=1) ------------- #
        if q is None:
            if getattr(op, "use_encoding", False):
                if op.enc_weights is None:
                    op.enc_weights = self._rng.choice([-1, 1], op.n_tx).astype(np.float32)
                    op.enc_delays = np.zeros(op.n_tx, np.int32)
                    op.renew_encoded_obs()
                res_raw = (op.forward(m, kind=kind) - op.get_field("obs_data"))
                res = res_raw * self._w
                if self._residual_callback is not None:
                    self._residual_callback(res_raw)
            else:
                raise ValueError("q cannot be None when encoding disabled")
        else:
            res = q
            self._last_encodings = None

        return self._single_grad(m, res, kind)

    # ------------------------------------------------------------------ #
    def _single_grad(self, m: np.ndarray, res: np.ndarray, kind: str) -> np.ndarray:
        """
        Compute the gradient contribution for a single (possibly encoded) shot set.

        This method ensures a fresh forward cache for the given (m, kind), builds the
        sensitivity kernel K from the current velocity `model_c`, then accumulates
        either
            ∫ (λ_t * u_t * K) dt    if use_first_deriv_product is True, or
            ∫ (λ * u_tt * K) dt     otherwise,
        over time using trapezoidal integration. Accumulation is performed per shot
        and summed.

        Parameters
        ----------
        m : np.ndarray
            Current model array.
        res : np.ndarray
            Weighted residual(s), shape depends on operator mode:
              - Non-encoding: res has shape (Tx, ...); forward fields are
                `op.get_forward_fields()` with shape (Tx, nt, ny, nx).
              - Encoding: res corresponds to the encoded shot; forward fields
                are indexed as `[0]` (nt, ny, nx).
        kind : {'c', 'alpha'}
            Parameter to differentiate.

        Returns
        -------
        np.ndarray
            Real-valued gradient (float64) with shape (ny, nx).

        Notes
        -----
        - Central differences with edge padding are used for time derivatives.
        - If `op.drop_self_rx` is True, uses `op.adjoint_one_tx` per transmitter; else
          calls `op.adjoint` with a single-shot batch slice.
        """
        op, dt = self._op, self._op.dt

        # --- check and prepare forward cache ---
        need_fwd = (op._cache is None) or (op.get_forward_fields() is None)
        if not need_fwd:
            if kind == "c":
                need_fwd = not np.array_equal(m, op.model_c)
            else:
                need_fwd = not np.array_equal(m, op.model_a)
        if need_fwd:
            # print(f"[TD-Grad]   (re)running forward({kind}) to refresh cache")
            op.forward(m, kind=kind)

        # sensitivity kernel K
        c_cur = op.model_c  # 声速用于灵敏度核
        if kind == "c":
            K = self._kc(c_cur, op)  # shape (ny, nx), float64
        else:
            K = self._ka(c_cur, op)

        g = np.zeros_like(c_cur, np.float64)

        def _accum_c(u: np.ndarray, lam: np.ndarray) -> np.ndarray:
            if self._use_first_deriv_product:
                u_t = self._d_dt(u, dt)[1:-1]
                lam_t = self._d_dt(lam, dt)[1:-1]
                return -np.trapz(lam_t * u_t * K[None], dx=dt, axis=0)
            else:
                u_tt = self._d2_dt(u, dt)[1:-1]
                lam_crop = lam[1:-1]
                return np.trapz(lam_crop * u_tt * K[None], dx=dt, axis=0)

        def _accum_a(u: np.ndarray, lam: np.ndarray) -> np.ndarray:
            if self._use_first_deriv_product:
                lam_t = self._d_dt(lam, dt)[1:-1]
                return np.trapz(lam_t * u * K[None], dx=dt, axis=0)
            else:
                lam_crop = lam[1:-1]
                return np.trapz(self._d_dt(lam_crop, dt) * u * K[None], dx=dt, axis=0)

        # --- accumulate over shots ---
        if not getattr(op, "use_encoding", False):
            WF = op.get_forward_fields().astype(np.float64)  # (Tx, nt, ny, nx)
            print(f"[TD-Grad] non-encoding, Tx={WF.shape[0]}, res shape={res.shape}")
            if getattr(op, "drop_self_rx", False):
                for tx in range(WF.shape[0]):
                    u = WF[tx]
                    lam = op.adjoint_one_tx(res[tx], tx)  # (nt, ny, nx)
                    g += _accum_c(u, lam) if kind == "c" else _accum_a(u, lam)
            else:
                for tx in range(WF.shape[0]):
                    u = WF[tx]
                    lam = op.adjoint(res[tx:tx + 1])
                    g += _accum_c(u, lam) if kind == "c" else _accum_a(u, lam)
        else:
            u = op.get_forward_fields().astype(np.float64)[0]
            lam = op.adjoint(res)
            g += _accum_c(u, lam) if kind == "c" else _accum_a(u, lam)

        return g  # float64

    # ------------------------------------------------------------------ #
    @staticmethod
    def _d_dt(field: np.ndarray, dt: float) -> np.ndarray:
        pad = np.pad(field, ((1, 1), (0, 0), (0, 0)), mode="edge")
        return (pad[2:] - pad[:-2]) / (2.0 * dt)

    @staticmethod
    def _d2_dt(field: np.ndarray, dt: float) -> np.ndarray:
        pad = np.pad(field, ((1, 1), (0, 0), (0, 0)), mode="edge")
        return (pad[2:] - 2.0 * pad[1:-1] + pad[:-2]) / (dt * dt)

    @staticmethod
    def _default_kernel_c(c: np.ndarray, _: WaveOperator) -> np.ndarray:
        # 2 / c^3
        c64 = c.astype(np.float64, copy=False)
        return 2.0 / (c64 * c64 * c64)

    @staticmethod
    def _default_kernel_a(c: np.ndarray, _: WaveOperator) -> np.ndarray:
        # constant 2.0
        return 2.0 * np.ones_like(c, np.float64)

    def set_residual_callback(self, fn: Callable[[np.ndarray], None]):
        """Callback after computing residual (possibly K-loop stack)."""
        self._residual_callback = fn
