
from __future__ import annotations

import numpy as np
from types import SimpleNamespace
from typing import Union, Optional, List, Tuple

from ..operator.base import Operator
from .base import Function
from ..gradient.base import GradientEvaluator



class NonlinearLS(Function):
    """
    Weighted least-squares objective with optional source-encoding synchronization.

    This wrapper exposes a consistent API for computing
    - residuals: r(m) = F(m) − d,
    - objective values: Φ(m) = ½ ‖w · r(m)‖²,
    - gradients: ∇Φ(m),

    and supports source encoding with either:
    - reusing the exact encodings from the most recent gradient evaluation, or
    - drawing fresh random encodings and aggregating across K trials.

    Parameters
    ----------
    op : Operator
        Forward operator providing:
          - forward(m, kind=...): predicted data,
          - get_field("obs_data"): observed data,
          - (optional) encoding fields and helpers when source encoding is enabled:
              .use_encoding : bool
              .n_tx : int
              .tau_step : int
              .enc_weights : np.ndarray
              .enc_delays  : np.ndarray
              .renew_encoded_obs()
    grad_eval : GradientEvaluator
        Gradient evaluator providing:
          - evaluate(m, q_or_None, kind=...),
          - set_residual_callback(fn),
          - (optional) attributes:
              ._K : int (number of encodings),
              .get_last_encodings() → List[(w_enc, d_enc)].
    weight : int | float | np.ndarray, optional
        Scalar or array broadcastable to the residual shape. Must be positive
        if scalar. Default 1.0.
    sync_value : bool | None, optional
        If True, value() attempts to use the same encodings as the last gradient
        call (when available). If False, value() samples fresh encodings. If None,
        defaults to True.
    reuse_last : bool, optional
        When `sync_value=True` and encodings exist in the gradient evaluator,
        control whether they are actually reused. Default True.
    K_value : int | None, optional
        If sampling fresh encodings, how many K trials to aggregate over for
        value(). If None, uses `grad_eval._K` when > 1; otherwise no K-loop.
    seed : int | None, optional
        RNG seed used when drawing fresh encodings.
    normalize : bool, optional
        If True, normalize gradients to unit max magnitude before returning.

    Attributes
    ----------
    _op : Operator
    _ge : GradientEvaluator
    _w : scalar or ndarray
    _sync_value : bool
    _reuse_last : bool
    _K_value : int | None
    _rng : numpy.random.Generator
    normalize : bool
    _last_misfit : float | None
    _cache : types.SimpleNamespace | None
        Holds the most recent (model copy, forward result, kind) for residual reuse.

    Notes
    -----
    - If `grad_eval` exposes `_w`, it is set to `weight` on construction.
    - `grad_eval.set_residual_callback(self.cache_from_residual)` is registered
      so the misfit can be cached directly from residual computations done inside
      the gradient evaluator.
    """
    def __init__(
            self,
            op: Operator,
            *,
            grad_eval: GradientEvaluator,
            weight: Union[int, float, np.ndarray] = 1.0,
            sync_value: Optional[bool] = True,
            reuse_last: bool = True,
            K_value: Optional[int] = None,
            seed: Optional[int] = None,
            normalize: bool = False,
    ):
        """
        Parameters
        ----------
        op : Operator
        grad_eval : GradientEvaluator
        weight : scalar or array broadcastable to residual shape
        sync_value : Optional[bool]
            If True, use last gradient‐cached encodings; if False, sample fresh.
            Defaults to True.
        reuse_last : bool
            When sync_value=True and encodings exist, whether to actually reuse them.
        K_value : Optional[int]
            If sampling fresh, how many random encodings to average.
            If None, defaults to grad_eval._K (if >1).
        seed : Optional[int]
            RNG seed for fresh sampling.
        normalize : bool
            Whether to normalize the gradient to unit max magnitude.
        """
        if isinstance(weight, (int, float)) and weight <= 0:
            raise ValueError("weight must be positive")

        self._op = op
        self._ge = grad_eval
        self._w = weight
        self._cache: SimpleNamespace | None = None

        # sync_value default True
        self._sync_value = True if sync_value is None else bool(sync_value)
        self._reuse_last = bool(reuse_last)

        # K_value default from gradient evaluator's _K
        if K_value is None:
            ge_K = getattr(self._ge, "_K", None)
            self._K_value = int(ge_K) if (ge_K and ge_K > 1) else None
        else:
            self._K_value = int(K_value) if K_value > 0 else None

        self._rng = np.random.default_rng(seed)
        self.normalize = bool(normalize)

        # pass weight into gradient evaluator if supported
        if hasattr(self._ge, "_w"):
            self._ge._w = weight

        # cache for last misfit
        self._last_misfit: Optional[float] = None
        grad_eval.set_residual_callback(self.cache_from_residual)

    # ------------------------- public API ------------------------------ #
    def set_sync_value(self, flag: bool) -> None:
        self._sync_value = bool(flag)

    def set_reuse_last(self, flag: bool) -> None:
        self._reuse_last = bool(flag)

    def set_K_value(self, K: Optional[int]) -> None:
        self._K_value = int(K) if (K is not None and K > 0) else None

    @property
    def last_misfit(self) -> float:
        """
        Return the most recently computed misfit Φ(m).
        Must call value(m) or gradient(m) first.
        """
        if self._last_misfit is None:
            raise RuntimeError("No misfit cached. Call value(m) or gradient(m) first.")
        return self._last_misfit

    # ------------------------------------------------------------------ #
    def residual(self, m: np.ndarray, *, kind: str=None) -> np.ndarray:
        """Compute and cache r = F(m) − d, return residual."""
        Fm = self._op.forward(m, kind=kind)
        D = self._op.get_field("obs_data")
        self._cache = SimpleNamespace(model=m.copy(), F=Fm, kind=kind)
        return Fm - D

    def value(self, m: np.ndarray, *, kind: str=None) -> float:
        """
        Evaluate the objective Φ(m) = ½ ‖w · r(m)‖² with optional encoding logic.

        Behavior
        --------
        - If source encoding is enabled on the operator and K>1 on the gradient
          evaluator:
            * When encodings are available and `sync_value=True`, reuse those
              encodings to ensure consistency with the most recent gradient call.
            * Otherwise, draw fresh random encodings and aggregate across K trials.
          In both cases the aggregation is a **sum** over trials (to match the gradient
          path), not an average.
        - Otherwise (deterministic path), compute a single forward and residual.

        Parameters
        ----------
        m : np.ndarray
            Model array.
        kind : str
            Parameterization selector passed to the operator.

        Returns
        -------
        float
            Objective value Φ(m) (or the K-sum thereof when encoding is used).

        Side Effects
        ------------
        - Updates `self._last_misfit`.
        - Updates the internal residual cache for potential reuse.

        Notes
        -----
        - The weight `w` is applied elementwise (broadcasted) to the residual.
        - When encoding is reused, the operator's encoding fields are temporarily
          overwritten and then restored.
        """
        use_enc = bool(getattr(self._op, "use_encoding", False))
        K = getattr(self._ge, "_K", None) or 0
        has_K = K > 1

        if use_enc and has_K:
            # if sync_value=True, reuse last encodings if available
            enc_list = self._ge.get_last_encodings()
            if enc_list:
                mis = self._value_reuse_last(m, enc_list, kind=kind)
                self._last_misfit = mis
                return mis
            else:
                mis = self._value_fresh_K(m, K, kind=kind)
                self._last_misfit = mis
                return mis

        # deterministic fallback
        r = self.residual(m, kind=kind)
        rw = r * self._w
        mis = 0.5 * np.vdot(rw, rw).real
        self._last_misfit = mis
        return mis

    def gradient(self, m: np.ndarray, *, kind: str=None) -> np.ndarray:
        """
        Compute the gradient ∇Φ(m) and cache the corresponding misfit.

        Behavior
        --------
        - Encoding path (operator.use_encoding and gradient evaluator has K>1):
            Calls `grad_eval.evaluate(m, None, kind=...)`. The evaluator is expected
            to use its internal encodings and residual callback to keep the misfit
            cache in sync.
        - Deterministic path (no encoding or K<=1):
            Ensures a consistent residual for the given (m, kind), forms q = w · r,
            sets `self._last_misfit = ½ ⟨q, q⟩`, and calls
            `grad_eval.evaluate(m, q, kind=...)`.

        If `normalize=True`, the returned gradient is scaled by 1 / max(|g|), when
        that maximum is positive.

        Parameters
        ----------
        m : np.ndarray
            Model array.
        kind : str
            Parameterization selector passed to the operator.

        Returns
        -------
        np.ndarray
            The gradient array.

        Notes
        -----
        - This method may recompute the residual if none was cached or if the cached
          model/kind mismatch the current inputs.
        - The gradient evaluator may ignore `q` on the encoding path (it receives None).
        """
        use_enc = bool(getattr(self._op, "use_encoding", False))
        has_K = (getattr(self._ge, "_K", None) or 0) > 1

        if use_enc and has_K:
            g = self._ge.evaluate(m, None, kind=kind)
            if self.normalize:
                mval = np.max(np.abs(g))
                if mval > 0:
                    g = g / mval
            # The "misfit" is maintained in the encoding path by either the "residual_callback" or the "value()".
            return g

        # non-encoding or K<=1 path
        # If the cache does not exist or the kind is inconsistent, recalculate the residual.
        if self._cache is None or (getattr(self._cache, "kind", None) != kind) or (
                not np.array_equal(self._cache.model, m)):
            r = self.residual(m, kind=kind)
        else:
            Fm = self._cache.F
            D = self._op.get_field("obs_data")
            r = Fm - D

        q = r * self._w
        self._last_misfit = 0.5 * np.vdot(q, q).real

        g = self._ge.evaluate(m, q, kind=kind)
        if self.normalize:
            mval = np.max(np.abs(g))
            if mval > 0:
                g = g / mval
        return g

    # ======================= internal helpers ============================ #
    def _value_reuse_last(
            self,
            m: np.ndarray,
            enc_list: List[Tuple[np.ndarray, np.ndarray]],
            *,
            kind: str=None,
    ) -> float:
        op = self._op
        wbak, dbak = getattr(op, "enc_weights", None), getattr(op, "enc_delays", None)
        try:
            vals = []
            for w_enc, d_enc in enc_list:
                op.enc_weights = w_enc.copy()
                op.enc_delays = d_enc.copy()
                op.renew_encoded_obs()
                Fm = op.forward(m, kind=kind)
                r = Fm - op.get_field("obs_data")
                vals.append(0.5 * np.vdot(r * self._w, r * self._w).real)
            return float(np.sum(vals))  # consistent with gradient K-sum
        finally:
            op.enc_weights, op.enc_delays = wbak, dbak
            if getattr(op, "use_encoding", False) and wbak is not None:
                op.renew_encoded_obs()

    def _value_fresh_K(self, m: np.ndarray, K: int, *, kind: str) -> float:
        op = self._op
        tau = int(getattr(op, "tau_step", 0))
        wbak, dbak = getattr(op, "enc_weights", None), getattr(op, "enc_delays", None)
        try:
            vals = []
            for _ in range(K):
                op.enc_weights = self._rng.choice([-1, 1], op.n_tx).astype(np.float32)
                op.enc_delays = (
                    self._rng.integers(0, tau + 1, op.n_tx, dtype=np.int32)
                    if tau > 0 else np.zeros(op.n_tx, np.int32)
                )
                op.renew_encoded_obs()
                Fm = op.forward(m, kind=kind)
                r = Fm - op.get_field("obs_data")
                vals.append(0.5 * np.vdot(r * self._w, r * self._w).real)
            return float(np.sum(vals))
        finally:
            op.enc_weights, op.enc_delays = wbak, dbak
            if getattr(op, "use_encoding", False) and wbak is not None:
                op.renew_encoded_obs()

    def value_from_residual(self, r: np.ndarray) -> float:
        """
        Compute Φ(m) = ½‖w·r‖² from residual r.
        """
        rw = r * self._w
        return 0.5 * np.vdot(rw, rw).real

    def cache_from_residual(self, r: np.ndarray) -> float:
        mis = self.value_from_residual(r)
        self._last_misfit = mis
        return mis
