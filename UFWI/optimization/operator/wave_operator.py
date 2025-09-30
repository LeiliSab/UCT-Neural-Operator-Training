# File: UFWI/optimization/operator/wave_operator.py

from __future__ import annotations

import contextlib
import copy
import os
import time
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, List

import numpy as np
from UFWI.data import AcquisitionData, ImageData
from UFWI.geometry import TransducerArray2D, ImageGrid2D
from .base import Operator
from UFWI.signals import Pulse, GaussianModulatedPulse

# ----------------------------------------------------------------------
# k-Wave C/gpu backend
try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu, kspaceFirstOrder2DC
    from kwave.options.simulation_options import SimulationOptions
    from kwave.options.simulation_execution_options import (
        SimulationExecutionOptions,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("kwave-python C backend not found") from exc


# =========================================================================
#   main operator
# =========================================================================

class WaveOperator(Operator):
    """
    k-Wave 2D forward/adjoint operator with optional source encoding.

    This operator wraps the k-Wave C/GPU backends to provide:
    - Forward simulation producing sensor traces (optionally full wavefields).
    - Adjoint backpropagation from sensor residuals (batch or single-Tx).
    - Optional source encoding using ±1 weights and integer sample delays.
    - Robust handling of element-order ↔ k-Wave-order index mappings.

    Parameters
    ----------
    data : AcquisitionData
        Acquisition container with geometry (ImageGrid2D, TransducerArray2D) and,
        optionally, observed traces. Traces are assumed in **element order**
        (Tx, Rx, T).
    medium_params : dict
        Parameters for `kWaveMedium` (e.g., sound_speed, density, alpha_coeff,
        alpha_power, alpha_mode). Missing keys are filled with reasonable defaults:
        sound_speed=1500 m/s, density=1000 kg/m³, alpha_coeff=zeros, power≈1.01,
        no dispersion.
    record_time : float
        Time (seconds) to record per shot (pre-delay); total simulation window is
        `record_time + tau_max`.
    tau_max : float, optional
        Maximum source-encoding delay (seconds). Default 0.0.
    use_encoding : bool, optional
        If True, uses a single encoded shot per forward/adjoint (Tx dimension=1).
    drop_self_rx : bool, optional
        If True in non-encoding mode, exclude the transmitter’s own receiver
        location when forming the sensor mask. Ignored when `use_encoding=True`.
    record_full_wf : bool, optional
        If True, also record the full pressure field over the domain for every
        shot (costly but required for adjoint-state kernels that need u(x,t)).
    cfl : float, optional
        CFL number used to build the k-Wave time step.
    c_ref : float, optional
        Reference sound speed (m/s) passed to `makeTime`. Default 1580.
    pml_size : int, optional
    pml_alpha : float, optional
        PML thickness/strength for both x and y. Default 10 / 10.0.
    encoding_seed : int | None, optional
        RNG seed for random encoding weights/delays.
    pulse : Pulse | None, optional
        Source pulse object. If None, uses a GaussianModulatedPulse(f0=5e5, frac_bw=0.75).
    scale_source_terms : bool, optional
        Whether to scale source terms inside k-Wave (kept here for compatibility).
    src_mode : {'additive', 'dirichlet'}, optional
        k-Wave source mode for `kSource.p_mode`. Default 'additive'.
    pml_inside : bool, optional
        Whether to place PML inside the computational domain.
    use_gpu : bool, optional
        Choose GPU (`kspace_first_order_2d_gpu`) or CPU (`kspaceFirstOrder2DC`) solver.
    verbose : bool, optional
        If True, let k-Wave print; otherwise suppress k-Wave stdout/stderr.

    Attributes
    ----------
    img_grid : ImageGrid2D
        Original imaging grid (kept for coordinate/index conversions).
    grid : kWaveGrid
        k-Wave computational grid used for simulations.
    medium : kWaveMedium
        k-Wave medium; updated when switching `kind`/model.
    model_c : np.ndarray
    model_a : np.ndarray
        Current real 2D parameter maps (float32) for velocity and attenuation.
    nt : int
    dt : float
    time_axis : np.ndarray
        Discrete time information created via `grid.makeTime`.
    tx_pos : np.ndarray
        (2, n_tx) coordinates of active transmitters.
    src_mask : np.ndarray (ny, nx) bool
    rx_mask : np.ndarray (ny, nx) bool
        k-Wave masks for TX and RX on the grid (Fortran/column-major indexing).
    idx_elem2kw : np.ndarray
    idx_kw2elem : np.ndarray
        Index maps between element order and k-Wave order (Rx dimension).
    use_encoding : bool
    drop_self_rx : bool
    enc_weights : np.ndarray | None
    enc_delays : np.ndarray | None
    tau_step : int
        Integer delay bound in samples (`round(tau_max / dt)`).
    record_full_wf : bool
    _fields : dict
        Field pool with (for example): 'obs_data', 'obs_data_kw', 'fwd_data',
        'fwd_data_kw', 'wf_fwd', 'wf_adj'.
    _cache : SimpleNamespace | None
        Forward cache with last wavefields and model copies.

    Notes
    -----
    - Input/output trace tensors presented to the user are always arranged in
      **element order** (Tx, Rx, T). Internally, k-Wave uses **k-Wave order**
      (column-major flattening of the 2D grid), and this class maintains
      fast conversions between the two.
    - In encoding mode, the forward returns `Tx=1` (single encoded shot).
    """
    # ------------------------------------------------------------------
    def __init__(
            self,
            data: AcquisitionData,
            medium_params: dict,
            record_time: float,
            *,
            tau_max: float = 0.0,
            use_encoding: bool = False,
            drop_self_rx: bool = False,
            record_full_wf: bool = True,
            cfl: float = 0.3,
            c_ref: float = 1580,
            pml_size: int = 10,
            pml_alpha: float = 10.0,
            encoding_seed: Optional[int] = None,
            # ---------- new opts -----------------------------------------
            pulse: Pulse | None = None,
            scale_source_terms: bool = True,
            src_mode: str = "additive",  # 'additive' or 'dirichlet'
            pml_inside: bool = False,
            use_gpu: bool = False,
            verbose: bool = False,
    ):
        super().__init__()

        # ---------- cache / field pool -------------------------------
        self._fields: Dict[str, np.ndarray] = {}

        # ---------- 1. grid ------------------------------------------
        img_grid = data.grid
        self.nx, self.ny = img_grid.nx, img_grid.ny
        self.dx, self.dy = img_grid.spacing
        self.grid = kWaveGrid(N=[self.ny, self.nx], spacing=[self.dy, self.dx])
        self.img_grid = img_grid  # keep reference to original grid

        # ---------- 2. medium ----------------------------------------
        med = dict(medium_params)
        tx_array = data.tx_array
        self.tx_array = tx_array
        self.c_ref = float(c_ref)  # reference sound speed for time step calculation
        c0 = 1500 # default sound speed in water [m/s]
        med.setdefault("sound_speed", np.full((self.ny, self.nx), c0, np.float32))
        med.setdefault("density", np.full((self.ny, self.nx), 1000.0, np.float32))
        med.setdefault("alpha_coeff", np.zeros((self.ny, self.nx), np.float32))
        med.setdefault("alpha_power", 1.01)
        med.setdefault("alpha_mode", "no_dispersion")
        self.medium = kWaveMedium(**med)

        # cache current model (always real 2D arrays)
        self.model_c = np.array(self.medium.sound_speed, dtype=np.float32, copy=True)
        self.model_a = np.array(self.medium.alpha_coeff, dtype=np.float32, copy=True)

        # ---------- 3. time axis (allow space for delay) --------------
        self.tau_max = max(float(tau_max), 0.0)
        T_total = record_time + self.tau_max
        self.grid.makeTime(c_ref, cfl, T_total)
        self.nt, self.dt = int(self.grid.Nt), float(self.grid.dt)

        self.time_axis = np.linspace(0, self.nt * self.dt, self.nt, endpoint=False)

        # deep-copy template grid for k-Wave C backend (avoids resize)
        self._grid_tpl = copy.deepcopy(self.grid)

        # ---------- 4. options ---------------------------------------
        self.sim_opts = SimulationOptions(
            save_to_disk=True,
            pml_x_size=pml_size,
            pml_y_size=pml_size,
            pml_x_alpha=pml_alpha,
            pml_y_alpha=pml_alpha,
            scale_source_terms=bool(scale_source_terms),
            data_cast="single",
            pml_inside=pml_inside,
        )

        self._k_solver = kspace_first_order_2d_gpu if use_gpu else kspaceFirstOrder2DC

        self.exec_opts = SimulationExecutionOptions(is_gpu_simulation=use_gpu)

        # ---------- 5. geometry --------------------------------------
        # only tx positions are needed for transmitter
        self.tx_pos = tx_array.tx_positions.astype(float)  # Keep it for subsequent use in the next launches.
        self.n_tx = self.tx_pos.shape[1]

        # Source mask (k-Wave uses Fortran column-major linear indexing)
        self.src_mask = tx_array.get_tx_mask(self.img_grid).T
        lin_idx = np.flatnonzero(self.src_mask.flatten(order="F"))
        self._row_map: Dict[int, int] = {idx: k for k, idx in enumerate(lin_idx)}

        # The "global element index" sequence of the transmitter
        # (consistent with the column sequence of tx_positions)
        self.tx_elem_indices = np.flatnonzero(tx_array.is_tx)

        # Full ring Rx info (using built-in mask directly)
        self.rx_mask = tx_array.get_rx_mask(self.img_grid).T

        # The linear index and quantity of Rx in k-Wave order (Fortran column-major flat)
        self._rx_lin_idx_full = np.flatnonzero(self.rx_mask.flatten(order="F"))
        self.n_rx_full = int(self.rx_mask.sum())
        self._rx_lin_idx = self._rx_lin_idx_full
        self.n_rx = self.n_rx_full  # The number of external Rx is always a whole ring.

        # ---------- 6. receiver masks -------------------------------
        # The array elements are traversed in sequence; any element that
        # falls on a grid point where rx_mask is True is regarded as an "active receiver".
        lin_each = []
        rx_elem_indices = []
        for i in range(tx_array.n_elements):
            x, y = tx_array.positions[:, i]
            ix, iy = self._xy2idx((x, y))
            if self.rx_mask[iy, ix]:
                rx_elem_indices.append(i)  # Record the "global element index" corresponding to this receiver.
                lin_each.append(iy + ix * self.ny)  # Fortran: iy + ix*ny
        lin_each = np.asarray(lin_each, dtype=np.int64)

        lin_full = self._rx_lin_idx_full
        # The position of "Array element order (only the enumeration order of the receiving subset) → k-Wave order"
        self.idx_elem2kw = np.searchsorted(lin_full, lin_each).astype(np.int64)
        # Anti-permutation: k-Wave order → "Element order (only for the receiving subset)"
        self.idx_kw2elem = np.empty(self.n_rx_full, dtype=np.int64)
        self.idx_kw2elem[self.idx_elem2kw] = np.arange(self.n_rx_full, dtype=np.int64)

        # (Optionally) If it is necessary to return from the "receiving subset enumeration index"
        # to the "global element index" later, keep a mapping.
        self._rx_elem_indices = np.asarray(rx_elem_indices, dtype=np.int64)

        # --- per-Tx mask (only valid when sequential) ----------------
        self.use_encoding = bool(use_encoding)
        self.drop_self_rx = bool(drop_self_rx)
        self.rng = np.random.default_rng(encoding_seed)
        self.enc_weights: Optional[np.ndarray] = None  # ±1 weights
        self.enc_delays: Optional[np.ndarray] = None  # integer samples
        self.tau_step = int(round(self.tau_max / self.dt))  # ≥0

        self.rx_mask_per_tx: Optional[np.ndarray] = None
        if self.use_encoding and self.drop_self_rx:
            print(
                "[TDO-FWD][WARN ] drop_self_rx=True is ignored when use_encoding=True. Use weighting in the misfit instead.")
            self.drop_self_rx = False

        if self.drop_self_rx and not self.use_encoding:
            self.rx_mask_per_tx = np.empty((self.n_tx, self.ny, self.nx), dtype=bool)
            # Note: The tx_idx of get_rx_mask_for_tx is the "global element index".
            for k, elem_idx in enumerate(self.tx_elem_indices):
                self.rx_mask_per_tx[k] = tx_array.get_rx_mask_for_tx(self.img_grid, elem_idx, True).T

        # ---------- 7. pulse -----------------------------------------
        # Directly receive the Pulse object; if not provided, use the default Gaussian-modulated pulse.
        if pulse is None:
            default_pulse = GaussianModulatedPulse(f0=5e5, frac_bw=0.75, amp=1.0)
            w = default_pulse.sample(self.dt, self.nt)  # shape=(nt,)
        else:
            w = pulse.sample(self.dt, self.nt)  # shape=(nt,)

        type = pulse.__class__.__name__
        attrs = pulse.__dict__
        self._pulse_info = {
            "Type": type,
            'attrs': {k: v for k, v in attrs.items()},
        }

        # If the length is less than nt, pad zeros at the end; if it is longer, truncate it to nt.
        if w.size < self.nt:
            w = np.pad(w, (0, self.nt - w.size))
        elif w.size > self.nt:
            w = w[: self.nt]

        # Save as (1, nt) for k-Wave to use.
        self.pulse = w[np.newaxis, :].astype(np.float32)

        # ---------- 8. sensors  --------------------------
        self.record_full_wf = bool(record_full_wf)
        self.snk_rx = kSensor(mask=self.rx_mask, record=["p"])

        # ---------- 9. runtime cache ---------------------------------
        self._cache: Optional[SimpleNamespace] = None
        self.obs_data_full: Optional[np.ndarray] = None  # (Tx, n_rx_full, nt) in k-Wave order

        # ---------- 10. user-provided obs ----------------------------
        # Agreement: The Rx dimension of the AcquisitionData.array passed in from the outside is the "element sequence".
        if data.array is not None:
            if data.array.shape[0] != self.n_tx:
                print(f"[TDO-FWD][ERROR] Tx dim mismatch: got {data.array.shape[0]}, expect {self.n_tx}")
                raise ValueError("Tx dimension mismatch in AcquisitionData")
            if data.array.shape[1] != self.n_rx_full:
                print(f"[TDO-FWD][ERROR] Rx dim mismatch: got {data.array.shape[1]}, expect {self.n_rx_full}")
                raise ValueError(
                    f"AcquisitionData second dim must be {self.n_rx_full} (full-ring)."
                )
            # Element sequence → k-Wave sequence
            data_kw = data.array[:, self.idx_kw2elem, :].astype(np.float64, copy=False)
            if self.use_encoding:
                self.set_obs(data_kw[:, self.idx_elem2kw, :])  # The expected element sequence of set_obs; reverse recovery here.
            else:
                # The internal sequence of kw is saved, and the external field provides the element sequence.
                self.obs_data_full = data_kw
                self._fields["obs_data"] = self.obs_data_full[:, self.idx_elem2kw, :]
                self._fields["obs_data_kw"] = self.obs_data_full

        # ---------- 11. misc -----------------------------------------
        self.src_mode = str(src_mode).lower()

        self._src_scale = 1.0 if scale_source_terms else 1.0

        # ---- Init print (one line, coordinated with gradient style) --
        print(
            "[WaveOperator] Init → "
            f"Tx={self.n_tx}, Rx={self.n_rx_full}, nt={self.nt}, dt={self.dt:.3e}s, "
            f"grid=({self.ny}×{self.nx},{self.dy:.2e}×{self.dx:.2e}m), "
            f"c_ref={self.c_ref:.1f}m/s, cfl={cfl}, pml={pml_size}, enc={self.use_encoding}, "
            f"tau_max={self.tau_max:.3e}s, dsrx={self.drop_self_rx}, full_wf={self.record_full_wf}, "
            f"src_mode={self.src_mode}, rx_order=element,"
            f" pulse={self._pulse_info}"
        )

        self.verbose = verbose

    # ================================================================
    # helpers
    # ================================================================
    def _xy2idx(self, xy) -> Tuple[int, int]:
        """Convert (x,y) [m] to (ix,iy) indices."""
        return self.img_grid.coord2index(*xy)

    @staticmethod
    def _reshape(raw, nt, ny, nx):
        return raw.reshape((nt, ny, nx), order="F")

    def _update_medium_from_model(self, model: np.ndarray, *, kind: str) -> None:
        """
        Update one parameter from a single model array.
        kind: 'c' or 'alpha'
        """
        if kind not in ("c", "alpha"):
            raise ValueError("kind must be 'c' or 'alpha'")
        arr = np.asarray(model, dtype=np.float32)
        if arr.shape != (self.ny, self.nx):
            raise ValueError(f"{kind} shape mismatch, got {arr.shape}, expect {(self.ny, self.nx)}")
        if kind == "c":
            self.medium.sound_speed = arr
            self.model_c = arr
        else:
            # IMPORTANT: alpha is expected in k-Wave units: dB / (MHz^y · cm)
            self.medium.alpha_coeff = arr
            self.model_a = arr

    # ================================================================
    # observation helpers
    # ================================================================
    def set_obs(self, data_tx_rx_t: np.ndarray) -> None:
        """
        Set up the observation (with the external input of **element order** (Tx, n_rx, nt)),
        internally save it in k-Wave order for encoding/returning; and store both element/kw views in _fields simultaneously.
        """
        if data_tx_rx_t.shape[0] != self.n_tx:
            print(f"[TDO-ENC][ERROR] Tx dim mismatch: got {data_tx_rx_t.shape[0]}, expect {self.n_tx}")
            raise ValueError("Tx dimension mismatch in set_obs()")
        if data_tx_rx_t.shape[1] != self.n_rx_full:
            print(f"[TDO-ENC][ERROR] Rx dim mismatch: got {data_tx_rx_t.shape[1]}, expect {self.n_rx_full}")
            raise ValueError(
                f"Expected obs to have n_rx={self.n_rx_full}, got {data_tx_rx_t.shape[1]}"
            )
        pad_len = 0
        if data_tx_rx_t.shape[-1] < self.nt:
            pad_len = self.nt - data_tx_rx_t.shape[-1]
            data_tx_rx_t = np.pad(data_tx_rx_t, ((0, 0), (0, 0), (0, pad_len)))

        # Element sequence → k-Wave sequence
        data_kw = data_tx_rx_t[:, self.idx_kw2elem, :]
        self.obs_data_full = data_kw.astype(np.float64, copy=False)

        # External field: Element sequence
        if self.use_encoding:
            if self.enc_weights is not None:
                self.renew_encoded_obs()
            else:
                enc_kw = self.obs_data_full  # Not encoded yet
                self._fields["obs_data"] = enc_kw[:, self.idx_elem2kw, :]
                self._fields["obs_data_kw"] = enc_kw
        else:
            self._fields["obs_data"] = self.obs_data_full[:, self.idx_elem2kw, :]
            self._fields["obs_data_kw"] = self.obs_data_full

        # logging
        Tx, Rx, nt = data_tx_rx_t.shape
        print(f"[TDO-ENC] set_obs | shape={Tx}×{Rx}×{nt}, order=element, pad={pad_len} samp")

    def _shift_right(self, arr: np.ndarray, d: int) -> np.ndarray:
        """Return arr shifted right by d (0-padding), arr shape (..., nt)."""
        if d == 0:
            return arr
        return np.pad(arr, ((0, 0), (d, 0)))[:, : self.nt]

    def get_encoded_obs(self) -> np.ndarray:
        """
        Return the encoded observation in the **k-Wave order** (1, n_rx, nt),
        for internal calculation purposes only. For external viewing, please use _fields['obs_data'] (element order).
        """
        if (
                self.obs_data_full is None
                or self.enc_weights is None
                or self.enc_delays is None
        ):
            raise RuntimeError("obs_data_full or encoding vectors not ready")
        enc = np.zeros((1, self.n_rx_full, self.nt), dtype=self.obs_data_full.dtype)
        for w, d, shot in zip(self.enc_weights, self.enc_delays, self.obs_data_full):
            enc += w * self._shift_right(shot, int(d))
        return enc

    def renew_encoded_obs(self):
        if self.use_encoding:
            enc_kw = self.get_encoded_obs()  # k-Wave order (1, n_rx_full, nt)
            self._fields["obs_data"] = enc_kw[:, self.idx_elem2kw, :]  # element order (1, n_rx, nt)
            self._fields["obs_data_kw"] = enc_kw
            # logging
            dmin = int(np.min(self.enc_delays)) if self.enc_delays is not None else 0
            dmax = int(np.max(self.enc_delays)) if self.enc_delays is not None else 0
            # print(f"[TDO-ENC] renew   | shape=1×{self.n_rx_full}×{self.nt}, order=element, delays={dmin}-{dmax} samp, weights=±1")

    def get_field(self, key: str) -> np.ndarray:
        if key not in self._fields:
            raise KeyError(f"field '{key}' not found")
        return self._fields[key]

    def _n_rx_print(self, tx_idx: Optional[int] = None) -> int:
        # In encoding mode, it is always a complete loop;
        # when not encoding and discarding from self-reception, count according to the current TX mask.
        if self.use_encoding:
            return self.n_rx_full
        if self.drop_self_rx and tx_idx is not None:
            return int(self.rx_mask_per_tx[tx_idx].sum())
        return self.n_rx_full

    # ================================================================
    # k-Wave low-level wrapper
    # ================================================================
    def _run_sim(
            self,
            src_mask,
            p_mat,
            *,
            full_wf: bool,
            sensor_mask_override: Optional[np.ndarray] = None,
    ):
        """Run kspaceFirstOrder2D with fresh grid & medium each time.
        """
        p_mat = np.ascontiguousarray(p_mat, dtype=np.float32)

        grid = copy.deepcopy(self._grid_tpl)
        medium = copy.deepcopy(self.medium)

        ks = kSource()
        ks.mask = ks.p_mask = src_mask
        ks.p = p_mat
        ks.p_mode = self.src_mode

        # The mask collected by k-Wave (when full_wf=True, to obtain the full wave field, the sensor is still set to the entire domain)
        if full_wf:
            raw_mask = np.ones((self.ny, self.nx), bool)
        else:
            raw_mask = sensor_mask_override if sensor_mask_override is not None else self.rx_mask

        sensor = kSensor(mask=raw_mask, record=["p"])

        if self.verbose:
            out = self._k_solver(
                grid, ks, sensor, medium, self.sim_opts, self.exec_opts
            )
        else:
            _null = open(os.devnull, "w")
            with contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
                out = self._k_solver(
                    grid, ks, sensor, medium, self.sim_opts, self.exec_opts
                )
            _null.close()

        # ------------------ full wavefield branch --------------------
        if full_wf:
            # out["p"] is (nt, ny*nx)
            raw = out["p"]
            wf = self._reshape(raw, self.nt, self.ny, self.nx)

            if sensor_mask_override is None:
                # full ring
                lin_cur = self._rx_lin_idx_full
                rec_used = raw[:, lin_cur].T  # (n_rx_full, nt) in k-Wave order
                rec_full = rec_used
            else:
                lin_full = self._rx_lin_idx_full
                lin_cur = np.flatnonzero(sensor_mask_override.flatten(order="F"))
                rec_used = raw[:, lin_cur].T  # (n_rx_cur, nt)
                pos = np.searchsorted(lin_full, lin_cur)
                rec_full = np.zeros((self.n_rx_full, self.nt), dtype=rec_used.dtype)
                rec_full[pos, :] = rec_used

            return wf, rec_full  # rec_full is (n_rx_full, nt) in k-Wave order

        # ------------------ only sensor traces branch ----------------
        raw = out["p"]  # (nt, n_rx_current)

        if sensor_mask_override is None:
            return None, raw.T  # (n_rx_full, nt) in k-Wave order

        lin_full = self._rx_lin_idx_full
        lin_cur = np.flatnonzero(sensor_mask_override.flatten(order="F"))
        pos = np.searchsorted(lin_full, lin_cur)
        rec_full = np.zeros((self.n_rx_full, self.nt), dtype=raw.dtype)
        rec_full[pos, :] = raw.T
        return None, rec_full  # k-Wave order (n_rx_full, nt)

    # ================================================================
    # forward
    # ================================================================
    def forward(self, model: np.ndarray, *, kind: str = "c") -> np.ndarray:
        """
        Compute forward sensor traces after updating exactly one parameter.

        Parameters
        ----------
        model : np.ndarray, shape (ny, nx), dtype float32
            Parameter map to apply (velocity or attenuation).
        kind : {'c', 'alpha'}, optional
            Which parameter `model` represents. Default 'c'.

        Returns
        -------
        np.ndarray
            Predicted traces in **element order**, shape:
              - (1, n_rx_full, nt) in encoding mode,
              - (n_tx, n_rx_full, nt) in sequential (non-encoding) mode.

        Side Effects
        ------------
        - Updates internal medium and model caches.
        - Populates `_fields['fwd_data']` (element order) and `_fields['fwd_data_kw']`
          (k-Wave order). If `record_full_wf=True`, also stores `_fields['wf_fwd']`.
        """
        self._update_medium_from_model(model, kind=kind)
        return self._forward_impl()

    def _forward_impl(self) -> np.ndarray:
        """Run forward using the CURRENT self.medium (no parameter update here)."""
        # Generate encoding vectors once per forward run series
        if self.use_encoding and self.enc_weights is None:
            self.enc_weights = self.rng.choice([-1, 1], self.n_tx).astype(np.float32)
            self.enc_delays = (
                self.rng.integers(0, self.tau_step + 1, self.n_tx, dtype=np.int32)
                if self.tau_step > 0
                else np.zeros(self.n_tx, np.int32)
            )
            if self.obs_data_full is not None:
                self.renew_encoded_obs()

        WF_list, REC_list = [], []
        full = self.record_full_wf

        # ---------- encoded -----------------------------------------
        if self.use_encoding:
            t0 = time.time()

            p_mat = np.zeros((self.src_mask.sum(), self.nt), np.float32)
            for w, d, (x, y) in zip(self.enc_weights, self.enc_delays, self.tx_pos.T):
                ix, iy = self._xy2idx((x, y))
                row = self._row_map[iy + ix * self.ny]
                end = d + self.pulse.shape[1]
                if end > self.nt:
                    end = self.nt
                seg_len = end - d
                if seg_len > 0:
                    p_mat[row, d:end] = w * self.pulse[0, :seg_len] / self._src_scale

            wf, rec_kw = self._run_sim(self.src_mask, p_mat, full_wf=full)
            if wf is not None:
                WF_list.append(wf)
            REC_list.append(rec_kw[None])  # k-Wave 顺序

            elapsed = time.time() - t0
            dmin = int(np.min(self.enc_delays)) if self.enc_delays is not None else 0
            dmax = int(np.max(self.enc_delays)) if self.enc_delays is not None else 0

            nrx = self._n_rx_print()
            print(
                f"[TDO-FWD] enc | nt={self.nt}, rx={nrx}, delays={dmin * self.dt * 1e6:.1f}–{dmax * self.dt * 1e6:.1f} µs | done in {elapsed:.2f}s"
            )

        # ---------- sequential --------------------------------------
        else:
            mask_single = np.zeros_like(self.src_mask)
            # progress width
            w = len(str(self.n_tx))
            t_sum = 0.0
            t_all_start = time.time()

            for i, (x, y) in enumerate(self.tx_pos.T):
                start_i = time.time()
                mask_single[:] = False
                ix, iy = self._xy2idx((x, y))
                mask_single[iy, ix] = True

                # Each independent pulse of Tx avoids side effects caused by modifications at the lower level.
                pulse_single = (self.pulse / self._src_scale).astype(np.float32, copy=True)

                if self.drop_self_rx:
                    wf, rec_kw = self._run_sim(
                        mask_single,
                        pulse_single,
                        full_wf=full,
                        sensor_mask_override=self.rx_mask_per_tx[i],
                    )
                else:
                    wf, rec_kw = self._run_sim(mask_single, pulse_single, full_wf=full)

                if wf is not None:
                    WF_list.append(wf)
                REC_list.append(rec_kw[None])  # k-Wave order

                t_i = time.time() - start_i
                t_sum += t_i

                nrx = self._n_rx_print(i)  # current Tx's Rx count
                print(
                    f"[TDO-FWD] {i + 1:0{w}d}/{self.n_tx} | tx=({x:+.3f},{y:+.3f})m | nt={self.nt}, rx={nrx} |  done in {t_i:.2f}s"
                )

            total = time.time() - t_all_start
            mean_shot = t_sum / self.n_tx if self.n_tx > 0 else 0.0

            print(f"[TDO-FWD] Done | shots={self.n_tx}, total={total:.2f}s, mean/shot={mean_shot:.2f}s")

        # k-Wave returns (nt, n_rx_full) in column-major order
        Fm_kw = np.concatenate(REC_list, axis=0).astype(np.float64)  # (Tx or 1, n_rx_full, nt)

        # Convert to element order (Tx, n_rx, nt) for external use
        Fm = Fm_kw[:, self.idx_elem2kw, :]

        self._cache = SimpleNamespace(
            model_c=self.model_c.copy(),
            model_a=self.model_a.copy(),
            WF=(np.stack(WF_list) if WF_list else None),
        )
        self._fields["fwd_data"] = Fm  # element order (Tx, n_rx, nt)
        self._fields["fwd_data_kw"] = Fm_kw  # k-Wave order (Tx or 1, n_rx_full, nt)
        if full and WF_list:
            self._fields["wf_fwd"] = self._cache.WF
        return Fm

    # ================================================================
    # adjoint (batch / encoded)
    # ================================================================
    def adjoint(self, residual: np.ndarray) -> np.ndarray:
        """
        Back-propagate sensor residuals (batch version).

        Parameters
        ----------
        residual : np.ndarray
            Residual traces in **element order**:
              - (1, n_rx_full, nt) when `use_encoding=True`,
              - (n_tx, n_rx_full, nt) otherwise.

        Returns
        -------
        np.ndarray
            Adjoint wavefield λ(t, y, x) with shape (nt, ny, nx), dtype float64.

        Notes
        -----
        - Residuals are converted to k-Wave order, time-reversed, integrated over
          time (cumulative sum × dt) to obtain pressure, and used as a k-Wave source.
        - In non-encoding mode, loops over Tx; when `drop_self_rx=True`, the sensor
          subset for that Tx is respected.
        - Stores the result in `_fields['wf_adj']`.
        """
        lam = np.zeros((self.nt, self.ny, self.nx), np.float32)

        # element order → k-Wave order
        residual_kw = residual[:, self.idx_kw2elem, :].astype(np.float32, copy=False)

        if self.use_encoding:
            t0 = time.time()
            q_rev = residual_kw[0][:, ::-1]  # (n_rx_full, nt)

            q_rev = np.cumsum(q_rev, axis=1) * self.dt  # integrate to get pressure
            p_src = q_rev / self._src_scale

            wf, _ = self._run_sim(self.rx_mask, p_src, full_wf=True)
            lam += wf[::-1]

            elapsed = time.time() - t0

            nrx = self._n_rx_print()

            print(f"[TDO-ADJ] enc | nt={self.nt}, rx={nrx} | done in {elapsed:.2f}s")
        else:
            N = residual_kw.shape[0]
            w = len(str(N))
            t_sum = 0.0
            t_all_start = time.time()

            for tx in range(N):
                start_i = time.time()
                if self.drop_self_rx:
                    lin_full = self._rx_lin_idx_full
                    lin_cur = np.flatnonzero(self.rx_mask_per_tx[tx].flatten(order="F"))
                    pos = np.searchsorted(lin_full, lin_cur)

                    q_rev_used = residual_kw[tx][pos][:, ::-1]

                    q_rev_used = np.cumsum(q_rev_used, axis=1) * self.dt  # integrate to get pressure
                    p_src = q_rev_used / self._src_scale

                    wf, _ = self._run_sim(
                        self.rx_mask_per_tx[tx],
                        p_src,
                        full_wf=True,
                        sensor_mask_override=self.rx_mask_per_tx[tx],
                    )
                else:
                    q_rev = residual_kw[tx][:, ::-1]
                    q_rev = np.cumsum(q_rev, axis=1) * self.dt  # integrate to get pressure
                    p_src = q_rev / self._src_scale
                    wf, _ = self._run_sim(self.rx_mask, p_src, full_wf=True)

                lam += wf[::-1]

                t_i = time.time() - start_i
                t_sum += t_i

                nrx = self._n_rx_print(tx)

                print(
                    f"[TDO-ADJ] {tx + 1:0{w}d}/{N} | nt={self.nt}, rx={nrx} | "
                    f"done in {t_i:.2f}s"
                )

            total = time.time() - t_all_start
            mean_shot = t_sum / N if N > 0 else 0.0
            print(f"[TDO-ADJ] Done | shots={N}, total={total:.2f}s, mean/shot={mean_shot:.2f}s")

        lam = lam.astype(np.float64)
        self._fields["wf_adj"] = lam
        return lam

    # ================================================================
    # NEW: single-Tx adjoint for drop_self_rx=True
    # ================================================================
    def adjoint_one_tx(self, residual_tx: np.ndarray, tx_idx: int) -> np.ndarray:
        """
        Back-propagate the residual for ONE Tx (non-encoding mode).

        Parameters
        ----------
        residual_tx : (n_rx_full, nt) array
            Residual trace for the given Tx,
        tx_idx : int
            The global Tx index (0-based).

        Returns
        -------
        lam : (nt, ny, nx) float64
            The adjoint wavefield corresponding to this Tx residual.
        """
        if self.use_encoding:
            raise RuntimeError("adjoint_one_tx() is only meant for non-encoding mode.")
        if not self.drop_self_rx:
            return self.adjoint(residual_tx[None, ...])

        if residual_tx.shape[0] != self.n_rx_full or residual_tx.shape[1] != self.nt:
            print(
                f"[TDO-ADJ][ERROR] residual_tx shape mismatch: got {residual_tx.shape}, expect ({self.n_rx_full},{self.nt})")
            raise ValueError(
                f"residual_tx must be (n_rx_full={self.n_rx_full}, nt={self.nt})"
            )
        if tx_idx < 0 or tx_idx >= self.n_tx:
            print(f"[TDO-ADJ][ERROR] tx_idx out of range: got {tx_idx}, valid [0,{self.n_tx})")
            raise IndexError(f"tx_idx out of range [0, {self.n_tx})")

        start = time.time()

        lam = np.zeros((self.nt, self.ny, self.nx), np.float32)

        residual_kw = residual_tx[self.idx_kw2elem].astype(np.float32, copy=False)

        lin_full = self._rx_lin_idx_full
        lin_cur = np.flatnonzero(self.rx_mask_per_tx[tx_idx].flatten(order="F"))
        pos = np.searchsorted(lin_full, lin_cur)

        q_rev_used = residual_kw[pos][:, ::-1]
        q_rev_used = np.cumsum(q_rev_used, axis=1) * self.dt  # integrate to get pressure
        p_src = q_rev_used / self._src_scale

        wf, _ = self._run_sim(
            self.rx_mask_per_tx[tx_idx],
            p_src,
            full_wf=True,
            sensor_mask_override=self.rx_mask_per_tx[tx_idx],
        )

        lam += wf[::-1]
        lam = lam.astype(np.float64)

        end = time.time()

        nrx = self._n_rx_print(tx_idx)

        print(f"[TDO-ADJ] one_tx | tx={tx_idx + 1}/{self.n_tx} | nt={self.nt}, rx={nrx} | done in {end - start:.2f}s")
        return lam

    # ================================================================
    # public hooks
    # ================================================================
    def get_forward_fields(self) -> np.ndarray:
        if self._cache is None or self._cache.WF is None:
            raise RuntimeError("forward(record_full_wf=True) was not run")
        return self._cache.WF

    def simulate(self, *, time_axis=None) -> AcquisitionData:
        """
        Run forward simulation with the CURRENT medium (self.model_c / self.model_a)
        and return a new AcquisitionData object.
        """
        t0 = time.time()
        data = self._forward_impl()  # (Tx, n_rx, nt) element order
        t_fwd = time.time() - t0
        if time_axis is None:
            time_axis = self.time_axis
        acq = AcquisitionData(
            array=data,
            tx_array=self.tx_array,
            grid=self.img_grid,
            time=time_axis,
        )
        Tx, Rx, nt = data.shape
        print(f"[TDO-SIM] simulate | forward={t_fwd:.2f}s → AcquisitionData(Tx={Tx}, Rx={Rx}, nt={nt})")
        return acq

