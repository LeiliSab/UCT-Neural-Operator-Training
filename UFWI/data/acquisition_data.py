import scipy.fft as sfft

from pathlib import Path
import numpy as np
from UFWI.data.data_container import DataContainer
import matplotlib.pyplot as plt
from ..geometry import ImageGrid2D, TransducerArray2D
import json


class AcquisitionData(DataContainer):
    """
    Initialize an AcquisitionData container.

    Parameters
    ----------
    array : np.ndarray | None
        Data array of shape (Tx, Rx, T) or (Tx, Rx, F). If None, creates an
        empty container that only carries geometry/axis metadata.
    tx_array : TransducerArray2D
        Transducer geometry and roles (positions/is_tx/is_rx).
    grid : ImageGrid2D
        Imaging grid (xi, yi).
    time : np.ndarray | None, optional
        1D time axis for time-domain data; length must match array.shape[-1].
        Mutually exclusive with `freqs`.
    freqs : np.ndarray | None, optional
        1D frequency axis for frequency-domain data; length must match
        array.shape[-1]. Mutually exclusive with `time`.
    c0 : float, optional
        Reference sound speed in m/s. Stored at ctx["c0"]. Default is 1500.0.
    **ctx : Any
        Additional metadata stored in `self.ctx`.

    Raises
    ------
    ValueError
        If `array` is not None and neither `time` nor `freqs` is provided.
    """

    # ------------------------------------------------------------------
    def __init__(self, array, *, tx_array, grid,
                 time: np.ndarray | None = None,
                 freqs: np.ndarray | None = None,
                 c0: float = 1500.0,  # sound speed
                 **ctx):

        if array is not None and (time is None and freqs is None):
            raise ValueError("Specify at least one of 'time' or 'freqs'")

        if c0 is not None:
            ctx = dict(ctx)
            ctx["c0"] = float(c0)

        super().__init__(array,
                         grid=grid,
                         tx_array=tx_array,
                         **ctx)

        self.time: np.ndarray | None = time
        self.freqs: np.ndarray | None = freqs

        self.mode = "time" if time is not None else "freqs"

    @classmethod
    def from_geometry(cls, grid, tx_array, **kwargs):
        """
        Construct an empty AcquisitionData from geometry.

        Parameters
        ----------
        grid : ImageGrid2D
            Imaging grid.
        tx_array : TransducerArray2D
            Transducer array geometry.
        **kwargs : Any
            Additional keyword arguments forwarded to `__init__` (e.g., time/freqs/c0/ctx).

        Returns
        -------
        AcquisitionData
            An instance with `array=None`.
        """
        return cls(array=None, grid=grid, tx_array=tx_array, **kwargs)

    # -------- setters --------
    def set_time(self, time: np.ndarray):
        """
        Set the time axis and clear the frequency axis.

        Parameters
        ----------
        time : np.ndarray
            1D time axis. If `array` is not None, its length must match array.shape[-1].

        Raises
        ------
        ValueError
            If `array` is not None and `time.size != array.shape[-1]`.
        """
        time = np.asarray(time, float)
        if self.array is not None and time.size != self.array.shape[-1]:
            raise ValueError("time length must match array.shape[-1]")
        self.time = time
        self.freqs = None

    def set_freqs(self, freqs: np.ndarray):
        """
        Set the frequency axis and clear the time axis.

        Parameters
        ----------
        freqs : np.ndarray
            1D frequency axis. If `array` is not None, its length must match array.shape[-1].

        Raises
        ------
        ValueError
            If `array` is not None and `freqs.size != array.shape[-1]`.
        """
        freqs = np.asarray(freqs, float)
        if self.array is not None and freqs.size != self.array.shape[-1]:
            raise ValueError("freqs length must match array.shape[-1]")
        self.freqs = freqs
        self.time = None

    def set_array(self, array: np.ndarray):
        """
        Assign the main data array without touching geometry or axes.

        Parameters
        ----------
        array : np.ndarray
            Array of shape (Tx, Rx, T) or (Tx, Rx, F). The original dtype is preserved.

        Notes
        -----
        This does not validate or synchronize the lengths of `time`/`freqs`. Call
        `set_time` or `set_freqs` if needed to ensure consistency.
        """
        self.array = np.asarray(array, copy=False)

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    def slice_frequency(self, f_idx: int):
        """
        Return the (Tx, Rx) slice at a given frequency index.

        Parameters
        ----------
        f_idx : int
            Index into the frequency axis (`self.freqs`).

        Returns
        -------
        np.ndarray
            A (Tx, Rx) real/complex slice.

        Raises
        ------
        RuntimeError
            If the dataset is time-domain (`self.freqs is None`).
        """
        if self.freqs is None:
            raise RuntimeError("This dataset is time-domain; no frequency axis.")
        return self.array[:, :, f_idx]

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save(self, path, *, compressed: bool = False) -> Path:
        """
        Save the entire AcquisitionData payload to an .npz file:
        - array
        - geometry: ImageGrid2D (xi, yi) and TransducerArray2D (positions, is_tx, is_rx)
        - axes: time or freqs (exactly one)
        - ctx: JSON-serialized dict (pickle-free)
        - meta: a tiny dict including version, mode

        Parameters
        ----------
        path : str | Path
        compressed : bool
            If True, use np.savez_compressed (smaller file, slightly slower).

        Returns
        -------
        Path : final file path written.
        """
        path = Path(path).with_suffix(".npz")

        payload = {
            "array": self.array,
            # --- geometry ---
            "grid_xi": getattr(self.grid, "xi", None),
            "grid_yi": getattr(self.grid, "yi", None),
            "tx_positions": self.tx_array.positions if self.tx_array is not None else None,
            "is_tx": self.tx_array.is_tx if self.tx_array is not None else None,
            "is_rx": self.tx_array.is_rx if self.tx_array is not None else None,
            # --- axes ---
            "time": self.time if self.time is not None else None,
            "freqs": self.freqs if self.freqs is not None else None,
            # --- meta / ctx ---
            "meta": np.array(
                {"version": "1.0", "mode": self.mode}, dtype=object
            ),
            "ctx_json": np.array(json.dumps(self.ctx, ensure_ascii=False), dtype=object),
        }

        saver = np.savez_compressed if compressed else np.savez
        saver(path, **payload)
        print(f"[info] AcquisitionData saved → {path} (mode={self.mode}, compressed={compressed})")
        return path

    @classmethod
    def load(cls, path) -> "AcquisitionData":
        """
        Load from an .npz written by `save()` and return a fully constructed
        AcquisitionData object (including geometry and ctx).

        Notes
        -----
        - This loader does NOT require allow_pickle=True. `ctx` is JSON.
        - If `meta['mode']` is 'time', we expect 'time' to be present; if 'freqs', expect 'freqs'.
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            array = z["array"] if "array" in z.files else None

            xi = z["grid_x"] if "grid_x" in z.files else None
            yi = z["grid_y"] if "grid_y" in z.files else None
            if xi is None or yi is None:
                raise ValueError("Missing grid_xi / grid_yi in file; cannot reconstruct ImageGrid2D.")
            grid = ImageGrid2D(xi=xi, yi=yi)

            tx_positions = z["tx_positions"] if "tx_positions" in z.files else None
            is_tx = z["is_tx"] if "is_tx" in z.files else None
            is_rx = z["is_rx"] if "is_rx" in z.files else None
            if tx_positions is None or is_tx is None or is_rx is None:
                raise ValueError("Missing transducer array fields; cannot reconstruct TransducerArray2D.")
            tx_array = TransducerArray2D(
                positions=tx_positions,
                is_tx=is_tx.astype(bool),
                is_rx=is_rx.astype(bool),
            )

            # meta / ctx
            meta = z["meta"].item() if "meta" in z.files else {"version": "unknown", "mode": "none"}
            try:
                ctx = json.loads(z["ctx_json"].item()) if "ctx_json" in z.files else {}
            except Exception:
                ctx = {}

            time = z["time"] if ("time" in z.files) else None
            freqs = z["freqs"] if ("freqs" in z.files) else None

        # construct AcquisitionData
        return cls(
            array=array,
            tx_array=tx_array,
            grid=grid,
            time=time,
            freqs=freqs,
            **ctx,
        )

    def fft(self) -> "AcquisitionData":
        """
        Compute FFT along the last axis and return a new frequency-domain AcquisitionData.

        Behavior
        --------
        - Uses `rfft`/`rfftfreq` for real arrays; otherwise `fft`/`fftfreq`.
        - The frequency axis is derived from the time axis; requires at least 2 samples.
        - Records `fft_meta = {'n_time', 'dt', 'kind'}` in the returned object's ctx
          to aid `ifft()` reconstruction.

        Returns
        -------
        AcquisitionData
            A new frequency-domain object (`time=None`, populated `freqs`).

        Raises
        ------
        RuntimeError
            If `array` is None or `time` is not set.
        ValueError
            If the time axis has fewer than 2 samples.
        """
        if self.array is None:
            raise RuntimeError("array is None")
        if self.time is None:
            raise RuntimeError("`time` is required for fft()")

        t = np.asarray(self.time, float)
        if t.size < 2:
            raise ValueError("time axis must have at least 2 samples")
        dt = float(np.mean(np.diff(t)))
        N = int(self.array.shape[-1])

        use_rfft = np.isrealobj(self.array)
        if use_rfft:
            spec = sfft.rfft(self.array, axis=-1)
            freqs = sfft.rfftfreq(N, d=dt)
            kind = "rfft"
        else:
            spec = sfft.fft(self.array, axis=-1)
            freqs = sfft.fftfreq(N, d=dt)
            kind = "fft"

        ctx_new = dict(self.ctx)
        ctx_new["fft_meta"] = {"n_time": N, "dt": dt, "kind": kind}

        return AcquisitionData(
            array=spec,
            tx_array=self.tx_array,
            grid=self.grid,
            time=None,
            freqs=freqs.astype(float, copy=False),
            **ctx_new,
        )

    def ifft(self) -> "AcquisitionData":
        """
        Compute IFFT along the last axis and return a new time-domain AcquisitionData.

        Behavior
        --------
        - Chooses `irfft` vs `ifft` from `ctx['fft_meta']['kind']`, or infers by checking
          for negative frequencies.
        - If `fft_meta['n_time']` and `['dt']` exist, they are used; otherwise `dt` is
          estimated from `freqs` and the target length `n` is inferred.
        - Produces `time = np.arange(n) * dt` and removes `fft_meta` from ctx.

        Returns
        -------
        AcquisitionData
            A new time-domain object (`freqs=None`, populated `time`).

        Raises
        ------
        RuntimeError
            If the frequency-domain array or `freqs` is missing.
        """
        if self.array is None or self.freqs is None:
            raise RuntimeError("need frequency-domain array and `freqs` for ifft()")

        meta = self.ctx.get("fft_meta", {})
        kind = meta.get("kind")
        F = int(self.array.shape[-1])

        if kind is None:
            use_rfft = bool(np.all(self.freqs >= -1e-15))
        else:
            use_rfft = (kind == "rfft")

        # target length `n` for IFFT
        if "n_time" in meta:
            n = int(meta["n_time"])
        else:
            n = 2 * (F - 1) if use_rfft else F

        # dt for IFFT
        if meta.get("dt") is not None:
            dt = float(meta["dt"])
        else:
            df = float(np.mean(np.diff(self.freqs)))
            dt = 1.0 / (n * df)

        if use_rfft:
            y = sfft.irfft(self.array, n=n, axis=-1)
        else:
            y = sfft.ifft(self.array, n=n, axis=-1)
        y = np.real_if_close(y, tol=1e3)  # 去掉数值虚部

        # make time axis
        t = np.arange(n, dtype=float) * dt
        ctx_new = dict(self.ctx)
        ctx_new.pop("fft_meta", None)

        return AcquisitionData(
            array=y,
            tx_array=self.tx_array,
            grid=self.grid,
            time=t,
            freqs=None,
            **ctx_new,
        )

    def show_trace(self, tx: int, rx: int,
                   *, xunit="s", ax=None, figure_size=None,
                   **line_kw):
        """
        Plot a single Tx–Rx waveform (time domain).

        Parameters
        ----------
        tx : int
            Transmitter index.
        rx : int
            Receiver index.
        xunit : {'s', 'idx'}, optional
            X-axis units: 's' for seconds (requires `time`), 'idx' for sample index.
        ax : matplotlib.axes.Axes | None, optional
            Axes to draw on. If None, a new figure/axes is created.
        figure_size : tuple | None, optional
            Figure size in inches when creating a new figure.
        **line_kw : Any
            Keyword arguments forwarded to `Axes.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes used for plotting.

        Raises
        ------
        RuntimeError
            If `array` is None.
        ValueError
            If `xunit='s'` but `time` is not set.
        """
        if self.array is None:
            raise RuntimeError("array is None")

        trace = self.array[tx, rx]

        if figure_size is None:
            figure_size = (6, 3)

        if ax is None:
            fig, ax = plt.subplots(figsize=figure_size)

        if xunit == "s":
            if self.time is None:
                raise ValueError("time axis not set")
            x = self.time
            ax.set_xlabel("t [s]")
        else:
            x = np.arange(trace.size)
            ax.set_xlabel("sample #")

        ax.plot(x, trace, **line_kw)
        ax.set_title(f"Tx {tx}, Rx {rx}")
        ax.set_ylabel("pressure")
        plt.show()
        return ax

    def show_traces(self, tx: int, rx_list=None,
                    *, norm=False, ax=None, figure_size=None,
                    **plot_kw):
        """
        Overlay multiple Rx waveforms for a given Tx (time domain).

        Parameters
        ----------
        tx : int
            Transmitter index.
        rx_list : Sequence[int] | None, optional
            List of receiver indices. Defaults to all receivers.
        norm : bool, optional
            Normalize each trace by its absolute maximum. Default False.
        ax : matplotlib.axes.Axes | None, optional
            Axes to draw on. If None, a new figure/axes is created.
        figure_size : tuple | None, optional
            Figure size in inches when creating a new figure.
        **plot_kw : Any
            Keyword arguments forwarded to `Axes.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes used for plotting.

        Raises
        ------
        RuntimeError
            If `array` is None.
        """
        if self.array is None:
            raise RuntimeError("array is None")

        if figure_size is None:
            figure_size = (6, 3)

        if rx_list is None:
            rx_list = np.arange(self.array.shape[1])

        if ax is None:
            fig, ax = plt.subplots(figsize=figure_size)

        for r in rx_list:
            y = self.array[tx, r]
            if norm:
                y = y / np.max(np.abs(y) + 1e-30)
            ax.plot(self.time if self.time is not None else np.arange(y.size),
                    y, **plot_kw)

        ax.set_title(f"Tx {tx} – {len(rx_list)} traces")
        plt.show()
        return ax

    def show_sinogram(self, *, t_idx=None, t_val=None,
                      mode="max", ax=None, figure_size=None,
                      **imshow_kw):
        """
        Visualize an intensity map over (Tx, Rx) (a.k.a. sinogram).

        Parameters
        ----------
        t_idx : int | None, optional
            Time index used when `mode='sample'`.
        t_val : float | None, optional
            Time in seconds used when `mode='sample'` (requires `time`).
        mode : {'max', 'sample'}, optional
            'max' : per-trace absolute maximum.
            'sample' : absolute value at the given time index/value.
        ax : matplotlib.axes.Axes | None, optional
            Axes to draw on. If None, a new figure/axes is created.
        figure_size : tuple | None, optional
            Figure size in inches when creating a new figure.
        **imshow_kw : Any
            Keyword arguments forwarded to `Axes.imshow`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes used for plotting.

        Raises
        ------
        RuntimeError
            If `array` is None.
        ValueError
            If `mode='sample'` and neither `t_idx` nor (`t_val` with `time` set) is provided.
        """
        if self.array is None:
            raise RuntimeError("array is None")

        if figure_size is None:
            figure_size = (8, 6)

        if mode == "max":
            img = np.abs(self.array).max(axis=-1)
        else:  # sample at time
            if t_idx is None:
                if t_val is None or self.time is None:
                    raise ValueError("Need t_idx or t_val (with self.time)")
                t_idx = np.argmin(np.abs(self.time - t_val))
            img = self.array[..., t_idx]

        if ax is None:
            fig, ax = plt.subplots(figsize=figure_size)
        im = ax.imshow(img, cmap="inferno", origin="lower", **imshow_kw)
        ax.set_xlabel("Rx index")
        ax.set_ylabel("Tx index")
        ax.set_title(f"Sinogram – {mode}")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.show()
        return ax

    def show_spectrum(self, tx: int, rx: int,
                      *, xunit="Hz", ax=None, figure_size=None,
                      **line_kw):
        """
        Plot the magnitude spectrum for a single Tx–Rx pair (frequency domain).

        Parameters
        ----------
        tx : int
            Transmitter index.
        rx : int
            Receiver index.
        xunit : {'Hz', 'idx'}, optional
            X-axis units: 'Hz' for frequency (default), 'idx' for frequency-bin index.
        ax : matplotlib.axes.Axes | None, optional
            Axes to draw on. If None, a new figure/axes is created.
        figure_size : tuple | None, optional
            Figure size in inches when creating a new figure.
        **line_kw : Any
            Keyword arguments forwarded to `Axes.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes used for plotting.

        Raises
        ------
        RuntimeError
            If `array` is None or if the dataset is time-domain (`freqs` is not set).
        """
        if self.array is None:
            raise RuntimeError("array is None")
        if self.freqs is None:
            raise RuntimeError("This dataset is time-domain; no frequency axis.")

        spec = self.array[tx, rx, :]
        mag = np.abs(spec)

        if figure_size is None:
            figure_size = (6, 3)

        if ax is None:
            fig, ax = plt.subplots(figsize=figure_size)

        if xunit == "Hz":
            x = self.freqs
            ax.set_xlabel("f [Hz]")
        else:
            x = np.arange(mag.size)
            ax.set_xlabel("freq index")

        ax.plot(x, mag, **line_kw)
        ax.set_title(f"Tx {tx}, Rx {rx} — spectrum")
        ax.set_ylabel("magnitude")
        plt.show()
        return ax

    @property
    def c0(self) -> float:
        """
        Reference sound speed (m/s).

        Returns
        -------
        float
            `ctx['c0']` if present, otherwise 1500.0.
        """
        return float(self.ctx.get("c0", 1500.0))

    @c0.setter
    def c0(self, val: float) -> None:
        """
        Set the reference sound speed (m/s).

        Parameters
        ----------
        val : float
            Reference sound speed to store in `ctx['c0']`.
        """
        self.ctx["c0"] = float(val)
