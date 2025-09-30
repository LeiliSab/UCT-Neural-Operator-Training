import copy
import warnings

import numpy as np
import matplotlib.pyplot as plt
from UFWI.data.data_container import DataContainer

from scipy.ndimage import zoom as ndi_zoom
from scipy.interpolate import RegularGridInterpolator
from typing import Optional


class ImageData(DataContainer):
    """
    Image container for 2-D models with built-in change history.

    This class extends `DataContainer` by keeping a chronological history of
    the image array after each update, enabling inspection, plotting, and
    (resampling) operations while preserving past states.

    Parameters
    ----------
    array : np.ndarray
        The initial image array (2-D). History starts with a copy of this.
    grid : Any, optional
        Optional grid object (e.g., ImageGrid2D) providing spatial metadata
        such as coordinates and `extent`.
    tx_array : Any, optional
        Optional transducer array geometry (e.g., TransducerArray2D).
    max_history : int | None, optional
        Maximum number of snapshots to keep. If None, the history grows
        without an automatic cap.
    **ctx : Any
        Additional metadata stored in `self.ctx`.

    Attributes
    ----------
    array : np.ndarray
        The current image (mutable).
    history : list[np.ndarray]
        Snapshots of the image over time. `history[-1]` is always equal to
        `array.copy()` right after an update.
    _max_history : int | None
        Maximum number of snapshots to retain.

    Notes
    -----
    - History stores copies, not views.
    - Some operations (e.g., resampling) return a *new* `ImageData`
      carrying a new grid and metadata describing the operation in `ctx`.
    """

    def __init__(self, array, *, grid=None, tx_array=None, max_history=None, **ctx):
        super().__init__(array, grid=grid, tx_array=tx_array, **ctx)
        self._max_history = max_history
        self.history = [self.array.copy()]

    # ----------------------------------------------------------------
    # current image
    # ----------------------------------------------------------------
    @property
    def current(self):
        """
        Return the current image array (no copy).

        Returns
        -------
        np.ndarray
            The current image stored in `self.array`.

        Notes
        -----
        Mutating the returned array mutates the internal state. Use `array.copy()`
        if you need an independent buffer.
        """
        return self.array

    # ----------------------------------------------------------------
    # update & history
    # ----------------------------------------------------------------
    def update(self, new_array):
        """
        Replace the current image and append to history.

        Parameters
        ----------
        new_array : np.ndarray
            New image to become the current `array`. A copy of this array is
            appended to `history`.

        Notes
        -----
        If `max_history` was provided at construction time and the number of
        snapshots exceeds that limit, the oldest snapshot is dropped.
        """
        self.array = np.asarray(new_array)
        self.history.append(self.array.copy())
        if self._max_history is not None and len(self.history) > self._max_history:
            self.history.pop(0)

    # ----------------------------------------------------------------
    # I/O
    # ----------------------------------------------------------------
    def save(self, path: str):
        """
        Save the current image and its history to a `.npz` file.

        Parameters
        ----------
        path : str
            Output file path. The file will contain:
              - `current`: the current array (`self.array`)
              - `history`: a stacked ndarray from `self.history`
              - any extra metadata stored in `self.ctx` (as additional fields)

        Raises
        ------
        ValueError
            If the arrays in `self.history` cannot be stacked due to shape mismatch.

        Notes
        -----
        All snapshots in history must have identical shapes for `np.stack` to
        succeed. Consider clearing or normalizing history if shapes differ.
        """
        np.savez(path,
                 current=self.array,
                 history=np.stack(self.history),
                 **self.ctx)

    def show(self, idx: int = -1, *, ax=None,
             cmap="seismic", vmin=None, vmax=None,
             underlay_tx: bool = True,
             component: str = "auto",
             tx_style: dict | None = None,
             title: str | None = None,
             figsize: tuple[int, int] | None = None,
             show_grid: bool = False,
             **imshow_kw):
        """
        Visualize a snapshot from history with optional overlays.

        Parameters
        ----------
        idx : int, optional
            History index to display (default -1 = latest snapshot).
        ax : matplotlib.axes.Axes | None, optional
            Axes to draw on. If None, a new figure and axes are created.
        cmap : str, optional
            Colormap passed to `imshow`. Default is "seismic".
        vmin, vmax : float | None, optional
            Color limits for `imshow`.
        underlay_tx : bool, optional
            If True and `tx_array` is available, overlay transducer positions.
        component : {"auto","real","imag","abs","phase"}, optional
            For complex arrays, choose which component to display.
            - "auto": show real part if the imaginary part is globally small;
              otherwise show magnitude.
            - "real", "imag": display respective component.
            - "abs" (or "magnitude", "mag"): display `np.abs(img)`.
            - "phase" (or "angle"): display `np.angle(img)`.
        tx_style : dict | None, optional
            Keyword arguments forwarded to `Axes.scatter` for the transducer overlay
            (e.g., `marker`, `s`, `linewidths`). Defaults are applied and then updated
            with this dict.
        title : str | None, optional
            Plot title. If None, a default title including the snapshot index is used.
        figsize : tuple[int, int] | None, optional
            Figure size in inches when creating a new figure. Defaults to `(6, 6)` for
            2-D and `(8, 6)` for >2-D.
        show_grid : bool, optional
            If True and `grid` is available, draw dashed grid lines at `grid.xi` and
            `grid.yi`.
        **imshow_kw : Any
            Additional keyword arguments forwarded to `Axes.imshow`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes used for plotting.

        Notes
        -----
        - If `grid.extent` is present, it is passed to `imshow` to place the image
          in physical coordinates.
        - Transducer overlay: TX-only (triangle), RX-only (square), and TX|RX (star)
          are drawn using `tx_array.positions`, `is_tx`, and `is_rx`, with a legend.
        """
        img = self.history[idx]

        # check if img is complex
        if np.iscomplexobj(img):
            comp = component.lower()
            if comp == "real":
                disp = img.real
            elif comp == "imag":
                disp = img.imag
            elif comp in ("abs", "magnitude", "mag"):
                disp = np.abs(img)
            elif comp in ("phase", "angle"):
                disp = np.angle(img)
            else:  # auto
                # if the imaginary part is globally small, show real part;
                if np.max(np.abs(img.imag)) < 1e-6 * max(1.0, np.max(np.abs(img.real))):
                    disp = img.real
                else:
                    disp = np.abs(img)
        else:
            disp = img

        if figsize is None:
            figsize = (6, 6) if disp.ndim == 2 else (8, 6)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        extent = getattr(self.grid, "extent", None)
        im = ax.imshow(disp, extent=extent, origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       **imshow_kw)

        # underlay transducer positions
        if underlay_tx and getattr(self, "tx_array", None) is not None:
            # default style for transducer markers
            base_style = dict(s=50, linewidths=0.5)
            if tx_style:
                base_style.update(tx_style)

            # get positions and transducer flags
            positions = self.tx_array.positions  # shape (2, N)
            is_tx = self.tx_array.is_tx
            is_rx = self.tx_array.is_rx

            # mask for different transducer states
            mask_tx_only = is_tx & ~is_rx
            mask_rx_only = ~is_tx & is_rx
            mask_both = is_tx & is_rx

            # only Tx: red triangles
            if mask_tx_only.any():
                ax.scatter(
                    positions[0, mask_tx_only],
                    positions[1, mask_tx_only],
                    marker='^',
                    label='TX', **base_style
                )
            # only Rx: blue squares
            if mask_rx_only.any():
                ax.scatter(
                    positions[0, mask_rx_only],
                    positions[1, mask_rx_only],
                    marker='s',
                    label='RX', **base_style
                )
            # both Tx and Rx: green stars
            if mask_both.any():
                ax.scatter(
                    positions[0, mask_both],
                    positions[1, mask_both],
                    marker='*',
                    label='TX|RX', **base_style
                )

            ax.legend(loc='upper right', frameon=True)

        # show grid lines if available
        if show_grid and getattr(self, "grid", None) is not None:
            xi, yi = self.grid.xi, self.grid.yi
            gkw = dict(color="black", linestyle="--", linewidth=0.5)
            for x in xi:
                ax.axvline(x, **gkw)
            for y in yi:
                ax.axhline(y, **gkw)

        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{self.__class__.__name__}  (snapshot {idx})" if title is None else title)
        plt.tight_layout()
        plt.show()
        return ax

    def _require_src_grid(self, origin_grid=None):
        """
        Resolve and validate the source grid for physical-coordinate operations.

        Parameters
        ----------
        origin_grid : Any | None, optional
            If provided, use this as the source grid; otherwise fall back to
            `self.grid`.

        Returns
        -------
        Any
            The resolved source grid to interpret physical coordinates.

        Raises
        ------
        ValueError
            If no grid can be resolved, if the image is not 2-D, or if the image
            shape is inconsistent with the source grid dimensions `(ny, nx)`.

        Notes
        -----
        This method ensures:
        - a source grid exists (either `origin_grid` or `self.grid`);
        - the image is 2-D (resampling routines here are 2-D);
        - `(src_grid.ny, src_grid.nx) == self.array.shape`.
        """
        src_grid = origin_grid if origin_grid is not None else self.grid
        if src_grid is None:
            raise ValueError(
                "ImageData has no associated grid; please explicitly provide the source grid (for physical coordinate resampling) via \"origin_grid\""
            )
        # check if the grid is 2-D
        if self.array.ndim != 2:
            raise ValueError("Only 2-D images are supported for resampling.")
        if (src_grid.ny, src_grid.nx) != self.array.shape:
            raise ValueError(
                f"The shape of array={self.array.shape} is inconsistent with origin_grid(shape={(src_grid.ny, src_grid.nx)}); "
                f"please provide a source grid that matches the image size."
            )
        return src_grid

    def _resample_interp_(self, new_grid, *, method: str = "linear",
                          fill_value: float | None = None) -> np.ndarray:
        """
        Coordinate-based resampling via interpolation (recommended for upsampling).

        Resamples the image from the source grid (`self.grid`) to `new_grid` using
        `scipy.interpolate.RegularGridInterpolator` in physical coordinates.

        Parameters
        ----------
        new_grid : Any
            Target grid providing `nx`, `ny`, and a `meshgrid(indexing="xy")` method,
            as well as coordinate arrays compatible with the source grid.
        method : {"linear","nearest"}, optional
            Interpolation method for `RegularGridInterpolator`. Default is "linear".
        fill_value : float | None, optional
            Value to use for points outside the source domain (when `bounds_error=False`).
            If None, values are extrapolated.

        Returns
        -------
        np.ndarray
            Resampled 2-D image of shape `(new_grid.ny, new_grid.nx)`. Complex images
            are supported by interpolating real and imaginary parts separately.

        Raises
        ------
        ValueError
            If the image is not 2-D or if `self.grid` is None.

        Notes
        -----
        - Source coordinates are `(yi_src, xi_src) = (self.grid.yi, self.grid.xi)`.
        - Target coordinates come from `new_grid.meshgrid(indexing="xy")`.
        """
        if self.array.ndim != 2:
            raise ValueError("Only 2-D images are supported here.")
        if self.grid is None:
            raise ValueError("Cannot do coordinate-based interpolation: source grid is None.")
        yi_src, xi_src = self.grid.yi, self.grid.xi
        f = RegularGridInterpolator(
            (yi_src, xi_src),
            np.asarray(self.array, dtype=float),
            method=method, bounds_error=False, fill_value=fill_value,
        )
        X_tgt, Y_tgt = new_grid.meshgrid(indexing="xy")
        pts = np.column_stack([Y_tgt.ravel(), X_tgt.ravel()])
        if np.iscomplexobj(self.array):
            re = f(pts).reshape(new_grid.ny, new_grid.nx)
            f_im = RegularGridInterpolator((yi_src, xi_src), np.asarray(self.array.imag, float),
                                           method=method, bounds_error=False, fill_value=fill_value)
            im = f_im(pts).reshape(new_grid.ny, new_grid.nx)
            return re + 1j * im
        else:
            return f(pts).reshape(new_grid.ny, new_grid.nx)

    def _resample_zoom_(self, new_grid, *, order: int = 1,
                        allow_stretch: bool = False) -> np.ndarray:
        """
        Pixel-based scaling using `scipy.ndimage.zoom` (fast).

        Resamples by scaling the array using shape ratios. When a source grid is
        available, scaling is only allowed if `grid.extent` matches `new_grid.extent`
        (or `allow_stretch=True`).

        Parameters
        ----------
        new_grid : Any
            Target grid providing `nx`, `ny`, and optionally `extent` for validation.
        order : int, optional
            Spline interpolation order passed to `ndi_zoom` (0–5 typical). Default 1.
        allow_stretch : bool, optional
            If False and both grids have `extent`, a mismatch raises `ValueError`.
            If True, scaling proceeds regardless of extent differences.

        Returns
        -------
        np.ndarray
            Resampled 2-D image of shape `(new_grid.ny, new_grid.nx)`. Complex images
            are supported by zooming real and imaginary parts separately.

        Raises
        ------
        ValueError
            If the image is not 2-D, or if grid extents differ and `allow_stretch=False`.

        Notes
        -----
        This method operates in index space and ignores physical coordinates unless
        extents are validated.
        """
        if self.array.ndim != 2:
            raise ValueError("Only 2-D images are supported here.")

        # if self.grid is None, we can always zoom
        if (self.grid is not None) and (not allow_stretch):
            if getattr(self.grid, "extent", None) != getattr(new_grid, "extent", None):
                raise ValueError("Grid extents differ. Set allow_stretch=True to force pixel scaling.")

        ny_src, nx_src = self.array.shape[-2], self.array.shape[-1]
        zy = new_grid.ny / ny_src
        zx = new_grid.nx / nx_src

        if np.iscomplexobj(self.array):
            re = ndi_zoom(self.array.real, zoom=(zy, zx), order=order, prefilter=True)
            im = ndi_zoom(self.array.imag, zoom=(zy, zx), order=order, prefilter=True)
            return re + 1j * im
        else:
            return ndi_zoom(self.array, zoom=(zy, zx), order=order, prefilter=True)

    def _resample_to(self, new_grid, *, mode: str = "interp",
                     method: str = "linear", order: int = 1,
                     allow_stretch: bool = False, copy_ctx: bool = True) -> "ImageData":
        """
        Internal helper to resample to `new_grid` using either interpolation or zoom.

        Parameters
        ----------
        new_grid : Any
            Target grid.
        mode : {"interp","zoom"}, optional
            Resampling mode. "interp" uses coordinate-based interpolation; "zoom"
            uses pixel scaling.
        method : {"linear","nearest"}, optional
            Interpolation method when `mode="interp"`.
        order : int, optional
            Spline order when `mode="zoom"`. Default 1.
        allow_stretch : bool, optional
            Passed to `_resample_zoom_` to allow extent mismatch. Default False.
        copy_ctx : bool, optional
            If True, start from a copy of `self.ctx`; otherwise start with an empty
            context. In both cases, a `resample` entry describing the operation is
            added.

        Returns
        -------
        ImageData
            A new `ImageData` with the resampled array, `new_grid`, the same
            `tx_array`, and updated context (e.g., `ctx["resample"]` describes
            the mode and parameters used).

        Raises
        ------
        ValueError
            If `mode` is not one of {"interp","zoom"}.

        Notes
        -----
        - If `mode="interp"` but `self.grid` is None, the method falls back to
          pixel scaling, emits a `RuntimeWarning`, and records the fallback in
          `ctx["resample"]`.
        """

        if mode == "interp":
            if self.grid is None:
                warnings.warn("Source grid is None; falling back to pixel scaling (zoom).",
                              RuntimeWarning)
                arr_new = self._resample_zoom_(new_grid, order=order, allow_stretch=True)
                meta = {"resample": {"mode": "zoom(fallback)", "order": order,
                                     "allow_stretch": True, "note": "no source grid"}}
            else:
                arr_new = self._resample_interp_(new_grid, method=method)
                meta = {"resample": {"mode": "interp", "method": method}}
        elif mode == "zoom":
            arr_new = self._resample_zoom_(new_grid, order=order, allow_stretch=allow_stretch)
            meta = {"resample": {"mode": "zoom", "order": order,
                                 "allow_stretch": bool(allow_stretch)}}
        else:
            raise ValueError("mode must be 'interp' or 'zoom'.")

        ctx_new = dict(self.ctx) if copy_ctx else {}
        ctx_new.update(meta)

        return ImageData(
            array=arr_new, grid=new_grid, tx_array=self.tx_array,
            max_history=None, **ctx_new,
        )

    def downsample_to(self, new_grid, *, order: int = 1, allow_stretch: bool = False) -> "ImageData":
        """
        Downsample to a coarser grid using fast pixel scaling.

        Parameters
        ----------
        new_grid : Any
            Target coarser grid.
        order : int, optional
            Spline order for pixel scaling (zoom). Default 1.
        allow_stretch : bool, optional
            Allow extent mismatch when using pixel scaling. Default False.

        Returns
        -------
        ImageData
            A new `ImageData` resampled to `new_grid` via pixel scaling.

        Raises
        ------
        ValueError
            If the target grid is finer than the source (based on grid attributes
            or array shapes when `self.grid` is None).

        Notes
        -----
        This method enforces that the target is coarser:
        - With a source grid: `new_grid.nx <= self.grid.nx` and `new_grid.ny <= self.grid.ny`.
        - Without a source grid: compare array shape to `new_grid` shape.
        """
        if self.grid is not None:
            if new_grid.nx > self.grid.nx or new_grid.ny > self.grid.ny:
                raise ValueError("Target grid is finer; use upsample_to().")
        else:
            ny_src, nx_src = self.array.shape[-2], self.array.shape[-1]
            if new_grid.nx > nx_src or new_grid.ny > ny_src:
                raise ValueError("Target grid is finer; use upsample_to().")

        return self._resample_to(new_grid, mode="zoom", order=order, allow_stretch=allow_stretch)

    def upsample_to(self, new_grid, *, method: str = "linear") -> "ImageData":
        """
        Upsample to a finer grid.

        If a source grid is available, performs coordinate-based interpolation.
        If no source grid is available, automatically falls back to pixel scaling
        (zoom) while still using the `"interp"` code path (a warning is emitted
        and the fallback is recorded in context).

        Parameters
        ----------
        new_grid : Any
            Target finer grid.
        method : {"linear","nearest"}, optional
            Interpolation method when a source grid is present. Default "linear".

        Returns
        -------
        ImageData
            A new `ImageData` resampled to the finer `new_grid`.

        Raises
        ------
        ValueError
            If the target grid is not strictly finer than the source
            (based on grid attributes or array shape when `self.grid` is None).

        Notes
        -----
        - With a source grid: requires `new_grid.nx >= self.grid.nx` and
          `new_grid.ny >= self.grid.ny`.
        - Without a source grid: requires `new_grid.shape >= self.array.shape`
          (per dimension).
        """
        if self.grid is not None:
            if new_grid.nx < self.grid.nx or new_grid.ny < self.grid.ny:
                raise ValueError("Target grid is coarser; use downsample_to().")
            return self._resample_to(new_grid, mode="interp", method=method)
        else:
            # no source grid, check array shape
            ny_src, nx_src = self.array.shape[-2], self.array.shape[-1]
            if new_grid.nx <= nx_src and new_grid.ny <= ny_src:
                raise ValueError("Target grid is not finer; use downsample_to() instead.")
            # automatically fall back to pixel scaling
            return self._resample_to(new_grid, mode="interp", method=method)
