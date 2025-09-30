import copy
import numpy as np
import matplotlib.pyplot as plt


class DataContainer:
    """
    Base class for data containers: carries a NumPy array plus optional geometry
    information (`grid`, `tx_array`) and a free-form context dict (`ctx`).
    """

    def __init__(self, array: np.ndarray, *, grid=None, tx_array=None, **ctx):
        self.array = None if array is None else np.asarray(array)
        self.grid = grid  # ImageGrid2D or None
        self.tx_array = tx_array  # TransducerArray2D or None
        self.ctx = dict(ctx)  # Free-form context dict

    # ------------------------------------------------------------------
    # clone / copy
    # ------------------------------------------------------------------
    def clone(self):
        """Create a deep copy, preserving subclass type and metadata."""
        return self.__class__(
            self.array.copy(),
            grid=copy.deepcopy(self.grid),
            tx_array=copy.deepcopy(self.tx_array),
            **copy.deepcopy(self.ctx),
        )

    copy = clone  # alias

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def save(self, path: str):
        """
        Save array (+ctx) to disk.  '.npz' → numpy.savez，否则 numpy.save (.npy).
        """
        if path.endswith(".npz"):
            np.savez(path, array=self.array, **self.ctx)
        else:
            np.save(path, self.array)

    @classmethod
    def load(cls, path: str):
        """Load from .npz produced by `save()`."""
        data = np.load(path, allow_pickle=True)
        arr = data["array"]
        ctx = {k: data[k].item() for k in data.files if k != "array"}
        return cls(arr, **ctx)

    # ------------------------------------------------------------------
    # shape / slice helpers
    # ------------------------------------------------------------------
    def _wrap(self, arr):
        """Internal helper: wrap ndarray into same-type container,携几何/ctx"""
        return self.__class__(arr, grid=self.grid, tx_array=self.tx_array, **copy.deepcopy(self.ctx))

    def reshape(self, new_shape):
        return self._wrap(self.array.reshape(new_shape))

    def get_slice(self, index, axis=0):
        slicer = [slice(None)] * self.array.ndim
        slicer[axis] = index
        return self._wrap(self.array[tuple(slicer)])

    def reorder(self, order):
        return self._wrap(np.transpose(self.array, order))

    def pad(self, pad_width, mode="constant", **kwargs):
        return self._wrap(np.pad(self.array, pad_width, mode=mode, **kwargs))

    def crop(self, slices):
        return self._wrap(self.array[slices])

    def fft(self, axis=-1):
        return self._wrap(np.fft.fft(self.array, axis=axis))

    def ifft(self, axis=-1):
        return self._wrap(np.fft.ifft(self.array, axis=axis))

    # ------------------------------------------------------------------
    # arithmetic
    # ------------------------------------------------------------------
    def _binary_op(self, other, op):
        arr = op(self.array, other.array) if isinstance(other, DataContainer) else op(
            self.array, other
        )
        return self._wrap(arr)

    def __add__(self, other):
        return self._binary_op(other, np.add)

    def __sub__(self, other):
        return self._binary_op(other, np.subtract)

    def __mul__(self, other):
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other):
        return self._binary_op(other, np.divide)

    def __neg__(self):
        return self._wrap(-self.array)

    # ------------------------------------------------------------------
    # reductions
    # ------------------------------------------------------------------
    def sum(self, axis=None):
        return self.array.sum(axis=axis)

    def mean(self, axis=None):
        return self.array.mean(axis=axis)

    def max(self, axis=None):
        return self.array.max(axis=axis)

    def min(self, axis=None):
        return self.array.min(axis=axis)

    # -------- basic visualisation ---------------------------------
    def show(self, *, ax=None, cmap="viridis", title=None, **imshow_kw):
        """
        Generic visualiser: if array is 2-D → imshow, 1-D → plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Use existing axis; otherwise create new figure.
        cmap : str
            Colormap for 2-D data.
        **imshow_kw
            Extra kwargs to plt.imshow (e.g., vmin/vmax, extent,…)
        """
        a = self.array
        if ax is None:
            fig, ax = plt.subplots()

        if a.ndim == 1:
            ax.plot(a, **imshow_kw)
        elif a.ndim == 2:
            extent = getattr(self, "grid", None)
            extent = extent.extent if extent is not None else None
            im = ax.imshow(a,
                           cmap=cmap,
                           origin="lower",
                           extent=extent,
                           **imshow_kw)
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            raise NotImplementedError("show() only handles 1-D / 2-D arrays")

        ax.set_title(title or self.__class__.__name__)
        return ax
