import numpy as np
from UFWI.geometry.base import Geometry


class ImageGrid2D(Geometry):
    """2-D image grid geometry.

    Construction options (mutually exclusive)
    -----------------------------------------
    1) Explicit coordinates
       >>> ImageGrid2D(xi=<1-D array>, yi=<1-D array>)

    2) Uniform grid by dimensions
       >>> ImageGrid2D(nx=128, ny=128, dx=5e-4)  # centred about 0

    3) Uniform grid by extent (half-widths)
       >>> ImageGrid2D(dx=5e-4, xmax=0.032)      # centred about 0; choose largest odd n within bounds

    Notes
    -----
    Internally, coordinates are always stored centred about zero following the k-Wave
    convention:
        xi[j] = (j - (nx - 1)/2) * dx,
        yi[i] = (i - (ny - 1)/2) * dy.
    """

    def __init__(self, *,
                 # explicit coords
                 xi: np.ndarray | None = None,
                 yi: np.ndarray | None = None,
                 # uniform by dims
                 nx: int | None = None,
                 ny: int | None = None,
                 dx: float | None = None,
                 dy: float | None = None,
                 # uniform by extent (half-widths)
                 xmax: float | None = None,
                 ymax: float | None = None,
                 # limits
                 n_max: int | None = None):

        # -------------------------------------------------------------
        # 1) explicit coordinate arrays
        # -------------------------------------------------------------
        if xi is not None:
            xi = np.asarray(xi, float).ravel()
            yi = xi.copy() if yi is None else np.asarray(yi, float).ravel()
            if xi.size < 2 or yi.size < 2:
                raise ValueError("xi/yi must contain ≥2 points")
            # enforce uniform step (take mean) and recentre about 0
            self.dx = float(np.mean(np.diff(xi)))
            self.dy = float(np.mean(np.diff(yi)))
            xi = xi - float(xi.mean())
            yi = yi - float(yi.mean())

        # -------------------------------------------------------------
        # 2) build from nx,ny,dx,dy  (k-Wave style: centred about 0)
        # -------------------------------------------------------------
        elif nx is not None or ny is not None:
            if dx is None:
                raise ValueError("dx must be specified when using nx/ny path")
            if nx is None:
                raise ValueError("nx must be specified when using nx/ny path")
            dy = dx if dy is None else dy
            ny = nx if ny is None else ny

            jx = np.arange(nx, dtype=float)
            jy = np.arange(ny, dtype=float)

            if nx % 2 == 0:
                xi = (jx - nx / 2.0) * dx  # k-Wave 偶数公式
            else:
                xi = (jx - (nx - 1) / 2.0) * dx  # k-Wave 奇数公式
            if ny % 2 == 0:
                yi = (jy - ny / 2.0) * dy
            else:
                yi = (jy - (ny - 1) / 2.0) * dy

            self.dx, self.dy = float(dx), float(dy)

        # -------------------------------------------------------------
        # 3) build from spacing + half-widths (choose largest odd n ≤ bounds)
        # -------------------------------------------------------------
        else:
            if dx is None or xmax is None:
                raise ValueError("must supply either (xi,yi) or (nx,dx) or (dx,xmax)")
            dy = dx if dy is None else dy
            ymax = xmax if ymax is None else ymax

            # choose n so that max |x| ≤ xmax 且对称；用奇数以包含中心点
            nx = int(2 * np.floor(xmax / dx) + 1)
            ny = int(2 * np.floor(ymax / dy) + 1)
            if nx < 2 or ny < 2:
                raise ValueError("xmax/ymax too small for given dx/dy")

            jx = np.arange(nx, dtype=float)
            jy = np.arange(ny, dtype=float)
            xi = (jx - (nx - 1) / 2.0) * dx
            yi = (jy - (ny - 1) / 2.0) * dy

            self.dx, self.dy = float(dx), float(dy)

        # -------------------------------------------------------------
        # safety check
        # -------------------------------------------------------------
        if n_max is not None and xi.size * yi.size > n_max:
            raise ValueError("grid too large; reduce nx, ny or increase spacing (dx, dy)")

        # -------------------------------------------------------------
        # finalise
        # -------------------------------------------------------------
        self.xi, self.yi = xi, yi
        extent = (float(xi.min()), float(xi.max()), float(yi.min()), float(yi.max()))
        super().__init__(shape=(yi.size, xi.size), extent=extent)

    # ----------------------------------------------------------------
    # derived properties
    # ----------------------------------------------------------------
    @property
    def nx(self) -> int:
        return self.xi.size

    @property
    def ny(self) -> int:
        return self.yi.size

    @property
    def spacing(self) -> tuple[float, float]:
        return self.dx, self.dy

    # @property
    # def extent(self) -> tuple[float, float, float, float]:
    #     """Return the spatial extent as (xmin, xmax, ymin, ymax)."""
    #     return float(self.xi.min()), float(self.xi.max()), float(self.yi.min()), float(self.yi.max())

    # ----------------------------------------------------------------
    # helper methods
    # ----------------------------------------------------------------
    def coord2index(self, x: float, y: float) -> tuple[int, int]:
        ix = int(round((x - self.xi[0]) / self.dx))
        iy = int(round((y - self.yi[0]) / self.dy))
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            raise ValueError("coordinate out of grid")
        return ix, iy

    def index2coord(self, ix: int, iy: int) -> tuple[float, float]:
        return float(self.xi[ix]), float(self.yi[iy])

    def meshgrid(self, indexing: str = "ij"):
        return np.meshgrid(self.xi, self.yi, indexing=indexing)

    # ---- helper ----
    def _dx_min(self) -> float:
        return min(self.dx, self.dy)

    # ---- 1) max resolvable frequency (space-limited) ----
    def max_f(self, c_min: float, ppw: float | None = None) -> dict:
        """
        Return spatial-Nyquist-limited max frequency, and (optionally)
        a PPW-limited max frequency. c_max in m/s.
        """
        dxm = self._dx_min()
        f_nyq_space = c_min / (2.0 * dxm)
        out = {"f_max": f_nyq_space}
        if ppw is not None and ppw > 0:
            f_ppw = c_min / (ppw * dxm)
            out["f_ppw"] = f_ppw
            out["f_safe"] = min(f_nyq_space, f_ppw)
        else:
            out["f_safe"] = f_nyq_space
        return out

    # ---- 3) spectral wave-number vectors (rad/m) ----
    def kx(self) -> np.ndarray:
        return 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)

    def ky(self) -> np.ndarray:
        return 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)

    def kmesh(self, indexing: str = "xy"):
        KX, KY = np.meshgrid(self.kx(), self.ky(), indexing=indexing)
        return KX, KY, np.hypot(KX, KY)

