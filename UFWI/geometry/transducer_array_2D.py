# File: UFWI/geometry/transducer_array.py
import numpy as np
from typing import List, Sequence, Optional
from UFWI.geometry.base import Geometry
from UFWI.geometry.transducer import Transducer
from UFWI.geometry.image_grid_2D import ImageGrid2D


class TransducerArray2D(Geometry):
    """
    Array of transducer elements, each of which may be TX, RX, or both.

    Construction options (mutually exclusive):
      - transducers : list[Transducer]
      - positions, is_tx, is_rx arrays

    Parameters
    ----------
    transducers : list of Transducer, optional
        Pre-built Transducer objects.
    positions : array_like, shape (2, N), optional
        Stack of (x, y) coordinates for each element.
    is_tx : sequence of bool, shape (N,), optional
        Flags indicating transmitter role (element-level boolean vector).
    is_rx : sequence of bool, shape (N,), optional
        Flags indicating receiver role (element-level boolean vector).
    """

    def __init__(
            self,
            *,
            transducers: Optional[List[Transducer]] = None,
            positions: Optional[np.ndarray] = None,
            is_tx: Optional[Sequence[bool]] = None,
            is_rx: Optional[Sequence[bool]] = None,
    ):
        if transducers is not None:
            # Build from Transducer instances
            pts = np.stack([t.position for t in transducers], axis=1)
            tx_flags = [t.is_tx for t in transducers]
            rx_flags = [t.is_rx for t in transducers]
            self._transducer_objs: Optional[List[Transducer]] = list(transducers)
        else:
            if positions is None or is_tx is None or is_rx is None:
                raise ValueError("Either transducers or (positions,is_tx,is_rx) must be provided")
            pts = np.asarray(positions, float)
            if pts.ndim != 2 or pts.shape[0] != 2:
                raise ValueError("positions must be shape (2, N)")
            tx_flags = np.asarray(is_tx, bool).ravel()
            rx_flags = np.asarray(is_rx, bool).ravel()
            if tx_flags.size != pts.shape[1] or rx_flags.size != pts.shape[1]:
                raise ValueError("is_tx/is_rx must have length N")
            self._transducer_objs = None  # constructed from arrays — create on demand

        # Store raw data
        self.positions = pts  # shape (2, N)
        self.is_tx = np.asarray(tx_flags, bool)
        self.is_rx = np.asarray(rx_flags, bool)

        # Dynamic receive flags (element-level boolean vector, can be gated at runtime)
        self.rx_flags = self.is_rx.copy()

        # Base class: N elements
        super().__init__(shape=(pts.shape[1],), extent=None)

    @property
    def n_elements(self) -> int:
        """Total number of elements in the array."""
        return self.positions.shape[1]

    @property
    def n_tx(self) -> int:
        """Number of active transmitters."""
        return int(self.is_tx.sum())

    @property
    def n_rx(self) -> int:
        """Number of active receivers (based on rx_flags)."""
        return int(self.rx_flags.sum())

    @property
    def tx_positions(self) -> np.ndarray:
        """(2, n_tx) array of transmitter coordinates."""
        return self.positions[:, self.is_tx]

    @property
    def rx_positions(self) -> np.ndarray:
        """(2, n_rx) array of active receiver coordinates."""
        return self.positions[:, self.rx_flags]

    @property
    def transducers(self) -> List[Transducer]:
        """
        Return a list of Transducer objects reflecting the current state.
        If constructed from arrays, these are created on demand.
        """
        if self._transducer_objs is None:
            self._transducer_objs = self.get_transducers(assign_ids=True)
        return self._transducer_objs

    def set_rx_flags(self, flags: Sequence[bool]) -> None:
        """
        Update which elements are actively used as receivers (element-level boolean vector).
        Only those with is_rx=True can be turned on.

        Parameters
        ----------
        flags : sequence of bool
            Element-level boolean vector of length N indicating active receivers.
        """
        f = np.asarray(flags, bool).ravel()
        if f.size != self.n_elements:
            raise ValueError("flags length must equal number of elements")
        # Enforce only nominal receivers
        self.rx_flags = f & self.is_rx

    def _get_rx_flags(self) -> np.ndarray:
        """Return a copy of the current element-level boolean vector for active receivers."""
        return self.rx_flags.copy()

    def _get_rx_flags_for_tx(self, tx_idx: int, drop_self_rx: bool) -> np.ndarray:
        """
        Element-level boolean vector of receivers when element tx_idx fires.

        Parameters
        ----------
        tx_idx : int
            Index of transmitting element.
        drop_self_rx : bool
            If True and that element is a receiver, its flag is set False.

        Returns
        -------
        flags : ndarray of bool
            Element-level boolean vector of active receivers for this TX.
        """
        flags = self.rx_flags.copy()
        if drop_self_rx and self.is_rx[tx_idx]:
            flags[tx_idx] = False
        return flags

    def geometric_tofs(self, c0: float) -> np.ndarray:
        """
        Compute time-of-flight (TOF) matrix between every TX→RX pair.

        Parameters
        ----------
        c0 : float
            Wave propagation speed in m/s.

        Returns
        -------
        tofs : ndarray, shape (n_rx, n_tx)
            tofs[i,j] = ||rx_i - tx_j|| / c0
        """
        tx = self.tx_positions
        rx = self.positions[:, self.is_rx]  # full-ring receivers
        dx = rx[0, :, None] - tx[0, None, :]
        dy = rx[1, :, None] - tx[1, None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        return dist / float(c0)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(elements={self.n_elements}, "
            f"Tx={self.n_tx}, Rx={self.n_rx})"
        )

    def get_rx_mask(self, img_grid: ImageGrid2D) -> np.ndarray:
        """
        Return a boolean mask for receiver positions on the given grid.

        Returns
        -------
        mask : ndarray of bool, shape (nx, ny)
            True where a receiver element is located.
        """
        mask = np.zeros((img_grid.nx, img_grid.ny), dtype=bool)  # (x, y)
        for (x, y), use in zip(self.positions.T, self.is_rx):
            if not use:
                continue
            ix, iy = img_grid.coord2index(x, y)
            if not (0 <= ix < img_grid.nx and 0 <= iy < img_grid.ny):
                raise ValueError("Receiver outside grid")
            mask[ix, iy] = True
        return mask

    def get_rx_mask_for_tx(self, img_grid: ImageGrid2D, tx_idx: int, drop: bool) -> np.ndarray:
        """
        Return a boolean mask for receiver positions for a given tx on the given grid.

        Returns
        -------
        mask : ndarray of bool, shape (nx, ny)
        """
        mask = np.zeros((img_grid.nx, img_grid.ny), dtype=bool)  # (x, y)
        rx_flags = self._get_rx_flags_for_tx(tx_idx, drop)  # 修正：调用私有方法
        for (x, y), use in zip(self.positions.T, rx_flags):
            if not use:
                continue
            ix, iy = img_grid.coord2index(x, y)
            if not (0 <= ix < img_grid.nx and 0 <= iy < img_grid.ny):
                raise ValueError("Receiver outside grid")
            mask[ix, iy] = True
        return mask

    def get_tx_mask(self, img_grid: ImageGrid2D) -> np.ndarray:
        """
        Return a boolean mask for transmitter positions on the given grid.

        Returns
        -------
        mask : ndarray of bool, shape (nx, ny)
            True where a transmitter element is located.
        """
        mask = np.zeros((img_grid.nx, img_grid.ny), dtype=bool)  # (x, y)
        for (x, y), use in zip(self.positions.T, self.is_tx):
            if not use:
                continue
            ix, iy = img_grid.coord2index(x, y)
            if not (0 <= ix < img_grid.nx and 0 <= iy < img_grid.ny):
                raise ValueError("Transmitter outside grid")
            mask[ix, iy] = True
        return mask


    def append(self, transducer: Transducer) -> None:
        """
        Append a single Transducer to the current array.

        Parameters
        ----------
        transducer : Transducer
            The transducer object to append. Its position, TX flag, and RX flag
            will be added to the internal arrays. If this array was originally
            constructed from Transducer objects, the internal list is updated.
        """
        if not isinstance(transducer, Transducer):
            raise TypeError("append_transducer expects a Transducer object")

        # Append position and flags
        new_pos = np.column_stack([self.positions, transducer.position])
        new_is_tx = np.r_[self.is_tx, transducer.is_tx]
        new_is_rx = np.r_[self.is_rx, transducer.is_rx]
        new_rx_flags = np.r_[self.rx_flags, transducer.is_rx]

        self.positions = new_pos
        self.is_tx = new_is_tx
        self.is_rx = new_is_rx
        self.rx_flags = new_rx_flags

        # If we were in object mode, append to list; else keep None so it's regenerated on demand
        if self._transducer_objs is not None:
            self._transducer_objs.append(transducer)

        # Update base class shape
        self.shape = (self.positions.shape[1],)


    def edit(
        self,
        *,
        index: Optional[int] = None,
        id: Optional[object] = None,
        position: Optional[Sequence[float]] = None,
        is_tx: Optional[bool] = None,
        is_rx: Optional[bool] = None,
        new_id: Optional[object] = None,
    ) -> None:
        """
        Edit a transducer's attributes in-place.

        You can locate the target by either its index (0-based) or its id.
        If both are given, index takes precedence.

        Parameters
        ----------
        index : int, optional
            The array index of the transducer to edit.
        id : object, optional
            The id of the transducer to edit (only used if index is None).
        position : sequence of float, optional
            New (x, y) position in meters.
        is_tx : bool, optional
            New transmitter flag.
        is_rx : bool, optional
            New receiver flag.
        new_id : object, optional
            New id for the transducer.

        Raises
        ------
        ValueError
            If no matching transducer is found.
        """
        # Find target index
        if index is not None:
            if not (0 <= index < self.n_elements):
                raise IndexError("index out of range")
            idx = index
        elif id is not None:
            # Look in object list if exists, else generate and search
            trans_list = self.transducers
            matches = [i for i, t in enumerate(trans_list) if t.id == id]
            if not matches:
                raise ValueError(f"No transducer found with id={id!r}")
            idx = matches[0]
        else:
            raise ValueError("Either index or id must be provided")

        # Update positions and flags in array representation
        if position is not None:
            pos_arr = np.asarray(position, float).ravel()
            if pos_arr.size != 2:
                raise ValueError("position must be length-2")
            self.positions[:, idx] = pos_arr

        if is_tx is not None:
            self.is_tx[idx] = bool(is_tx)
        if is_rx is not None:
            self.is_rx[idx] = bool(is_rx)
            self.rx_flags[idx] = bool(is_rx)

        if new_id is not None and self._transducer_objs is not None:
            self._transducer_objs[idx].id = new_id

        # If we have an object list, update it too
        if self._transducer_objs is not None:
            t = self._transducer_objs[idx]
            if position is not None:
                t.position = pos_arr
            if is_tx is not None:
                t.is_tx = bool(is_tx)
            if is_rx is not None:
                t.is_rx = bool(is_rx)
            if new_id is not None:
                t.id = new_id

    # ------------------------------------------------------------------
    # Class constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_ring_array_2D(
            cls,
            grid: ImageGrid2D,
            r: Optional[float] = None,
            n: int = 256,
            *,
            start_angle: float = 0.0
    ) -> "TransducerArray2D":
        """
        Build an equally spaced circular (ring) array centered at the grid origin.

        If `radius` is None, the maximum radius that fits inside the grid
        (minus a half-grid spacing margin) is used. If `radius` is provided,
        it must not exceed this maximum. All elements are TX and RX (full-ring).

        Parameters
        ----------
        grid : ImageGrid2D
            Image grid providing extent and spacing.
        radius : float or None
            Ring radius in meters. If None, use the largest inscribed radius.
        n_elements : int, default 256
            Number of transducer elements placed uniformly on the ring.
        start_angle : float, default 0.0
            Starting angle (radians) for the first element.

        Returns
        -------
        TransducerArray2D
            Constructed full-ring array with is_tx=is_rx=True.
        """
        # Compute maximum inscribed radius from grid extent
        xmin, xmax, ymin, ymax = grid.extent
        width = float(xmax - xmin)
        height = float(ymax - ymin)
        # Small margin to keep indices inside bounds when mapping to grid
        try:
            dx = float(getattr(grid, "dx"))
            dy = float(getattr(grid, "dy"))
            margin = 0.5 * min(dx, dy)
        except Exception:
            margin = 0.0
        r_max = 0.5 * min(width, height) - margin
        r_max = max(r_max, 0.0)

        # Determine radius
        if r is None:
            r = r_max
        else:
            r = float(r)
            if r <= 0:
                raise ValueError("radius must be positive")
            if r > r_max + 1e-12:
                raise ValueError(
                    f"radius={r:.6g} m exceeds grid limit r_max={r_max:.6g} m"
                )

        # Uniform angles and positions (centered at (0,0))
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + float(start_angle)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        positions = np.vstack([x, y]).astype(np.float32)

        # All elements are both TX and RX
        flags = np.ones(n, dtype=bool)
        return cls(positions=positions, is_tx=flags, is_rx=flags)

    # ------------------------------------------------------------------
    # Accessor: build Transducer objects from current arrays
    # ------------------------------------------------------------------
    def get_transducers(self, *, assign_ids: bool = True) -> List[Transducer]:
        """
        Return a list of Transducer objects reflecting the CURRENT state
        (positions/is_tx/is_rx). Useful when the array was constructed from
        raw arrays and you want synchronized Transducer instances.

        Parameters
        ----------
        assign_ids : bool, default True
            If True, set each transducer.id = its index.

        Returns
        -------
        list[Transducer]
        """
        transducers: List[Transducer] = []
        N = self.n_elements
        for i in range(N):
            t = Transducer(
                self.positions[:, i],
                is_tx=bool(self.is_tx[i]),
                is_rx=bool(self.is_rx[i]),
                id=(i if assign_ids else None),
            )
            transducers.append(t)
        return transducers

    def print_transducers(self) -> None:
        """
        Print one line of information for each transducer element.
        """
        for i, t in enumerate(self.get_transducers(assign_ids=True)):
            print(f"[{i:03d}] {t}")



    def attach_to_grid(self, img_grid: ImageGrid2D) -> "TransducerArray2D":
        """
        Snap all transducer positions to the nearest grid node (same logic as mask building)
        and return a NEW TransducerArray2D whose positions lie exactly on grid nodes.

        Parameters
        ----------
        img_grid : ImageGrid2D
            Target image grid that defines the discrete node locations.

        Returns
        -------
        TransducerArray2D
            A new array with positions replaced by their nearest grid-node coordinates.
            TX/RX flags are preserved.

        Raises
        ------
        ValueError
            If any element lies outside the grid bounds (consistent with get_*_mask).
        NotImplementedError
            If the grid does not provide index2coord (to avoid ambiguous half-cell offsets).
        """
        # Require an inverse mapping to ensure coordinates exactly match mask nodes
        if not hasattr(img_grid, "index2coord"):
            raise NotImplementedError(
                "ImageGrid2D.index2coord(ix, iy) is required for attach_to_grid "
                "to guarantee exact node coordinates consistent with mask generation."
            )

        snapped = np.zeros_like(self.positions, dtype=np.float32)  # (2, N)
        for i, (x, y) in enumerate(self.positions.T):
            ix, iy = img_grid.coord2index(float(x), float(y))
            if not (0 <= ix < img_grid.nx and 0 <= iy < img_grid.ny):
                raise ValueError("Element outside grid")
            xg, yg = img_grid.index2coord(ix, iy)  # exact node coordinate
            snapped[:, i] = (float(xg), float(yg))

        # Preserve TX/RX flags; RX gating (rx_flags) in the new instance defaults to is_rx
        return TransducerArray2D(
            positions=snapped,
            is_tx=self.is_tx.copy(),
            is_rx=self.is_rx.copy(),
        )
