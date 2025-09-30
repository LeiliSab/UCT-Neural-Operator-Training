import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import coo_matrix
import scipy.sparse as sp

try:
    import cupy as cp
    from .decompBlockLU import decompBlockLU as decompBlockLU_gpu
    from .applyBlockLU import applyBlockLU as applyBlockLU_gpu

    _GPU_AVAILABLE = True
except ImportError:
    cp = None
    _GPU_AVAILABLE = False

from .stencilOptParams import stencilOptParams


class HelmholtzSolver:
    """
    Solver for the Helmholtz equation with Perfectly Matched Layer (PML),
    using 9-point finite-difference and block-LU factorization on GPU (optional).

    Layout & Indexing Convention:
      - We keep shape=(Ny, Nx) in Python just as a labeling,
        but effectively we treat the flatten index as col-major:
        lin_idx(x, y) = y + Ny * x
        => 'x' is the outer loop, 'y' is the inner loop of the flatten array.
      - This matches the idea of Nx blocks, each block is (Ny x Ny).
      - In the final applyBlockLU, we also do
        row_idx = y_idx + Ny * x_idx + Nx*Ny * src_idx
        so that x increments in strides of Ny, y increments in stride=1,
        i.e. col-major.
    """

    def __init__(self, x, y, vel, atten, f,
                 signConvention, a0, L_PML, canUseGPU=False):
        # Basic parameters
        self.x = x;
        self.y = y
        self.vel = vel;
        self.atten = atten
        self.f = f;
        self.signConvention = signConvention
        self.a0 = a0;
        self.L_PML = L_PML
        self.canUseGPU = bool(canUseGPU and _GPU_AVAILABLE)

        # Grid size
        if x.ndim == 1:
            self.Nx = x.size
        else:
            self.Nx = x.shape[1]
        if y.ndim == 1:
            self.Ny = y.size
        else:
            self.Ny = y.shape[1]

        # Grid spacing
        self.h = float(np.mean(np.diff(x.ravel())))
        self.gh = float(np.mean(np.diff(y.ravel())))
        self.g = self.gh / self.h
        self.xmin, self.xmax = float(np.min(x)), float(np.max(x))
        self.ymin, self.ymax = float(np.min(y)), float(np.max(y))

        # Complex velocity and wavenumber
        SI = atten / (2 * np.pi)
        self.V = 1.0 / (1.0 / vel + 1j * SI * np.sign(signConvention))
        self.k = (2 * np.pi * f) / self.V

        # PML
        xe = np.linspace(self.xmin, self.xmax, 2 * (self.Nx - 1) + 1)
        ye = np.linspace(self.ymin, self.ymax, 2 * (self.Ny - 1) + 1)
        Xe, Ye = np.meshgrid(xe, ye, indexing='xy')
        xctr, xspan = 0.5 * (self.xmin + self.xmax), 0.5 * (self.xmax - self.xmin)
        yctr, yspan = 0.5 * (self.ymin + self.ymax), 0.5 * (self.ymax - self.ymin)
        sx = 2 * np.pi * a0 * f * (np.maximum(np.abs(Xe - xctr) - xspan + L_PML, 0) / L_PML) ** 2
        sy = 2 * np.pi * a0 * f * (np.maximum(np.abs(Ye - yctr) - yspan + L_PML, 0) / L_PML) ** 2
        ex = 1.0 + 1j * sx * np.sign(signConvention) / (2 * np.pi * f)
        ey = 1.0 + 1j * sy * np.sign(signConvention) / (2 * np.pi * f)
        bigA, bigB, bigC = ey / ex, ex / ey, ex * ey
        self.A = bigA[0::2, 1::2]
        self.B = bigB[1::2, 0::2]
        self.C = bigC[0::2, 0::2]

        # Stencil parameters
        self.b, self.d, self.e = stencilOptParams(
            np.min(vel), np.max(vel), f, self.h, self.g
        )

        # Assemble the sparse matrix
        self._populate_sparse_matrix()
        self.PML = self.C

        # GPU block-LU path
        if self.canUseGPU:
            self._compute_block_lu_gpu()

    def _populate_sparse_matrix(self):
        """
        We use col-major flatten: lin_idx(x,y) = y + Ny*x
        => Nx blocks, each block is Ny x Ny
        => shape = Nx*Ny x Nx*Ny
        => row-major python view is irrelevant, but we physically store it in coo/csc.
        """
        Nx, Ny = self.Nx, self.Ny
        num_elements = 9 * (Nx - 2) * (Ny - 2) + (Nx * Ny - (Nx - 2) * (Ny - 2))
        rows = np.zeros(num_elements, dtype=np.float64)
        cols = np.zeros(num_elements, dtype=np.float64)
        vals = np.zeros(num_elements, dtype=np.complex128)

        def lin_idx(x, y):
            # col-major index
            return y + Ny * x

        val_idx = 0
        for x_idx in range(Nx):
            for y_idx in range(Ny):
                if (x_idx == 0 or x_idx == Nx - 1 or y_idx == 0 or y_idx == Ny - 1):
                    # boundary
                    idx = lin_idx(x_idx, y_idx)
                    rows[val_idx] = idx
                    cols[val_idx] = idx
                    vals[val_idx] = 1.0
                    val_idx += 1
                else:
                    idxC = lin_idx(x_idx, y_idx)
                    # center
                    rows[val_idx] = idxC
                    cols[val_idx] = idxC
                    vals[val_idx] = (1 - self.d - self.e) * self.C[y_idx, x_idx] * (self.k[y_idx, x_idx] ** 2) \
                                    - self.b * (self.A[y_idx, x_idx] + self.A[y_idx, x_idx - 1]
                                                + self.B[y_idx, x_idx] / (self.g ** 2)
                                                + self.B[y_idx - 1, x_idx] / (self.g ** 2)) / (self.h ** 2)
                    val_idx += 1
                    # left
                    rows[val_idx] = idxC
                    cols[val_idx] = lin_idx(x_idx - 1, y_idx)
                    vals[val_idx] = (
                            (self.b * self.A[y_idx, x_idx - 1]
                             - ((1 - self.b) / 2) * (
                                     self.B[y_idx, x_idx - 1] / (self.g ** 2) + self.B[y_idx - 1, x_idx - 1] / (
                                     self.g ** 2))
                             ) / (self.h ** 2)
                            + (self.d / 4) * self.C[y_idx, x_idx - 1] * (self.k[y_idx, x_idx - 1] ** 2)
                    )
                    val_idx += 1
                    # right
                    rows[val_idx] = idxC
                    cols[val_idx] = lin_idx(x_idx + 1, y_idx)
                    vals[val_idx] = (
                            (self.b * self.A[y_idx, x_idx]
                             - ((1 - self.b) / 2) * (
                                     self.B[y_idx, x_idx + 1] / (self.g ** 2) + self.B[y_idx - 1, x_idx + 1] / (
                                     self.g ** 2))
                             ) / (self.h ** 2)
                            + (self.d / 4) * self.C[y_idx, x_idx + 1] * (self.k[y_idx, x_idx + 1] ** 2)
                    )
                    val_idx += 1
                    # down
                    rows[val_idx] = idxC
                    cols[val_idx] = lin_idx(x_idx, y_idx - 1)
                    vals[val_idx] = (
                            (self.b * self.B[y_idx - 1, x_idx] / (self.g ** 2)
                             - ((1 - self.b) / 2) * (self.A[y_idx - 1, x_idx] + self.A[y_idx - 1, x_idx - 1])
                             ) / (self.h ** 2)
                            + (self.d / 4) * self.C[y_idx - 1, x_idx] * (self.k[y_idx - 1, x_idx] ** 2)
                    )
                    val_idx += 1
                    # up
                    rows[val_idx] = idxC
                    cols[val_idx] = lin_idx(x_idx, y_idx + 1)
                    vals[val_idx] = (
                            (self.b * self.B[y_idx, x_idx] / (self.g ** 2)
                             - ((1 - self.b) / 2) * (self.A[y_idx + 1, x_idx] + self.A[y_idx + 1, x_idx - 1])
                             ) / (self.h ** 2)
                            + (self.d / 4) * self.C[y_idx + 1, x_idx] * (self.k[y_idx + 1, x_idx] ** 2)
                    )
                    val_idx += 1
                    # bottom-left
                    rows[val_idx] = idxC
                    cols[val_idx] = lin_idx(x_idx - 1, y_idx - 1)
                    vals[val_idx] = (
                            ((1 - self.b) / 2) * (
                            self.A[y_idx - 1, x_idx - 1] + self.B[y_idx - 1, x_idx - 1] / (self.g ** 2)) / (
                                    self.h ** 2)
                            + (self.e / 4) * self.C[y_idx - 1, x_idx - 1] * (self.k[y_idx - 1, x_idx - 1] ** 2)
                    )
                    val_idx += 1
                    # bottom-right
                    rows[val_idx] = idxC
                    cols[val_idx] = lin_idx(x_idx + 1, y_idx - 1)
                    vals[val_idx] = (
                            ((1 - self.b) / 2) * (
                            self.A[y_idx - 1, x_idx] + self.B[y_idx - 1, x_idx + 1] / (self.g ** 2)) / (self.h ** 2)
                            + (self.e / 4) * self.C[y_idx - 1, x_idx + 1] * (self.k[y_idx - 1, x_idx + 1] ** 2)
                    )
                    val_idx += 1
                    # top-left
                    rows[val_idx] = idxC
                    cols[val_idx] = lin_idx(x_idx - 1, y_idx + 1)
                    vals[val_idx] = (
                            ((1 - self.b) / 2) * (
                            self.A[y_idx + 1, x_idx - 1] + self.B[y_idx, x_idx - 1] / (self.g ** 2)) / (self.h ** 2)
                            + (self.e / 4) * self.C[y_idx + 1, x_idx - 1] * (self.k[y_idx + 1, x_idx - 1] ** 2)
                    )
                    val_idx += 1
                    # top-right
                    rows[val_idx] = idxC
                    cols[val_idx] = lin_idx(x_idx + 1, y_idx + 1)
                    vals[val_idx] = (
                            ((1 - self.b) / 2) * (
                            self.A[y_idx + 1, x_idx] + self.B[y_idx, x_idx + 1] / (self.g ** 2)) / (self.h ** 2)
                            + (self.e / 4) * self.C[y_idx + 1, x_idx + 1] * (self.k[y_idx + 1, x_idx + 1] ** 2)
                    )
                    val_idx += 1

        coo = coo_matrix((vals, (rows, cols)), shape=(Nx * Ny, Nx * Ny))
        self.HelmholtzEqn = coo.tocsc()

    def _compute_block_lu_gpu(self):
        """
        In col-major flatten: offset = y + Ny*x
        We want Nx blocks, each block is (Ny x Ny).
        => block j => row range = [ j*Ny : (j+1)*Ny ], col range = [ j*Ny : (j+1)*Ny ]
        etc.
        """
        Nx, Ny = self.Nx, self.Ny

        # shapes
        Dd = cp.zeros((Ny, Nx), dtype=cp.complex64)
        Dl = cp.zeros((Ny - 1, Nx), dtype=cp.complex64)
        Du = cp.zeros((Ny - 1, Nx), dtype=cp.complex64)

        self.Ld = cp.zeros((Ny, Nx - 1), dtype=cp.complex64)
        self.Ll = cp.zeros((Ny - 1, Nx - 1), dtype=cp.complex64)
        self.Lu = cp.zeros((Ny - 1, Nx - 1), dtype=cp.complex64)

        self.Ud = cp.zeros((Ny, Nx - 1), dtype=cp.complex64)
        self.Ul = cp.zeros((Ny - 1, Nx - 1), dtype=cp.complex64)
        self.Uu = cp.zeros((Ny - 1, Nx - 1), dtype=cp.complex64)

        # flatten csc
        # col-major => block j => row_start=j*Ny, row_end=(j+1)*Ny
        # => get Nx blocks along the x direction
        for j in range(Nx - 1):
            row_start = j * Ny
            row_end = (j + 1) * Ny

            D_block = self.HelmholtzEqn[row_start:row_end, row_start:row_end]
            L_block = self.HelmholtzEqn[(j + 1) * Ny: (j + 2) * Ny, row_start:row_end]
            U_block = self.HelmholtzEqn[row_start:row_end, (j + 1) * Ny: (j + 2) * Ny]

            # diag of D => Dd[:,j], etc
            Dd[:, j] = cp.asarray(D_block.diagonal())
            Dl[:, j] = cp.asarray(D_block.diagonal(-1))
            Du[:, j] = cp.asarray(D_block.diagonal(1))

            self.Ld[:, j] = cp.asarray(L_block.diagonal())
            self.Ll[:, j] = cp.asarray(L_block.diagonal(-1))
            self.Lu[:, j] = cp.asarray(L_block.diagonal(1))

            self.Ud[:, j] = cp.asarray(U_block.diagonal())
            self.Ul[:, j] = cp.asarray(U_block.diagonal(-1))
            self.Uu[:, j] = cp.asarray(U_block.diagonal(1))

        # last block j=Nx-1
        j = Nx - 1
        row_start = j * Ny
        row_end = (j + 1) * Ny
        D_block = self.HelmholtzEqn[row_start:row_end, row_start:row_end]
        Dd[:, j] = cp.asarray(D_block.diagonal())
        Dl[:, j] = cp.asarray(D_block.diagonal(-1))
        Du[:, j] = cp.asarray(D_block.diagonal(1))

        self.invT = decompBlockLU(self.Ld, self.Ll, self.Lu,
                                  Dd, Dl, Du,
                                  self.Ud, self.Ul, self.Uu)

    def solve(self, src, adjoint=False):
        """
        Solve with block-LU or fallback to spsolve if GPU is not used
        src: shape=(Ny, Nx, K)
        BUT col-major kernel => row_idx = y_idx + Ny*x_idx + Nx*Ny* src_idx
        => x steps by +Ny, y steps by +1
        """
        Ny, Nx, K = src.shape
        if (Nx, Ny) != (self.Nx, self.Ny):
            raise ValueError("Dimension mismatch")

        if self.canUseGPU:
            src_gpu = cp.asarray(src, dtype=cp.complex64)
            wv_gpu = applyBlockLU_gpu(
                src_gpu,
                self.Ld, self.Ll, self.Lu,
                self.Ud, self.Ul, self.Uu,
                self.invT,
                adjoint
            )
            sf = 8 * (np.pi ** 2) * (self.f ** 2)
            P_gpu = cp.asarray(self.PML, dtype=cp.complex64)
            V_gpu = cp.asarray(self.V, dtype=cp.complex64)
            mat = sf * (P_gpu / V_gpu).astype(cp.complex64)
            virt_gpu = mat[..., None] * wv_gpu
            return cp.asnumpy(wv_gpu), cp.asnumpy(virt_gpu)

        else:
            # CPU path: one-time splu + repeated solve
            if not hasattr(self, '_cpu_lu'):
                self._cpu_lu = spla.splu(self.HelmholtzEqn)

            # Fortran-order flatten
            SRC = np.asarray(src, dtype=np.complex64, order='F')
            rhs = SRC.reshape(Ny * Nx, K, order='F')

            # trans='N' → solve A x = b; trans='H' → solve Aᴴ x = b
            trans_flag = 'H' if adjoint else 'N'
            sol = self._cpu_lu.solve(rhs, trans=trans_flag)

            # Restore shape
            wv = sol.reshape((Ny, Nx, K), order='F')

            # Virtual source
            sf = 8 * (np.pi ** 2) * (self.f ** 2)
            mat = sf * (self.PML / self.V)
            virt = mat[..., None] * wv

            return wv, virt
