import cupy as cp
from cupy.cuda import cusolver, cublas

################################################################################
# 1) The CUDA kernels (corresponding to the kernels in MEX) all assume that the 
# arrays are in column-major (F-order) linear index.
################################################################################
_cplx_header = r'''
////////////////////////////////////////////////////////////////////////////////
// Basic float2 operations in complex form
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ float2 cplx_make(float r, float i){
    return make_float2(r, i);
}
__device__ __forceinline__ float2 cplx_set0(){
    return make_float2(0.0f, 0.0f);
}
__device__ __forceinline__ float2 cplx_add(float2 a, float2 b){
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ float2 cplx_sub(float2 a, float2 b){
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ float2 cplx_mul(float2 a, float2 b){
    // (a.x + i*a.y)*(b.x + i*b.y)
    return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
'''

# -----------------------------------------------------------------------------
# Construct a tridiagonal matrix
# -----------------------------------------------------------------------------
_triDiagMat_src = _cplx_header + r'''
extern "C" __global__
void triDiagMat(const float2* Ad,
                const float2* Al,
                const float2* Au,
                float2*       A,
                int           N)
{
    unsigned int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < N && col_idx < N) {
        int idx = row_idx + N*col_idx;
        float2 z = cplx_set0();
        if      (row_idx == col_idx    ) A[idx] = Ad[row_idx];
        else if (row_idx == col_idx + 1) A[idx] = Al[col_idx];
        else if (row_idx == col_idx - 1) A[idx] = Au[row_idx];
        else                             A[idx] = z;
    }
}
'''.strip()

# -----------------------------------------------------------------------------
# Copy internal block (set boundary to 0)
# -----------------------------------------------------------------------------
_copyInterior_src = _cplx_header + r'''
extern "C" __global__
void copyInterior(const float2* A,
                  float2*       B,
                  int           N)
{
    unsigned int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < N && col_idx < N) {
        int idx = row_idx + N*col_idx;
        if (row_idx==0 || row_idx==N-1 || col_idx==0 || col_idx==N-1)
            B[idx] = cplx_set0();
        else
            B[idx] = A[idx];
    }
}
'''.strip()

# -----------------------------------------------------------------------------
# Initialize the internal identity matrix
# -----------------------------------------------------------------------------
_initializeIdentity_src = _cplx_header + r'''
extern "C" __global__
void initializeIdentity(float2* A,
                        int     N)
{
    unsigned int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < N && col_idx < N) {
        int idx = row_idx + N*col_idx;
        if (row_idx!=0 && row_idx!=N-1 && row_idx==col_idx)
            A[idx] = cplx_make(1.0f, 0.0f);
        else
            A[idx] = cplx_set0();
    }
}
'''.strip()

# -----------------------------------------------------------------------------
# C = D - B*A (Left-multiplication by tridiagonal)
#       —— The boundary threads of the block are responsible for writing to the 
# left/right neighbors in shared memory to avoid race conditions.
# -----------------------------------------------------------------------------
_triDiagMultLeftPlusD_src = _cplx_header + r'''
extern "C" __global__
void triDiagMultLeftPlusD(const float2* A,   // size×size  read
                          const float2* Bd,  // size       read
                          const float2* Bl,  // size-1     read
                          const float2* Bu,  // size-1     read
                          const float2* Dd,  // size       read
                          const float2* Dl,  // size-1     read
                          const float2* Du,  // size-1     read
                          float2*       C,   // size×size  write
                          int           N)
{
    __shared__ float2 sA[32][34];
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        // center
        sA[threadIdx.y][threadIdx.x+1] = A[row + N*col];
        // Left neighbor (first column thread write within block)
        if (threadIdx.x==0) {
            sA[threadIdx.y][0] = (row>0) ? A[(row-1) + N*col] : cplx_set0();
        }
        // Right neighbor (last column thread write within block)
        if (threadIdx.x==blockDim.x-1) {
            sA[threadIdx.y][threadIdx.x+2] = (row<N-1) ? A[(row+1) + N*col] : cplx_set0();
        }
        __syncthreads();

        // Tridiagonal D(row, col)
        float2 Dval = cplx_set0();
        if      (row == col  ) Dval = Dd[row];
        else if (row == col+1) Dval = Dl[col];
        else if (row == col-1) Dval = Du[row];

        // Three-diagonal B row vector
        float2 Bd_ = Bd[row];
        float2 Bl_ = (row==0)   ? cplx_set0() : Bl[row-1];
        float2 Bu_ = (row<N-1)  ? Bu[row]     : cplx_set0();

        float2 center = sA[threadIdx.y][threadIdx.x+1];
                float2 left   = (row==0) ? cplx_set0() : sA[threadIdx.y][threadIdx.x];
                float2 right  = (row==N-1) ? cplx_set0() : sA[threadIdx.y][threadIdx.x+2];

        float2 res = cplx_sub(Dval,
                              cplx_add(cplx_mul(Bd_, center),
                                       cplx_add(cplx_mul(Bl_, left),
                                                cplx_mul(Bu_, right))));
        C[row + N*col] = res;
    }
}
'''.strip()

# -----------------------------------------------------------------------------
# C = A * B (Right-multiplication of tridiagonal)
# -----------------------------------------------------------------------------
_triDiagMultRight_src = _cplx_header + r'''
extern "C" __global__
void triDiagMultRight(const float2* A,   // numRows×numCols  read
                      const float2* Bd,  // numCols          read
                      const float2* Bl,  // numCols-1        read
                      const float2* Bu,  // numCols-1        read
                      float2*       C,   // numRows×numCols  write
                      int           numRows,
                      int           numCols)
{
    __shared__ float2 sA[32][34];
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < numRows && col < numCols) {
        // center
        sA[threadIdx.x][threadIdx.y+1] = A[row + numRows*col];
        // lift neighbor
        if (threadIdx.y==0) {
            sA[threadIdx.x][0] = (col>0) ? A[row + numRows*(col-1)] : cplx_set0();
        }
        // right neighbor
        if (threadIdx.y==blockDim.y-1) {
            sA[threadIdx.x][threadIdx.y+2] = (col<numCols-1) ? A[row + numRows*(col+1)] : cplx_set0();
        }
        __syncthreads();

        float2 Bd_ = Bd[col];
        float2 Bu_ = (col==0)          ? cplx_set0() : Bu[col-1];
        float2 Bl_ = (col==numCols-1)  ? cplx_set0() : Bl[col];

        float2 center = sA[threadIdx.x][threadIdx.y+1];
                float2 left   = (col==0) ? cplx_set0() : sA[threadIdx.x][threadIdx.y];
                float2 right  = (col==numCols-1) ? cplx_set0() : sA[threadIdx.x][threadIdx.y+2];

        float2 res = cplx_add(cplx_mul(Bd_, center),
                              cplx_add(cplx_mul(Bu_, left),
                                       cplx_mul(Bl_, right)));
        C[row + numRows*col] = res;
    }
}
'''.strip()

# -----------------------------------------------------------------------------
# Compile the kernel
#     
# -----------------------------------------------------------------------------
triDiagMat_kernel           = cp.RawKernel(_triDiagMat_src,           "triDiagMat",           options=("-fmad=false",))
copyInterior_kernel         = cp.RawKernel(_copyInterior_src,         "copyInterior",         options=("-fmad=false",))
initializeIdentity_kernel   = cp.RawKernel(_initializeIdentity_src,   "initializeIdentity",   options=("-fmad=false",))
triDiagMultLeftPlusD_kernel = cp.RawKernel(_triDiagMultLeftPlusD_src, "triDiagMultLeftPlusD", options=("-fmad=false",))
triDiagMultRight_kernel     = cp.RawKernel(_triDiagMultRight_src,     "triDiagMultRight",     options=("-fmad=false",))

################################################################################
# 2) helper: F‑order complex64 ↔ float2 view
################################################################################

def toFloat2F(arr: cp.ndarray):
    """flatten(arr,'F') → view float32 → (N,2)"""
    return arr.ravel(order='F').view(cp.float32).reshape(-1, 2)

def fromFloat2F(arr_f2: cp.ndarray, shape, out: cp.ndarray):
    """float2 array → Fortran‑order complex64"""
    out[...] = arr_f2.reshape(-1).view(cp.complex64).reshape(shape, order='F')

################################################################################
# 3) Main function: block LU with cuSOLVER inversion
################################################################################

def decompBlockLU(Ld, Ll, Lu,
                  Dd, Dl, Du,
                  Ud, Ul, Uu):
    Ny, Nx = Dd.shape

    # -- ensure complex64 & F‑order ------------------------------------------------
    Ld = cp.asfortranarray(Ld, dtype=cp.complex64); Ll = cp.asfortranarray(Ll, dtype=cp.complex64); Lu = cp.asfortranarray(Lu, dtype=cp.complex64)
    Dd = cp.asfortranarray(Dd, dtype=cp.complex64); Dl = cp.asfortranarray(Dl, dtype=cp.complex64); Du = cp.asfortranarray(Du, dtype=cp.complex64)
    Ud = cp.asfortranarray(Ud, dtype=cp.complex64); Ul = cp.asfortranarray(Ul, dtype=cp.complex64); Uu = cp.asfortranarray(Uu, dtype=cp.complex64)

    # -- buffers ------------------------------------------------------------------
    T         = cp.zeros((Ny, Ny),     dtype=cp.complex64, order='F')
    Tinterior = cp.zeros((Ny, Ny),     dtype=cp.complex64, order='F')
    invT      = cp.zeros((Ny, Ny, Nx), dtype=cp.complex64, order='F')

    threads = (32, 32)
    blocks  = ((Ny + 31)//32, (Ny + 31)//32)

    # Step 1: initialize T ← diag(Dd[:,0],Dl[:,0],Du[:,0])
    triDiagMat_kernel(blocks, threads,
                      (toFloat2F(Dd[:,0]), toFloat2F(Dl[:,0]), toFloat2F(Du[:,0]),
                       toFloat2F(T), Ny))
    fromFloat2F(toFloat2F(T), T.shape, T)

    # cuSOLVER handle & const
    handle = cusolver.create()
    lda, n = Ny, Ny-2
    ipiv   = cp.empty((Ny,), dtype=cp.int32)
    info   = cp.empty((1,),  dtype=cp.int32)

    byte_off = (1 + lda*1) * 8  # Shift to the internal block (row=1,col=1)

    # Step 2: Loop through the col ---------------------------------------------------------------
    for j in range(1, Nx):
        # 2.1 copyInterior
        copyInterior_kernel(blocks, threads, (toFloat2F(T), toFloat2F(Tinterior), Ny))
        fromFloat2F(toFloat2F(Tinterior), Tinterior.shape, Tinterior)

        # 2.2 Initialize the internal unit of invT(:,:,j-1)
        slice_prev = invT[:, :, j-1]
        initializeIdentity_kernel(blocks, threads, (toFloat2F(slice_prev), Ny))
        fromFloat2F(toFloat2F(slice_prev), slice_prev.shape, slice_prev)

        # 2.3 invert Interior(T)
        sub_ptr = int(Tinterior.data.ptr + byte_off)
        lwork   = cusolver.cgetrf_bufferSize(handle, n, n, sub_ptr, lda)
        work    = cp.empty((lwork,), dtype=cp.complex64)
        inv_ptr = int(invT.data.ptr + ((j-1)*Ny*Ny + 1 + lda*1) * 8)
        cusolver.cgetrf(handle, n, n, sub_ptr, lda,
                        work.data.ptr, ipiv.data.ptr, info.data.ptr)
        cusolver.cgetrs(handle, cublas.CUBLAS_OP_N, n, n,
                        sub_ptr, lda, ipiv.data.ptr, inv_ptr, lda,
                        info.data.ptr)

        # 2.4 Tinterior ← invT_prev * U(j-1)
        triDiagMultRight_kernel(blocks, threads,
            (toFloat2F(slice_prev),
             toFloat2F(Ud[:, j-1]), toFloat2F(Ul[:, j-1]), toFloat2F(Uu[:, j-1]),
             toFloat2F(Tinterior), Ny, Ny))
        fromFloat2F(toFloat2F(Tinterior), Tinterior.shape, Tinterior)

        # 2.5 T ← D(j) − L(j-1) * Tinterior
        triDiagMultLeftPlusD_kernel(blocks, threads,
            (toFloat2F(Tinterior),
             toFloat2F(Ld[:, j-1]), toFloat2F(Ll[:, j-1]), toFloat2F(Lu[:, j-1]),
             toFloat2F(Dd[:, j]),   toFloat2F(Dl[:, j]),   toFloat2F(Du[:, j]),
             toFloat2F(T), Ny))
        fromFloat2F(toFloat2F(T), T.shape, T)

    # -------------------------------------------------------------------------
    # Step 3: the last col
    # -------------------------------------------------------------------------
    copyInterior_kernel(blocks, threads, (toFloat2F(T), toFloat2F(Tinterior), Ny))
    fromFloat2F(toFloat2F(Tinterior), Tinterior.shape, Tinterior)

    slice_last = invT[:, :, Nx-1]
    initializeIdentity_kernel(blocks, threads, (toFloat2F(slice_last), Ny))
    fromFloat2F(toFloat2F(slice_last), slice_last.shape, slice_last)

    sub_ptr = int(Tinterior.data.ptr + byte_off)
    lwork   = cusolver.cgetrf_bufferSize(handle, n, n, sub_ptr, lda)
    work    = cp.empty((lwork,), dtype=cp.complex64)
    inv_ptr = int(invT.data.ptr + ((Nx-1)*Ny*Ny + 1 + lda*1) * 8)
    cusolver.cgetrf(handle, n, n, sub_ptr, lda,
                    work.data.ptr, ipiv.data.ptr, info.data.ptr)
    cusolver.cgetrs(handle, cublas.CUBLAS_OP_N, n, n,
                    sub_ptr, lda, ipiv.data.ptr, inv_ptr, lda,
                    info.data.ptr)

    return invT
