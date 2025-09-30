import cupy as cp

# Thread block size
threadsPerBlock1D = 32

# Basic operations of the complex float2 
_cplx_header = r'''
////////////////////////////////////////////////////////////////////////////////
// simulate several operations of cuComplex (such as addition, subtraction, multiplication, conjugation, etc.) in CUDA.
////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ float2 cplx_make(float r, float i){
    return make_float2(r, i);
}
__device__ __forceinline__ float2 cplx_add(float2 a, float2 b){
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ float2 cplx_sub(float2 a, float2 b){
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ float2 cplx_mul(float2 a, float2 b){
    // (a.x + i a.y)*(b.x + i b.y)
    return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
__device__ __forceinline__ float2 cplx_conj(float2 a){
    return make_float2(a.x, -a.y);
}
__device__ __forceinline__ float2 cplx_set0(){
    return make_float2(0.0f, 0.0f);
}
'''.strip()

_applyL_src = _cplx_header + r'''
extern "C" __global__
void applyL(
    const float2* __restrict__ SRC,
    const float2* __restrict__ Ld,
    const float2* __restrict__ Ll,
    const float2* __restrict__ Lu,
    const float2* __restrict__ u,
    float2*       tmp,
    int x_idx,
    int Nx, int Ny, int numSources)
{
    // __shared__ su[threadsPerBlock1D][threadsPerBlock1D+2] => 32x34
    __shared__ float2 su[32][34];

    unsigned int src_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_idx   = blockIdx.y * blockDim.y + threadIdx.y;

    if ((y_idx < Ny) && (src_idx < numSources)) {
        // row_idx = y_idx + Ny*x_idx + Nx*Ny*src_idx
        int row_idx = y_idx + Ny*x_idx + Nx*Ny*src_idx;

        // su[threadIdx.x][threadIdx.y+1] = u[row_idx - Ny];
        su[threadIdx.x][threadIdx.y+1] = u[row_idx - Ny];

        // boundary
        if ((threadIdx.y==0) && (y_idx>0)) {
            su[threadIdx.x][0] = u[row_idx - Ny - 1];
        }
        else if ((threadIdx.y==(blockDim.y-1)) && (y_idx<(Ny-1))) {
            su[threadIdx.x][threadIdx.y+2] = u[row_idx - Ny + 1];
        }
        __syncthreads();

        float2 valSRC = SRC[row_idx];
        float2 ldval  = Ld[y_idx + Ny*(x_idx-1)]; // Ld[y_idx + Ny*(x_idx-1)]
        float2 center = su[threadIdx.x][threadIdx.y+1];

        float2 out = cplx_set0();
        if (y_idx == 0) {
            float2 luval = Lu[y_idx + (Ny-1)*(x_idx-1)];
            float2 right = su[threadIdx.x][threadIdx.y+2];
            // result = SRC - (ld*center + lu*right)
            float2 t1 = cplx_mul(ldval, center);
            float2 t2 = cplx_mul(luval, right);
            out = cplx_sub(valSRC, cplx_add(t1, t2));
        }
        else if (y_idx == Ny-1) {
            float2 llval = Ll[(y_idx-1) + (Ny-1)*(x_idx-1)];
            float2 left  = su[threadIdx.x][threadIdx.y];
            float2 t1 = cplx_mul(ldval, center);
            float2 t2 = cplx_mul(llval, left);
            out = cplx_sub(valSRC, cplx_add(t1, t2));
        }
        else {
            float2 llval = Ll[(y_idx-1) + (Ny-1)*(x_idx-1)];
            float2 luval = Lu[y_idx + (Ny-1)*(x_idx-1)];
            float2 left  = su[threadIdx.x][threadIdx.y];
            float2 right = su[threadIdx.x][threadIdx.y+2];

            float2 t1 = cplx_mul(ldval, center);
            float2 t2 = cplx_mul(llval, left);
            float2 t3 = cplx_mul(luval, right);
            out = cplx_sub(valSRC, cplx_add(t1, cplx_add(t2, t3)));
        }
        tmp[y_idx + Ny*src_idx] = out;
    }
}
'''.strip()

_applyU_src = _cplx_header + r'''
extern "C" __global__
void applyU(
    const float2* __restrict__ Ud,
    const float2* __restrict__ Ul,
    const float2* __restrict__ Uu,
    const float2* __restrict__ u,
    float2*       tmp,
    int x_idx,
    int Nx, int Ny, int numSources)
{
    __shared__ float2 su[32][34];

    unsigned int src_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_idx   = blockIdx.y * blockDim.y + threadIdx.y;

    if (y_idx < Ny && src_idx < numSources) {
        int row_idx = y_idx + Ny*x_idx + Nx*Ny*src_idx;
        su[threadIdx.x][threadIdx.y+1] = u[row_idx + Ny];
        if ((threadIdx.y==0) && (y_idx>0)) {
            su[threadIdx.x][0] = u[row_idx + Ny - 1];
        }
        else if ((threadIdx.y==(blockDim.y-1)) && (y_idx<(Ny-1))) {
            su[threadIdx.x][threadIdx.y+2] = u[row_idx + Ny + 1];
        }
        __syncthreads();

        float2 udval = Ud[y_idx + Ny*x_idx];
        float2 center = su[threadIdx.x][threadIdx.y+1];
        float2 out = cplx_set0();
        if (y_idx == 0) {
            float2 uuval = Uu[y_idx + (Ny-1)*x_idx];
            float2 right = su[threadIdx.x][threadIdx.y+2];
            out = cplx_add(cplx_mul(udval, center),
                           cplx_mul(uuval, right));
        }
        else if (y_idx == Ny-1) {
            float2 ulval = Ul[(y_idx-1) + (Ny-1)*x_idx];
            float2 left  = su[threadIdx.x][threadIdx.y];
            out = cplx_add(cplx_mul(udval, center),
                           cplx_mul(ulval, left));
        }
        else {
            float2 ulval = Ul[(y_idx-1) + (Ny-1)*x_idx];
            float2 uuval = Uu[y_idx + (Ny-1)*x_idx];
            float2 left  = su[threadIdx.x][threadIdx.y];
            float2 right = su[threadIdx.x][threadIdx.y+2];
            out = cplx_add(cplx_mul(udval, center),
                  cplx_add(cplx_mul(ulval, left),
                           cplx_mul(uuval, right)));
        }
        tmp[y_idx + Ny*src_idx] = out;
    }
}
'''.strip()

_applyUadj_src = _cplx_header + r'''
extern "C" __global__
void applyUadj(
    const float2* __restrict__ SRC,
    const float2* __restrict__ Ud,
    const float2* __restrict__ Ul,
    const float2* __restrict__ Uu,
    float2*       u,
    const float2* __restrict__ tmp,
    int x_idx,
    int Nx, int Ny, int numSources)
{
    __shared__ float2 sTmp[32][34];
    unsigned int src_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_idx   = blockIdx.y * blockDim.y + threadIdx.y;
    if (y_idx < Ny && src_idx < numSources) {
        int row_idx = y_idx + Ny*x_idx + Nx*Ny*src_idx;
        sTmp[threadIdx.x][threadIdx.y+1] = tmp[y_idx + Ny*src_idx];
        if ((threadIdx.y==0) && (y_idx>0)) {
            sTmp[threadIdx.x][0] = tmp[y_idx + Ny*src_idx - 1];
        }
        else if ((threadIdx.y==(blockDim.y-1)) && (y_idx<(Ny-1))) {
            sTmp[threadIdx.x][threadIdx.y+2] = tmp[y_idx + Ny*src_idx + 1];
        }
        __syncthreads();

        float2 valSRC = SRC[row_idx];
        // conjf(Ud[y_idx + Ny*(x_idx-1)])
        float2 udval = Ud[y_idx + Ny*(x_idx-1)];
        float2 cud = cplx_conj(udval);

        float2 center = sTmp[threadIdx.x][threadIdx.y+1];
        float2 out = cplx_set0();
        if (y_idx==0) {
            float2 ulval = Ul[y_idx + (Ny-1)*(x_idx-1)];
            float2 cul   = cplx_conj(ulval);
            float2 right = sTmp[threadIdx.x][threadIdx.y+2];
            float2 t1 = cplx_mul(cud, center);
            float2 t2 = cplx_mul(cul, right);
            out = cplx_sub(valSRC, cplx_add(t1, t2));
        }
        else if (y_idx==(Ny-1)) {
            float2 uuval = Uu[(y_idx-1) + (Ny-1)*(x_idx-1)];
            float2 cuu   = cplx_conj(uuval);
            float2 left  = sTmp[threadIdx.x][threadIdx.y];
            float2 t1 = cplx_mul(cud, center);
            float2 t2 = cplx_mul(cuu, left);
            out = cplx_sub(valSRC, cplx_add(t1, t2));
        }
        else {
            float2 uuval = Uu[(y_idx-1) + (Ny-1)*(x_idx-1)];
            float2 ulval = Ul[y_idx + (Ny-1)*(x_idx-1)];
            float2 cuu   = cplx_conj(uuval);
            float2 cul   = cplx_conj(ulval);
            float2 left  = sTmp[threadIdx.x][threadIdx.y];
            float2 right = sTmp[threadIdx.x][threadIdx.y+2];
            float2 t1 = cplx_mul(cud, center);
            float2 t2 = cplx_mul(cuu, left);
            float2 t3 = cplx_mul(cul, right);
            out = cplx_sub(valSRC, cplx_add(t1, cplx_add(t2, t3)));
        }
        u[row_idx] = out;
    }
}
'''.strip()

_applyLadj_src = _cplx_header + r'''
extern "C" __global__
void applyLadj(
    const float2* __restrict__ Ld,
    const float2* __restrict__ Ll,
    const float2* __restrict__ Lu,
    const float2* __restrict__ u,
    float2*       tmp,
    int x_idx,
    int Nx, int Ny, int numSources)
{
    __shared__ float2 su[32][34];
    unsigned int src_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_idx   = blockIdx.y * blockDim.y + threadIdx.y;
    if (y_idx < Ny && src_idx < numSources) {
        int row_idx = y_idx + Ny*x_idx + Nx*Ny*src_idx;
        su[threadIdx.x][threadIdx.y+1] = u[row_idx + Ny];
        if ((threadIdx.y==0) && (y_idx>0)) {
            su[threadIdx.x][0] = u[row_idx + Ny - 1];
        }
        else if ((threadIdx.y==(blockDim.y-1)) && (y_idx<(Ny-1))) {
            su[threadIdx.x][threadIdx.y+2] = u[row_idx + Ny + 1];
        }
        __syncthreads();

        float2 valU = u[row_idx];
        float2 ldval = Ld[y_idx + Ny*x_idx];
        float2 cld   = cplx_conj(ldval);
        float2 center= su[threadIdx.x][threadIdx.y+1];
        float2 out   = cplx_set0();

        if (y_idx==0) {
            float2 llval = Ll[y_idx + (Ny-1)*x_idx];
            float2 cll   = cplx_conj(llval);
            float2 right = su[threadIdx.x][threadIdx.y+2];
            float2 t1 = cplx_mul(cld, center);
            float2 t2 = cplx_mul(cll, right);
            out = cplx_sub(valU, cplx_add(t1, t2));
        }
        else if (y_idx==(Ny-1)) {
            float2 luval = Lu[(y_idx-1) + (Ny-1)*x_idx];
            float2 clu   = cplx_conj(luval);
            float2 left  = su[threadIdx.x][threadIdx.y];
            float2 t1 = cplx_mul(cld, center);
            float2 t2 = cplx_mul(clu, left);
            out = cplx_sub(valU, cplx_add(t1, t2));
        }
        else {
            float2 luval = Lu[(y_idx-1) + (Ny-1)*x_idx];
            float2 llval = Ll[y_idx + (Ny-1)*x_idx];
            float2 clu   = cplx_conj(luval);
            float2 cll   = cplx_conj(llval);
            float2 left  = su[threadIdx.x][threadIdx.y];
            float2 right = su[threadIdx.x][threadIdx.y+2];
            float2 t1 = cplx_mul(cld, center);
            float2 t2 = cplx_mul(clu, left);
            float2 t3 = cplx_mul(cll, right);
            out = cplx_sub(valU, cplx_add(t1, cplx_add(t2, t3)));
        }
        tmp[y_idx + Ny*src_idx] = out;
    }
}
'''.strip()

# complie the kernel
applyL_kernel = cp.RawKernel(_applyL_src, "applyL")
applyU_kernel = cp.RawKernel(_applyU_src, "applyU")
applyUadj_kernel = cp.RawKernel(_applyUadj_src, "applyUadj")
applyLadj_kernel = cp.RawKernel(_applyLadj_src, "applyLadj")


def toFloat2F(arr: cp.ndarray):
    """
    Flatten the `arr` (complex, Fortran-order, shape=...) into a 
    one-dimensional array (traversed in column order), and then
    treat it as `float32[2]` => return a `cp.float32` array with shape=(-1, 2).
    """
    arr_1d = arr.ravel(order='F')
    return arr_1d.view(cp.float32).reshape(-1, 2)


def fromFloat2F(arr_f2: cp.ndarray, shape, out: cp.ndarray):
    """
    Restore the `float2` array written by the kernel back to `out` (complex, Fortran-order).
    """
    arr_f1d = arr_f2.reshape(-1)  # float32 one dim
    arr_cplx = arr_f1d.view(cp.complex64)  # complex64
    arr_cplx_3d = arr_cplx.reshape(shape, order='F')
    out[...] = arr_cplx_3d


def applyInvTpostL(invT_x, tmp, u_x):
    """
    applyInvTpostL: alpha=1, beta=0
    => u_x = invT_x @ tmp
    where:
      invT_x: shape (Ny,Ny)
      tmp:    shape (Ny, numSources)
      u_x:    shape (Ny, numSources)
    """
    # (Ny,Ny) @ (Ny,numSources) => (Ny, numSources)
    u_x[...] = invT_x @ tmp


def applyInvTpostU(invT_x, tmp, u_x):
    """
    applyInvTpostU: alpha=-1, beta=1
    => u_x -= invT_x @ tmp
    """
    u_x[...] = u_x - invT_x @ tmp


def applyInvTpreU(invT_x, u_xm1, tmp):
    """
    applyInvTpreU: conjTranspose( invT_x ) * u_xm1
    => tmp = invT_x^H @ u_xm1
    """
    # (Ny,Ny), conj().T => (Ny,Ny), @ (Ny,numSources) => (Ny,numSources)
    u_conjT = invT_x.conj().T
    tmp[...] = u_conjT @ u_xm1


def applyInvTpostLadj(invT_x, tmp, u_x):
    """
    applyInvTpostLadj: conjTranspose( invT_x ) * tmp => u_x
    => alpha=1, beta=0
    """
    u_conjT = invT_x.conj().T
    u_x[...] = u_conjT @ tmp


def applyBlockLU(SRC, Ld, Ll, Lu, Ud, Ul, Uu, invT, adjHelmholtzEqn):
    """
    - Parameters
    SRC: (Ny, Nx, num_sources) complex, single
    Ld, Ll, Lu, Ud, Ul, Uu: Three-diagonal parameters (the same as those generated by the previous block LU)
    invT: (Ny, Ny, Nx)
    adjHelmholtzEqn: bool, determines whether to take the "adjoint" branch or the "forward" branch
    - Returns
    u: of the same shape (Ny, Nx, num_sources), single complex
    """
    Ny, Nx, num_sources = SRC.shape

    # ensure cp.complex64, Fortran-order
    SRC = cp.asfortranarray(SRC, dtype=cp.complex64)
    Ld = cp.asfortranarray(Ld, dtype=cp.complex64)
    Ll = cp.asfortranarray(Ll, dtype=cp.complex64)
    Lu = cp.asfortranarray(Lu, dtype=cp.complex64)
    Ud = cp.asfortranarray(Ud, dtype=cp.complex64)
    Ul = cp.asfortranarray(Ul, dtype=cp.complex64)
    Uu = cp.asfortranarray(Uu, dtype=cp.complex64)
    invT = cp.asfortranarray(invT, dtype=cp.complex64)  # (Ny,Ny,Nx)

    # Allocation result u (same dimension)
    u = cp.zeros_like(SRC)  # (Ny, Nx, num_sources), F-order

    # allocation tmp, shape=(Ny, num_sources)
    tmp = cp.zeros((Ny, num_sources), dtype=cp.complex64, order='F')

    # set kernel grid
    threadsPerBlock = (32, 32)
    grid_x = (num_sources + threadsPerBlock[0] - 1) // threadsPerBlock[0]
    grid_y = (Ny + threadsPerBlock[1] - 1) // threadsPerBlock[1]
    blocksPerGrid = (grid_x, grid_y)

    # Encapsulate a function to flatten when passing the kernel.
    def callKernel(kernel, argsShapes):
        """
        argsShapes: [(arr, shapeAfter)] => 先 flatten => kernel => 再 unflatten
        kernel(...) => raw param tuple
        """
        flatArgs = []
        for arr, shapeAfter in argsShapes:
            arrF2 = toFloat2F(arr)
            flatArgs.append(arrF2)
        return flatArgs

    if adjHelmholtzEqn:
        # ----------------------------------------------------
        # Adjoint Helmholtz
        #   Forward pass: x in [1..Nx-1]
        #       1) tmp = invT[..., x_idx-1]^H @ u[..., x_idx-1]
        #       2) applyUadj <<<kernel>>> => u[..., x_idx] = ...
        #
        #   Backward pass: x in [Nx-2..0]
        #       1) applyLadj <<<kernel>>> => tmp = ...
        #       2) applyInvTpostLadj => u[..., x_idx] = invT[..., x_idx]^H @ tmp
        # ----------------------------------------------------

        # ============ Forward Pass ============
        for x_idx in range(1, Nx):
            # 1) tmp = invT[..., x_idx-1]^H @ u[..., x_idx-1]
            applyInvTpreU(
                invT[:, :, x_idx - 1],  # invT_x (Ny,Ny)
                u[:, x_idx - 1, :],  # u_xm1  (Ny,num_sources)
                tmp  # tmp    (Ny,num_sources)
            )

            # 2) applyUadj <<<kernel>>> => u[..., x_idx] = ...
            #    need to first flatten -> float2 for SRC, Ud, Ul, Uu, u, tmp
            SRC_flat = toFloat2F(SRC)
            Ud_flat = toFloat2F(Ud)
            Ul_flat = toFloat2F(Ul)
            Uu_flat = toFloat2F(Uu)
            u_flat = toFloat2F(u)
            tmp_flat = toFloat2F(tmp)

            applyUadj_kernel(
                blocksPerGrid, threadsPerBlock,
                (
                    SRC_flat,  # const float2* SRC
                    Ud_flat,  # const float2* Ud
                    Ul_flat,  # const float2* Ul
                    Uu_flat,  # const float2* Uu
                    u_flat,  # float2* u
                    tmp_flat,  # const float2* tmp
                    x_idx,
                    Nx, Ny, num_sources
                )
            )
            # The kernel wrote u_flat, so it needs to be restored.
            fromFloat2F(u_flat, u.shape, u)

        # ============ Backward Pass ============
        for x_idx in range(Nx - 2, -1, -1):
            # 1) applyLadj <<<kernel>>> => tmp = ...
            Ld_flat = toFloat2F(Ld)
            Ll_flat = toFloat2F(Ll)
            Lu_flat = toFloat2F(Lu)
            u_flat = toFloat2F(u)
            tmp_flat = toFloat2F(tmp)

            applyLadj_kernel(
                blocksPerGrid, threadsPerBlock,
                (
                    Ld_flat, Ll_flat, Lu_flat,
                    u_flat,
                    tmp_flat,
                    x_idx,
                    Nx, Ny, num_sources
                )
            )
  
            fromFloat2F(tmp_flat, tmp.shape, tmp)

            # 2) applyInvTpostLadj => u[..., x_idx] = invT[..., x_idx]^H @ tmp
            applyInvTpostLadj(
                invT[:, :, x_idx],  # invT_x
                tmp,  # tmp
                u[:, x_idx, :]  # u_x
            )
    else:
        # ----------------------------------------------------
        # Forward Helmholtz
        #   Forward pass: x in [1..Nx-1]
        #       applyL => tmp = SRC - L * u[..., x_idx-1]
        #       applyInvTpostL => u[..., x_idx] = invT[..., x_idx] @ tmp
        #   Backward pass: x in [Nx-2..0]
        #       applyU => tmp = U * u[..., x_idx+1]
        #       applyInvTpostU => u[..., x_idx] -= invT[..., x_idx] @ tmp
        # ----------------------------------------------------
        # forward pass
        for x_idx in range(1, Nx):
            # (a) applyL<<<kernel>>> => tmp
            SRC_flat = toFloat2F(SRC)
            Ld_flat = toFloat2F(Ld)
            Ll_flat = toFloat2F(Ll)
            Lu_flat = toFloat2F(Lu)
            u_flat = toFloat2F(u)
            tmp_flat = toFloat2F(tmp)

            applyL_kernel(
                blocksPerGrid, threadsPerBlock,
                (
                    SRC_flat, Ld_flat, Ll_flat, Lu_flat,
                    u_flat, tmp_flat,
                    x_idx, Nx, Ny, num_sources
                )
            )
            fromFloat2F(tmp_flat, tmp.shape, tmp)

            # (b) applyInvTpostL => u[..., x_idx] = invT[..., x_idx] @ tmp
            applyInvTpostL(invT[:, :, x_idx], tmp, u[:, x_idx, :])

        # backward pass
        for x_idx in range(Nx - 2, -1, -1):
            # (a) applyU<<<kernel>>> => tmp = U * u[..., x_idx+1]
            Ud_flat = toFloat2F(Ud)
            Ul_flat = toFloat2F(Ul)
            Uu_flat = toFloat2F(Uu)
            u_flat = toFloat2F(u)
            tmp_flat = toFloat2F(tmp)

            applyU_kernel(
                blocksPerGrid, threadsPerBlock,
                (
                    Ud_flat, Ul_flat, Uu_flat,
                    u_flat, tmp_flat,
                    x_idx, Nx, Ny, num_sources
                )
            )
            fromFloat2F(tmp_flat, tmp.shape, tmp)

            # (b) applyInvTpostU => u[..., x_idx] -= invT[..., x_idx] @ tmp
            applyInvTpostU(invT[:, :, x_idx], tmp, u[:, x_idx, :])

    return u
