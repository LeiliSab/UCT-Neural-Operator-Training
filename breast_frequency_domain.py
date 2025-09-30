import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import savemat

from UFWI.io import load_mat, save_results
from UFWI.geometry import TransducerArray2D, ImageGrid2D
from UFWI.data import AcquisitionData, ImageData
from UFWI.processors import (
    GaussianTimeWindow,
    DTFT,
    PhaseScreenCorrection,
    DownSample,
    AcceptanceMask,
    MagnitudeOutlierFilter,
    Pipeline
)
from UFWI.optimization.function.least_squares import NonlinearLS
from UFWI.optimization.algorithm.cg import CG
from UFWI.optimization.operator.helmholtz import HelmholtzOperator
from UFWI.optimization.gradient.adjoint_helmholtz import HelmholtzAdjointGrad
# from UFWI.utils.InversionVisualizer import InversionVisualizer  # Visualization wrapper
from UFWI.utils.visulizer_multi_mode import Visualizer

# ------------------------------------------------------------------------------
# (1) Load raw k-Wave data and construct AcquisitionData and ImageGeometry
# ------------------------------------------------------------------------------
raw_mat = Path("SampleData/kWave_BreastCT.mat")
raw = load_mat(raw_mat)

pos = raw["transducerPositionsXY"]  # (2, N)
N = pos.shape[1]
ones = np.ones(N, dtype=bool)
tx_array = TransducerArray2D(
    positions=pos.astype(np.float32),
    is_tx=ones, is_rx=ones
)

# Define imaging grid
# give grid spacing and half-width to ImageGeometry and it makes grid automatically
dxi = 0.6e-3
xmax = 120e-3
img_grid = ImageGrid2D(dx=dxi, xmax=xmax)
c0 = 1540.0  # Speed of sound in water

# Construct AcquisitionData
acq_data = AcquisitionData(
    array=raw["full_dataset"].transpose(2, 1, 0),  # (Tx,Rx,T)
    time=raw["time"],  # (T,)
    tx_array=tx_array,
    grid=img_grid,
    c0=c0
)

# ------------------------------------------------------------------------------
# (2) Define frequency list & preprocessing pipeline
# ------------------------------------------------------------------------------
f_sos = np.arange(0.3, 1.3, 0.05) * 1e6  # Frequencies for SoS-only stage
f_att = np.arange(0.325, 1.325, 0.05) * 1e6  # Frequencies for attenuation stage
freqs = np.concatenate([f_sos, f_att])  # All frequencies (Nfreq,)

pipe = Pipeline(
    stages=[
        GaussianTimeWindow(),
        DTFT(freqs),
        PhaseScreenCorrection(img_grid),
        DownSample(step=1),
        AcceptanceMask(delta=63),
        MagnitudeOutlierFilter(threshold=0.99),
    ],
    verbose=True
)

# Apply all preprocessing to the acquisition data
acq_data = pipe(acq_data)  # Resulting shape: (Tx, Rx, Nfreq)

# ------------------------------------------------------------------------------
# (3) Prepare iteration counts for SoS/Atten per frequency
# ------------------------------------------------------------------------------
Tx, Rx, Nfreq = acq_data.array.shape
n_sos = f_sos.size
n_att = f_att.size
assert n_sos + n_att == Nfreq

# Run 3 SoS iterations for all 40 frequencies,
# and 3 attenuation iterations for the latter 20 frequencies
niterSoSPerFreq = np.array([3] * n_sos + [3] * n_att)
niterAttenPerFreq = np.array([0] * n_sos + [3] * n_att)
total_iters = int(np.sum(niterSoSPerFreq) + np.sum(niterAttenPerFreq))
print(f"SoS iterations per frequency: {niterSoSPerFreq}, Atten iterations: {niterAttenPerFreq}")
print(f"Total number of iterations: {total_iters} (SoS + Atten)")

# ------------------------------------------------------------------------------
# (4) Initialize complex slowness model using ImageData
# ------------------------------------------------------------------------------
c_init = 1480.0
atten_init = 0.0

Nxi, Nyi = img_grid.nx, img_grid.ny
SLOW_INIT = (1.0 / c_init) + 1j * (atten_init / (2.0 * np.pi))
slow0 = np.full((Nyi, Nxi), SLOW_INIT, dtype=np.complex128)
slow_data = ImageData(array=slow0, grid=img_grid)

# ------------------------------------------------------------------------------
# (5) Create visualizer (InversionVisualizer)
# ------------------------------------------------------------------------------
# Load ground truth for comparison
C_true = raw["C"]  # (Nyi, Nxi)
atten_true = raw["atten"]  # (Nyi, Nxi)
# viz = InversionVisualizer(img_grid.xi, img_grid.yi, C_true, atten_true)

viz = Visualizer(
    xi=img_grid.xi, yi=img_grid.yi,
    C_true=C_true, atten_true=atten_true,
    mode="both",
    baseline=1500,
    sign_conv=-1,  # 与算子一致
    atten_unit='Np/(Hz·m)'
)

# ------------------------------------------------------------------------------
# (6) Loop over each frequency, use CG_Time.solve(...) in two stages
#     "Print time per iteration + automatic plotting"
# ------------------------------------------------------------------------------
cg = CG(c1=1e-4, shrink=0.5, max_ls=20)

for idx_f in range(Nfreq):
    print(f"\n=== Processing frequency idx_f = {idx_f}, f = {freqs[idx_f] / 1e6:.3f} MHz ===")
    n_sos = niterSoSPerFreq[idx_f]
    n_att = niterAttenPerFreq[idx_f]

    operator = HelmholtzOperator(acq_data, idx_f,
                                 sign_conv=-1, pml_alpha=10.0, pml_size=9.0e-3)
    grad = HelmholtzAdjointGrad(
        operator,
        deriv_fn=lambda m, op: 8 * np.pi ** 2 * op.get_field("freq") ** 2 * (
                op.get_field("PML") / op.get_field("V"))
    )
    fun = NonlinearLS(operator, grad_eval=grad)

    # —— SoS-only stage: update only real part → mode="real" ——
    if n_sos > 0:
        cg.solve(fun, slow_data,
                 n_iter=n_sos,
                 mode="real",
                 viz=viz,
                 do_print_time=True)

    # —— Atten-only stage: update only imaginary part → mode="imag" ——
    if n_att > 0:
        cg.solve(fun, slow_data,
                 n_iter=n_att,
                 mode="imag",
                 viz=viz,
                 do_print_time=True)

# ------------------------------------------------------------------------------
# (7) Take a snapshot of the Recorder and save it under the variable name.
# ------------------------------------------------------------------------------
rec = cg.get_record()
VEL_ESTIM_ITER = rec["vel"]
ATTEN_ESTIM_ITER = rec["atten"]
GRAD_IMG_ITER = rec["grad"]
SEARCH_DIR_ITER = rec["search"]

# ------------------------------------------------------------------------------
# (8) Save the final result + intermediate snapshots
# ------------------------------------------------------------------------------
Path("Results").mkdir(exist_ok=True)
savemat("Results/kWave_BreastCT_WaveformInversionResults.mat", {
    "xi": img_grid.xi,
    "yi": img_grid.yi,
    "fDATA": freqs.reshape(1, -1),
    "niterAttenPerFreq": niterAttenPerFreq.reshape(1, -1),
    "niterSoSPerFreq": niterSoSPerFreq.reshape(1, -1),
    "VEL_ESTIM_ITER": VEL_ESTIM_ITER,
    "ATTEN_ESTIM_ITER": ATTEN_ESTIM_ITER,
    "GRAD_IMG_ITER": GRAD_IMG_ITER,
    "SEARCH_DIR_ITER": SEARCH_DIR_ITER,
}, do_compression=True)

print("Results saved to Results/kWave_BreastCT_WaveformInversionResults.mat")
