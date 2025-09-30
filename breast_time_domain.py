import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from UFWI.geometry import ImageGrid2D, TransducerArray2D
from UFWI.data import AcquisitionData
from UFWI.data.image_data import ImageData
from UFWI.optimization.operator import WaveOperator
from UFWI.optimization.gradient import AdjointStateGrad
from UFWI.optimization.function import NonlinearLS
from UFWI.optimization.algorithm import GD, CG_Time
from UFWI.utils.visulizer_multi_mode import Visualizer
from UFWI.signals import GaussianModulatedPulse
from scipy.io import loadmat
import matplotlib
matplotlib.use("TkAgg")


# Settings
USE_ENCODING = True  # Whether to use source encoding
K = 80  # Number of random encoding loops
TAU_MAX = 0.0  # Maximum random delay (seconds)
DROP_SELF_RX = True  # Drop self-Rx when not encoding
NORMALIZE = True  # Whether to normalize gradient

N_ITER = 20  # Number of iterations
SAVE_DIR = Path("output")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# Lower frequency & coarse grid
f0 = 0.3e6  # Center frequency 0.3 MHz
Nx = Ny = 240  # Number of grid points
dx = dy = 1.0e-3  # Grid spacing 1 mm
c0 = 1500.0

# (1) Load and downsample true sound-speed model
# For get c_ref and record time
mat = loadmat("UFWI/Phantom/C_true.mat")
model_raw = mat['C_true']  # Original (1601,1601)
print("Original model shape:", model_raw.shape)

# (2) make ImageGrid2D and compute record time
img_grid = ImageGrid2D(nx=Nx, ny=Ny, dx=dx)

# Downsample using ImageGrid2D
img_raw = ImageData(model_raw)
img_true = img_raw.downsample_to(new_grid=img_grid)
model_true = img_true.array
print("Downsampled model shape:", model_true.shape)
c_ref = model_true.max() # Reference sound speed for time step
extent = img_grid.extent
c_min = float(model_true.min())
record_time = 1.3 * (extent[1] - extent[0]) / c_min
print(f"record_time = {record_time * 1e3:.2f} ms")

# Construct 512-element ring array & mask
n_tx = 512
radius = 110e-3  # 110 mm
tx_array = TransducerArray2D.from_ring_array_2D(r=radius, grid=img_grid, n=n_tx)

# Visualize true sound speed + sensors
# true_image_data = ImageData(array=model_true, tx_array=tx_array, grid=img_grid)
# true_image_data.show()


# Synthesize observations & save
fname = SAVE_DIR / f"d_obs_240x240_1mm_0p3MHz_new_512.npz"

dat = np.load(fname, allow_pickle=True)
d_obs = dat["array"]
t_vec = dat["time"]
print("Load data, shape =", d_obs.shape, "time vector shape =", t_vec.shape)

# construct AcquisitionData
acq_inv_data = AcquisitionData(array=d_obs,
                               tx_array=tx_array,
                               grid=img_grid,
                               time=t_vec)

# construct medium parameters
medium_params = {
    "density": np.ones((Ny, Nx), np.float32) * 1000.0,
    "alpha_coeff": np.zeros((Ny, Nx), np.float32),
    "alpha_power": 1.01,
    "alpha_mode": "no_dispersion",
}

# Construct WaveOperator
pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)

op_inv = WaveOperator(
    data=acq_inv_data,
    medium_params=medium_params,
    record_time=record_time,
    record_full_wf=True,
    use_encoding=USE_ENCODING,
    drop_self_rx=DROP_SELF_RX,
    pulse=pulse,
    c_ref=c_ref,
)

# -------- gradient evaluator & LS function ---------------------- #
grad_eval = AdjointStateGrad(
    op_inv,
    K=(K if (USE_ENCODING and K is not None and K > 1) else None),
    seed=0,
)

f_ls = NonlinearLS(
    op_inv,
    grad_eval=grad_eval,
    normalize=NORMALIZE,
)

model_bg = np.full((Ny, Nx), c0, np.float64)
model_init = model_bg.astype(np.float64)
img_init = ImageData(model_init) # Initial guess for optimization

eta0 = 6.0e-1

# construct Visualizer
x = (np.arange(Nx) - Nx / 2) * dx
y = (np.arange(Ny) - Ny / 2) * dy
atten_zero = np.zeros_like(model_true)
viz = Visualizer(x, y, model_true, atten_zero, mode="vel")

# =================================================================
#                    ►►   optimisation   ◄◄
# =================================================================
USE = "GD"  # Options are 'GD' or 'CG_Time'

if USE == "CG_Time":
    print("\n>>> Using CG_Time algorithm (PR+BB1/BB2+Armijo)")
    op = CG_Time(viz=viz)
else:
    print("\n>>> Using GD algorithm (constant step size)")

    kappa = 50
    op = GD(
        lr=kappa * eta0,
        backtrack=False,
        max_bt=12,
        schedule_fn=lambda k, lr0: lr0,  # Still using constant step size
        viz=viz
    )

op.solve(fun=f_ls, m0=img_init, kind='c', n_iter=N_ITER)
op.save_record(SAVE_DIR / f"record_{USE}_breast_512.npz")


# =================================================================
#                    ►►   Final visualization   ◄◄
# =================================================================
rec = op.get_record()
vel_all = rec["vel"]  # (Ny, Nx, N_ITER)
grad_all = rec["grad"].real  # (Ny, Nx, N_ITER)

vel_all = np.array(vel_all, np.float64)

# —— Compute min/max sound speed of true model for color scale range in both maps —— #
vmin_c = model_true.min()
vmax_c = model_true.max()

# Gradient color scale (99th percentile)
abs99 = np.percentile(np.abs(grad_all), 99)
vmin_grad, vmax_grad = -abs99, abs99

extent = (-Nx / 2 * dx, Nx / 2 * dx, -Ny / 2 * dy, Ny / 2 * dy)
tx_x, tx_y = op_inv.tx_pos

# Plotting
fig, ax = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

# 1) True sound speed map
im0 = ax[0].imshow(
    model_true,
    extent=extent, origin='lower',
    cmap='seismic', vmin=vmin_c, vmax=vmax_c
)
ax[0].set_title("True sound speed c")
ax[0].set_xlabel("x [m]")
ax[0].set_ylabel("y [m]")
ax[0].scatter(tx_x, tx_y, marker='*', s=30, color='lime')
fig.colorbar(im0, ax=ax[0], fraction=0.046)

# 2) Final reconstruction sound speed map (same color scale as true model)
im1 = ax[1].imshow(
    vel_all[..., -1],
    extent=extent, origin='lower',
    cmap='seismic', vmin=vmin_c, vmax=vmax_c
)
ax[1].set_title(f"Final reconstruction @ iter {N_ITER}")
ax[1].set_xlabel("x [m]")
ax[1].set_ylabel("y [m]")
ax[1].scatter(tx_x, tx_y, marker='*', s=30, color='lime')
fig.colorbar(im1, ax=ax[1], fraction=0.046)

# 3) Final gradient map (using original gradient color scale settings)
im2 = ax[2].imshow(
    grad_all[..., -1],
    extent=extent, origin='lower',
    cmap='seismic', vmin=vmin_grad, vmax=vmax_grad
)
ax[2].set_title(f"Final gradient @ iter {N_ITER}")
ax[2].set_xlabel("x [m]")
ax[2].set_ylabel("y [m]")
ax[2].scatter(tx_x, tx_y, marker='*', s=30, color='lime')
fig.colorbar(im2, ax=ax[2], fraction=0.046)

out_fig = SAVE_DIR / "final_delta_c_grad_1.png"
plt.savefig(out_fig, dpi=220)
print(f"\n[info] Final figure saved → {out_fig}")

rec = op.get_record()
vel_all = rec["vel"]  # (Ny, Nx, N_ITER)
grad_all = rec["grad"].real  # (Ny, Nx, N_ITER)

misfit_all = rec["misfit"][1, :]

# -------- Compute differences & color scales --------
# Sound speed difference
vel_diff_all = vel_all.real - c0
# Symmetric color scale range
v_abs = 200
vmin_rec, vmax_rec = -v_abs, v_abs
# Gradient color scale (99th percentile)
abs99 = np.percentile(np.abs(grad_all), 99)
vmin_grad, vmax_grad = -abs99, abs99

# -------- Visualization --------
n_rows = 1 + N_ITER
n_cols = 2
extent = (-Nx / 2 * dx, Nx / 2 * dx, -Ny / 2 * dy, Ny / 2 * dy)
tx_x, tx_y = op_inv.tx_pos

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(10, 3.2 * n_rows),
                         constrained_layout=True)
axes = np.atleast_2d(axes)

# Row 0 Left: True Δc
ax = axes[0, 0]
im = ax.imshow(model_true - c0,
               extent=extent, origin='lower',
               cmap='seismic', vmin=vmin_rec, vmax=vmax_rec)
ax.set_title("True Δc = c_true - c₀ [m/s]")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.scatter(tx_x, tx_y, marker='*', s=30, color='lime')
fig.colorbar(im, ax=ax, fraction=0.046)

# Row 0 Right: Initial gradient
ax = axes[0, 1]
im = ax.imshow(grad_all[..., 0],
               extent=extent, origin='lower',
               cmap='seismic', vmin=vmin_grad, vmax=vmax_grad)
ax.set_title("Initial gradient ∂Φ/∂c")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.scatter(tx_x, tx_y, marker='*', s=30, color='lime')
fig.colorbar(im, ax=ax, fraction=0.046)

# Rows 1..N_ITER: Reconstruction Δc and gradient
for i in range(1, n_rows):
    # Δc
    ax = axes[i, 0]
    im = ax.imshow(vel_diff_all[..., i - 1],
                   extent=extent, origin='lower',
                   cmap='seismic', vmin=vmin_rec, vmax=vmax_rec)

    ax.set_title(
        f"Reconstruction Δc @ iter {i} \n misfit = {misfit_all[i - 1]:.3e} "
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.scatter(tx_x, tx_y, marker='*', s=30, color='lime')
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Gradient
    ax = axes[i, 1]
    im = ax.imshow(grad_all[..., i - 1],
                   extent=extent, origin='lower',
                   cmap='seismic', vmin=vmin_grad, vmax=vmax_grad)
    ax.set_title(f"Gradient @ iter {i}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.scatter(tx_x, tx_y, marker='*', s=30, color='lime')
    fig.colorbar(im, ax=ax, fraction=0.046)

out_fig = SAVE_DIR / "timeline_delta_c_grad.png"
plt.savefig(out_fig, dpi=220)
print(f"\n[info] Figure saved → {out_fig}")

plt.ioff()
plt.show()


