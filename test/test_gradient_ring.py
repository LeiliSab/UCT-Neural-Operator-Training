import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from UFWI.geometry import ImageGrid2D, TransducerArray2D
from UFWI.data import AcquisitionData, ImageData
from UFWI.optimization.operator.wave_operator import WaveOperator
from UFWI.signals import GaussianModulatedPulse

from UFWI.optimization.gradient.time_grad import AdjointStateGrad
from UFWI.optimization.function.least_squares import NonlinearLS

import matplotlib
matplotlib.use("TkAgg")  # interactive plotting

# ------------------------------------------------------------
# 1) Smaller grid (keep physical positions/size the same)
# ------------------------------------------------------------
Nx = Ny = 160
dx = dy = 1.5e-3
c0 = 1500.0
record_time = 1.2 * Nx * dx / c0
n_tx = 64
f0 = 0.1e6

img_grid = ImageGrid2D(dx=dx, nx=Nx, ny=Ny)

print(img_grid.max_f(c_min=1300, ppw=4))

# Background and true models (centers/radii specified in meters)
model_bg = np.full((Ny, Nx), c0, np.float64)
X, Y = img_grid.meshgrid(indexing="xy")
model_true = model_bg.copy()

model_true[((X - 0.03) ** 2 + (Y - 0.03) ** 2) < 0.012 ** 2] = c0 + 200 # faster blob
model_true[((X + 0.02) ** 2 + (Y + 0.016) ** 2) < 0.010 ** 2] = c0 - 200  # slower blob

# ------------------------------------------------------------
# 2) Build ring and three acquisition setups
#    A) top–bottom single pair (1 TX, 1 RX)
#    B) single TX (top) → all RX
#    C) full ring: all TX & all RX
# ------------------------------------------------------------
ring = TransducerArray2D.from_ring_array_2D(
    grid=img_grid,
    r=(min(Nx, Ny) // 2 - 10) * dx,   # inside the grid
    n=n_tx
)

img = ImageData(array=model_true, grid=img_grid, tx_array=ring)  # ImageData for visualization
img.show()

pos = ring.positions
top_idx = int(np.argmax(pos[1]))      # "top" element index
bot_idx = int(np.argmin(pos[1]))      # "bottom" element index

# A) top–bottom single pair
pair_positions = np.column_stack([pos[:, top_idx], pos[:, bot_idx]])
is_tx_pair = np.array([True,  False], dtype=bool)
is_rx_pair = np.array([False, True ], dtype=bool)
array_pair = TransducerArray2D(positions=pair_positions, is_tx=is_tx_pair, is_rx=is_rx_pair)

# B) one TX (top), all RX
is_tx_1 = np.zeros(ring.n_elements, dtype=bool); is_tx_1[top_idx] = True
is_rx_all = np.ones(ring.n_elements, dtype=bool)
array_1tx_allrx = TransducerArray2D(positions=ring.positions, is_tx=is_tx_1, is_rx=is_rx_all)

# C) full ring (all TX & all RX)
array_full = ring  # already TX=RX=True from constructor

acq_pair = AcquisitionData.from_geometry(grid=img_grid, tx_array=array_pair)
acq_1tx  = AcquisitionData.from_geometry(grid=img_grid, tx_array=array_1tx_allrx)
acq_full = AcquisitionData.from_geometry(grid=img_grid, tx_array=array_full)

# ------------------------------------------------------------
# 3) Operator options (sequential; no source encoding)
# ------------------------------------------------------------
pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)
op_opts = dict(
    record_time=record_time,
    pulse=pulse,
    use_encoding=False,
    drop_self_rx=True,
    record_full_wf=True,
    cfl=0.2, c_ref=c0,
    pml_size=10, pml_alpha=10.0,
    scale_source_terms=True,
    src_mode="additive",
    pml_inside=False,
    use_gpu=False,
    verbose=False,
)

# ------------------------------------------------------------
# 4) Build three operators
# ------------------------------------------------------------
op_pair = WaveOperator(
    data=acq_pair,
    medium_params={"sound_speed": model_true.astype(np.float32)},
    **op_opts
)
op_1tx = WaveOperator(
    data=acq_1tx,
    medium_params={"sound_speed": model_true.astype(np.float32)},
    **op_opts
)
op_full = WaveOperator(
    data=acq_full,
    medium_params={"sound_speed": model_true.astype(np.float32)},
    **op_opts
)

# ------------------------------------------------------------
# 5) Utility: gradient at background for a given operator
#    (forward@true → set_obs → gradient@background)
# ------------------------------------------------------------
def compute_grad_at_background(op, model_true, model_bg):
    d_true = op.forward(model_true.astype(np.float32), kind="c")
    op.set_obs(d_true)
    ge = AdjointStateGrad(op, use_first_deriv_product=True)
    fun = NonlinearLS(op, grad_eval=ge, weight=1.0, sync_value=True)
    g = fun.gradient(model_bg, kind="c")   # (ny, nx), float64
    return g

# Three gradients
G_pair       = compute_grad_at_background(op_pair, model_true, model_bg)      # 1 TX, 1 RX
G_1tx_allrx  = compute_grad_at_background(op_1tx,  model_true, model_bg)      # 1 TX, all RX
G_full       = compute_grad_at_background(op_full, model_true, model_bg)      # all TX & RX



# ------------------------------------------------------------
# 6) Plot: one row, three columns (shared colorbar placed outside)
# ------------------------------------------------------------
extent = img_grid.extent
vals = np.concatenate([
    np.abs(G_pair).ravel(),
    np.abs(G_1tx_allrx).ravel(),
    np.abs(G_full).ravel()
])
A = np.percentile(vals, 99.5)
normG = colors.TwoSlopeNorm(vmin=-A, vcenter=0.0, vmax=A)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')  # no overlap with constrained layout

im0 = axes[0].imshow(G_pair, origin="lower", extent=extent, cmap="seismic", norm=normG)
axes[0].set_title("Gradient: top–bottom single pair")
axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]")

im1 = axes[1].imshow(G_1tx_allrx, origin="lower", extent=extent, cmap="seismic", norm=normG)
axes[1].set_title("Gradient: single TX → all RX")
axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]")

im2 = axes[2].imshow(G_full, origin="lower", extent=extent, cmap="seismic", norm=normG)
axes[2].set_title("Gradient: all TX & all RX (sum)")
axes[2].set_xlabel("x [m]"); axes[2].set_ylabel("y [m]")

# One shared colorbar to the right of all axes, with padding
cbar = fig.colorbar(im2, ax=axes, location='right', fraction=0.046, pad=0.02)
cbar.set_label("gradient amplitude (arb.)")

# Save first, then show
plt.savefig("gradients_triptych.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved figure -> gradients_triptych.png")


# only plot G_pair
fig, ax = plt.subplots(figsize=(6, 4.8), layout='constrained')
im = ax.imshow(G_pair, origin="lower", extent=extent, cmap="seismic")
ax.set_title("Gradient: top–bottom single pair")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
# Shared colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label("gradient amplitude (arb.)")
# Save and show
plt.savefig("gradient_single_pair.png", dpi=300, bbox_inches="tight")
plt.show()

