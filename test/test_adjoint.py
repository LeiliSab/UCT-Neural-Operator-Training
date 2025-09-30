import numpy as np
import matplotlib.pyplot as plt

from UFWI.geometry import ImageGrid2D, TransducerArray2D
from UFWI.data import AcquisitionData
from UFWI.optimization.operator.wave_operator import WaveOperator
from UFWI.signals import GaussianModulatedPulse

import matplotlib
# Use TkAgg backend for interactive plotting
matplotlib.use("TkAgg")

# ------------------------------------------------------------
# 1) Grid & models
# ------------------------------------------------------------
Nx = Ny = 240
dx = dy = 1e-3
c0 = 1500.0
record_time = 1.2 * Nx * dx / c0
n_tx = 64
f0 = 0.3e6

# Imaging grid (centered about 0)
img_grid = ImageGrid2D(dx=dx, nx=Nx, ny=Ny)

# Background model
model_bg = np.full((Ny, Nx), c0, np.float64)

# True model = background + two circular speed anomalies
X, Y = img_grid.meshgrid(indexing="xy")
model_true = model_bg.copy()
model_true[((X - 0.01) ** 2 + (Y - 0.01) ** 2) < 0.006 ** 2] = c0 + 100
model_true[(X ** 2 + (Y + 0.01) ** 2) < 0.005 ** 2] = c0 - 100  # slower blob

# ------------------------------------------------------------
# 2) Build a 64-element ring; pick a top TX and bottom RX
# ------------------------------------------------------------
ring = TransducerArray2D.from_ring_array_2D(
    grid=img_grid,
    r=(min(Nx, Ny) // 2 - 2) * dx,   # radius inside the grid
    n=n_tx
)

# Choose the "top" element (max y) as TX and "bottom" (min y) as RX
pos = ring.positions  # shape (2, N)
top_idx = int(np.argmax(pos[1]))     # highest y
bot_idx = int(np.argmin(pos[1]))     # lowest y

tx_pos = pos[:, top_idx].reshape(2, 1)
rx_pos = pos[:, bot_idx].reshape(2, 1)

# Build a 2-element array: one TX at top, one RX at bottom
pair_positions = np.hstack([tx_pos, rx_pos])  # (2, 2)
is_tx = np.array([True,  False], dtype=bool)
is_rx = np.array([False, True ], dtype=bool)
pair_array = TransducerArray2D(positions=pair_positions, is_tx=is_tx, is_rx=is_rx)

# Acquisition container using the 1-Tx / 1-Rx pair
acq_pair = AcquisitionData.from_geometry(grid=img_grid, tx_array=pair_array)

# ------------------------------------------------------------
# 3) Forward operator (configured with the true model for medium params)
# ------------------------------------------------------------
medium_params = {
    "sound_speed": model_true.astype(np.float32),  # k-Wave expects float32
    # (density/alpha can be omitted -> defaults)
}
pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)

op = WaveOperator(
    data=acq_pair,
    medium_params=medium_params,
    record_time=record_time,
    pulse=pulse,
    use_encoding=False,
    drop_self_rx=False,
    record_full_wf=True,    # record full wavefield if you want to inspect forward too
    cfl=0.2,
    c_ref=c0,
    pml_size=10,
    pml_alpha=10.0,
    scale_source_terms=True,
    src_mode="additive",
    pml_inside=False,
    use_gpu=False,          # set True if your k-Wave GPU backend is available
    verbose=False
)

# ------------------------------------------------------------
# 4) Forward with TRUE model → observation d_true
# ------------------------------------------------------------
Fm_true = op.forward(model_true.astype(np.float32), kind="c")  # (Tx=1, n_rx=1, nt)
t = op.time_axis                                               # (nt,)

# ------------------------------------------------------------
# 5) Choose the same time picks as before (uniform in 10%..90% of the record)
# ------------------------------------------------------------
n = 6
t_sel = np.linspace(0.1, 0.9, n) * t[-1]
idx_sel = [int(np.argmin(np.abs(t - ts))) for ts in t_sel]

# ------------------------------------------------------------
# 6) Forward with BACKGROUND model, form residual, run adjoint
#     residual = F(c_bg) - d_true
# ------------------------------------------------------------
Fm_bg = op.forward(model_bg.astype(np.float32), kind="c")  # (1,1,nt) element order
residual = Fm_bg - Fm_true                                 # (1,1,nt)
lam = op.adjoint(residual)                                 # (nt, ny, nx) float64

# ------------------------------------------------------------
# 7) Plot ONE ROW of adjoint snapshots at the same selected times
#     Use a shared, robust color scale with zero-centered norm
# ------------------------------------------------------------
from matplotlib import colors

extent = img_grid.extent  # (xmin, xmax, ymin, ymax)

# Robust, shared scale over selected frames (keeps zero as white)
A = np.percentile(np.abs(lam[idx_sel]), 99.5)
norm = colors.TwoSlopeNorm(vmin=-A, vcenter=0.0, vmax=A)

fig = plt.figure(figsize=(2.2*n + 2.5, 3.2))
gs = fig.add_gridspec(1, n, wspace=0.25)

for k, idx in enumerate(idx_sel):
    ax = fig.add_subplot(gs[0, k])
    im = ax.imshow(lam[idx], origin="lower", extent=extent, cmap="seismic", norm=norm)
    # overlay TX / RX for reference
    ax.scatter(pair_positions[0, 0], pair_positions[1, 0], marker="^", s=40, edgecolors="k", label="TX")
    ax.scatter(pair_positions[0, 1], pair_positions[1, 1], marker="s", s=40, edgecolors="k", label="RX")
    if k == 0:
        ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.set_title(f"Adjoint  t = {t[idx]*1e3:.2f} ms")
    ax.set_xlabel("x [m]")
    if k == 0:
        ax.set_ylabel("y [m]")

# Shared colorbar
cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
plt.colorbar(im, cax=cax, label="adjoint field λ")
plt.tight_layout(rect=[0.03, 0.03, 0.9, 0.97])
plt.show()
