import numpy as np
import matplotlib.pyplot as plt

from UFWI.geometry import ImageGrid2D, TransducerArray2D
from UFWI.data import AcquisitionData
from UFWI.optimization.operator.wave_operator import WaveOperator
from UFWI.signals import GaussianModulatedPulse

import matplotlib
matplotlib.use("TkAgg")  # interactive plotting

# ------------------------------------------------------------
# 1) Grid & models
# ------------------------------------------------------------
Nx = Ny = 240
dx = dy = 1e-3
c0 = 1500.0
record_time = 1.2 * Nx * dx / c0
n_tx = 64
f0 = 0.3e6

# Imaging grid (centered at 0)
img_grid = ImageGrid2D(dx=dx, nx=Nx, ny=Ny)

print(img_grid.max_f(c_min=1300, ppw=4))


# Background model
model_bg = np.full((Ny, Nx), c0, np.float64)

# True model = background + two circular anomalies
X, Y = img_grid.meshgrid(indexing="xy")
model_true = model_bg.copy()
model_true[((X - 0.03) ** 2 + (Y - 0.03) ** 2) < 0.012 ** 2] = c0 + 200  # faster blob
model_true[((X + 0.02) ** 2 + (Y + 0.016) ** 2) < 0.010 ** 2] = c0 - 200  # slower blob

# ------------------------------------------------------------
# 2) 64-element ring → pick top TX & bottom RX (1Tx/1Rx)
# ------------------------------------------------------------
ring = TransducerArray2D.from_ring_array_2D(
    grid=img_grid,
    r=(min(Nx, Ny) // 2 - 10) * dx,
    n=n_tx
)
pos = ring.positions
top_idx = int(np.argmax(pos[1]))      # highest y
bot_idx = int(np.argmin(pos[1]))      # lowest y

tx_pos = pos[:, top_idx].reshape(2, 1)
rx_pos = pos[:, bot_idx].reshape(2, 1)

pair_positions = np.hstack([tx_pos, rx_pos])  # (2, 2)
is_tx = np.array([True,  False], dtype=bool)
is_rx = np.array([False, True ], dtype=bool)
pair_array = TransducerArray2D(positions=pair_positions, is_tx=is_tx, is_rx=is_rx)

# Acquisition container for the 1Tx/1Rx pair
acq_pair = AcquisitionData.from_geometry(grid=img_grid, tx_array=pair_array)

# ------------------------------------------------------------
# 3) Forward operator (configured with the true model as medium params)
# ------------------------------------------------------------
medium_params = {"sound_speed": model_true.astype(np.float32)}
pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)

op = WaveOperator(
    data=acq_pair,
    medium_params=medium_params,
    record_time=record_time,
    pulse=pulse,
    use_encoding=False,
    drop_self_rx=False,
    record_full_wf=True,
    cfl=0.2,
    c_ref=c0,
    pml_size=10,
    pml_alpha=10.0,
    scale_source_terms=True,
    src_mode="additive",
    pml_inside=False,
    use_gpu=False,
    verbose=False
)

# ------------------------------------------------------------
# 4) Forward with TRUE model → observation d_true
# ------------------------------------------------------------
Fm_true = op.forward(model_true.astype(np.float32), kind="c")  # (1,1,nt)
t = op.time_axis                                               # (nt,)

# ------------------------------------------------------------
# 5) Same time picks (6 instants, 10%..90% of the record)
# ------------------------------------------------------------
n = 6
t_sel = np.linspace(0.1, 0.9, n) * t[-1]
idx_sel = [int(np.argmin(np.abs(t - ts))) for ts in t_sel]

# ------------------------------------------------------------
# 6) Forward with BACKGROUND, residual, adjoint
#    residual = F(c_bg) - d_true
# ------------------------------------------------------------
Fm_bg = op.forward(model_bg.astype(np.float32), kind="c")  # overwrites cached forward fields
residual = Fm_bg - Fm_true                                 # (1,1,nt) in element order
lam = op.adjoint(residual)                                 # (nt, ny, nx), float64

# Cached forward wavefield corresponds to the LAST forward call (background)
u = op.get_forward_fields()[0].astype(np.float64)          # (nt, ny, nx)

# ------------------------------------------------------------
# 7) Time derivatives and kernel frames
#    Use first-derivative product:  (2/c^3) * u_t * lam_t
# ------------------------------------------------------------
def d_dt(field, dt):
    pad = np.pad(field, ((1,1),(0,0),(0,0)), mode="edge")
    return (pad[2:] - pad[:-2]) / (2.0 * dt)

u_t_full   = d_dt(u, op.dt)        # (nt, ny, nx)
lam_t_full = d_dt(lam, op.dt)      # (nt, ny, nx)

# Kernel factor K = 2 / c^3 at the model where gradient is evaluated (background)
c_bg = model_bg.astype(np.float64)
K = 2.0 / (c_bg * c_bg * c_bg)     # (ny, nx)

# Per-time kernel frames for the selected instants
kernel_frames = [K[None, ...] * (u_t_full[idx][None, ...]) * (lam_t_full[idx][None, ...]) for idx in idx_sel]
kernel_frames = np.squeeze(np.stack(kernel_frames, axis=0))   # (n, ny, nx)

# ------------------------------------------------------------
# 8) Figure 1: three rows — u_t, lam_t, kernel frames (shared robust norms per row)
# ------------------------------------------------------------
from matplotlib import colors
extent = img_grid.extent

# Robust, zero-centered norms per row
A_ut   = np.percentile(np.abs(u_t_full[idx_sel]),   99.5)
A_lamt = np.percentile(np.abs(lam_t_full[idx_sel]), 99.5)
A_ker  = np.percentile(np.abs(kernel_frames),       99.5)

norm_ut   = colors.TwoSlopeNorm(vmin=-A_ut,   vcenter=0.0, vmax=A_ut)
norm_lamt = colors.TwoSlopeNorm(vmin=-A_lamt, vcenter=0.0, vmax=A_lamt)
norm_ker  = colors.TwoSlopeNorm(vmin=-A_ker,  vcenter=0.0, vmax=A_ker)

fig = plt.figure(figsize=(2.3*n + 2.6, 9.5))
gs = fig.add_gridspec(3, n, hspace=0.32, wspace=0.25,
                      height_ratios=[1, 1, 1])

# Row 1: u_t
for col, idx in enumerate(idx_sel):
    ax = fig.add_subplot(gs[0, col])
    im1 = ax.imshow(u_t_full[idx], origin="lower", extent=extent, cmap="seismic", norm=norm_ut)
    ax.scatter(pair_positions[0,0], pair_positions[1,0], marker='^', s=40, edgecolors='k', label='TX')
    ax.scatter(pair_positions[0,1], pair_positions[1,1], marker='s', s=40, edgecolors='k', label='RX')
    if col == 0: ax.legend(loc='upper right', fontsize=8, frameon=True)
    ax.set_title(f"t = {t[idx]*1e3:.2f} ms")
    ax.set_ylabel("u_t")
    ax.set_xlabel("x [m]")

# Row 2: lam_t
for col, idx in enumerate(idx_sel):
    ax = fig.add_subplot(gs[1, col])
    im2 = ax.imshow(lam_t_full[idx], origin="lower", extent=extent, cmap="seismic", norm=norm_lamt)
    ax.scatter(pair_positions[0,0], pair_positions[1,0], marker='^', s=40, edgecolors='k')
    ax.scatter(pair_positions[0,1], pair_positions[1,1], marker='s', s=40, edgecolors='k')
    ax.set_ylabel("lam_t")
    ax.set_xlabel("x [m]")

# Row 3: kernel frames 2/c^3 * u_t * lam_t
for col, idx in enumerate(idx_sel):
    ax = fig.add_subplot(gs[2, col])
    im3 = ax.imshow(kernel_frames[col], origin="lower", extent=extent, cmap="seismic", norm=norm_ker)
    ax.scatter(pair_positions[0,0], pair_positions[1,0], marker='^', s=40, edgecolors='k')
    ax.scatter(pair_positions[0,1], pair_positions[1,1], marker='s', s=40, edgecolors='k')
    ax.set_ylabel("2/c^3 * p_t * lam_t")
    ax.set_xlabel("x [m]")

# Colorbars (one per row)
cax1 = fig.add_axes([0.92, 0.71, 0.015, 0.20]); plt.colorbar(im1, cax=cax1, label="∂p/∂t")
cax2 = fig.add_axes([0.92, 0.41, 0.015, 0.20]); plt.colorbar(im2, cax=cax2, label="∂λ/∂t")
cax3 = fig.add_axes([0.92, 0.11, 0.015, 0.20]); plt.colorbar(im3, cax=cax3, label="(2/c³)·∂p/∂t·∂λ/∂t")

plt.tight_layout(rect=[0.03, 0.03, 0.9, 0.97])
plt.show()

# ------------------------------------------------------------
# 9) Figure 2: time-integrated kernel  G(x,y) = ∫ (2/c^3) u_t λ_t dt
# ------------------------------------------------------------
# Use trimmed arrays for integration (avoid padded edges)
u_t_int   = u_t_full[1:-1]
lam_t_int = lam_t_full[1:-1]

G = np.trapz((K[None, ...] * u_t_int * lam_t_int), dx=op.dt, axis=0)  # (ny, nx)

# A_G = np.percentile(np.abs(G), 1)
A_G = 1e-13
norm_G = colors.TwoSlopeNorm(vmin=-A_G, vcenter=0.0, vmax=A_G)

plt.figure(figsize=(6.2, 5.4))
imG = plt.imshow(G, origin="lower", extent=extent, cmap="seismic", norm=norm_G)
plt.scatter(pair_positions[0,0], pair_positions[1,0], marker='^', s=50, edgecolors='k', label='TX')
plt.scatter(pair_positions[0,1], pair_positions[1,1], marker='s', s=45, edgecolors='k', label='RX')
plt.title("Time-integrated kernel  G(x,y) = ∫ (2/c³)·∂p/∂t·∂λ/∂t dt")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.legend(loc='upper right', fontsize=9, frameon=True)
plt.colorbar(imG, fraction=0.046, label="G amplitude")
plt.tight_layout()
plt.show()
