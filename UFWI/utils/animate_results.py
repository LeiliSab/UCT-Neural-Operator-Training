# animate_results.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

# Load the inversion results (MAT v7) with scipy.io.loadmat
res_path = '../../Results/kWave_BreastCT_WaveformInversionResults_cpu.mat'
res = loadmat(res_path)

# loadmat wraps each variable in an extra array dimension; .squeeze() removes it
xi    = res['xi'].squeeze()        # (Nx,)
yi    = res['yi'].squeeze()        # (Ny,)
VEL   = res['VEL_ESTIM_ITER']      # (Ny, Nx, Niter)
ATT   = res['ATTEN_ESTIM_ITER']    # (Ny, Nx, Niter)
GRAD  = res['GRAD_IMG_ITER']       # (Ny, Nx, Niter)
SRCH  = res['SEARCH_DIR_ITER']     # (Ny, Nx, Niter)
nIter = VEL.shape[2]


# Load the original dataset (MAT v7.3) with h5py
orig_path = '../../SampleData/kWave_BreastCT.mat'
with h5py.File(orig_path, 'r') as f0:
    def _load(key):
        arr = np.array(f0[key])
        # Transpose if multi-dimensional to match MATLAB layout
        return arr.T if arr.ndim > 1 else arr
    C      = _load('C')
    atten0 = _load('atten')
    xi0    = _load('xi_orig')
    yi0    = _load('yi_orig')


# Compute display ranges
# velocity global range
vmin_vel, vmax_vel = float(VEL.min()), float(VEL.max())

# convert attenuation to display units (dB/(MHz·mm))
Np2dB      = 20 / np.log(10)
slow2atten = 1e6 / 1e2
ATT_vis    = Np2dB * slow2atten * ATT
vmin_att, vmax_att = float(ATT_vis.min()), float(ATT_vis.max())


# Create the plotting canvas
plt.ion()
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

# top-left: Estimated Velocity
im_vel = ax[0,0].imshow(
    VEL[:,:,0],
    extent=[xi.min(), xi.max(), yi.max(), yi.min()],
    vmin=vmin_vel, vmax=vmax_vel,
    cmap='gray', aspect='auto'
)
ax[0,0].set_title('Estimated Velocity')

# top-middle: True Velocity
im_tvel = ax[0,1].imshow(
    C,
    extent=[xi0.min(), xi0.max(), yi0.max(), yi0.min()],
    vmin=vmin_vel, vmax=vmax_vel,
    cmap='gray', aspect='auto'
)
ax[0,1].set_title('True Velocity')

# top-right: Search Direction
im_s = ax[0,2].imshow(
    SRCH[:,:,0],
    extent=[xi.min(), xi.max(), yi.max(), yi.min()],
    cmap='gray', aspect='auto'
)
ax[0,2].set_title('Search Direction')

# bottom-left: Estimated Attenuation
im_att = ax[1,0].imshow(
    ATT_vis[:,:,0],
    extent=[xi.min(), xi.max(), yi.max(), yi.min()],
    vmin=vmin_att, vmax=vmax_att,
    cmap='gray', aspect='auto'
)
ax[1,0].set_title('Estimated Attenuation')

# bottom-middle: True Attenuation
im_tatt = ax[1,1].imshow(
    atten0,
    extent=[xi0.min(), xi0.max(), yi0.max(), yi0.min()],
    vmin=vmin_att, vmax=vmax_att,
    cmap='gray', aspect='auto'
)
ax[1,1].set_title('True Attenuation')

# bottom-right: Gradient
im_g = ax[1,2].imshow(
    -GRAD[:,:,0],  # invert sign to match MATLAB convention
    extent=[xi.min(), xi.max(), yi.max(), yi.min()],
    cmap='gray', aspect='auto'
)
ax[1,2].set_title('Gradient')

for a in ax.ravel():
    a.set_xlabel('x [m]')
    a.set_ylabel('y [m]')

plt.tight_layout()
plt.pause(0.1)


# Animated update loop
for k in range(nIter):
    # update estimated velocity
    im_vel.set_data(VEL[:,:,k])
    ax[0,0].set_title(f'Estimated Velocity    (Iter {k+1}/{nIter})')

    # update search direction and adjust color limits dynamically
    sd = SRCH[:,:,k]
    im_s.set_data(sd)
    im_s.set_clim(sd.min(), sd.max())
    ax[0,2].set_title(f'Search Direction      (Iter {k+1}/{nIter})')

    # update attenuation
    im_att.set_data(ATT_vis[:,:,k])
    ax[1,0].set_title(f'Estimated Attenuation (Iter {k+1}/{nIter})')

    # update gradient and adjust color limits dynamically
    g = -GRAD[:,:,k]
    im_g.set_data(g)
    im_g.set_clim(g.min(), g.max())
    ax[1,2].set_title(f'Gradient              (Iter {k+1}/{nIter})')

    fig.canvas.draw_idle()
    plt.pause(2)

plt.ioff()
plt.show()