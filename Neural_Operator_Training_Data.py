##### THIS WORKS FOR N_ITERS = 180 (IDX_F), AND 368 TRANSDUCERS #############
#### UP TO LINE 210 WE SIMULATE AND SAVE TIME-DOMAIN DATA #########
#### FROM LINE 211 WE LOAD THE DATA AND PROCESS IN FREQUENCY DOMAIN,CREATING SCATTERED AND INCIDENT FIELDS #####
#### FROM LINE 512 WE SAVE TRAINING DATA FOR NEURAL OPERATOR #####
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
print(f"=== SCRIPT STARTED AT {datetime.now().strftime('%H:%M:%S')} - VERSION CHECK ===")
from UFWI.geometry import ImageGrid2D, TransducerArray2D
from UFWI.data import AcquisitionData
from UFWI.data.image_data import ImageData
from UFWI.optimization.operator import WaveOperator
from UFWI.signals import GaussianModulatedPulse
from scipy.io import loadmat
import matplotlib
matplotlib.use("TkAgg")

# Settings
SAVE_DIR = Path("output")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# Lower frequency & coarse grid
f0 = 0.3e6  # Center frequency 0.3 MHz
Nx = 616
Ny = 485 
Nz = 719 # Number of grid points
dx = dy = 1.0e-3  # Grid spacing 1 mm
c0 = 1500.0

# (1) Load and downsample true sound-speed model

with open("UFWI/Phantom/mergedPhantom.DAT", "rb") as f:
    data = np.fromfile(f, dtype=np.uint8)
data = data.reshape((Nx, Ny, Nz), order="F")

X = Nx // 2
model_raw = data[X, :, :]

"""
mat = loadmat("UFWI/Phantom/C_true.mat")
model_raw = mat['C_true']  # Original (1601,1601)
print("Original model shape:", model_raw.shape)
"""


# Acoustic property mapping
# Using the actual integer labels from the data: [0, 2, 3, 4, 5]
sound_speed_map = {
    0: 1500.0,  # Background / water
    2: 1540.0,  # Fibroglandular
    3: 1450.0,  # Fat
    4: 1555.0,  # Skin
    5: 1548.0,  # Vessel
}
density_map = {
    0: 1000.0,  # Background / water
    2: 1040.0,  # Fibroglandular
    3: 911.0,   # Fat
    4: 1100.0,  # Skin
    5: 945.0,   # Vessel
}

# Create sound speed array
model_raw_ss = np.zeros_like(model_raw, dtype=np.float32)
model_raw_rho = np.zeros_like(model_raw, dtype=np.float32)

print("model_raw shape:", model_raw.shape)
print("model_raw dtype:", model_raw.dtype)
print("Unique values in model_raw:", np.unique(model_raw))

for label, ss_val in sound_speed_map.items():
    mask = (model_raw == label)
    count = np.sum(mask)
    print(f"Label {label}: found {count} pixels")
    model_raw_ss[mask] = ss_val
    model_raw_rho[mask] = density_map[label]

unmapped_mask = (model_raw_ss == 0)
print("Unmapped pixels after mapping:", np.sum(unmapped_mask))

# (2) make ImageGrid2D and compute record time
img_grid = ImageGrid2D(nx=120, ny=120, dx=1.0e-3)

print(f"ImageGrid2D created: nx={img_grid.nx}, ny={img_grid.ny}")
print(f"Grid extent: {img_grid.extent}")

# Downsample using ImageGrid2D
img_raw = ImageData(model_raw_ss)
img_true = img_raw.downsample_to(new_grid=img_grid)
model_core = img_true.array
print("Downsampled model shape:", model_core.shape)
print("model_true min:", model_core.min(), "max:", model_core.max())

# Replace zero/negative values with fat tissue speed (1450.0)
model_core[model_core <= 0] = 1450.0
print("After fixing zeros - model_core min:", model_core.min(), "max:", model_core.max())


print(f"Core model shape (before padding): {model_core.shape}")
print(f"Core model range: {model_core.min():.1f} - {model_core.max():.1f} m/s")
    
# Add 60 pixels of 1500 m/s padding on all sides to make 180x180
padding_width = 30  # 30 pixels on each side to go from 120x120 to 180x180
padding_value = 1500.0  # Water/background speed
    
model_true = np.pad(model_core, 
                    pad_width=padding_width, 
                    mode='constant', 
                    constant_values=padding_value)
    
print(f"After padding: {model_true.shape}")
print(f"Padding: {padding_width} pixels ({padding_width * img_grid.dx * 1000:.1f} mm) on each side")
print(f"Final model range: {model_true.min():.1f} - {model_true.max():.1f} m/s")
    
# Create new 180x180 grid for the padded model
img_grid_padded = ImageGrid2D(nx=180, ny=180, dx=1.0e-3)
print(f"Padded grid: {img_grid_padded.nx} x {img_grid_padded.ny}")
print(f"Padded grid physical size: {img_grid_padded.nx*img_grid_padded.dx*1000:.1f} x {img_grid_padded.ny*img_grid_padded.dy*1000:.1f} mm")
    
# Create transducer array on the larger padded grid
n_tx = 368  # Can use more transducers now with larger grid
radius = 70e-3  # Can use larger radius now
tx_array = TransducerArray2D.from_ring_array_2D(r=radius, grid=img_grid_padded, n=n_tx)

c_ref = model_true.max()


c_min = float(model_true.min())

# Compute record time
extent = img_grid_padded.extent

print(f"c_min = {c_min}")
record_time = 1.3 * (extent[1] - extent[0]) / c_min
print(f"record_time = {record_time * 1e3:.2f} ms")


# (4) AcquisitionData
acq_data = AcquisitionData.from_geometry(tx_array=tx_array, grid=img_grid_padded)

# (5) Visualize true sound speed + sensors
# true_image_data = ImageData(array=model_true, tx_array=tx_array, grid=img_grid)
# true_image_data.show()

# (6) Construct WaveOperator
# Use the downsampled grid dimensions instead of original Ny, Nx
grid_ny, grid_nx = model_true.shape
medium_params = {
    "sound_speed": model_true.astype(np.float32),
    "density": np.ones((grid_ny, grid_nx), np.float32) * 1000.0,
    "alpha_coeff": np.zeros((grid_ny, grid_nx), np.float32),
    "alpha_power": 1.01,
    "alpha_mode": "no_dispersion",
}

pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)

op_true = WaveOperator(
    data=acq_data,
    medium_params=medium_params,
    record_time=record_time,
    record_full_wf=False,
    use_encoding=False,
    drop_self_rx=False,
    pulse=pulse,
    c_ref=c_ref,
)

# (7) Load existing acquisition data 
# Note: We'll use 128x128 time-domain data but process on 180x180 frequency grid
fname_128 = SAVE_DIR / f"d_obs_128x128_1mm_0p3MHz_new_368.npz"

    

print(f"Error loading data: {e}")
print("Will need to run time-domain simulation...")
acq_sim = op_true.simulate()
fname = SAVE_DIR / f"d_obs_180x180_1mm_0p3MHz_new_368.npz"
acq_sim.save(fname)
acq_data_for_freq = acq_sim


###########SIMULATE#######################

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
"""
raw_mat = Path("SampleData/kWave_BreastCT.mat")
raw = load_mat(raw_mat)

pos = raw["transducerPositionsXY"]  # (2, N)
N = pos.shape[1]
ones = np.ones(N, dtype=bool)
tx_array = TransducerArray2D(
    positions=pos.astype(np.float32),
    is_tx=ones, is_rx=ones
)
"""
# Define imaging grid
# give grid spacing and half-width to ImageGeometry and it makes grid automatically
# dxi = 0.6e-3
# xmax = 120e-3
# img_grid = ImageGrid2D(dx=dxi, xmax=xmax)
c0 = 1500.0  # Speed of sound in water
"""
# Construct AcquisitionData
acq_data = AcquisitionData(
    array=raw["full_dataset"].transpose(2, 1, 0),  # (Tx,Rx,T)
    time=raw["time"],  # (T,)
    tx_array=tx_array,
    grid=img_grid,
    c0=c0
)
"""
# ------------------------------------------------------------------------------
# (2) Define frequency list & preprocessing pipeline
# ------------------------------------------------------------------------------

f_sos = np.arange(0.25, 0.35, 0.02) * 1e6  # Closer to simulation frequency
f_att = np.arange(0.255, 0.355, 0.02) * 1e6  # Slightly offset for attenuation
freqs = np.concatenate([f_sos, f_att])  # All frequencies (Nfreq,)

print(f"Frequency range: {freqs.min()/1e6:.3f} to {freqs.max()/1e6:.3f} MHz")
print(f"Simulation frequency: 0.3 MHz")

pipe = Pipeline(
    stages=[
        GaussianTimeWindow(),
        DTFT(freqs),
        PhaseScreenCorrection(img_grid_padded),
        DownSample(step=1),
        AcceptanceMask(delta=63),
        MagnitudeOutlierFilter(threshold=0.99),
    ],
    verbose=True
)

# Apply all preprocessing to the simulated acquisition data (which has time info)
acq_data_processed = pipe(acq_data_for_freq)  # Resulting shape: (Tx, Rx, Nfreq)

# Note: acq_data_processed.save() has JSON serialization issues with numpy arrays in ctx
# The processed data is ready for use in the inversion loop below

# ------------------------------------------------------------------------------
# (3) Prepare iteration counts for SoS/Atten per frequency
# ------------------------------------------------------------------------------
Tx, Rx, Nfreq = acq_data_processed.array.shape
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

Nxi, Nyi = img_grid_padded.nx, img_grid_padded.ny
print(f"Grid sizes: Nxi={Nxi}, Nyi={Nyi}")
SLOW_INIT = (1.0 / c_init) + 1j * (atten_init / (2.0 * np.pi))
slow0 = np.full((Nyi, Nxi), SLOW_INIT, dtype=np.complex128)
slow_data = ImageData(array=slow0, grid=img_grid_padded)

# ------------------------------------------------------------------------------
# (5) Create visualizer (InversionVisualizer)
# ------------------------------------------------------------------------------
# Load ground truth for comparison
C_true = model_true  # (Nyi, Nxi)
atten_true = np.zeros_like(model_true)  # (Nyi, Nxi)
# viz = InversionVisualizer(img_grid.xi, img_grid.yi, C_true, atten_true)

viz = Visualizer(
    xi=img_grid_padded.xi, yi=img_grid_padded.yi,
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


for idx_f in range(Nfreq):  # Original loop over all frequencies 
# for idx_f in range(3, 6):  # If you want to only process frequencies 3-5 to regenerate with 180x180 grid
    print(f"\n=== Processing frequency idx_f = {idx_f}, f = {freqs[idx_f] / 1e6:.3f} MHz ===")
    print(f"Grid size: {Nxi}x{Nyi} (should be 180x180)")
    n_sos = niterSoSPerFreq[idx_f]
    n_att = niterAttenPerFreq[idx_f]

    operator = HelmholtzOperator(acq_data_processed, idx_f,
                                 sign_conv=-1, pml_alpha=10.0, pml_size=9.0e-3)
    grad = HelmholtzAdjointGrad(
        operator,
        deriv_fn=lambda m, op: 8 * np.pi ** 2 * op.get_field("freq") ** 2 * (
                op.get_field("PML") / op.get_field("V"))
    )
    fun = NonlinearLS(operator, grad_eval=grad)



    # =========================================================================
    # GENERATE INCIDENT FIELDS FOR ALL SOURCES AT THIS FREQUENCY
    # Using homogeneous background sound speed c0 and zero attenuation
    # This solves: Helmholtz equation with eta = (c0, 0) for each source
    # =========================================================================

    # Create homogeneous slowness with background sound speed c0 = 1500 m/s
    c0_background = 1500.0  # Background sound speed (water)
    homogeneous_slowness = np.full((Nyi, Nxi), 
                                  (1.0 / c0_background) + 1j * 0.0,  # eta = (c0, 0)
                                  dtype=np.complex128)
    
    print(f"Computing incident fields with homogeneous c0 = {c0_background} m/s")
    print(f"Frequency: {freqs[idx_f]/1e6:.3f} MHz")
    
    # Solve Helmholtz forward problem for all sources with homogeneous background
    _ = operator.forward(homogeneous_slowness) # solves and caches the fields
    incident_fields = operator._cache.WF.copy()  # (ny, nx, n_tx) - INCIDENT FIELDS
    
    print(f"Generated incident fields: shape {incident_fields.shape}") # shape(180,180,368)
    print(f"Expected shape: ({Nyi}, {Nxi}, n_transmitters)")
    print(f"Magnitude range: {np.abs(incident_fields).min():.2e} to {np.abs(incident_fields).max():.2e}") # 0.00e+00 to 4.21e-07
    
    # Save incident fields for this frequency
    incident_fields_dir = SAVE_DIR / "incident_fields"
    incident_fields_dir.mkdir(exist_ok=True)
    
    np.save(incident_fields_dir / f"incident_fields_freq_{idx_f:02d}_f_{freqs[idx_f]/1e6:.3f}MHz.npy", 
            incident_fields)
    
    print(f"Saved incident fields to: incident_fields_freq_{idx_f:02d}_f_{freqs[idx_f]/1e6:.3f}MHz.npy")

    
     # =========================================================================
    # GENERATE SCATTERED WAVEFIELDS FOR ALL SOURCES AT THIS FREQUENCY
    # Using heterogeneous sound-speed map (not homogeneous background)
    # This solves: Helmholtz equation with the true sound speed distribution
    # =========================================================================
    
    print(f"Computing scattered wavefields through heterogeneous medium")
    print(f"Frequency: {freqs[idx_f]/1e6:.3f} MHz")
    
    # Create heterogeneous slowness model from true sound speed model
    atten_hetero = 0.0  # No attenuation
    slow_hetero = (1.0 / model_true) + 1j * (atten_hetero / (2.0 * np.pi))
    print(f"Heterogeneous slowness model shape: {slow_hetero.shape}") #(180, 180)
    print(f"Sound speed range: {model_true.min():.1f} to {model_true.max():.1f} m/s") #1450.0 to 1555.0 m/s
    
    # Solve Helmholtz forward problem for all sources with heterogeneous medium
    _ = operator.forward(slow_hetero) # solves and caches the fields
    total_wavefields = operator._cache.WF.copy()  # (ny, nx, n_tx) - TOTAL WAVEFIELDS
    
    scattered_fields = total_wavefields - incident_fields
    print(f"Scattered field magnitude range: {np.abs(scattered_fields).min():.2e} to {np.abs(scattered_fields).max():.2e}")
    
    # Save scattered fields
    scattered_fields_dir = SAVE_DIR / "scattered_fields"
    scattered_fields_dir.mkdir(exist_ok=True)
    
    np.save(scattered_fields_dir / f"scattered_fields_freq_{idx_f:02d}_f_{freqs[idx_f]/1e6:.3f}MHz.npy", 
            scattered_fields)
    
    print(f"Saved scattered fields to: scattered_fields_freq_{idx_f:02d}_f_{freqs[idx_f]/1e6:.3f}MHz.npy")


    print(f"Generated scattered wavefields: shape {scattered_fields.shape}") # Shape []
    print(f"Expected shape: ({Nyi}, {Nxi}, n_transmitters)")
    print(f"Magnitude range: {np.abs(scattered_fields).min():.2e} to {np.abs(scattered_fields).max():.2e}")



    tx_examples = [50, 184, 276]  

    for tx in tx_examples:
        field_tx = scattered_fields[:, :, tx]   # shape (128, 128), complex values - TOTAL WAVEFIELD

        # Magnitude, Phase, and Real Part
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        
        # Magnitude
        im0 = axs[0].imshow(np.abs(field_tx), cmap='inferno', origin='lower')
        axs[0].set_title(f"Tx {tx} | Magnitude")
        plt.colorbar(im0, ax=axs[0])

        # Phase
        im1 = axs[1].imshow(np.angle(field_tx), cmap='twilight', origin='lower')
        axs[1].set_title(f"Tx {tx} | Phase")
        plt.colorbar(im1, ax=axs[1])
        
        # Real Part
        im2 = axs[2].imshow(np.real(field_tx), cmap='seismic', origin='lower')
        axs[2].set_title(f"Tx {tx} | Real Part")
        plt.colorbar(im2, ax=axs[2])

        plt.suptitle(f"TOTAL WAVEFIELD (Heterogeneous Medium) - TX {tx} at {freqs[idx_f]/1e6:.3f} MHz")
        plt.tight_layout()
        plt.savefig(f"total_wavefield_freq_{idx_f:02d}_tx{tx}.png", dpi=150)
        plt.close()

    
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
    "xi": img_grid_padded.xi,
    "yi": img_grid_padded.yi,
    "fDATA": freqs.reshape(1, -1),
    "niterAttenPerFreq": niterAttenPerFreq.reshape(1, -1),
    "niterSoSPerFreq": niterSoSPerFreq.reshape(1, -1),
    "VEL_ESTIM_ITER": VEL_ESTIM_ITER,
    "ATTEN_ESTIM_ITER": ATTEN_ESTIM_ITER,
    "GRAD_IMG_ITER": GRAD_IMG_ITER,
    "SEARCH_DIR_ITER": SEARCH_DIR_ITER,
}, do_compression=True)

print("Results saved to Results/kWave_BreastCT_WaveformInversionResults.mat")

# ------------------------------------------------------------------------------
# (9) Create Neural Operator Training Data
#     Collect (ui, us, c(x,y)) tuples for all processed frequencies
# ------------------------------------------------------------------------------
print("\n=== CREATING NEURAL OPERATOR TRAINING DATA ===")

# Create directory for neural operator data
neural_data_dir = SAVE_DIR / "neural_operator_data"
neural_data_dir.mkdir(exist_ok=True)

# Collect training data for all processed frequencies (3-5)
training_data = []

processed_frequencies = range(Nfreq)  # Original: all frequencies
# processed_frequencies = range(3, 6)  # To save time and only iterate over 3 frequencies

for idx_f in processed_frequencies:
    freq_value = freqs[idx_f]
    print(f"Collecting data for frequency {idx_f}: {freq_value/1e6:.3f} MHz")
    
    # Load incident fields for this frequency
    incident_file = SAVE_DIR / "incident_fields" / f"incident_fields_freq_{idx_f:02d}_f_{freq_value/1e6:.3f}MHz.npy"
    if incident_file.exists():
        ui = np.load(incident_file)  # Shape: (ny, nx, n_tx)
        print(f"  Loaded incident fields: {ui.shape}")
    else:
        print(f"Incident fields not found for frequency {idx_f}")
        continue
    
    # Load scattered fields for this frequency  
    scattered_file = SAVE_DIR / "scattered_fields" / f"scattered_fields_freq_{idx_f:02d}_f_{freq_value/1e6:.3f}MHz.npy"
    if scattered_file.exists():
        us = np.load(scattered_file)  # Shape: (ny, nx, n_tx)
        print(f"  Loaded scattered fields: {us.shape}")
    else:
        print(f"Scattered fields not found for frequency {idx_f}")
        continue
    
    # Sound speed map (same for all frequencies)
    c_xy = model_true.copy()  # Shape: (ny, nx)
    print(f"  Sound speed map: {c_xy.shape}, range: {c_xy.min():.1f} to {c_xy.max():.1f} m/s")
    
    # Verify shapes match
    if ui.shape[:2] != us.shape[:2] or ui.shape[:2] != c_xy.shape:
        print(f" Shape mismatch: ui={ui.shape}, us={us.shape}, c_xy={c_xy.shape}")
        continue
    
    # Create training tuples for each transmitter
    n_tx = ui.shape[2]
    for tx_idx in range(n_tx):
        training_tuple = {
            'ui': ui[:, :, tx_idx],      # Incident field for this transmitter (ny, nx)
            'us': us[:, :, tx_idx],      # Scattered field for this transmitter (ny, nx)
            'c_xy': c_xy,                # Sound speed map (ny, nx)
        }
        training_data.append(training_tuple)
    
    print(f"  Created {n_tx} training tuples for frequency {freq_value/1e6:.3f} MHz")

print(f"\nTotal training tuples created: {len(training_data)}")
print(f"Each tuple contains:")
print(f"  - ui: incident field ({model_true.shape}) complex")
print(f"  - us: scattered field ({model_true.shape}) complex") 
print(f"  - c_xy: sound speed map ({model_true.shape}) real")


# Save training data in compressed format - ORGANIZED BY FREQUENCY
training_data_file = neural_data_dir / "neural_operator_training_data.npz"

# Prepare arrays with your preferred structure: [N_freqs, 180, 180, 368]
n_freqs = len(processed_frequencies)
ny, nx = model_true.shape
n_tx_actual = 368  # Number of transmitters

print(f"Creating tensor structure: [{n_freqs}, {ny}, {nx}, {n_tx_actual}]")

# Initialize arrays with frequency-organized structure
ui_tensor = np.zeros((n_freqs, ny, nx, n_tx_actual), dtype=np.complex128)
us_tensor = np.zeros((n_freqs, ny, nx, n_tx_actual), dtype=np.complex128)
c_xy_tensor = np.zeros((n_freqs, ny, nx), dtype=np.float64)  # Same for all freqs
frequency_array = np.zeros(n_freqs, dtype=np.float64)

# Fill tensors by reorganizing the data
freq_idx_map = {processed_frequencies[i]: i for i in range(n_freqs)}

for sample in training_data:
    freq_idx = freq_idx_map[sample['freq_index']]  # Map to tensor frequency index
    tx_idx = sample['tx_index']
    
    ui_tensor[freq_idx, :, :, tx_idx] = sample['ui']
    us_tensor[freq_idx, :, :, tx_idx] = sample['us']
    c_xy_tensor[freq_idx, :, :] = sample['c_xy']  # Same for all TX at this freq
    frequency_array[freq_idx] = sample['frequency']

print(f"Tensor organization complete!")
print(f"  ui_tensor shape: {ui_tensor.shape} (N_freqs, ny, nx, n_tx)")
print(f"  us_tensor shape: {us_tensor.shape} (N_freqs, ny, nx, n_tx)")
print(f"  c_xy_tensor shape: {c_xy_tensor.shape} (N_freqs, ny, nx)")

# Save tensor format
np.savez_compressed(
    training_data_file,
    # Tensor format [N_freqs, ny, nx, n_tx]
    ui_tensor=ui_tensor,                # Incident fields (n_freqs, ny, nx, n_tx)
    us_tensor=us_tensor,                # Scattered fields (n_freqs, ny, nx, n_tx)  
    c_xy_tensor=c_xy_tensor,            # Sound speed maps (n_freqs, ny, nx)
    frequency_array=frequency_array,    # Frequency values (n_freqs,)
    

    # Metadata
    grid_spacing=img_grid_padded.dx,           # Grid spacing (scalar)
    grid_extent=img_grid_padded.extent,        # Grid extent (tuple)
    n_frequencies=n_freqs,              # Number of frequencies
    n_transmitters=n_tx_actual,         # Number of transmitters
    processed_freq_indices=list(processed_frequencies),  # Which frequency indices were processed
    description="Neural operator training data: tensor format"
)

print(f"\n Training data saved to: {training_data_file}")
print(f"File size: {training_data_file.stat().st_size / 1024**2:.1f} MB")

# Print sample statistics
print(f"\n Training Data Statistics:")
print(f"\nTensor Format:")
print(f"Incident fields (ui_tensor):")
print(f"  Shape: {ui_tensor.shape} (N_freqs, ny, nx, n_tx)")
print(f"  Magnitude range: {np.abs(ui_tensor).min():.2e} to {np.abs(ui_tensor).max():.2e}")

print(f"Scattered fields (us_tensor):")
print(f"  Shape: {us_tensor.shape} (N_freqs, ny, nx, n_tx)")  
print(f"  Magnitude range: {np.abs(us_tensor).min():.2e} to {np.abs(us_tensor).max():.2e}")

print(f"Sound speed maps (c_xy_tensor):")
print(f"  Shape: {c_xy_tensor.shape} (N_freqs, ny, nx)")
print(f"  Value range: {c_xy_tensor.min():.1f} to {c_xy_tensor.max():.1f} m/s")


print(f"\n Neural operator training data creation complete")
print(f"Use this data to train neural operators that map (ui, c_xy) → us")
