# single_grad_run_encoded_grid.py
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless-safe; change to "TkAgg" for interactive windows
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.io import loadmat

from UFWI.geometry import ImageGrid2D, TransducerArray2D
from UFWI.data import AcquisitionData
from UFWI.data.image_data import ImageData
from UFWI.signals import GaussianModulatedPulse
from UFWI.optimization.operator.wave_operator import WaveOperator
from UFWI.optimization.gradient.time_grad import AdjointStateGrad
from UFWI.optimization.function.least_squares import NonlinearLS


def build_obs_if_needed(dfile: Path,
                        img_grid: ImageGrid2D,
                        tx_array: TransducerArray2D,
                        model_true: np.ndarray,
                        record_time: float,
                        pulse,
                        c_ref: float):
    """Synthesize observation once (non-encoding, sequential) and save to npz."""
    if dfile.exists():
        return
    print("[info] No observation npz found; synthesizing once with TRUE model…")
    Ny, Nx = model_true.shape
    medium_true = {
        "sound_speed": np.array(model_true, np.float32),
        "density": np.full((Ny, Nx), 1000.0, np.float32),
        "alpha_coeff": np.zeros((Ny, Nx), np.float32),
        "alpha_power": 1.01,
        "alpha_mode": "no_dispersion",
    }
    op_true = WaveOperator(
        data=AcquisitionData.from_geometry(tx_array=tx_array, grid=img_grid),
        medium_params=medium_true,
        record_time=record_time,
        record_full_wf=False,
        use_encoding=False,       # synthesize plain shots
        drop_self_rx=True,        # typical config
        pulse=pulse,
        c_ref=c_ref,
        use_gpu=False,
    )
    acq_sim = op_true.simulate()
    np.savez(dfile, array=acq_sim.array, time=acq_sim.time)
    print(f"[info] ✓ observation saved → {dfile}")


def run_gradient_encoded(op: WaveOperator,
                         init_model: np.ndarray,
                         K: int | None,
                         seed: int = 1234) -> tuple[np.ndarray, float]:
    """
    Use AdjointStateGrad's internal K-loop.
    - If tau_max == 0: random ±1 weights, no delays.
    - If tau_max  > 0: random ±1 weights + random integer delays in [0, tau_step].
    Returns (gradient, misfit_sum) with NO averaging over K (just the native output).
    """
    ge = AdjointStateGrad(op, K=K, seed=seed, use_first_deriv_product=True)
    fun = NonlinearLS(op, grad_eval=ge, weight=1.0, sync_value=True)
    g = fun.gradient(init_model, kind="c")
    return g, float(fun.last_misfit)


def run_single_realization_with_random_wts_and_delays(op: WaveOperator,
                                                      init_model: np.ndarray,
                                                      seed: int = 2025) -> tuple[np.ndarray, float]:
    """
    For the K=1 (single realization) WITH delays case:
    Manually set random ±1 weights and random delays, then run a single gradient.
    """
    rng = np.random.default_rng(seed)
    # Random ±1 weights
    op.enc_weights = rng.choice([-1, 1], size=op.n_tx).astype(np.float32)
    # Random delays within [0, tau_step]
    if getattr(op, "tau_step", 0) > 0:
        op.enc_delays = rng.integers(0, op.tau_step + 1, size=op.n_tx, dtype=np.int32)
    else:
        op.enc_delays = np.zeros(op.n_tx, np.int32)
    op.renew_encoded_obs()

    ge = AdjointStateGrad(op, K=None, seed=seed, use_first_deriv_product=True)
    fun = NonlinearLS(op, grad_eval=ge, weight=1.0, sync_value=True)
    g = fun.gradient(init_model, kind="c")
    return g, float(fun.last_misfit)


def main():
    # ---------------- basic config ----------------
    SAVE_DIR = Path("output_single_grad")
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    # grid / physics
    Nx = Ny = 120
    dx = dy = 2e-3
    f0 = 0.3e6
    c0 = 1500.0

    # array
    n_tx = 256
    radius = 110e-3

    # data file
    dfile = Path("output") / "d_obs_for_single.npz"
    dfile.parent.mkdir(exist_ok=True, parents=True)

    # ---------------- true model (for record_time & plotting) ----------------
    mat = loadmat("../UFWI/Phantom/C_true.mat")
    model_raw = mat["C_true"]  # (1601,1601)
    img_grid = ImageGrid2D(nx=Nx, ny=Ny, dx=dx)
    img_true = ImageData(model_raw).downsample_to(new_grid=img_grid)
    model_true = img_true.array.astype(np.float64)
    c_ref = float(model_true.max())

    extent = img_grid.extent
    c_min = float(model_true.min())
    record_time = 1.3 * (extent[1] - extent[0]) / c_min

    # ring
    tx_array = TransducerArray2D.from_ring_array_2D(grid=img_grid, n=n_tx, r=radius)

    # pulse
    pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)

    # ---------------- synthesize observation if missing ----------------
    build_obs_if_needed(dfile, img_grid, tx_array, model_true, record_time, pulse, c_ref)

    # ---------------- load observation & build background operator base ----------------
    dat = np.load(dfile, allow_pickle=True)
    d_obs = dat["array"]           # (Tx, n_rx, nt) element order
    t_vec = dat["time"]

    # initial/background model
    Ny, Nx = model_true.shape
    model_init = np.full((Ny, Nx), c0, np.float64)
    medium_bg = {
        "sound_speed": np.full((Ny, Nx), c0, np.float32),
        "density": np.full((Ny, Nx), 1000.0, np.float32),
        "alpha_coeff": np.zeros((Ny, Nx), np.float32),
        "alpha_power": 1.01,
        "alpha_mode": "no_dispersion",
    }

    # ---------- load non-encoding baseline gradient (must exist in SAVE_DIR) ----------
    # ref_file = SAVE_DIR / "single_grad_result.npz"
    # if not ref_file.exists():
    #     raise FileNotFoundError(
    #         f"Baseline non-encoding gradient not found: {ref_file}\n"
    #         f"Please run your non-encoding script first to produce this file."
    #     )
    # ref = np.load(ref_file, allow_pickle=True)
    # g_ref = ref["grad"].astype(np.float64)

    # ---------- helper to build encoding operator with a given tau_max ----------
    def make_op(tau_max_sec: float) -> WaveOperator:
        return WaveOperator(
            data=AcquisitionData(array=d_obs, tx_array=tx_array, grid=img_grid, time=t_vec),
            medium_params=medium_bg,
            record_time=record_time,
            record_full_wf=True,
            use_encoding=True,         # enable encoding for all tests
            tau_max=tau_max_sec,       # =0: no delays; >0: allow delays
            drop_self_rx=True,         # ignored when use_encoding=True
            pulse=pulse,
            c_ref=c_ref,
            use_gpu=False,
            verbose=False,
        )

    # ================= Row 1: NO time delays (random ±1, no delays) =================
    op_no_delay = make_op(tau_max_sec=0.0)
    g_1,  _m1  = run_gradient_encoded(op_no_delay, model_init, K=None)  # K=1 realization
    g_6,  _m6  = run_gradient_encoded(op_no_delay, model_init, K=6)
    g_12, _m12 = run_gradient_encoded(op_no_delay, model_init, K=12)

    # ================= Row 2: WITH time delays (random ±1 + random delays) =================
    tau_max_sec = 8.0 / f0  # ~8 cycles
    op_delay = make_op(tau_max_sec=tau_max_sec)
    g_1d,  _m1d  = run_single_realization_with_random_wts_and_delays(op_delay, model_init)
    g_6d,  _m6d  = run_gradient_encoded(op_delay, model_init, K=6)
    g_12d, _m12d = run_gradient_encoded(op_delay, model_init, K=12)

    # ---------------- gradients to display & simple scalar errors (mean absolute difference) ----------
    grads = [g_1, g_6, g_12, g_1d, g_6d, g_12d]
    # panel_err = [float(np.mean(np.abs(g - g_ref))) for g in grads]  # simplest error
    #
    # # ---------------- save numeric results ----------------
    # np.savez(
    #     SAVE_DIR / "encoding_grid_vs_baseline.npz",
    #     grad_noDelay_K1=g_1,
    #     grad_noDelay_K6=g_6,
    #     grad_noDelay_K12=g_12,
    #     grad_delay_K1=g_1d,
    #     grad_delay_K6=g_6d,
    #     grad_delay_K12=g_12d,
    #     grad_ref=g_ref,
    #     err_noDelay_K1=panel_err[0],
    #     err_noDelay_K6=panel_err[1],
    #     err_noDelay_K12=panel_err[2],
    #     err_delay_K1=panel_err[3],
    #     err_delay_K6=panel_err[4],
    #     err_delay_K12=panel_err[5],
    # )
    # print(f"[info] ✓ results saved → {SAVE_DIR/'encoding_grid_vs_baseline.npz'}")

    # ---------------- plotting: 2 rows × 3 columns of GRADIENT maps, corner shows simple error ----------
    extent_xy = (-(Nx/2)*dx, (Nx/2)*dx, -(Ny/2)*dy, (Ny/2)*dy)
    G_stack = np.stack(grads, axis=0)
    A = np.percentile(np.abs(G_stack), 99.5)
    norm = colors.TwoSlopeNorm(vmin=-A, vcenter=0.0, vmax=A)

    titles_row1 = ["No delay — K=1", "No delay — K=6", "No delay — K=12"]
    titles_row2 = ["With delays — K=1", "With delays — K=6", "With delays — K=12"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    ims = []

    # Row 1
    for j in range(3):
        ax = axes[0, j]
        im = ax.imshow(grads[j], origin="lower", extent=extent_xy, cmap="seismic", norm=norm)
        ims.append(im)
        ax.set_title(titles_row1[j])
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        # ax.text(0.02, 0.04, f"error={panel_err[j]:.3e}",
        #         transform=ax.transAxes, fontsize=10, color="white",
        #         bbox=dict(facecolor="black", alpha=0.45, pad=5, edgecolor="none"))

    # Row 2
    for j in range(3):
        ax = axes[1, j]
        im = ax.imshow(grads[3 + j], origin="lower", extent=extent_xy, cmap="seismic", norm=norm)
        ims.append(im)
        ax.set_title(titles_row2[j])
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        # ax.text(0.02, 0.04, f"error={panel_err[3 + j]:.3e}",
        #         transform=ax.transAxes, fontsize=10, color="white",
        #         bbox=dict(facecolor="black", alpha=0.45, pad=5, edgecolor="none"))

    # Single shared colorbar
    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), fraction=0.046, pad=0.02)
    cbar.set_label("∂Φ/∂c (arb.)")

    out_png = SAVE_DIR / "encoding_grid_gradients_2x3.png"
    plt.savefig(out_png, dpi=220)
    print(f"[info] ✓ figure saved → {out_png}")


if __name__ == "__main__":
    main()
