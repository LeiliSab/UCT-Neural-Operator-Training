import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path
from UFWI.geometry import TransducerArray2D, ImageGrid2D
from UFWI.data import AcquisitionData
from UFWI.data.image_data import ImageData
from UFWI.optimization.operator.wave_operator import WaveOperator
from UFWI.optimization.gradient.time_grad import AdjointStateGrad
from UFWI.optimization.function.least_squares import NonlinearLS
from UFWI.optimization.algorithm.gd import GD
from UFWI.optimization.algorithm.cg_time import CG_Time
from UFWI.utils.visulizer_multi_mode import Visualizer
from UFWI.signals import GaussianModulatedPulse

# ====================== One-click configuration ====================== #
USE_ENCODING = True            # Whether to use source encoding
K = 16                         # Number of random encoding averages (if K > 1)
TAU_MAX = 0.0                  # Maximum random delay (seconds); 0 → ±1 synchronous encoding
DROP_SELF_RX = True            # Drop self-receiver in non-encoding mode
N_ITER = 10                    # Number of iterations
SAVE_DIR = Path("output")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
f0 = 5e5
normalize = True


def main():
    # -------- Grid parameters -------------------------------------- #
    Nx = Ny = 128
    dx = dy = 5.0e-4  # 0.5 mm
    c0 = 1500.0
    record_time = 1.2 * Nx * dx / c0
    n_tx = 64

    # Build imaging grid
    img_grid = ImageGrid2D(dx=dx, nx=Nx, ny=Ny)

    # -------- True / initial models -------------------------------- #
    model_bg = np.full((Ny, Nx), c0, np.float64)
    # Extract coordinates
    X, Y = img_grid.meshgrid()

    model_true = model_bg.copy()
    model_true[((X - 0.01) ** 2 + (Y - 0.01) ** 2) < 0.006 ** 2] = c0 + 100
    model_true[(X ** 2 + (Y + 0.01) ** 2) < 0.005 ** 2] = c0 - 100

    # -------- Build acquisition data container --------------------- #
    tx_array = TransducerArray2D.from_ring_array_2D(grid=img_grid, r=(min(Nx, Ny) // 2 - 2) * dx,
                                                    n=n_tx)
    acq_data = AcquisitionData.from_geometry(grid=img_grid, tx_array=tx_array)

    # ================================================================
    # 1) Generate “true” data (non-encoding + drop self-receive)
    # ================================================================
    pulse = GaussianModulatedPulse(f0=f0, frac_bw=0.75, amp=1.0)
    op_true = WaveOperator(
        acq_data,
        {"sound_speed": model_true},  # pass sound speed via medium_params
        record_time,
        use_encoding=False,
        record_full_wf=False,
        pml_size=10,
        cfl=0.2,
        drop_self_rx=True,
        pulse=pulse
    )

    fname = SAVE_DIR / f"acq_sim_ring_full_{n_tx}_new3.npz"

    # -------- Simulate data ---------------------------------------- #
    acq_sim = op_true.simulate()
    acq_sim.save(fname)
    # return

    # ================================================================
    # Load synthetic data for inversion
    # ================================================================
    dat = np.load(fname, allow_pickle=True)
    d_obs = dat["array"]
    t_vec = dat["time"]
    print("✓ Loaded synthetic data, shape =", d_obs.shape, "time vector shape =", t_vec.shape)

    # -------- Wrap inversion data ---------------------------------- #
    acq_inv_data = AcquisitionData(array=d_obs,
                                   tx_array=acq_data.tx_array,
                                   grid=acq_data.grid,
                                   time=t_vec)

    print(acq_inv_data.array.shape, "shape of acquisition data array")
    print(acq_inv_data.time.shape, "shape of acquisition data time vector")

    # ================================================================
    # Inversion operator configuration
    # ================================================================
    op_inv = WaveOperator(
        acq_inv_data,
        {"sound_speed": c0},
        record_time,
        use_encoding=USE_ENCODING,
        tau_max=float(TAU_MAX if USE_ENCODING else 0.0),
        record_full_wf=True,
        pml_size=10,
        cfl=0.2,
        drop_self_rx=bool(DROP_SELF_RX and not USE_ENCODING),
        pulse=pulse
    )

    # -------- Construct least squares & gradient -------------------- #
    w = 1.0
    grad_eval = AdjointStateGrad(
        op_inv,
        K=(K if (USE_ENCODING and K > 1) else None),
        seed=0,
    )
    f_ls = NonlinearLS(
        op_inv,
        grad_eval=grad_eval,
        weight=w,
        normalize=normalize,
    )

    # -------- Initial image ---------------------------------------- #
    model_init = model_bg.astype(np.float64)
    img_init = ImageData(model_init)

    # -------- Choose optimizer ------------------------------------- #
    eta0 = 6.0e-1
    viz = Visualizer(X, Y, model_true, np.zeros_like(model_true), mode="vel")

    USE = "GD"

    if USE == "GD":
        print("\n>>> Using GD algorithm")
        step0 = 50 * eta0
        solver = GD(
            lr=step0,
            backtrack=False,
            max_bt=12,
            schedule_fn=lambda k, lr: lr,
            viz=viz
        )
    else:
        print("\n>>> Using CG_Time algorithm")
        solver = CG_Time(viz=viz)

    solver.solve(kind='c', fun=f_ls, m0=img_init, n_iter=N_ITER)

    solver.save_record(SAVE_DIR / f"record_{('ENC' if USE_ENCODING else 'NOENC')}.npz")


    # ================================================================
    # Visualization of results
    # ================================================================
    rec = solver.get_record()
    vel_all = np.array(rec["vel"], dtype=np.float64)
    grad_all = np.array(rec["grad"], dtype=np.float64).real

    vmin_c, vmax_c = model_true.min(), model_true.max()
    abs99 = np.percentile(np.abs(grad_all), 99)
    vmin_grad, vmax_grad = -abs99, abs99
    extent = (-Nx / 2 * dx, Nx / 2 * dx, -Ny / 2 * dy, Ny / 2 * dy)
    tx_x, tx_y = op_inv.tx_pos

    fig, ax = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
    ims = [
        (model_true, "True c", vmin_c, vmax_c),
        (vel_all[..., -1], f"Recon it {N_ITER}", vmin_c, vmax_c),
        (grad_all[..., -1], f"Grad it {N_ITER}", vmin_grad, vmax_grad),
    ]
    for a, (arr, title, vmin, vmax) in zip(ax, ims):
        im = a.imshow(arr, extent=extent, origin='lower',
                      cmap='seismic', vmin=vmin, vmax=vmax)
        a.set_title(title)
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")
        a.scatter(tx_x, tx_y, marker='*', s=30, color='lime')
        fig.colorbar(im, ax=a, fraction=0.046)
    out_fig = SAVE_DIR / "final.png"
    plt.savefig(out_fig, dpi=220)
    print(f"[info] Final figure saved → {out_fig}")

    rec_misfit = rec["misfit"][1, :]
    vel_diff = vel_all.real - c0
    v_abs = 200
    vmin_rec, vmax_rec = -v_abs, v_abs

    n_rows, n_cols = 1 + N_ITER, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(model_true - c0, extent=extent, origin='lower',
                      cmap='seismic', vmin=vmin_rec, vmax=vmax_rec)
    axes[0, 0].set_title("True Δc")
    axes[0, 0].scatter(tx_x, tx_y, marker='*', s=30, color='lime')
    fig.colorbar(axes[0, 0].images[-1], ax=axes[0, 0], fraction=0.046)

    axes[0, 1].imshow(grad_all[..., 0], extent=extent, origin='lower',
                      cmap='seismic', vmin=vmin_grad, vmax=vmax_grad)
    axes[0, 1].set_title("Initial gradient")
    axes[0, 1].scatter(tx_x, tx_y, marker='*', s=30, color='lime')
    fig.colorbar(axes[0, 1].images[-1], ax=axes[0, 1], fraction=0.046)

    for i in range(1, n_rows):
        im0 = axes[i, 0].imshow(vel_diff[..., i - 1], extent=extent, origin='lower',
                                cmap='seismic', vmin=vmin_rec, vmax=vmax_rec)
        axes[i, 0].set_title(f"Recon Δc @ iter {i}\nmisfit={rec_misfit[i - 1]:.2e}")
        axes[i, 0].scatter(tx_x, tx_y, marker='*', s=30, color='lime')
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046)

        im1 = axes[i, 1].imshow(grad_all[..., i - 1], extent=extent, origin='lower',
                                cmap='seismic', vmin=vmin_grad, vmax=vmax_grad)
        axes[i, 1].set_title(f"Grad @ iter {i}")
        axes[i, 1].scatter(tx_x, tx_y, marker='*', s=30, color='lime')
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046)

    out_fig2 = SAVE_DIR / "timeline.png"
    plt.savefig(out_fig2, dpi=220)
    print(f"[info] Saved timeline → {out_fig2}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()