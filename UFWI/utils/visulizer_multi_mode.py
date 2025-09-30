import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from typing import Literal
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for interactive plotting


class Visualizer:
    """
    Live visualization of velocity / attenuation during iterative inversions.
    """

    def __init__(
        self,
        xi: np.ndarray,
        yi: np.ndarray,
        C_true: np.ndarray,
        atten_true: np.ndarray,
        mode: Literal["vel", "atten", "both"] = "both",
        baseline: float = 1500.0,
        vel_cmap: str = "PRGn",
        sync_clim: bool = True,
        sign_conv: int = 1,
        atten_unit: str = "dB/(MHz·cm)"
    ):
        # -------- basic checks / attributes -------------------------
        if mode not in ("vel", "atten", "both"):
            raise ValueError("mode must be 'vel', 'atten' or 'both'")

        self.mode = mode
        self.baseline = baseline
        self.vel_cmap = vel_cmap
        self.sync_clim = bool(sync_clim)
        self.sign_conv = int(np.sign(sign_conv)) or 1
        self.global_iter = 0

        # -------- color handling ------------------------------------
        # velocity: use TwoSlopeNorm centered at baseline
        vmin_vel, vmax_vel = float(C_true.min()), float(C_true.max())
        self.true_vel_norm = TwoSlopeNorm(vmin=vmin_vel, vcenter=baseline, vmax=vmax_vel)

        # attenuation: record true min/max for sync_clim
        self.atten_vmin_true = float(atten_true.min())
        self.atten_vmax_true = float(atten_true.max())

        # -------- figure & axes -------------------------------------
        if mode == "both":
            self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=False)
        else:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=False)

        # ============================================================
        # velocity panels
        # ============================================================
        self.im_true_vel = None
        self.im_est_vel = None
        self.im_grad_vel = None
        self.im_search_vel = None  # only used when mode=='vel'

        if mode in ("vel", "both"):
            if mode == "vel":
                ax_t, ax_e, ax_g, ax_s = self.axes[0, 0], self.axes[1, 0], self.axes[0, 1], self.axes[1, 1]
            else:
                ax_t, ax_e, ax_g, ax_s = self.axes[0, 0], self.axes[0, 1], self.axes[0, 2], self.axes[1, 2]

            # -- true velocity ---------------------------------------
            self.im_true_vel = ax_t.imshow(
                C_true,
                extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                origin="lower",
                aspect="equal",
                cmap=self.vel_cmap,
                norm=self.true_vel_norm,
            )
            ax_t.set_title("True Velocity")
            self.fig.colorbar(self.im_true_vel, ax=ax_t, label="Velocity (m/s)", shrink=0.6)

            # -- estimated velocity ----------------------------------
            if self.sync_clim:
                # synchronized colorbar: reuse the same TwoSlopeNorm as the truth
                self.im_est_vel = ax_e.imshow(
                    np.zeros_like(C_true),
                    extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                    origin="lower",
                    aspect="equal",
                    cmap=self.vel_cmap,
                    norm=self.true_vel_norm,
                )
            else:
                # independent colorbar: initialize placeholder, dynamically set_clim in update()
                self.im_est_vel = ax_e.imshow(
                    np.zeros_like(C_true),
                    extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                    origin="lower",
                    aspect="equal",
                    cmap=self.vel_cmap,
                    vmin=vmin_vel,
                    vmax=vmax_vel,
                )
            ax_e.set_title("Estimated Velocity")
            self.fig.colorbar(self.im_est_vel, ax=ax_e, label="Velocity (m/s)", shrink=0.6)

            # -- gradient --------------------------------------------
            self.im_grad_vel = ax_g.imshow(
                np.zeros_like(C_true),
                extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                origin="lower",
                aspect="equal",
                cmap="seismic",
            )
            ax_g.set_title("Gradient")
            self.fig.colorbar(self.im_grad_vel, ax=ax_g, label="Amplitude", shrink=0.6)

            # -- search direction ------------------------------------
            # Same approach as for “Gradient”: draw only one image.
            # • In 'both' mode: do not draw here in the velocity panel (to avoid duplication).
            # • In 'vel'  mode: draw here (bottom-right of the 2×2 layout).
            if mode == "vel":
                self.im_search_vel = ax_s.imshow(
                    np.zeros_like(C_true),
                    extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                    origin="lower",
                    aspect="equal",
                    cmap="seismic",
                )
                ax_s.set_title("Search Direction")
                self.fig.colorbar(self.im_search_vel, ax=ax_s, label="Amplitude", shrink=0.6)

        # ============================================================
        # attenuation panels
        # ============================================================
        self.im_true_atten = None
        self.im_est_atten = None
        self.im_search = None  # used in both/atten modes (unified Search Dir panel)

        if mode in ("atten", "both"):
            if mode == "atten":
                ax_ta, ax_ea, ax_ga_unused, ax_sa = self.axes[0, 0], self.axes[1, 0], self.axes[0, 1], self.axes[1, 1]
            else:
                # both: place on the bottom row (left/middle); bottom-right reserved for unified Search Direction
                ax_ta, ax_ea, ax_ga_unused, ax_sa = self.axes[1, 0], self.axes[1, 1], None, self.axes[1, 2]

            # -- true attenuation ------------------------------------
            self.im_true_atten = ax_ta.imshow(
                atten_true,
                extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                origin="lower",
                aspect="equal",
                cmap="viridis",
                vmin=self.atten_vmin_true,
                vmax=self.atten_vmax_true,
            )
            ax_ta.set_title("True Attenuation")
            self.fig.colorbar(self.im_true_atten, ax=ax_ta, label=f"{atten_unit}", shrink=0.6)

            # -- estimated attenuation -------------------------------
            if self.sync_clim:
                # synchronize to the range of the truth
                self.im_est_atten = ax_ea.imshow(
                    np.zeros_like(atten_true),
                    extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                    origin="lower",
                    aspect="equal",
                    cmap="viridis",
                    vmin=self.atten_vmin_true,
                    vmax=self.atten_vmax_true,
                )
            else:
                # independent colorbar: initialize placeholder, then set_clim dynamically in update()
                self.im_est_atten = ax_ea.imshow(
                    np.zeros_like(atten_true),
                    extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                    origin="lower",
                    aspect="equal",
                    cmap="viridis",
                    vmin=self.atten_vmin_true,
                    vmax=self.atten_vmax_true,
                )
            ax_ea.set_title("Estimated Attenuation")
            self.fig.colorbar(self.im_est_atten, ax=ax_ea, label=f"{atten_unit}", shrink=0.6)

            # -- unified search direction (both/atten) ----------------
            # Same as “Gradient”: draw only one image; in 'both' mode place at bottom-right.
            self.im_search = ax_sa.imshow(
                np.zeros_like(atten_true),
                extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                origin="lower",
                aspect="equal",
                cmap="seismic",
            )
            ax_sa.set_title("Search Direction")
            self.fig.colorbar(self.im_search, ax=ax_sa, label="Amplitude", shrink=0.6)

        # ------------------------------------------------------------
        self.fig.tight_layout()
        plt.ion()
        self.fig.canvas.draw()
        plt.pause(0.1)

    # -----------------------------------------------------------------
    @staticmethod
    def _safe_clim(img: np.ndarray, pad: float = 1e-6) -> tuple[float, float]:
        vmin, vmax = float(img.min()), float(img.max())
        if vmin == vmax:
            vmin -= pad
            vmax += pad
        return vmin, vmax

    # -----------------------------------------------------------------
    def update(self, *, vel_est=None, atten_est=None, grad=None, search_dir=None, title=None):
        self.global_iter += 1
        k = self.global_iter

        # =============== velocity ====================
        if self.mode in ("vel", "both") and vel_est is not None and self.im_est_vel is not None:
            self.im_est_vel.set_data(vel_est)

            # independent colorbar: update per frame
            if not self.sync_clim:
                vmin, vmax = self._safe_clim(vel_est)
                self.im_est_vel.set_clim(vmin, vmax)

            if title:
                self.im_est_vel.axes.set_title(f"Estimated Velocity {k}\n{title}")
            else:
                self.im_est_vel.axes.set_title(f"Estimated Velocity {k}")

        if self.mode in ("vel", "both") and grad is not None and self.im_grad_vel is not None:
            g = grad.real
            gmax = np.max(np.abs(g)) or 1.0
            self.im_grad_vel.set_data(g)
            self.im_grad_vel.set_clim(-gmax, gmax)
            self.im_grad_vel.axes.set_title(f"Gradient {k}")

        # Search Direction: same as “Gradient”, update only one place
        # • both/atten mode → self.im_search
        # • vel mode        → self.im_search_vel
        if search_dir is not None:
            s = search_dir.real
            smax = np.max(np.abs(s)) or 1.0
            if self.im_search is not None:
                self.im_search.set_data(s)
                self.im_search.set_clim(-smax, smax)
                self.im_search.axes.set_title(f"Search Direction {k}")
            elif self.im_search_vel is not None:
                self.im_search_vel.set_data(s)
                self.im_search_vel.set_clim(-smax, smax)
                self.im_search_vel.axes.set_title(f"Search Direction {k}")

        # =============== attenuation =================
        if self.mode in ("atten", "both") and atten_est is not None and self.im_est_atten is not None:
            a = self.sign_conv * atten_est  # display only; any unit conversion is handled upstream
            self.im_est_atten.set_data(a)

            if not self.sync_clim:
                avmin, avmax = self._safe_clim(a)
                self.im_est_atten.set_clim(avmin, avmax)

            if title:
                self.im_est_atten.axes.set_title(f"Estimated Attenuation {k}\n{title}")
            else:
                self.im_est_atten.axes.set_title(f"Estimated Attenuation {k}")

        # ------------------------------------------------------------
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()