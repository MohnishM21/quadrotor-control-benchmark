"""Figures for the README. Every figure is drawn twice, for GitHub's light and dark themes.

Colors: a fixed categorical order (one color per controller, never reassigned),
checked for color-vision-deficiency separation on both surfaces. The reference
is always a dashed neutral line, so it never competes with a controller color.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from .dynamics import P, Q  # noqa: E402
from .metrics import position_error  # noqa: E402
from .rotations import quat_to_rotmat  # noqa: E402
from .scenarios import SCENARIOS  # noqa: E402

CONTROLLERS = ["PID", "LQR", "Geometric", "MPC"]

THEMES = {
    "light": {
        "surface": "#ffffff", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e8e7e3",
        "series": {"PID": "#2a78d6", "LQR": "#eb6834", "Geometric": "#1baf7a", "MPC": "#eda100"},
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "crash": "#f0efec",
    },
    "dark": {
        "surface": "#0d1117", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#262b33",
        "series": {"PID": "#3987e5", "LQR": "#d95926", "Geometric": "#199e70", "MPC": "#c98500"},
        "ramp": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "crash": "#383835",
    },
}

TOP_VIEW = {"circle", "figure8", "figure8_fast", "mass_mismatch", "motor_lag", "wind", "sensor_noise"}


def _apply(theme):
    th = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": th["surface"], "axes.facecolor": th["surface"],
        "savefig.facecolor": th["surface"], "text.color": th["text"],
        "axes.labelcolor": th["muted"], "axes.edgecolor": th["grid"],
        "xtick.color": th["muted"], "ytick.color": th["muted"],
        "axes.grid": True, "grid.color": th["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    })
    return th


def _save(fig, out_dir, stem, theme):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}_{theme}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scenario(results, name, out_dir, theme):
    th = _apply(theme)
    scenario = SCENARIOS[name]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.suptitle(f"{name}: {scenario.description}", color=th["text"], fontsize=11, x=0.02, ha="left")

    any_run = next(r for (s, _), r in results.items() if s == name)
    if name in TOP_VIEW:
        ax_l.plot(any_run.ref_pos[:, 0], any_run.ref_pos[:, 1], "--", color=th["muted"], lw=1.2, label="reference")
        for c in CONTROLLERS:
            r = results.get((name, c))
            if r is not None:
                ax_l.plot(r.x[:, 0], r.x[:, 1], color=th["series"][c], label=c)
        ax_l.set_xlabel("x (m)")
        ax_l.set_ylabel("y (m)")
        ax_l.set_aspect("equal", adjustable="datalim")
        ax_l.set_title("Top view", loc="left")
    else:
        ax_l.plot(any_run.t, any_run.ref_pos[:, 2], "--", color=th["muted"], lw=1.2, label="reference")
        for c in CONTROLLERS:
            r = results.get((name, c))
            if r is not None:
                ax_l.plot(r.t, r.x[:, 2], color=th["series"][c], label=c)
        ax_l.set_xlabel("time (s)")
        ax_l.set_ylabel("altitude z (m)")
        ax_l.set_title("Altitude", loc="left")

    for c in CONTROLLERS:
        r = results.get((name, c))
        if r is not None:
            err = np.maximum(100 * position_error(r), 1e-4)
            ax_r.plot(r.t, err, color=th["series"][c], label=c + (" (crashed)" if r.crashed else ""))
    ax_r.set_yscale("log")
    ax_r.set_ylim(bottom=1e-4)
    ax_r.set_xlabel("time (s)")
    ax_r.set_ylabel("position error (cm, log scale)")
    ax_r.set_title("Tracking error", loc="left")
    handles, labels = ax_l.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9.5, labelcolor=th["text"])
    _save(fig, out_dir, name, theme)


def plot_summary(rows, out_dir, theme):
    th = _apply(theme)
    names = list(SCENARIOS)
    grid = np.full((len(names), len(CONTROLLERS)), np.nan)
    crashed = np.zeros_like(grid, dtype=bool)
    for r in rows:
        if r["scenario"] in names and r["controller"] in CONTROLLERS:
            i, j = names.index(r["scenario"]), CONTROLLERS.index(r["controller"])
            grid[i, j] = r["rmse_cm"]
            crashed[i, j] = r["status"] != "ok"

    finite = grid[np.isfinite(grid) & ~crashed]
    norm = LogNorm(vmin=max(finite.min(), 1e-2), vmax=finite.max())
    cmap = LinearSegmentedColormap.from_list("ramp", th["ramp"])

    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(names) + 1.4))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(names)):
        for j in range(len(CONTROLLERS)):
            v = grid[i, j]
            if crashed[i, j] or not np.isfinite(v):
                face, label, ink = th["crash"], "crash", th["text"]
            else:
                face = cmap(norm(max(v, norm.vmin)))
                label = f"{v:.1f}" if v >= 10 else f"{v:.2f}" if v >= 0.01 else "<0.01"
                lum = np.dot(face[:3], [0.2126, 0.7152, 0.0722])
                ink = "#0b0b0b" if lum > 0.45 else "#ffffff"
            # 2 px surface gap between cells
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.04), 0.96, 0.92, facecolor=face, edgecolor="none"))
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", color=ink, fontsize=9.5)
    ax.set_xlim(0, len(CONTROLLERS))
    ax.set_ylim(len(names), 0)
    ax.set_xticks(np.arange(len(CONTROLLERS)) + 0.5, CONTROLLERS)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(names)) + 0.5, names)
    ax.tick_params(length=0, labelsize=9.5)
    for label in ax.get_xticklabels():
        label.set_color(th["text"])
        label.set_fontweight("bold")
    ax.set_title("Position RMSE (cm), lower is better", loc="left", pad=26, color=th["text"])
    fig.tight_layout()
    _save(fig, out_dir, "summary_rmse", theme)


def _quad_lines(R, p, arm=0.13):
    """Body-frame X arms and thrust axis, scaled up so they are visible at room scale."""
    d = arm / np.sqrt(2)
    rotors = np.array([[d, -d, 0], [-d, d, 0], [-d, -d, 0], [d, d, 0]])
    world = (R @ rotors.T).T + p
    thrust_tip = p + R[:, 2] * arm * 0.9
    return world, thrust_tip


def make_gif(results, name, out_dir, theme, seconds=3.0, fps=25):
    th = _apply(theme)
    runs = [results[(name, c)] for c in CONTROLLERS if (name, c) in results]
    fig = plt.figure(figsize=(10, 3.6))
    axes = [fig.add_subplot(1, len(runs), k + 1, projection="3d") for k in range(len(runs))]
    fig.subplots_adjust(left=-0.02, right=1.02, bottom=-0.08, top=0.95, wspace=-0.15)
    title = fig.suptitle("", color=th["text"], fontsize=11, y=0.99)
    target = runs[0].ref_pos[0]

    def draw(frame):
        t_now = frame / fps
        title.set_text(f"Flip recovery from 170 deg roll    t = {t_now:.2f} s")
        for ax, r in zip(axes, runs):
            ax.cla()
            ax.set_facecolor(th["surface"])
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((0, 0, 0, 0))
                axis.line.set_color(th["grid"])
                axis.set_ticklabels([])
                axis._axinfo["grid"]["color"] = th["grid"]
            ax.set_xlim(target[0] - 0.4, target[0] + 0.4)
            ax.set_ylim(target[1] - 0.4, target[1] + 0.4)
            ax.set_zlim(target[2] - 0.8, target[2] + 0.3)
            ax.set_box_aspect((0.8, 0.8, 1.1), zoom=1.15)
            ax.view_init(elev=10, azim=-70)
            color = th["series"][r.controller]
            i = min(np.searchsorted(r.t, t_now), len(r.t) - 1)
            ax.plot(*r.x[: i + 1, P].T, color=color, lw=1.2, alpha=0.7)
            ax.scatter(*target, color=th["muted"], marker="x", s=30)
            R = quat_to_rotmat(r.x[i, Q])
            rotors, tip = _quad_lines(R, r.x[i, P])
            for a, b in ((0, 1), (2, 3)):
                ax.plot(*np.array([rotors[a], rotors[b]]).T, color=th["text"], lw=2.2)
            ax.scatter(*rotors.T, color=color, s=22, depthshade=False)
            ax.plot(*np.array([r.x[i, P], tip]).T, color=color, lw=2.0)
            ax.set_title(r.controller, color=th["text"], fontsize=11, fontweight="bold", y=0.93)
        return []

    anim = FuncAnimation(fig, draw, frames=int(seconds * fps), blit=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    anim.save(out_dir / f"{name}_{theme}.gif", writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def make_side_gif(results, name, out_dir, theme, seconds=3.0, fps=25):
    """Side view (x-z), one strip per controller. Suits a pure x-motion like long_step."""
    th = _apply(theme)
    runs = [results[(name, c)] for c in CONTROLLERS if (name, c) in results]
    target = runs[0].ref_pos[-1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 3.9))
    axes = axes.ravel()
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.12, hspace=0.35, wspace=0.12)
    title = fig.suptitle("", color=th["text"], fontsize=11)

    def draw(frame):
        t_now = frame / fps
        title.set_text(f"3 m step    t = {t_now:.2f} s")
        for ax, r in zip(axes, runs):
            ax.cla()
            color = th["series"][r.controller]
            ax.set_xlim(-0.4, target[0] + 0.5)
            ax.set_ylim(-0.05, 1.45)
            ax.set_aspect("equal")
            ax.set_yticks([0, 1])
            ax.axhline(0, color=th["muted"], lw=1.2)
            ax.scatter([target[0]], [target[2]], color=th["muted"], marker="x", s=30, zorder=3)
            i = min(np.searchsorted(r.t, t_now), len(r.t) - 1)
            ax.plot(r.x[: i + 1, 0], r.x[: i + 1, 2], color=color, lw=1.2, alpha=0.7)
            R = quat_to_rotmat(r.x[i, Q])
            c = r.x[i, P]
            arm = 0.16 * R[:, 0]
            tip = c + 0.16 * R[:, 2]
            ax.plot([c[0] - arm[0], c[0] + arm[0]], [c[2] - arm[2], c[2] + arm[2]], color=th["text"], lw=2.4)
            ax.plot([c[0], tip[0]], [c[2], tip[2]], color=color, lw=2.0)
            tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
            crashed = r.crashed and t_now >= r.t[-1]
            status = "CRASHED" if crashed else f"tilt {tilt:4.0f} deg"
            ax.text(0.01, 0.92, f"{r.controller}", transform=ax.transAxes, color=th["text"],
                    fontsize=10, fontweight="bold", va="top")
            ax.text(0.99, 0.92, status, transform=ax.transAxes, color=th["text"], fontsize=9.5,
                    ha="right", va="top", fontweight="bold" if crashed else "normal")
            ax.set_xticks([0, 1, 2, 3])
        for ax in axes[::2]:
            ax.set_ylabel("z (m)")
        for ax in axes[2:]:
            ax.set_xlabel("x (m)")
        return []

    anim = FuncAnimation(fig, draw, frames=int(seconds * fps), blit=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    anim.save(out_dir / f"{name}_{theme}.gif", writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def make_all(results, rows, out_dir: Path):
    for theme in THEMES:
        for name in SCENARIOS:
            if any(s == name for s, _ in results):
                plot_scenario(results, name, out_dir, theme)
        plot_summary(rows, out_dir, theme)
        if all(("flip_recovery", c) in results for c in CONTROLLERS):
            make_gif(results, "flip_recovery", out_dir, theme)
        if all(("long_step", c) in results for c in CONTROLLERS):
            make_side_gif(results, "long_step", out_dir, theme)
    print(f"figures written to {out_dir}")
