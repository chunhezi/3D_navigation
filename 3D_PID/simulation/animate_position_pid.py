"""Animate saved simulation CSV files without rerunning the controller."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "output")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--speed", type=float, default=7.5, help="Simulation seconds per playback second")
    parser.add_argument("--cases", nargs="+", default=["baseline", "demo_disturbance"])
    args = parser.parse_args()
    if args.fps <= 0 or not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("fps and speed must be positive and finite")
    output = args.input.resolve()
    if not 1 <= len(args.cases) <= 4 or any(Path(name).name != name for name in args.cases):
        parser.error("Provide one to four scenario names without directory components")
    summary = json.loads((output / "summary.json").read_text())
    mode = summary["parameters"].get("plant_mode", "ideal_force")
    runs = [np.loadtxt(output / f"{name}_states.csv", delimiter=",", skiprows=1, ndmin=2) for name in args.cases]
    baseline = runs[0]
    if any(baseline.shape != run.shape or not np.array_equal(baseline[:, 0], run[:, 0]) for run in runs):
        parser.error("Scenario CSV files must have matching timestamps and shapes")
    if len(baseline) < 2 or not np.all(np.isfinite(runs)):
        parser.error("CSV files must contain at least two finite samples")
    t = baseline[:, 0]
    target = baseline[:, 4:7]
    paths = [target, *(run[:, 1:4] for run in runs)]
    colors = ("#303846", "#1879c9", "#ed8a24", "#31956a", "#9056aa")[:len(paths)]
    names = {"baseline": "PI: no disturbance", "demo_disturbance": "PI: demo disturbance",
             "original_pi": "Original PI", "paper_rough": "Paper: rough only",
             "paper_hierarchical": "Paper: hierarchical (adapted)"}
    labels = ["Target", *(names.get(name, name) for name in args.cases)]
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=(10, 5.5), facecolor="white")
    ax = fig.add_subplot(121, projection="3d")
    error_ax = fig.add_subplot(122)
    ax.plot(*target.T, color=colors[0], linestyle="--", alpha=0.3, label="Full reference")
    all_points = np.concatenate(paths)
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    spans = np.maximum(upper - lower, 0.3)
    for setter, lo, hi, span in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lower, upper, spans):
        setter(lo - span * 0.08, hi + span * 0.08)
    ax.set_box_aspect(spans)
    ax.set(xlabel="X (cm)", ylabel="Y (cm)", zlabel="Z (cm)")
    ax.view_init(elev=23, azim=-58)
    trails, markers = [], []
    for color, label in zip(colors, labels):
        trails.append(ax.plot([], [], [], color=color, linewidth=1.8, label=label)[0])
        markers.append(ax.plot([], [], [], "o", color=color, markersize=6)[0])
    ax.legend(loc="upper left", fontsize=7)
    errors = [np.linalg.norm(run[:, 7:10], axis=1) for run in runs]
    error_lines = [error_ax.plot([], [], color=color, label=label)[0]
                   for color, label in zip(colors[1:], labels[1:])]
    error_ax.set(xlim=(t[0], t[-1]), ylim=(0, max(0.1, np.max(errors) * 1.15)),
                 xlabel="Simulation time (s)", ylabel="3D position error (cm)")
    error_ax.grid(alpha=0.25)
    error_ax.legend(fontsize=8, loc="upper right")
    clock_line = error_ax.axvline(t[0], color="gray", alpha=0.5)
    title = fig.suptitle("Position control comparison", fontsize=13)
    fig.text(0.5, 0.025, f"{mode} model (not identified) | Same simulation inputs | {args.speed:g}x playback",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    frame_count = max(2, round((t[-1] - t[0]) / args.speed * args.fps) + 1)
    indices = np.unique(np.rint(np.linspace(0, len(t) - 1, frame_count)).astype(int))

    def update(k):
        for path, trail, marker in zip(paths, trails, markers):
            trail.set_data_3d(*path[:k + 1].T)
            marker.set_data_3d(*path[k:k + 1].T)
        for line, error in zip(error_lines, errors):
            line.set_data(t[:k + 1], error[:k + 1])
        clock_line.set_xdata([t[k], t[k]])
        title.set_text(f"Position control comparison | t = {t[k]:5.1f} / {t[-1]:.1f} s")
        return *trails, *markers, *error_lines, clock_line, title

    animation = FuncAnimation(fig, update, frames=indices, interval=1000 / args.fps, blit=False)
    path = output / "trajectory.gif"
    animation.save(path, writer=PillowWriter(fps=args.fps), dpi=100)
    plt.close(fig)
    print(f"Saved {path} ({len(indices)} frames, {len(indices) / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
