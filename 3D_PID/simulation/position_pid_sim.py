"""Offline position PI baseline. No cameras, serial devices or GUI imports."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def load_leaf(name, relative_path):
    """Load independent project modules without core/__init__ side effects."""
    spec = importlib.util.spec_from_file_location(name, PROJECT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through this entry
    spec.loader.exec_module(module)
    return module


pid = load_leaf("_position_sim_pid", "core/controllers/pid_3d_controller.py")
reference = load_leaf("_position_sim_reference", "core/reference_generator_3d.py")
gravity = load_leaf("_position_sim_gravity", "core/gravity_compensation.py")


def validate(config):
    if config.get("plant_mode", "equilibrium") not in ("equilibrium", "ideal_force"):
        raise ValueError("plant_mode must be equilibrium or ideal_force")
    for key in ("duration_s", "dt_s", "pump_weak_s", "pump_strong_s"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("dynamic_viscosity_pa_s", "disturbance_start_s"):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("initial_position_cm", "initial_velocity_m_s", "disturbance_force_n"):
        value = np.asarray(config[key], dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must be a finite three-element vector")
    if config["reference"] not in ("project", "fixed", "step", "helix"):
        raise ValueError("Invalid reference mode")
    steps = config["duration_s"] / config["dt_s"]
    if steps < 1 or not np.isclose(steps, round(steps), rtol=0, atol=1e-8):
        raise ValueError("duration_s must be an integer multiple of dt_s")
    if min(int(config[k] / config["dt_s"]) for k in ("pump_weak_s", "pump_strong_s")) < 1:
        raise ValueError("Pump phases must each span at least one sample")


def plant_step(position_m, velocity_m_s, force_n, mass_kg, drag_n_s_m, dt_s):
    """Exact ZOH solution of m*p'' = force - c*p' (force excludes drag)."""
    if mass_kg <= 0 or drag_n_s_m < 0 or dt_s <= 0:
        raise ValueError("Invalid mass, drag or timestep")
    if drag_n_s_m == 0:
        accel = force_n / mass_kg
        return (position_m + velocity_m_s * dt_s + 0.5 * accel * dt_s**2,
                velocity_m_s + accel * dt_s)
    one_minus_decay = -np.expm1(-drag_n_s_m * dt_s / mass_kg)
    velocity_integral = mass_kg / drag_n_s_m * one_minus_decay
    terminal_velocity = force_n / drag_n_s_m
    return (position_m + velocity_integral * velocity_m_s
            + terminal_velocity * (dt_s - velocity_integral),
            (1 - one_minus_decay) * velocity_m_s + terminal_velocity * one_minus_decay)


def simulate(config, disturbed=False, controller_factory=None):
    validate(config)
    dt = config["dt_s"]
    count = round(config["duration_s"] / dt)
    params = gravity.gravity_parameter_snapshot()
    mass = params["BEAD_VOLUME_M3"] * params["OBJECT_DENSITY_KG_M3"]
    drag = 3 * np.pi * config["dynamic_viscosity_pa_s"] * params["BEAD_DIAMETER_M"]
    weight = np.array([0., 0., params["EFFECTIVE_GRAVITY_MINUS_BUOYANCY_N"]])
    feedforward = gravity.get_gravity_compensation_force()
    mode = config.get("plant_mode", "equilibrium")
    # Equilibrium mode uses commanded-force coordinates around a presumed
    # successful hover. This is an ASSUMPTION, not an identified actuator model.
    # It does not claim that physical gravity equals the empirical feedforward.
    equilibrium_command = feedforward.copy() if mode == "equilibrium" else weight.copy()
    pos = np.zeros((count + 1, 3))
    vel = np.zeros_like(pos)
    target = np.zeros_like(pos)
    force_names = (("p", "i", "command", "disturbance") if controller_factory is None
                   else ("feedback", "compensation", "command", "disturbance"))
    forces = {name: np.zeros((count, 3)) for name in force_names}
    learning_diagnostics = []
    pos[0] = np.asarray(config["initial_position_cm"]) / 100
    vel[0] = config["initial_velocity_m_s"]
    controller = (pid.PID3DController() if controller_factory is None
                  else controller_factory(mass, drag, dt))
    weak = int(config["pump_weak_s"] / dt)
    strong = int(config["pump_strong_s"] / dt)
    old_mode = reference.REFERENCE_MODE_3D
    try:
        if config["reference"] != "project":
            reference.REFERENCE_MODE_3D = config["reference"]
        reference.reset_reference_3d()
        # Follow live ordering: advance reference on a valid observation, then PI.
        with contextlib.redirect_stdout(io.StringIO()):
            reference.get_reference_3d(advance=False)
            for k in range(count):
                target[k], _, _ = reference.get_reference_3d(advance=True, dt_s=dt)
                result = controller.step(pos[k] * 100, target[k], dt)
                if controller_factory is None:
                    forces["p"][k] = result.p_force_xyz
                    forces["i"][k] = result.i_force_xyz
                    forces["command"][k] = result.pid_force_xyz + feedforward
                else:
                    forces["feedback"][k] = result.feedback_force_n
                    forces["compensation"][k] = result.compensation_force_n
                    forces["command"][k] = result.control_force_n + feedforward
                    learning_diagnostics.append([k * dt, *result.residual_cm, result.sigma_norm,
                                                 result.w1_norm, result.w2_norm, float(result.refined_active)])
                # Demo external force, NOT a calibration of pump speed/voltage.
                if (disturbed and k * dt >= config["disturbance_start_s"]
                        and k % (weak + strong) >= weak):
                    forces["disturbance"][k] = config["disturbance_force_n"]
                net = forces["command"][k] - equilibrium_command + forces["disturbance"][k]
                pos[k + 1], vel[k + 1] = plant_step(pos[k], vel[k], net, mass, drag, dt)
            target[-1], _, _ = reference.get_reference_3d(advance=False)
        ref_params = reference.reference_parameter_snapshot_3d()
        ref_params.update(FIXED_TARGET_3D_CM=reference.FIXED_TARGET_3D_CM.tolist(),
                          STEP_WAYPOINTS_3D_CM=reference.STEP_WAYPOINTS_3D_CM.tolist())
    finally:
        reference.REFERENCE_MODE_3D = old_mode
        reference.reset_reference_3d()
    position_cm = pos * 100
    error = target - position_cm
    if not all(np.all(np.isfinite(a)) for a in (pos, vel, error, *forces.values())):
        raise RuntimeError("Simulation produced non-finite values")
    return dict(time_s=np.arange(count + 1) * dt, position_cm=position_cm,
                velocity_m_s=vel, target_cm=target, error_cm=error, forces=forces,
                learning_diagnostics=np.asarray(learning_diagnostics),
                metrics=dict(rmse_xyz_cm=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                             rmse_3d_cm=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                             max_error_3d_cm=float(np.linalg.norm(error, axis=1).max()),
                             final_error_xyz_cm=error[-1].tolist()),
                parameters=dict(mass_kg=mass, drag_n_s_m=drag, gravity=params,
                                plant_mode=mode, model_identified=False,
                                equilibrium_command_n=equilibrium_command.tolist(),
                                physical_weight_minus_buoyancy_n=weight.tolist(),
                                incremental_force_gain_assumption=[1., 1., 1.],
                                controller=(pid.controller_parameter_snapshot() if controller_factory is None
                                            else controller.parameter_snapshot()), reference=ref_params,
                                weak_samples=weak, strong_samples=strong))


def save_results(results, config, output):
    output.mkdir(parents=True, exist_ok=True)
    summary = {"config": config, "parameters": next(iter(results.values()))["parameters"],
               "controller_parameters": {name: run["parameters"]["controller"] for name, run in results.items()},
               "metrics": {name: run["metrics"] for name, run in results.items()},
               "sampling": "Errors include initial and terminal states; terminal target is held. "
                           "Force CSV has one row per held control interval."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, run in results.items():
        np.savetxt(output / f"{name}_states.csv", np.column_stack([
            run["time_s"], run["position_cm"], run["target_cm"], run["error_cm"], run["velocity_m_s"]]),
            delimiter=",", comments="", header="time_s,x_cm,y_cm,z_cm,ref_x_cm,ref_y_cm,ref_z_cm,"
            "error_x_cm,error_y_cm,error_z_cm,vx_m_s,vy_m_s,vz_m_s")
        header = "time_s," + ",".join(f"{part}_{axis}_n" for part in run["forces"] for axis in "xyz")
        np.savetxt(output / f"{name}_forces.csv", np.column_stack([
            run["time_s"][:-1], *run["forces"].values()]), delimiter=",", comments="", header=header)
        if run["learning_diagnostics"].size:
            np.savetxt(output / f"{name}_learning.csv", run["learning_diagnostics"], delimiter=",", comments="",
                       header="time_s,residual_x_cm,residual_y_cm,residual_z_cm,sigma_norm,w1_norm,w2_norm,refined_active")


def plot_results(results, output, title=None):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = next(iter(results.values()))
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot(*first["target_cm"].T, "k--", label="Reference", linewidth=1.5)
    for name, run in results.items():
        ax.plot(*run["position_cm"].T, label=name, alpha=0.85)
    mode = first["parameters"]["plant_mode"]
    ax.set(xlabel="X (cm)", ylabel="Y (cm)", zlabel="Z (cm)",
           title=(f"Controller comparison: {mode} model" if title else f"Position PI: {mode} model (not identified)"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "trajectory.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for axis, letter in enumerate("XYZ"):
        disturbance_drawn = False
        axes[axis, 0].plot(first["time_s"], first["target_cm"][:, axis], "k--", label="Reference")
        for case_index, (name, run) in enumerate(results.items()):
            t = run["time_s"]
            color = f"C{case_index}"
            axes[axis, 0].plot(t, run["position_cm"][:, axis], label=name, color=color)
            axes[axis, 1].plot(t, run["error_cm"][:, axis], label=name, color=color)
            axes[axis, 2].step(t[:-1], run["forces"]["command"][:, axis] * 1e6,
                               where="post", label=f"{name}: command", color=color)
            if not disturbance_drawn and np.any(run["forces"]["disturbance"][:, axis]):
                axes[axis, 2].step(t[:-1], run["forces"]["disturbance"][:, axis] * 1e6,
                                   where="post", linestyle=":", label="External disturbance", color="black")
                disturbance_drawn = True
        for col, unit in enumerate(("position (cm)", "error (cm)", "force (uN)")):
            axes[axis, col].set_ylabel(f"{letter} {unit}")
            axes[axis, col].grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle(title or f"Original project PI | {mode} model (not identified) | no gain retuning")
    fig.tight_layout()
    fig.savefig(output / "response.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--reference", choices=("project", "fixed", "step", "helix"))
    parser.add_argument("--duration", type=float)
    parser.add_argument("--plant-mode", choices=("equilibrium", "ideal_force"))
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.reference is not None:
        config["reference"] = args.reference
    if args.duration is not None:
        config["duration_s"] = args.duration
    if args.plant_mode is not None:
        config["plant_mode"] = args.plant_mode
    try:
        results = {"baseline": simulate(config), "demo_disturbance": simulate(config, disturbed=True)}
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    save_results(results, config, output)
    if not args.no_plots:
        plot_results(results, output)
    for name, run in results.items():
        print(f"{name}: 3D RMSE = {run['metrics']['rmse_3d_cm']:.6f} cm; "
              f"final error XYZ = {np.round(run['error_cm'][-1], 6)} cm")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
