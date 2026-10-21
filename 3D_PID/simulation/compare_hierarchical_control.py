"""Original PI vs paper DRNN-EID adaptation, with identical disturbances."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from design_hierarchical_controller import design_gains
from position_pid_sim import HERE, PROJECT, load_leaf, plot_results, save_results, simulate

paper = load_leaf("_position_sim_hierarchical", "core/controllers/hierarchical_learning_controller.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--learning-config", type=Path, default=HERE / "hierarchical_config.json")
    parser.add_argument("--output", type=Path, default=HERE / "output" / "comparison")
    parser.add_argument("--reference", choices=("fixed", "step", "helix"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    learning = json.loads(args.learning_config.read_text())
    if args.reference:
        config["reference"] = args.reference
    gains = design_gains(**learning["offline_design"])

    def factory(refined):
        def create(mass, drag, dt):
            # Reduced nominal design; the plant still uses its original mass.
            return paper.HierarchicalLearningController(drag, gains, refined=refined, **learning["online_learning"])
        return create

    results = {"original_pi": simulate(config, disturbed=True),
               "paper_rough": simulate(config, disturbed=True, controller_factory=factory(False)),
               "paper_hierarchical": simulate(config, disturbed=True, controller_factory=factory(True))}
    original = results["original_pi"]
    for name, run in results.items():
        for key in ("time_s", "target_cm"):
            if not np.array_equal(original[key], run[key]):
                raise RuntimeError(f"Unfair comparison: different {key} for {name}")
        if not np.array_equal(original["forces"]["disturbance"], run["forces"]["disturbance"]):
            raise RuntimeError("Unfair comparison: different disturbances")
        error_norm = np.linalg.norm(run["error_cm"], axis=1)
        active = run["time_s"] >= config["disturbance_start_s"]
        if np.any(active):
            run["metrics"]["after_disturbance_rmse_3d_cm"] = float(np.sqrt(np.mean(error_norm[active]**2)))
    output = args.output.resolve()
    save_results(results, config, output)
    (output / "offline_design.json").write_text(json.dumps(gains, indent=2, allow_nan=False) + "\n")
    provenance = dict(learning_config=learning, source="User-provided paper PDF",
                      paper_path="1-s2.0-S0005109825006429-main.pdf",
                      implementation="Overdamped nominal design evaluated on the original inertial plant",
                      full_plant_certified=False)
    pdf = PROJECT.parent / provenance["paper_path"]
    if pdf.is_file():
        provenance["paper_sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
    (output / "paper_implementation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if not args.no_plots:
        plot_results(results, output, title="Same disturbance: original PI vs paper DRNN-EID adaptation (uncertified full plant)")
    for name, run in results.items():
        print(f"{name}: RMSE 3D={run['metrics']['rmse_3d_cm']:.6f} cm; "
              f"after disturbance={run['metrics'].get('after_disturbance_rmse_3d_cm', float('nan')):.6f} cm")
    print(f"SDP: {gains['status']}; full inertial plant certificate: False")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
