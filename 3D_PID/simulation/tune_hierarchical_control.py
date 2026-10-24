"""Reproducible nominal-simulation parameter search; never edits live gains."""
import copy
import itertools
import json

import numpy as np

from compare_hierarchical_control import paper
from design_hierarchical_controller import design_gains
from position_pid_sim import HERE, gravity, simulate


def metrics(run, config):
    e = np.linalg.norm(run["error_cm"], axis=1)
    mask = run["time_s"] >= config["disturbance_start_s"]
    inc = run["forces"]["command"] - gravity.get_gravity_compensation_force()
    return dict(rmse_cm=float(np.sqrt(np.mean(e**2))),
                post_disturbance_rmse_cm=float(np.sqrt(np.mean(e[mask]**2))),
                peak_incremental_force_n=float(np.linalg.norm(inc, axis=1).max()))


def evaluate(config, learning, gains):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return simulate(config, disturbed=True, controller_factory=lambda m, c, dt:
                        paper.HierarchicalLearningController(c, gains, **learning["online_learning"]))


def main():
    config = json.loads((HERE / "config.json").read_text())
    initial = json.loads((HERE / "hierarchical_config.before_tuning.json").read_text())
    output = HERE / "output" / "tuning"
    output.mkdir(parents=True, exist_ok=True)
    baseline = metrics(simulate(config, disturbed=True), config)
    before = metrics(evaluate(config, initial, design_gains(**initial["offline_design"])), config)
    candidates, rejected = [], []
    budget = 3 * baseline["peak_incremental_force_n"]

    def assess(learning, gains, phase):
        try:
            run = evaluate(config, learning, gains)
            values = metrics(run, config)
            snapshot = run["parameters"]["controller"]
            if values["peak_incremental_force_n"] > budget:
                raise ValueError("exceeds 3x original PI peak incremental force budget")
            if snapshot["discrete_bound_projection_count"]:
                raise ValueError("weight projection was triggered")
            score = 0.5 * values["rmse_cm"] / baseline["rmse_cm"] + 0.5 * values["post_disturbance_rmse_cm"] / baseline["post_disturbance_rmse_cm"]
            row = dict(score=score, metrics=values, learning=copy.deepcopy(learning), phase=phase)
            candidates.append(row)
            return row
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            rejected.append(dict(learning=copy.deepcopy(learning), reason=str(exc), phase=phase))

    grid = itertools.product((4., 8., 12.), (2., 3.5, 5.), (1., 4.))
    for index, (l, k, a) in enumerate(grid):
        offline = dict(observer_gain=l, max_feedback_gain=k, filter_rate=a, gamma=1.)
        try:
            gains = design_gains(**offline)
        except (RuntimeError, ValueError) as exc:
            rejected.append(dict(offline_design=offline, reason=str(exc)))
            continue
        for gamma in (300., 3000., 30000.):
            learning = copy.deepcopy(initial)
            learning["offline_design"] = offline
            learning["online_learning"]["gamma_learning"] = gamma
            assess(learning, gains, "coarse")
        print(f"Offline design {index + 1}/18; accepted={len(candidates)}", flush=True)
    if not candidates:
        raise RuntimeError("No candidate satisfies numerical and force constraints")
    finalists = sorted(candidates, key=lambda row: row["score"])[:3]
    for row in finalists:
        for phi, eta in itertools.product((0.3, 1., 3.), (5., 100., 2000.)):
            learning = copy.deepcopy(row["learning"])
            learning["online_learning"].update(phi=phi, eta1=eta, eta2=eta)
            assess(learning, design_gains(**learning["offline_design"]), "fine")
    (output / "search_candidates.json").write_text(json.dumps(dict(candidates=candidates, rejected=rejected), indent=2) + "\n")
    # Additional scenarios reject numerical failures; their error performance
    # is reported, not added to the training score. This is a validation screen,
    # not an independent claim of generalization.
    validation_configs = []
    for label, changes in (("half_disturbance", dict(disturbance_force_n=(np.array(config["disturbance_force_n"]) * 0.5).tolist())),
                           ("1.5x_disturbance", dict(disturbance_force_n=(np.array(config["disturbance_force_n"]) * 1.5).tolist())),
                           ("fixed", dict(reference="fixed")),
                           ("step_300s", dict(reference="step", duration_s=300.))):
        cfg = dict(config, **changes)
        original = metrics(simulate(cfg, disturbed=True), cfg)
        validation_configs.append((label, cfg, original))
    validation_rejected = []
    best, scenarios = None, []
    for candidate in sorted(candidates, key=lambda row: row["score"]):
        scenarios = []
        try:
            gains = design_gains(**candidate["learning"]["offline_design"])
            for label, cfg, original in validation_configs:
                run = evaluate(cfg, candidate["learning"], gains)
                projections = run["parameters"]["controller"]["discrete_bound_projection_count"]
                if projections or np.linalg.norm(run["error_cm"], axis=1).max() > 5.:
                    raise RuntimeError(f"Numerical screening failed for {label}")
                scenarios.append(dict(name=label, original_pi=original, selected=metrics(run, cfg),
                                      weight_projection_count=projections))
            best = candidate
            break
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            validation_rejected.append(dict(learning=candidate["learning"], reason=str(exc)))
    if best is None:
        raise RuntimeError("No candidate passes the additional numerical validation scenarios")
    report = dict(training_config=config, original_config=initial, original_pi=baseline,
                  previous_paper_parameters=before, force_budget_n=budget,
                  objective="0.5*(whole-run RMSE / PI RMSE) + 0.5*(post-disturbance RMSE / PI RMSE)",
                  selected=best, validation=scenarios, candidates=candidates, rejected=rejected,
                  validation_rejected=validation_rejected,
                  caveat="Nominal simulation tuning only; no hardware or sampled-data stability certification")
    (output / "search_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "selected_config.json").write_text(json.dumps(best["learning"], indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("original_pi", "previous_paper_parameters", "selected", "validation")}, indent=2))


if __name__ == "__main__":
    main()
