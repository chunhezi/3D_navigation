"""Numerical model and scenario-isolation checks; no hardware required."""
import json
import unittest

import numpy as np

from position_pid_sim import HERE, gravity, pid, plant_step, simulate


class SimulationTests(unittest.TestCase):
    def test_equilibrium_fixed_target_stays_at_rest_without_changing_live_parameters(self):
        config = json.loads((HERE / "config.json").read_text())
        config.update(reference="fixed", plant_mode="equilibrium")
        before = gravity.gravity_parameter_snapshot()
        run = simulate(config)
        np.testing.assert_allclose(run["error_cm"], 0., atol=1e-12)
        np.testing.assert_allclose(run["velocity_m_s"], 0., atol=1e-12)
        self.assertEqual(gravity.gravity_parameter_snapshot(), before)
        self.assertFalse(run["parameters"]["model_identified"])

    def test_original_offset_is_explained_by_ideal_force_model_assumption(self):
        config = json.loads((HERE / "config.json").read_text())
        config.update(reference="fixed", plant_mode="ideal_force")
        run = simulate(config)
        params = run["parameters"]
        surplus = (params["gravity"]["GRAVITY_COMPENSATION_N"]
                   - params["gravity"]["EFFECTIVE_GRAVITY_MINUS_BUOYANCY_N"])
        # With integral separation, the initial zero-error sample integrates
        # zero, then the large jump freezes Z integration at zero.
        self.assertAlmostEqual(run["error_cm"][-1, 2], -surplus / pid.KP_Z, places=6)
        self.assertAlmostEqual(params["gravity"]["GRAVITY_COMPENSATION_FACTOR"], 1.5)

    def test_constant_acceleration_without_drag(self):
        p, v = plant_step(np.zeros(3), np.ones(3), np.array([2., 0., -2.]), 2., 0., 0.5)
        np.testing.assert_allclose(p, [0.625, 0.5, 0.375])
        np.testing.assert_allclose(v, [1.5, 1., 0.5])

    def test_exact_drag_solution_is_independent_of_subdivision(self):
        args = (np.array([0., 1., 2.]), np.array([1., -1., 0.]), np.array([1e-6, 0., -1e-6]))
        p, v = plant_step(*args, 4e-6, 4.5e-4, 0.15)
        ps, vs = args[:2]
        for _ in range(150):
            ps, vs = plant_step(ps, vs, args[2], 4e-6, 4.5e-4, 0.001)
        np.testing.assert_allclose(p, ps, atol=1e-13)
        np.testing.assert_allclose(v, vs, atol=1e-13)

    def test_terminal_velocity_equilibrium(self):
        vel = np.array([0.01, -0.02, 0.03])
        p, v = plant_step(np.zeros(3), vel, 4.5e-4 * vel, 4e-6, 4.5e-4, 0.15)
        np.testing.assert_allclose(p, vel * 0.15)
        np.testing.assert_allclose(v, vel)

    def test_reference_modes_reset_between_runs(self):
        config = json.loads((HERE / "config.json").read_text())
        config.update(duration_s=3., disturbance_start_s=0., pump_weak_s=0.3, pump_strong_s=0.3)
        for mode in ("fixed", "step", "helix"):
            with self.subTest(mode=mode):
                config["reference"] = mode
                first = simulate(config)
                noisy = simulate(config, disturbed=True)
                again = simulate(config)
                np.testing.assert_array_equal(first["position_cm"], again["position_cm"])
                np.testing.assert_array_equal(first["target_cm"], noisy["target_cm"])
                self.assertGreater(np.max(np.abs(noisy["position_cm"] - first["position_cm"])), 0)
                config["disturbance_force_n"] = [0., 0., 0.]
                zero = simulate(config, disturbed=True)
                np.testing.assert_array_equal(first["position_cm"], zero["position_cm"])
                config["disturbance_force_n"] = [2e-6, -1e-6, 0.]


if __name__ == "__main__":
    unittest.main()
