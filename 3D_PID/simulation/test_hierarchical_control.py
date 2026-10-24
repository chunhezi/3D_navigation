import json
import unittest

import numpy as np

from design_hierarchical_controller import design_gains
from nominal_position_model import build_position_state_space
from position_pid_sim import HERE, load_leaf, simulate

module = load_leaf("_test_hierarchical_controller", "core/controllers/hierarchical_learning_controller.py")


class HierarchicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gains = design_gains()

    def controller(self, **kwargs):
        return module.HierarchicalLearningController(4.571e-4, self.gains, **kwargs)

    def test_sdp_certificate_has_strict_negative_inequalities(self):
        r = self.gains["residuals"]
        self.assertLess(r["kyp_equality_norm"], 1e-6)
        for k in ("kyp_lyapunov_max_eigenvalue", "eq25_max_eigenvalue", "eq47_max_eigenvalue"):
            self.assertLess(r[k], 0.)
        self.assertFalse(self.gains["full_inertial_plant_certified"])

    def test_second_order_position_model_has_zero_high_frequency_force_coefficient(self):
        model = build_position_state_space(3.927e-6, 4.571e-4, 0.15)
        np.testing.assert_array_equal(model.C @ model.B, np.zeros((3, 3)))
        self.assertGreater(np.linalg.norm(model.B), 0)

    def test_paper_gradients_match_frozen_error_surrogate_for_one_hidden_neuron(self):
        error = np.array([0.1, -0.2, 0.05])
        s = np.diag([-0.2, -0.3, -0.1])
        w1 = np.array([[0.2, -0.1, 0.3]])
        w2 = np.array([[0.1], [0.2], [-0.3]])
        k1, k2 = module.refined_gradients(error, s, w1, w2)
        base_phi = w2 @ np.tanh(w1 @ error)

        def loss(a, b):
            residual = error + s @ (b @ np.tanh(a @ error) - base_phi)
            return 0.5 * residual @ residual

        for weights, analytical, first in ((w1, k1, True), (w2, k2, False)):
            numerical = np.zeros_like(weights)
            for index in np.ndindex(weights.shape):
                plus, minus = weights.copy(), weights.copy()
                plus[index] += 1e-6
                minus[index] -= 1e-6
                numerical[index] = ((loss(plus, w2) - loss(minus, w2)) if first
                                    else (loss(w1, plus) - loss(w1, minus))) / 2e-6
            np.testing.assert_allclose(analytical, numerical, atol=1e-10, rtol=1e-5)

    def test_observer_does_not_invent_disturbance_on_nominal_position_ramp(self):
        controller = self.controller(refined=False)
        y = np.zeros(3)
        for _ in range(100):
            result = controller.step(y, np.array([0.2, -0.1, 0.3]), 0.15)
            np.testing.assert_allclose(result.residual_cm, 0., atol=1e-12)
            y += result.control_force_n / controller.force_scale * 0.15

    def test_learning_compensates_positive_constant_disturbance_and_reset_repeats(self):
        def run(controller):
            y = np.zeros(3)
            for _ in range(400):
                result = controller.step(y, np.zeros(3), 0.15)
                y += (result.control_force_n / controller.force_scale + np.array([0.2, 0., 0.])) * 0.15
            return y, result
        controller = self.controller()
        first, result = run(controller)
        self.assertGreater(result.compensation_force_n[0], 0.)
        # Same observer + same feedback without EID has y_ss =
        # d*(1/|Kf| + 1/L). Comparing to measured-state P instead would change
        # the architecture as well as disturbance compensation.
        uncompensated_error = 0.2 * (1 / abs(self.gains["kf"]) + 1 / self.gains["observer_gain"])
        self.assertLess(abs(first[0]), uncompensated_error)
        self.assertLessEqual(result.w2_norm, controller.weight_bound + 1e-12)
        controller.reset()
        again, _ = run(controller)
        np.testing.assert_array_equal(first, again)

    def test_sampling_weight_projection_prevents_boundary_overshoot(self):
        controller = self.controller(eta2=1e8, refined_start_s=0., weight_bound=0.2)
        controller.xhat = np.zeros(3)
        controller.time_s = 1.
        controller.last = (np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3))
        controller._advance(np.zeros(3), 0.15)
        self.assertLessEqual(np.linalg.norm(controller.w2, 2), 0.2 + 1e-12)
        self.assertGreater(controller.projection_count, 0)

    def test_same_plant_reference_and_disturbance_for_both_controllers(self):
        cfg = json.loads((HERE / "config.json").read_text())
        cfg.update(duration_s=12., disturbance_start_s=0.)
        original = simulate(cfg, disturbed=True)
        adapted = simulate(cfg, disturbed=True, controller_factory=lambda m, c, dt:
                           module.HierarchicalLearningController(c, self.gains))
        np.testing.assert_array_equal(original["target_cm"], adapted["target_cm"])
        np.testing.assert_array_equal(original["forces"]["disturbance"], adapted["forces"]["disturbance"])
        np.testing.assert_array_equal(original["position_cm"][0], adapted["position_cm"][0])
        self.assertEqual(original["parameters"]["mass_kg"], adapted["parameters"]["mass_kg"])


if __name__ == "__main__":
    unittest.main()
