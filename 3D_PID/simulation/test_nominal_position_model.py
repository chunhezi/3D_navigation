import unittest

import numpy as np

from nominal_position_model import build_position_state_space
from position_pid_sim import plant_step


class NominalPositionModelTests(unittest.TestCase):
    def test_discrete_model_matches_existing_simulation_plant(self):
        state = np.array([0.002, -0.001, 0.005, 0.001, -0.003, 0.002])
        force = np.array([2e-6, -1e-6, 0.5e-6])
        for drag in (0., 4.571e-4):
            with self.subTest(drag=drag):
                model = build_position_state_space(3.927e-6, drag, 0.15)
                p, v = plant_step(state[:3], state[3:], force, 3.927e-6, drag, 0.15)
                np.testing.assert_allclose(model.Ad @ state + model.Bd @ force, np.r_[p, v], atol=1e-14)
                np.testing.assert_array_equal(model.C @ state, state[:3])

    def test_position_measurement_makes_nominal_state_observable(self):
        model = build_position_state_space(3.927e-6, 4.571e-4, 0.15)
        self.assertEqual(np.linalg.matrix_rank(np.vstack((model.C, model.C @ model.Ad))), 6)

    def test_small_drag_approaches_constant_acceleration(self):
        model = build_position_state_space(1., 1e-12, 0.1)
        free = build_position_state_space(1., 0., 0.1)
        np.testing.assert_allclose(model.Ad, free.Ad, atol=1e-12)
        np.testing.assert_allclose(model.Bd, free.Bd, atol=1e-12)

    def test_invalid_parameters_rejected(self):
        for params in ((0, 1, 1), (1, -1, 1), (1, 1, 0), (float("nan"), 1, 1)):
            with self.subTest(params=params), self.assertRaises(ValueError):
                build_position_state_space(*params)


if __name__ == "__main__":
    unittest.main()
