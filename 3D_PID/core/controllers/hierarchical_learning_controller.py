"""Yu--Wu--Liu (Automatica 185, 112744) position-loop adaptation.

Paper equations (4)--(7), (12), (22), (27), (34)--(36). Nominal
overdamped coordinates: y in cm, normalized input in cm/s, A=0, B=C=I.
Output force is N. This file is not activated by the hardware runtime.
The reduced-model continuous-time certificate does not certify the actual
inertial plant, sampling, trajectory tracking, or hardware.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class HierarchicalResult:
    feedback_force_n: np.ndarray
    compensation_force_n: np.ndarray  # estimated EID, subtracted from feedback
    control_force_n: np.ndarray
    residual_cm: np.ndarray
    sigma_norm: float
    w1_norm: float
    w2_norm: float
    refined_active: bool


def refined_gradients(residual, sensitivity, w1, w2):
    """Literal kappa_1,kappa_2 of (35)--(39), including printed Xi=I-h*h.T.

    With one hidden neuron (the default), Xi is exactly the derivative of tanh.
    For more neurons this outer-product expression differs from the usual
    diagonal tanh Jacobian; do not silently replace the paper expression.
    """
    h = np.tanh(w1 @ residual)
    xi = np.eye(h.size) - np.outer(h, h)
    output_gradient = sensitivity.T @ residual
    kappa1 = np.outer(xi.T @ w2.T @ output_gradient, residual)
    kappa2 = np.outer(output_gradient, h)
    return kappa1, kappa2


class HierarchicalLearningController:
    """Position-only observer, DRNN EID estimator and state feedback.

    No true velocity or true disturbance is accepted by the interface.
    The observer uses u_f (not the compensated u), as required by equation (4).
    Tracking extension: u_f=Kf*(x_hat-reference), without reference derivatives.
    """

    def __init__(self, drag_n_s_m, gains, *, refined=True, gamma_learning=300.,
                 phi=1., mu=0., eta1=5., eta2=5., refined_start_s=5.,
                 weight_bound=4., hidden_neurons=1, seed=2026):
        values = [drag_n_s_m, gamma_learning, phi, mu, eta1, eta2, refined_start_s, weight_bound]
        if (not np.all(np.isfinite(values)) or drag_n_s_m <= 0 or gamma_learning <= 0
                or min(phi, mu, eta1, eta2, refined_start_s) < 0 or weight_bound <= 0
                or int(hidden_neurons) != hidden_neurons or hidden_neurons < 1):
            raise ValueError("Invalid hierarchical controller parameters")
        self.force_scale = float(drag_n_s_m) / 100.  # (cm/s) -> N
        self.gains = gains
        self.l, self.a = gains["observer_gain"], gains["filter_rate"]
        self.kl, self.kh, self.kf = gains["kl"], gains["kh"], gains["kf"]
        if not np.all(np.isfinite([self.l, self.a, self.kl, self.kh, self.kf])) or min(self.l, self.a) <= 0:
            raise ValueError("Invalid offline gains")
        self.refined, self.gamma_learning, self.phi, self.mu = refined, gamma_learning, phi, mu
        self.eta1, self.eta2 = eta1, eta2
        self.refined_start_s, self.weight_bound = refined_start_s, weight_bound
        self.hidden_neurons, self.seed = int(hidden_neurons), int(seed)
        self.ap = np.array([[-self.l, -1.], [0., -self.a]])
        self.bf = np.array([[0.], [self.a * self.kl]])
        self.reset()

    def reset(self):
        self.xhat = None
        self.xf = np.zeros(3)
        self.sigma = np.zeros((3, 3))
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.uniform(-0.1, 0.1, (self.hidden_neurons, 3))
        self.w2 = rng.uniform(-0.1, 0.1, (3, self.hidden_neurons))
        if np.linalg.norm(self.w2, 2) >= self.weight_bound:
            raise ValueError("Initial output weight must lie inside weight_bound")
        self.time_s = 0.
        self.last = None
        self.projection_count = 0

    def sensitivity(self, time_s):
        """Theorem 2: Cpsi*(exp(Apsi*t)-I)*inv(Apsi)*Bf; three equal axes."""
        gain = ((expm(self.ap * time_s) - np.eye(2)) @ np.linalg.solve(self.ap, self.bf))[0, 0]
        return gain * np.eye(3)

    def _advance(self, position, dt):
        previous_position, residual, uf, phi_total = self.last
        decay_l = np.exp(-self.l * dt)
        z_l = -np.expm1(-self.l * dt) / self.l
        # Causal FOH on the two measured endpoints of the *past* interval;
        # held u_f. This avoids inventing a velocity measurement and preserves
        # the observer on an undisturbed linear position ramp.
        self.xhat = (decay_l * self.xhat + z_l * uf
                     + (1 - decay_l) * previous_position
                     + (1 - z_l / dt) * (position - previous_position))
        decay_a = np.exp(-self.a * dt)
        self.xf = decay_a * self.xf + (1 - decay_a) * self.kl * phi_total
        error_norm_mu = np.linalg.norm(residual)**self.mu
        decay_phi = np.exp(-self.phi * dt)
        z_phi = dt if self.phi == 0 else -np.expm1(-self.phi * dt) / self.phi
        # (27), Gamma=gamma_learning*I, Phi=phi*I; symmetric sigma.
        self.sigma = (decay_phi * self.sigma - z_phi * self.gamma_learning
                      * error_norm_mu * np.outer(residual, residual))
        if self.refined and self.time_s >= self.refined_start_s:
            k1, k2 = refined_gradients(residual, self.sensitivity(self.time_s), self.w1, self.w2)
            self.w1 -= dt * self.eta1 * k1
            if np.linalg.norm(self.w2, 2) < self.weight_bound:
                next_w2 = self.w2 - dt * self.eta2 * k2
                norm = np.linalg.norm(next_w2, 2)
                if norm >= self.weight_bound:
                    # Discrete-step safeguard: continuous law (36) freezes at
                    # its bound, but Euler can jump across it in one sample.
                    next_w2 *= self.weight_bound / norm
                    self.projection_count += 1
                self.w2 = next_w2
        self.time_s += dt

    def step(self, current_position_cm, target_position_cm, dt_s):
        position, target = (np.asarray(v, dtype=float) for v in (current_position_cm, target_position_cm))
        dt = float(dt_s)
        if (position.shape != (3,) or target.shape != (3,)
                or not np.all(np.isfinite([position, target])) or not np.isfinite(dt) or dt <= 0):
            raise ValueError("Require finite 3D positions and positive finite dt")
        if self.xhat is None:
            self.xhat = position.copy()
        else:
            self._advance(position, dt)
        residual = position - self.xhat
        ws = np.linalg.norm(residual)**self.mu * self.sigma.T
        phi_s = ws @ residual
        active = self.refined and self.time_s >= self.refined_start_s
        phi_f = self.w2 @ np.tanh(self.w1 @ residual) if active else np.zeros(3)
        estimate = self.xf + self.kh * phi_s  # (12), CF=I
        uf = self.kf * (self.xhat - target)
        self.last = (position.copy(), residual.copy(), uf.copy(), (phi_s + phi_f).copy())
        feedback, compensation = self.force_scale * uf, self.force_scale * estimate
        states = [*self.xhat, *self.xf, *self.sigma.ravel(), *self.w1.ravel(), *self.w2.ravel()]
        if not np.all(np.isfinite(states)):
            raise RuntimeError("Non-finite DRNN state; revise sampling/learning settings")
        return HierarchicalResult(feedback_force_n=feedback, compensation_force_n=compensation,
                                  control_force_n=feedback - compensation, residual_cm=residual.copy(),
                                  sigma_norm=float(np.linalg.norm(self.sigma)),
                                  w1_norm=float(np.linalg.norm(self.w1, 2)),
                                  w2_norm=float(np.linalg.norm(self.w2, 2)), refined_active=bool(active))

    def parameter_snapshot(self):
        return dict(algorithm="Paper DRNN-EID, overdamped position adaptation",
                    paper_doi="10.1016/j.automatica.2025.112744", equations=[4, 5, 6, 7, 12, 22, 27, 34, 35, 36],
                    nominal_state_unit="cm", nominal_input_unit="cm/s", output_force_unit="N",
                    force_scale=self.force_scale, gains=self.gains, refined=self.refined,
                    gamma_learning=self.gamma_learning, phi=self.phi, mu=self.mu,
                    eta1=self.eta1, eta2=self.eta2, refined_start_s=self.refined_start_s,
                    weight_bound=self.weight_bound, hidden_neurons=self.hidden_neurons, seed=self.seed,
                    final_sigma=self.sigma.tolist(), final_W1=self.w1.tolist(), final_W2=self.w2.tolist(),
                    discrete_bound_projection_count=self.projection_count,
                    continuous_time_reduced_model_only=True, full_plant_stability_claim=False)
