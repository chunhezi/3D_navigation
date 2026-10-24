"""Offline scalar-axis SDP design for the paper's reduced nominal model.

Equations (24), (25), (47), specialized to A=0, B=C=1, R=0.
The three nominal axes use identical gains. The full inertial simulation plant
is not certified by these reduced-order continuous-time inequalities.
"""
import numpy as np


def design_gains(observer_gain=4., filter_rate=1., max_feedback_gain=2., gamma=1.):
    import cvxpy as cp

    l, a = float(observer_gain), float(filter_rate)
    if min(l, a, max_feedback_gain, gamma) <= 0:
        raise ValueError("Design parameters must be positive")
    ap = np.array([[-l, -1.], [0., -a]])
    c = np.array([[1., 0.]])
    q1 = cp.Variable((2, 2), symmetric=True)
    q2 = cp.Variable((1, 1), symmetric=True)
    u = cp.Variable((1, 1))
    kl, kh = cp.Variable(), cp.Variable()
    chi = cp.Variable()
    bs = cp.vstack([-cp.reshape(kh, (1, 1), order="C"),
                    a * cp.reshape(kl, (1, 1), order="C")])
    s1 = ap @ q1 + q1 @ ap.T + 0.01 * np.eye(2)
    s2 = u + u.T
    # R=0, Bg=0: the last two negative diagonal blocks of (25) decouple.
    stab = cp.bmat([[s1, gamma * q1 @ c.T * l],
                    [gamma * l * c @ q1, gamma**2 * s2]])
    hinf = cp.bmat([[s2, np.array([[l]]), q2],
                    [np.array([[l]]), -cp.reshape(chi, (1, 1), order="C"), np.ones((1, 1))],
                    [q2, np.ones((1, 1)), -cp.reshape(chi, (1, 1), order="C")]])
    constraints = [q1 >> 1e-5 * np.eye(2), q2 >> 1e-5 * np.eye(1), chi >= 1e-5,
                   ap @ q1 + q1 @ ap.T << -1e-5 * np.eye(2), bs == q1 @ c.T,
                   stab << -1e-5 * np.eye(3), hinf << -1e-5 * np.eye(3),
                   q1[0, 0] == 1., cp.trace(q1) <= 100., q2[0, 0] <= 100.,
                   u >= -max_feedback_gain * q2, u <= -0.1 * q2, kl <= -0.02]
    problem = cp.Problem(cp.Minimize(chi), constraints)
    problem.solve(solver="CLARABEL")
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"SDP design failed: {problem.status}")
    qv = q1.value
    klv, khv = float(kl.value), float(kh.value)
    # Check returned matrices numerically, not only the solver status.
    equality_residual = float(np.linalg.norm(np.array([[-khv], [a * klv]]) - qv @ c.T))
    residuals = dict(kyp_equality_norm=equality_residual,
                     q1_min_eigenvalue=float(np.linalg.eigvalsh(qv).min()),
                     q2_min_eigenvalue=float(q2.value[0, 0]),
                     kyp_lyapunov_max_eigenvalue=float(np.linalg.eigvalsh(ap @ qv + qv @ ap.T).max()),
                     eq25_max_eigenvalue=float(np.linalg.eigvalsh(stab.value).max()),
                     eq47_max_eigenvalue=float(np.linalg.eigvalsh(hinf.value).max()))
    if (equality_residual > 1e-6 or residuals["q1_min_eigenvalue"] <= 0
            or residuals["q2_min_eigenvalue"] <= 0
            or max(residuals[k] for k in ("kyp_lyapunov_max_eigenvalue", "eq25_max_eigenvalue",
                                           "eq47_max_eigenvalue")) >= 0):
        raise RuntimeError(f"Invalid numerical SDP certificate: {residuals}")
    return dict(observer_gain=l, filter_rate=a, kl=klv, kh=khv,
                kf=float((u.value / q2.value)[0, 0]), chi=float(chi.value),
                Q1=qv.tolist(), Q2=q2.value.tolist(), U=u.value.tolist(),
                solver=problem.solver_stats.solver_name, status=problem.status, residuals=residuals,
                adaptations="First-order nominal model; R=0; gamma fixed; Q1[0,0]=1; "
                "bounded feedback bandwidth and negative Kl for this application; "
                "gain stacking follows physical equations (12),(16).",
                full_inertial_plant_certified=False)


if __name__ == "__main__":
    import json
    print(json.dumps(design_gains(), indent=2))
