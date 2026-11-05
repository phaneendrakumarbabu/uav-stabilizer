"""Open-loop longitudinal stability analysis (linearised 6-DOF model at trim).
Numerical Jacobian of dynamics_6dof.dynamics at each trim point; longitudinal
states [u, w, q, theta]. Reports eigenvalues, short-period and phugoid damping
and the static margin. Open loop only: it shows how the payload degrades the
airframe; it is not a closed-loop proof. Assumed (not measured) coefficients."""
import numpy as np
import aircraft_params as P
from dynamics_6dof import dynamics, static_margin
from trim import find_trim

PITCH_BASE = (-2.0, -0.3, -0.4)

def long_matrix(state, controls, pl, eps=1e-6):
    idx = [3, 5, 10, 7]                      # u, w, q, theta
    A = np.zeros((4, 4))
    f0 = dynamics(state, controls, pl)
    for j, k in enumerate(idx):
        s = state.copy(); h = eps * max(1.0, abs(state[k])); s[k] += h
        A[:, j] = (dynamics(s, controls, pl)[idx] - f0[idx]) / h
    return A

def modes(eig):
    out = []
    for lam in eig:
        if lam.imag > 1e-9 or abs(lam.imag) < 1e-9:
            wn = abs(lam); zeta = -lam.real / wn if wn > 0 else np.nan
            out.append((lam, wn, zeta))
    return out


def closed_loop_matrix(state, controls, pl, eps=1e-6):
    """Numerical longitudinal PID closed-loop Jacobian.

    This is a local nominal analysis with ideal measurements, no actuator lag,
    and no estimator uncertainty. It answers a different question from the
    open-loop matrix: whether the implemented pitch feedback can stabilize a
    statically unstable airframe near a trim point.
    """
    idx = [3, 5, 10, 7]  # u, w, q, theta
    de0 = float(controls[0])
    kp, ki, kd = PITCH_BASE

    def closed_dynamics(z):
        s = state.copy()
        s[idx] = state[idx] + z[:4]
        # theta_ref perturbation is zero; pitch error is -delta theta and
        # error derivative is -delta q.
        pitch_error = -z[3]
        pitch_error_rate = -z[2]
        de = de0 + kp * pitch_error + ki * z[4] + kd * pitch_error_rate
        c = controls.copy()
        c[0] = de
        f = dynamics(s, c, pl)
        return np.r_[f[idx], pitch_error]

    z0 = np.zeros(5)
    f0 = closed_dynamics(z0)
    A = np.zeros((5, 5))
    for j in range(5):
        h = eps * max(1.0, abs(z0[j]))
        z = z0.copy()
        z[j] += h
        A[:, j] = (closed_dynamics(z) - f0) / h
    return A

if __name__ == "__main__":
    print("Va payload  static_margin  open_max_Re  open_stable  closed_max_Re  closed_stable")
    for Va in (12.0, 15.0, 20.0):
        for pl in (0.0, 0.2, 0.5, 0.7, 0.8, 1.0):
            tr = find_trim(Va, payload_level=pl)
            A = long_matrix(tr["state"], tr["controls"], pl)
            eig = np.linalg.eigvals(A)
            Acl = closed_loop_matrix(tr["state"], tr["controls"], pl)
            eig_cl = np.linalg.eigvals(Acl)
            sp = eig[int(np.argmax(np.abs(eig)))]
            print(f"{Va:4.0f} {pl:5.2f}  {static_margin(pl):8.4f}  "
                  f"{eig.real.max():11.4f}  {'yes' if eig.real.max() < 0 else 'NO ':10s}  "
                  f"{eig_cl.real.max():12.4f}  {'yes' if eig_cl.real.max() < 0 else 'NO ':13s}")
