"""Atmospheric disturbance models used by the UAV simulations.

Two modes are retained deliberately:

``ou``
    The original first-order Ornstein-Uhlenbeck model used by the published
    baseline grids.  It is deterministic for a seed and preserves the
    requested stationary RMS exactly.

``dryden``
    A Dryden-style spectral shaping model.  The longitudinal component uses
    the standard first-order filter and the lateral/vertical components use
    second-order rational filters.  The discrete filter input is normalized
    so the long-run RMS is close to the requested component sigma.

Keeping the modes explicit prevents a stronger disturbance model from
silently invalidating old evidence. New robustness studies should use
``model="dryden"`` and record that choice in their manifest.
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov


def one_cosine_gust(t, start_t, duration, amplitude, axis=2):
    """Return a finite-duration 1-cosine gust in NED coordinates.

    ``amplitude`` is the peak velocity increment (m/s); ``axis`` is 0=N,
    1=E, 2=D. This is a deterministic transient test, separate from the
    stochastic Dryden-style background model.
    """
    out = np.zeros(3, dtype=float)
    if duration <= 0.0 or t < start_t or t > start_t + duration:
        return out
    phase = (t - start_t) / duration
    out[int(axis)] = 0.5 * float(amplitude) * (1.0 - np.cos(2.0 * np.pi * phase))
    return out


class DrydenGust:
    def __init__(self, Va, intensity, seed=None,
                 Lu=200.0, Lv=200.0, Lw=50.0,
                 sigma_u_ref=1.5, sigma_v_ref=1.5, sigma_w_ref=1.0,
                 model="ou"):
        """
        Va        : airspeed (m/s), sets time constants tau = L / Va
        intensity : turbulence intensity scale factor (0 = calm)
        seed      : RNG seed for reproducibility
        Lu,Lv,Lw  : turbulence scale lengths (m) -- longitudinal, lateral, vertical
         sigma_*_ref : stationary RMS gust speed (m/s) at intensity == 1
        """
        model = str(model).lower()
        if model not in {"ou", "dryden"}:
            raise ValueError("gust model must be 'ou' or 'dryden'")
        self.model = model
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        Va_safe = max(Va, 1e-3)

        self.tau_u = Lu / Va_safe
        self.tau_v = Lv / Va_safe
        self.tau_w = Lw / Va_safe

        self.sigma_u = sigma_u_ref * intensity
        self.sigma_v = sigma_v_ref * intensity
        self.sigma_w = sigma_w_ref * intensity

        self.x = np.zeros(3)  # current [gu, gv, gw]
        self._va_for_filter = Va_safe
        self._dt_for_filter = 0.02
        self._filters = None
        if self.model == "dryden":
            self._filters = self._build_dryden_filters(Va_safe)

    @staticmethod
    def _discrete_filter(A, B, C, dt, target_sigma):
        """Discretize a shaping filter and normalize its stationary RMS."""
        # Exact zero-order-hold state transition for deterministic input. The
        # process-noise scaling is then normalized from the discrete Lyapunov
        # covariance, which is more repeatable than an Euler approximation.
        from scipy.linalg import expm
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float).reshape(A.shape[0], 1)
        C = np.asarray(C, dtype=float).reshape(1, A.shape[0])
        n = A.shape[0]
        aug = np.zeros((n + 1, n + 1))
        aug[:n, :n] = A
        aug[:n, n:] = B
        exp_aug = expm(aug * float(dt))
        Ad = exp_aug[:n, :n]
        Bd = exp_aug[:n, n:]
        covariance = solve_discrete_lyapunov(Ad, Bd @ Bd.T)
        variance = float((C @ covariance @ C.T).item())
        scale = float(target_sigma / np.sqrt(max(variance, 1e-15)))
        return {"A": Ad, "B": Bd[:, 0], "C": C[0], "scale": scale,
                "state": np.zeros(n)}

    def _build_dryden_filters(self, Va):
        filters = []
        # H_u(s) = K/(s+a), with K selected so that the unscaled output has
        # the requested Dryden low-frequency shape.
        a_u = Va / max(self.tau_u * Va, 1e-9)
        K_u = np.sqrt(2.0 * a_u)
        filters.append(self._discrete_filter(
            [[-a_u]], [[K_u]], [[1.0]], self._dt_for_filter, self.sigma_u))

        # H(s) = K (s + sqrt(3)a)/(s+a)^2.  This is the standard rational
        # Dryden-style lateral/vertical shaping form.  RMS normalization is
        # applied after discretization so intensity retains a clear meaning.
        for tau, sigma in ((self.tau_v, self.sigma_v),
                           (self.tau_w, self.sigma_w)):
            a = 1.0 / max(tau, 1e-9)
            A = [[0.0, 1.0], [-a * a, -2.0 * a]]
            B = [[0.0], [1.0]]
            C = [[a * a, np.sqrt(3.0) * a]]
            filters.append(self._discrete_filter(
                A, B, C, self._dt_for_filter, sigma))
        return filters

    def step(self, dt):
        """Advance the OU processes by dt and return a NED gust velocity
        [north, east, down]. The dynamics rotate it into body axes using the
        current attitude before computing relative airspeed."""
        if self.model == "dryden":
            # Filters are built lazily because DT is supplied to step().
            if self._filters is None or abs(self._dt_for_filter - float(dt)) > 1e-12:
                self._dt_for_filter = float(dt)
                self._filters = self._build_dryden_filters(
                    max(self.tau_u * max(self._va_for_filter, 1e-3), 1e-3)
                )
            eta = self.rng.standard_normal(3)
            values = []
            for filt, noise in zip(self._filters, eta):
                filt["state"] = (filt["A"] @ filt["state"]
                                  + filt["B"] * (filt["scale"] * noise))
                values.append(float(filt["C"] @ filt["state"]))
            self.x = np.asarray(values, dtype=float)
            return self.x.copy()

        taus = np.array([self.tau_u, self.tau_v, self.tau_w])
        sigmas = np.array([self.sigma_u, self.sigma_v, self.sigma_w])

        decay = np.exp(-dt / taus)
        # stationary-variance-preserving noise injection for OU discretization
        noise_scale = sigmas * np.sqrt(1.0 - decay ** 2)
        eta = self.rng.standard_normal(3)

        self.x = self.x * decay + noise_scale * eta
        return self.x.copy()

    def reset(self):
        self.x = np.zeros(3)
        if self._filters is not None:
            for filt in self._filters:
                filt["state"][:] = 0.0
