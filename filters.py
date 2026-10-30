"""Signal filters for smoothing noisy landmark positions."""
import math


class _LowPass:
    def __init__(self):
        self.value = None

    def __call__(self, x, alpha):
        self.value = x if self.value is None else alpha * x + (1 - alpha) * self.value
        return self.value


class OneEuroFilter:
    """One Euro filter (Casiez et al., 2012).

    Heavy smoothing when the signal moves slowly (kills jitter), light
    smoothing when it moves fast (kills lag). `min_cutoff` controls jitter,
    `beta` controls how quickly the filter opens up with speed.
    """

    def __init__(self, min_cutoff=1.5, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t_prev is None:
            self._t_prev = t
            self._dx(0.0, 1.0)
            return self._x(x, 1.0)

        dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t
        dx = (x - self._x.value) / dt
        edx = self._dx(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(x, self._alpha(cutoff, dt))


class PointFilter:
    """One Euro filter applied independently to x and y."""

    def __init__(self, **kwargs):
        self._fx = OneEuroFilter(**kwargs)
        self._fy = OneEuroFilter(**kwargs)

    def reset(self):
        self._fx.reset()
        self._fy.reset()

    def __call__(self, point, t):
        return self._fx(point[0], t), self._fy(point[1], t)
