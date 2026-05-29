"""
M01 — (S2, C0, G0, P1)
-----------------------
Simplest valid Kalman reconstruction.

State      : 2D  [height, drift_rate]
Classifier : C0  no quality weighting — all pixels trusted equally
GP         : G0  no lateral coupling
Pass       : P1  single pass (forward scan only)

This is the baseline. Every more complex model should outperform this
on images with artifacts. If it does not, something is wrong.

Typical RMSE improvement over corrupted input: 20-35% on crash images.
"""

import time
import numpy as np
from .._base import (
    SPMReconstructor,
    make_A_s2, make_C, make_Q_s2, make_P0_s2,
    build_quality_c0,
    kalman_forward, kalman_backward,
    DEFAULTS_S2,
)


class M01(SPMReconstructor):

    MODEL_ID           = "(S2, C0, G0, P1)"
    AXES               = {'S':'S2', 'C':'C0', 'G':'G0', 'P':'P1'}
    DEFAULTS           = DEFAULTS_S2.copy()
    REQUIRES_DUAL_PASS = False

    def _reconstruct(self, data, p):
        t0 = time.time()

        # State matrices
        A  = make_A_s2()
        C  = make_C(2)
        Q  = make_Q_s2(p)
        P0 = make_P0_s2()
        R  = np.array([[p['sigma_meas']**2]])

        # C0: trust all pixels equally
        quality = build_quality_c0(data.forward)

        # Forward Kalman pass
        xf, Pf, xp, Pp, mon = kalman_forward(
            data.forward, quality, A, C, Q, R, P0,
            p['burn_in'], p['alpha_R'])

        # Backward RTS smoother
        xs, Ps, sc = kalman_backward(xf, Pf, xp, Pp, A)
        mon['smooth_corr'] = sc

        return self._build_result(xs, Ps, quality, mon, p, t0)
