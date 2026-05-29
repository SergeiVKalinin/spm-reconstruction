"""
M19 — (S6, C0, G0, P1)
-----------------------
6D geometric state, no classifier, no GP, single-pass.

Upgrades the state from 2D [h, d] to 6D [s, sx, sk, sxx, sxk, skk].
No classifier — all pixels trusted equally. No GP coupling.

The 6D state enables parabolic extrapolation across crash gaps
(vs linear in S2 models) and recovers surface gradients and
curvatures as free outputs of the smoother.

New parameters vs M01
---------------------
sigma_sx, sigma_sxx, sigma_sxk, sigma_skk  (PHYSICAL, S6 extras)

Controlled comparisons
----------------------
M01 vs M19: isolates value of 6D geometric state (S2 → S6).
M02 vs M20: same comparison on dual-pass.
M19 vs M25: isolates classifier value (C0 → CT) on S6.
"""

import time
import numpy as np
from .._base import (
    SPMReconstructor,
    make_A_s6, make_C, make_Q_s6, make_P0_s6,
    build_quality_c0,
    kalman_forward, kalman_backward,
    DEFAULTS_S2, DEFAULTS_S6_EXTRA,
)


class M19(SPMReconstructor):

    MODEL_ID           = "(S6, C0, G0, P1)"
    AXES               = {'S':'S6', 'C':'C0', 'G':'G0', 'P':'P1'}
    DEFAULTS           = {**DEFAULTS_S2, **DEFAULTS_S6_EXTRA}
    REQUIRES_DUAL_PASS = False

    def _reconstruct(self, data, p):
        t0 = time.time()
        A  = make_A_s6()
        C  = make_C(6)
        Q  = make_Q_s6(p)
        P0 = make_P0_s6()
        R  = np.array([[p['sigma_meas']**2]])

        quality = build_quality_c0(data.forward)

        xf, Pf, xp, Pp, mon = kalman_forward(
            data.forward, quality, A, C, Q, R, P0,
            p['burn_in'], p['alpha_R'])

        xs, Ps, sc = kalman_backward(xf, Pf, xp, Pp, A)
        mon['smooth_corr'] = sc

        return self._build_result(
            xs, Ps, quality, mon, p, t0,
            diagnostics={
                'sx_map' : xs[:, :, 1].astype(np.float32),
                'sk_map' : xs[:, :, 2].astype(np.float32),
            })
