"""
M20 — (S6, C0, G0, P2)
-----------------------
6D geometric state, no classifier, no GP, dual-pass.

M19 with dual-pass. Simple information-weighted fusion
(equal weights under C0) before the 6D RTS smoother.

Controlled comparisons
----------------------
M19 vs M20: isolates dual-pass value on 6D state without classifier.
M02 vs M20: isolates state value (S2 → S6) on dual-pass C0.
M20 vs M26: isolates classifier value (C0 → CT) on S6 dual-pass.
"""

import time
import numpy as np
from .._base import (
    SPMReconstructor,
    make_A_s6, make_C, make_Q_s6, make_P0_s6,
    build_quality_c0,
    kalman_forward, kalman_backward,
    DEFAULTS_S2, DEFAULTS_S6_EXTRA, DEFAULTS_P2,
)


class M20(SPMReconstructor):

    MODEL_ID           = "(S6, C0, G0, P2)"
    AXES               = {'S':'S6', 'C':'C0', 'G':'G0', 'P':'P2'}
    DEFAULTS           = {**DEFAULTS_S2, **DEFAULTS_S6_EXTRA, **DEFAULTS_P2}
    REQUIRES_DUAL_PASS = True

    def _reconstruct(self, data, p):
        t0 = time.time()

        # C0: equal-weight fusion
        fused   = 0.5 * data.forward + 0.5 * data.backward
        quality = build_quality_c0(fused)
        R_fused = np.array([[p['sigma_meas']**2 / 2.0]])

        A  = make_A_s6()
        C  = make_C(6)
        Q  = make_Q_s6(p)
        P0 = make_P0_s6()

        xf, Pf, xp, Pp, mon = kalman_forward(
            fused, quality, A, C, Q, R_fused, P0,
            p['burn_in'], p['alpha_R'])

        xs, Ps, sc = kalman_backward(xf, Pf, xp, Pp, A)
        mon['smooth_corr'] = sc

        return self._build_result(
            xs, Ps, quality, mon, p, t0,
            diagnostics={
                'sx_map' : xs[:, :, 1].astype(np.float32),
                'sk_map' : xs[:, :, 2].astype(np.float32),
                'fused'  : fused,
            })
