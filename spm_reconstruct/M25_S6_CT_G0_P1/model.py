"""
M25 — (S6, CT, G0, P1)
-----------------------
6D geometric state, threshold classifier, no GP, single-pass.

Combines the S6 geometric state with the CT classifier.
No GP coupling. Bridge model between M19 (no classifier)
and M27 (full model with GP).

Controlled comparisons
----------------------
M19 vs M25: isolates classifier value (C0 → CT) on S6 single-pass.
M07 vs M25: isolates state value (S2 → S6) under CT single-pass.
M25 vs M27: isolates GP value (G0 → GC) on S6 CT single-pass.
M25 vs M26: isolates dual-pass value on S6 CT without GP.
"""

import time
import numpy as np
from .._base import (
    SPMReconstructor,
    make_A_s6, make_C, make_Q_s6, make_P0_s6,
    build_quality_ct, update_tip_scores,
    kalman_forward, kalman_backward,
    DEFAULTS_S2, DEFAULTS_S6_EXTRA, DEFAULTS_CT,
)


class M25(SPMReconstructor):

    MODEL_ID           = "(S6, CT, G0, P1)"
    AXES               = {'S':'S6', 'C':'CT', 'G':'G0', 'P':'P1'}
    DEFAULTS           = {**DEFAULTS_S2, **DEFAULTS_S6_EXTRA, **DEFAULTS_CT}
    REQUIRES_DUAL_PASS = False

    def _reconstruct(self, data, p):
        t0 = time.time()
        A  = make_A_s6()
        C  = make_C(6)
        Q  = make_Q_s6(p)
        P0 = make_P0_s6()
        R  = np.array([[p['sigma_meas']**2]])

        quality = build_quality_ct(data.forward, p)

        xf, Pf, xp, Pp, mon = kalman_forward(
            data.forward, quality, A, C, Q, R, P0,
            p['burn_in'], p['alpha_R'])

        quality = update_tip_scores(
            quality, mon['innov_rms'], p, p['burn_in'])

        xf, Pf, xp, Pp, mon = kalman_forward(
            data.forward, quality, A, C, Q, R, P0,
            p['burn_in'], p['alpha_R'])

        xs, Ps, sc = kalman_backward(xf, Pf, xp, Pp, A)
        mon['smooth_corr'] = sc

        return self._build_result(
            xs, Ps, quality, mon, p, t0,
            diagnostics={
                'sx_map': xs[:, :, 1].astype(np.float32),
                'sk_map': xs[:, :, 2].astype(np.float32),
            })
