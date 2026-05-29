"""
M07 — (S2, CT, G0, P1)
-----------------------
2D state, threshold classifier, no GP, single-pass.

Adds the CT classifier to M01. The classifier computes per-line
quality scores from crash deviation, oscillation std, cross-line
consistency, and tip-change innovation ratio. These quality scores
inflate R_eff at corrupted pixels, reducing their influence on the
Kalman update.

New parameters vs M01
---------------------
theta_c, alpha_c   — crash detection threshold and steepness
theta_o, alpha_o   — oscillation detection threshold and steepness
tau_cons           — cross-line consistency scale
theta_t, alpha_t   — tip-change threshold and steepness

Controlled comparison
---------------------
M01 vs M07: isolates the value of quality weighting (C0 → CT).
M07 vs M09: isolates the value of lateral GP coupling (G0 → GC).
M07 vs M25: isolates the value of 6D geometric state (S2 → S6).
"""

import time
import numpy as np
from .._base import (
    SPMReconstructor,
    make_A_s2, make_C, make_Q_s2, make_P0_s2,
    build_quality_ct, update_tip_scores,
    kalman_forward, kalman_backward,
    DEFAULTS_S2, DEFAULTS_CT,
)


class M07(SPMReconstructor):

    MODEL_ID           = "(S2, CT, G0, P1)"
    AXES               = {'S':'S2', 'C':'CT', 'G':'G0', 'P':'P1'}
    DEFAULTS           = {**DEFAULTS_S2, **DEFAULTS_CT}
    REQUIRES_DUAL_PASS = False

    def _reconstruct(self, data, p):
        t0 = time.time()
        A  = make_A_s2()
        C  = make_C(2)
        Q  = make_Q_s2(p)
        P0 = make_P0_s2()
        R  = np.array([[p['sigma_meas']**2]])

        # Stage 1: initial quality (crash + oscillation + consistency)
        quality = build_quality_ct(data.forward, p)

        # Stage 2: forward pass
        xf, Pf, xp, Pp, mon = kalman_forward(
            data.forward, quality, A, C, Q, R, P0,
            p['burn_in'], p['alpha_R'])

        # Stage 3: update quality with tip-change scores
        quality = update_tip_scores(
            quality, mon['innov_rms'], p, p['burn_in'])

        # Stage 4: second forward + backward with updated quality
        xf, Pf, xp, Pp, mon = kalman_forward(
            data.forward, quality, A, C, Q, R, P0,
            p['burn_in'], p['alpha_R'])

        xs, Ps, sc = kalman_backward(xf, Pf, xp, Pp, A)
        mon['smooth_corr'] = sc

        return self._build_result(xs, Ps, quality, mon, p, t0)
