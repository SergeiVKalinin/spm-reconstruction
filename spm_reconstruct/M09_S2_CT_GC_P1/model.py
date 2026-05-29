"""
M09 — (S2, CT, GC, P1)
-----------------------
2D state, threshold classifier, conditional GP, single-pass.

Adds the edge-aware bilateral GP smoother to M07. The GP provides
lateral information flow — clean pixels help reconstruct uncertain
(crash-damaged) neighbours within the same line. Applied conditionally:
only pixels where forward uncertainty exceeds gp_unc_thresh * sigma_meas
are smoothed. Clean pixels on clean lines are never touched.

New parameters vs M07
---------------------
beta           — edge sensitivity in bilateral smoother (TUNING)
gp_unc_thresh  — GP activation threshold × sigma_meas (TUNING)

Controlled comparisons
----------------------
M07 vs M09: isolates GP value (G0 → GC) on 2D CT single-pass.
M09 vs M10: isolates dual-pass value under 2D CT GC.
M09 vs M27: isolates state value (S2 → S6) under CT GC P1.
"""

import time
import numpy as np
from .._base import (
    SPMReconstructor,
    make_A_s2, make_C, make_Q_s2, make_P0_s2,
    build_quality_ct, update_tip_scores,
    kalman_forward, kalman_backward,
    estimate_ell, apply_gp,
    DEFAULTS_S2, DEFAULTS_CT, DEFAULTS_GC,
)


class M09(SPMReconstructor):

    MODEL_ID           = "(S2, CT, GC, P1)"
    AXES               = {'S':'S2', 'C':'CT', 'G':'GC', 'P':'P1'}
    DEFAULTS           = {**DEFAULTS_S2, **DEFAULTS_CT, **DEFAULTS_GC}
    REQUIRES_DUAL_PASS = False

    def _reconstruct(self, data, p):
        t0 = time.time()
        A  = make_A_s2()
        C  = make_C(2)
        Q  = make_Q_s2(p)
        P0 = make_P0_s2()
        R  = np.array([[p['sigma_meas']**2]])

        # Stage 1: initial quality
        quality = build_quality_ct(data.forward, p)

        # Stage 2: forward pass
        xf, Pf, xp, Pp, mon = kalman_forward(
            data.forward, quality, A, C, Q, R, P0,
            p['burn_in'], p['alpha_R'])

        # Stage 3: update quality with tip scores
        quality = update_tip_scores(
            quality, mon['innov_rms'], p, p['burn_in'])

        # Stage 4: estimate correlation length and apply GP
        ell, L = estimate_ell(data.forward, quality, data.Nx)
        xf, mon = apply_gp(xf, mon, quality, ell, L, p)

        # Stage 5: backward RTS
        xs, Ps, sc = kalman_backward(xf, Pf, xp, Pp, A)
        mon['smooth_corr'] = sc

        return self._build_result(
            xs, Ps, quality, mon, p, t0,
            diagnostics={'ell': ell, 'L': L,
                         'gp_mask': mon.get('gp_mask')})
