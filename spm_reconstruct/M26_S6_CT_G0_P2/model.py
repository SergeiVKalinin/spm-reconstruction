"""
M26 — (S6, CT, G0, P2)
-----------------------
6D geometric state, threshold classifier, no GP, dual-pass.

M25 extended to dual-pass. Quality-weighted fusion with
independent CT classification of each scan direction.

Controlled comparisons
----------------------
M25 vs M26: isolates dual-pass value on S6 CT without GP.
M08 vs M26: isolates state value (S2 → S6) on dual-pass CT.
M26 vs M28: isolates GP value (G0 → GC) on S6 dual-pass CT.
"""

import time
import numpy as np
from .._base import (
    SPMReconstructor,
    make_A_s6, make_C, make_Q_s6, make_P0_s6,
    build_quality_ct, update_tip_scores,
    kalman_forward, kalman_backward,
    DEFAULTS_S2, DEFAULTS_S6_EXTRA, DEFAULTS_CT, DEFAULTS_P2,
)


class M26(SPMReconstructor):

    MODEL_ID           = "(S6, CT, G0, P2)"
    AXES               = {'S':'S6', 'C':'CT', 'G':'G0', 'P':'P2'}
    DEFAULTS           = {**DEFAULTS_S2, **DEFAULTS_S6_EXTRA,
                          **DEFAULTS_CT, **DEFAULTS_P2}
    REQUIRES_DUAL_PASS = True

    def _reconstruct(self, data, p):
        t0  = time.time()
        fwd = data.forward
        bwd = data.backward

        q_fwd = build_quality_ct(fwd, p)
        q_bwd = build_quality_ct(bwd, p)
        w_fwd = q_fwd / (q_fwd + q_bwd + 1e-8)
        fused   = w_fwd * fwd + (1.0 - w_fwd) * bwd
        quality = np.maximum(q_fwd, q_bwd)
        R_fused = np.array([[p['sigma_meas']**2 / 2.0]])

        A  = make_A_s6()
        C  = make_C(6)
        Q  = make_Q_s6(p)
        P0 = make_P0_s6()

        xf, Pf, xp, Pp, mon = kalman_forward(
            fused, quality, A, C, Q, R_fused, P0,
            p['burn_in'], p['alpha_R'])

        quality = update_tip_scores(
            quality, mon['innov_rms'], p, p['burn_in'])

        xf, Pf, xp, Pp, mon = kalman_forward(
            fused, quality, A, C, Q, R_fused, P0,
            p['burn_in'], p['alpha_R'])

        xs, Ps, sc = kalman_backward(xf, Pf, xp, Pp, A)
        mon['smooth_corr'] = sc

        return self._build_result(
            xs, Ps, quality, mon, p, t0,
            diagnostics={
                'sx_map': xs[:, :, 1].astype(np.float32),
                'fused' : fused,
                'q_fwd' : q_fwd,
                'q_bwd' : q_bwd,
            })
