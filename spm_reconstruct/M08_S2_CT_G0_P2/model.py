"""
M08 — (S2, CT, G0, P2)
-----------------------
2D state, threshold classifier, no GP, dual-pass.

M07 extended to dual-pass acquisition. The CT classifier runs
independently on each scan direction, then quality-weighted
information fusion combines the two observations before the
RTS smoother. A crashed forward line with a clean backward line
is correctly handled: the backward observation dominates.

New parameters vs M07
---------------------
pid_corr_weight  — PID correction attenuation (TUNING, default 0.40)
drift_interline  — inter-scan lateral offset (PHYSICAL, default 0.30)

Controlled comparisons
----------------------
M07 vs M08: isolates dual-pass value under CT classifier.
M08 vs M10: isolates GP value (G0 → GC) on dual-pass CT model.
M08 vs M28: isolates state value (S2 → S6) on dual-pass CT model.
"""

import time
import numpy as np
from .._base import (
    SPMReconstructor,
    make_A_s2, make_C, make_Q_s2, make_P0_s2,
    build_quality_ct, update_tip_scores,
    kalman_forward, kalman_backward,
    DEFAULTS_S2, DEFAULTS_CT, DEFAULTS_P2,
)


class M08(SPMReconstructor):

    MODEL_ID           = "(S2, CT, G0, P2)"
    AXES               = {'S':'S2', 'C':'CT', 'G':'G0', 'P':'P2'}
    DEFAULTS           = {**DEFAULTS_S2, **DEFAULTS_CT, **DEFAULTS_P2}
    REQUIRES_DUAL_PASS = True

    def _reconstruct(self, data, p):
        t0  = time.time()
        fwd = data.forward
        bwd = data.backward

        # Stage 1: classify each scan independently
        q_fwd = build_quality_ct(fwd, p)
        q_bwd = build_quality_ct(bwd, p)

        # Stage 2: information-weighted fusion
        # Higher quality scan gets more weight at each pixel
        w_fwd = q_fwd / (q_fwd + q_bwd + 1e-8)
        w_bwd = 1.0 - w_fwd
        fused   = w_fwd * fwd + w_bwd * bwd
        quality = np.maximum(q_fwd, q_bwd)   # best quality at each pixel

        # Effective noise after fusion (weighted harmonic mean)
        R_base  = np.array([[p['sigma_meas']**2]])
        R_fused = np.array([[p['sigma_meas']**2 / 2.0]])

        A  = make_A_s2()
        C  = make_C(2)
        Q  = make_Q_s2(p)
        P0 = make_P0_s2()

        # Stage 3: forward pass on fused observation
        xf, Pf, xp, Pp, mon = kalman_forward(
            fused, quality, A, C, Q, R_fused, P0,
            p['burn_in'], p['alpha_R'])

        # Stage 4: update quality with tip-change scores
        quality = update_tip_scores(
            quality, mon['innov_rms'], p, p['burn_in'])

        # Stage 5: re-run with updated quality
        xf, Pf, xp, Pp, mon = kalman_forward(
            fused, quality, A, C, Q, R_fused, P0,
            p['burn_in'], p['alpha_R'])

        xs, Ps, sc = kalman_backward(xf, Pf, xp, Pp, A)
        mon['smooth_corr'] = sc

        return self._build_result(
            xs, Ps, quality, mon, p, t0,
            diagnostics={
                'q_fwd': q_fwd,
                'q_bwd': q_bwd,
                'fused': fused,
            })
