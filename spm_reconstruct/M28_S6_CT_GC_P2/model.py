"""
M28 — (S6, CT, GC, P2)
-----------------------
6D geometric state, threshold classifier, conditional GP, dual-pass.

The most complete dual-pass model. M27 extended to dual-pass:
quality-weighted fusion of two scans, then the full M27 pipeline
(6D state + CT + GP). The GP correlation length is estimated from
the fused observation, which is cleaner than either scan alone.

This is the current dual-pass notebook algorithm.

Controlled comparisons
----------------------
M27 vs M28: isolates dual-pass value on the most complete model.
M26 vs M28: isolates GP value (G0 → GC) on S6 dual-pass CT.
M10 vs M28: isolates state value (S2 → S6) on full dual-pass model.
"""

import time
import numpy as np
from .._base import (
    SPMReconstructor,
    make_A_s6, make_C, make_Q_s6, make_P0_s6,
    build_quality_ct, update_tip_scores,
    kalman_forward, kalman_backward,
    estimate_ell, apply_gp,
    DEFAULTS_S2, DEFAULTS_S6_EXTRA, DEFAULTS_CT, DEFAULTS_GC, DEFAULTS_P2,
)


class M28(SPMReconstructor):

    MODEL_ID           = "(S6, CT, GC, P2)"
    AXES               = {'S':'S6', 'C':'CT', 'G':'GC', 'P':'P2'}
    DEFAULTS           = {**DEFAULTS_S2, **DEFAULTS_S6_EXTRA,
                          **DEFAULTS_CT, **DEFAULTS_GC, **DEFAULTS_P2}
    REQUIRES_DUAL_PASS = True

    def _reconstruct(self, data, p):
        t0  = time.time()
        fwd = data.forward
        bwd = data.backward

        # Stage 1: classify each scan independently
        q_fwd = build_quality_ct(fwd, p)
        q_bwd = build_quality_ct(bwd, p)

        # Stage 2: quality-weighted fusion
        w_fwd   = q_fwd / (q_fwd + q_bwd + 1e-8)
        fused   = w_fwd * fwd + (1.0 - w_fwd) * bwd
        quality = np.maximum(q_fwd, q_bwd)
        R_fused = np.array([[p['sigma_meas']**2 / 2.0]])

        A  = make_A_s6()
        C  = make_C(6)
        Q  = make_Q_s6(p)
        P0 = make_P0_s6()

        # Stage 3: forward pass on fused observation
        xf, Pf, xp, Pp, mon = kalman_forward(
            fused, quality, A, C, Q, R_fused, P0,
            p['burn_in'], p['alpha_R'])

        # Stage 4: update quality with tip-change scores
        quality = update_tip_scores(
            quality, mon['innov_rms'], p, p['burn_in'])

        # Stage 5: GP on fused observation (cleaner ell estimate)
        ell, L = estimate_ell(fused, quality, data.Nx)
        xf, mon = apply_gp(xf, mon, quality, ell, L, p)

        # Stage 6: backward RTS
        xs, Ps, sc = kalman_backward(xf, Pf, xp, Pp, A)
        mon['smooth_corr'] = sc

        return self._build_result(
            xs, Ps, quality, mon, p, t0,
            diagnostics={
                'ell'    : ell,
                'L'      : L,
                'gp_mask': mon.get('gp_mask'),
                'q_fwd'  : q_fwd,
                'q_bwd'  : q_bwd,
                'fused'  : fused,
            })
