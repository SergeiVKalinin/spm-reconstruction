"""
M10 — (S2, CT, GC, P2)
-----------------------
2D state, threshold classifier, conditional GP, dual-pass.

Combines M09 (2D + CT + GC) with dual-pass acquisition.
The GP smoother benefits from dual-pass because the quality map
is derived from two independent observations, making it more
reliable at identifying which pixels are genuinely uncertain.

Controlled comparisons
----------------------
M09 vs M10: isolates dual-pass value under 2D CT GC.
M08 vs M10: isolates GP value (G0 → GC) on dual-pass CT model.
M10 vs M28: isolates state value (S2 → S6) on full dual-pass model.
"""

import time
import numpy as np
from .._base import (
    SPMReconstructor,
    make_A_s2, make_C, make_Q_s2, make_P0_s2,
    build_quality_ct, update_tip_scores,
    kalman_forward, kalman_backward,
    estimate_ell, apply_gp,
    DEFAULTS_S2, DEFAULTS_CT, DEFAULTS_GC, DEFAULTS_P2,
)


class M10(SPMReconstructor):

    MODEL_ID           = "(S2, CT, GC, P2)"
    AXES               = {'S':'S2', 'C':'CT', 'G':'GC', 'P':'P2'}
    DEFAULTS           = {**DEFAULTS_S2, **DEFAULTS_CT,
                          **DEFAULTS_GC, **DEFAULTS_P2}
    REQUIRES_DUAL_PASS = True

    def _reconstruct(self, data, p):
        t0  = time.time()
        fwd = data.forward
        bwd = data.backward

        # Stage 1: classify each scan
        q_fwd = build_quality_ct(fwd, p)
        q_bwd = build_quality_ct(bwd, p)

        # Stage 2: information-weighted fusion
        w_fwd   = q_fwd / (q_fwd + q_bwd + 1e-8)
        fused   = w_fwd * fwd + (1.0 - w_fwd) * bwd
        quality = np.maximum(q_fwd, q_bwd)
        R_fused = np.array([[p['sigma_meas']**2 / 2.0]])

        A  = make_A_s2()
        C  = make_C(2)
        Q  = make_Q_s2(p)
        P0 = make_P0_s2()

        # Stage 3: forward pass
        xf, Pf, xp, Pp, mon = kalman_forward(
            fused, quality, A, C, Q, R_fused, P0,
            p['burn_in'], p['alpha_R'])

        # Stage 4: tip-change scores
        quality = update_tip_scores(
            quality, mon['innov_rms'], p, p['burn_in'])

        # Stage 5: GP smoothing
        ell, L = estimate_ell(fused, quality, data.Nx)
        xf, mon = apply_gp(xf, mon, quality, ell, L, p)

        # Stage 6: backward RTS
        xs, Ps, sc = kalman_backward(xf, Pf, xp, Pp, A)
        mon['smooth_corr'] = sc

        return self._build_result(
            xs, Ps, quality, mon, p, t0,
            diagnostics={'ell': ell, 'L': L,
                         'gp_mask': mon.get('gp_mask'),
                         'fused': fused})
