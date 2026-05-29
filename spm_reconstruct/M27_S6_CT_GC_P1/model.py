"""
M27 — (S6, CT, GC, P1)
-----------------------
Most complete single-pass reconstruction model.

State      : S6  6D geometric [s, sx, sk, sxx, sxk, skk]
Classifier : CT  soft sigmoid threshold (crash + oscillation + consistency + tip)
GP         : GC  conditional edge-aware GP smoother along x
Pass       : P1  single pass (forward scan only)

Pipeline
--------
1. Build initial quality map (CT: crash + oscillation + consistency)
2. Forward Kalman pass with online R estimation
3. Update quality with tip-change scores (uses innovation RMS)
4. Estimate lateral correlation length ell from gradient autocorrelation
5. Conditional GP smoothing (only uncertain pixels)
6. Backward RTS smoother

This corresponds to the v3 notebook algorithm.
Typical RMSE improvement: 30-45% on images with mixed artifacts.
"""

import time
import numpy as np
from .._base import (
    SPMReconstructor,
    make_A_s6, make_C, make_Q_s6, make_P0_s6,
    build_quality_ct, update_tip_scores,
    kalman_forward, kalman_backward,
    estimate_ell, apply_gp,
    DEFAULTS_S2, DEFAULTS_S6_EXTRA, DEFAULTS_CT, DEFAULTS_GC,
)


class M27(SPMReconstructor):

    MODEL_ID           = "(S6, CT, GC, P1)"
    AXES               = {'S':'S6', 'C':'CT', 'G':'GC', 'P':'P1'}
    DEFAULTS           = {**DEFAULTS_S2, **DEFAULTS_S6_EXTRA,
                          **DEFAULTS_CT,  **DEFAULTS_GC}
    REQUIRES_DUAL_PASS = False

    def _reconstruct(self, data, p):
        t0 = time.time()

        # State matrices
        A  = make_A_s6()
        C  = make_C(6)
        Q  = make_Q_s6(p)
        P0 = make_P0_s6()
        R  = np.array([[p['sigma_meas']**2]])

        # Stage 1: initial quality (crash + oscillation + consistency)
        quality = build_quality_ct(data.forward, p)

        # Stage 2: forward Kalman pass
        xf, Pf, xp, Pp, mon = kalman_forward(
            data.forward, quality, A, C, Q, R, P0,
            p['burn_in'], p['alpha_R'])

        # Stage 3: update quality with tip-change scores
        quality = update_tip_scores(
            quality, mon['innov_rms'], p, p['burn_in'])

        # Stage 4: estimate correlation length and apply GP smoothing
        ell, L = estimate_ell(data.forward, quality, data.Nx)
        xf, mon = apply_gp(xf, mon, quality, ell, L, p)

        # Stage 5: backward RTS smoother
        xs, Ps, sc = kalman_backward(xf, Pf, xp, Pp, A)
        mon['smooth_corr'] = sc

        return self._build_result(
            xs, Ps, quality, mon, p, t0,
            diagnostics={
                'ell'    : ell,
                'L'      : L,
                'gp_mask': mon.get('gp_mask'),
            })
