"""
M02 — (S2, C0, G0, P2)
-----------------------
2D state, no classifier, no GP, dual-pass.

Identical to M01 except it uses both the forward and backward scans.
The two scans are fused with information-weighted averaging before
the RTS smoother runs. No quality weighting — both scans trusted equally.

New parameters vs M01
---------------------
pid_corr_weight : float — PID lag correction attenuation (TUNING, default 0.40)
drift_interline : float — sub-pixel lateral offset between scans (PHYSICAL, default 0.30)

What dual-pass adds
-------------------
- Noise reduced by ~sqrt(2) on clean pixels (two independent observations)
- PID lag partially corrected from the forward-backward difference signal
- Crashed lines: if crash appears in both scans, no gain; if only one,
  the other provides a clean observation for that line

Controlled comparison
---------------------
M01 vs M02: isolates the value of dual-pass acquisition at minimum complexity.
Any performance difference is purely from having two scan directions.
"""

import time
import numpy as np
from .._base import (
    SPMReconstructor,
    make_A_s2, make_C, make_Q_s2, make_P0_s2,
    build_quality_c0,
    kalman_forward, kalman_backward,
    DEFAULTS_S2, DEFAULTS_P2,
)


class M02(SPMReconstructor):

    MODEL_ID           = "(S2, C0, G0, P2)"
    AXES               = {'S':'S2', 'C':'C0', 'G':'G0', 'P':'P2'}
    DEFAULTS           = {**DEFAULTS_S2, **DEFAULTS_P2}
    REQUIRES_DUAL_PASS = True

    def _reconstruct(self, data, p):
        t0  = time.time()
        fwd = data.forward
        bwd = data.backward   # spatially aligned: bwd[:,x] matches fwd[:,x]

        # ── Step 1: estimate PID lag from fwd-bwd difference signal ───────────
        diff   = fwd - bwd                        # shape (Ny, Nx)
        pid_lag= self._estimate_pid_lag(diff)

        # ── Step 2: apply PID correction to forward scan ──────────────────────
        fwd_corr = self._pid_correct(fwd, pid_lag, p['pid_corr_weight'])

        # ── Step 3: information-weighted fusion ───────────────────────────────
        # C0: no classifier — equal weights, so simple average
        R_base = np.array([[p['sigma_meas']**2]])
        fused  = 0.5 * fwd_corr + 0.5 * bwd
        # Effective noise after fusion: sigma / sqrt(2)
        R_fused = np.array([[p['sigma_meas']**2 / 2.0]])

        # ── Step 4: Kalman/RTS on fused observation ───────────────────────────
        A  = make_A_s2()
        C  = make_C(2)
        Q  = make_Q_s2(p)
        P0 = make_P0_s2()

        quality = build_quality_c0(fused)

        xf, Pf, xp, Pp, mon = kalman_forward(
            fused, quality, A, C, Q, R_fused, P0,
            p['burn_in'], p['alpha_R'])

        xs, Ps, sc = kalman_backward(xf, Pf, xp, Pp, A)
        mon['smooth_corr'] = sc

        return self._build_result(
            xs, Ps, quality, mon, p, t0,
            diagnostics={
                'pid_lag_estimate' : pid_lag,
                'difference_signal': diff,
                'fused_observation': fused,
            })

    @staticmethod
    def _estimate_pid_lag(diff):
        """
        Estimate PID feedback lag from forward-backward difference signal.

        The difference (fwd - bwd) is proportional to the derivative of the
        surface convolved with the PID step response. The lag lambda is
        estimated by fitting an exponential to the autocorrelation of diff.

        Returns
        -------
        float — estimated lag length in pixels (0 if estimation fails)
        """
        Ny, Nx = diff.shape
        max_lag = min(30, Nx // 4)
        ac = np.zeros(max_lag)
        count = 0
        for k in range(Ny):
            d = diff[k] - diff[k].mean()
            if d.std() < 1e-8:
                continue
            full = np.correlate(d, d, 'full')
            mid  = len(full) // 2
            seg  = full[mid:mid + max_lag]
            if seg[0] > 0:
                ac += seg / seg[0]
                count += 1
        if count == 0:
            return 0.0
        ac /= count

        # Fit exp(-lag/lambda) to ac[1:]
        lags  = np.arange(1, max_lag)
        logac = np.log(np.maximum(ac[1:], 1e-10))
        valid = logac > np.log(0.05)
        if valid.sum() < 3:
            return 0.0
        coeffs = np.polyfit(lags[valid], logac[valid], 1)
        if coeffs[0] >= 0:
            return 0.0
        return float(max(-1.0 / coeffs[0], 0.0))

    @staticmethod
    def _pid_correct(fwd, lag, weight):
        """
        Partial correction of PID lag in the forward scan.

        The PID lag produces an exponential tail after step features.
        The correction is estimated from the forward-backward difference:
            correction ≈ weight * (fwd - lag_filtered_fwd)

        Parameters
        ----------
        fwd    : (Ny, Nx) forward scan
        lag    : float — estimated lag length in pixels
        weight : float — correction attenuation (0.4 = conservative)
        """
        if lag < 0.5:
            return fwd.copy()
        Ny, Nx = fwd.shape
        corrected = fwd.copy()
        alpha = 1.0 - np.exp(-1.0 / max(lag, 0.1))
        for k in range(Ny):
            # Causal low-pass (simulates PID response)
            smoothed = np.zeros(Nx)
            state = float(fwd[k, 0])
            for x in range(Nx):
                state = (1 - alpha) * state + alpha * fwd[k, x]
                smoothed[x] = state
            # Correction: move fwd toward the un-lagged version
            corrected[k] = fwd[k] + weight * (fwd[k] - smoothed)
        return corrected.astype(np.float32)
