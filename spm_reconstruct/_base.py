"""
spm_reconstruct/_base.py
------------------------
Shared mathematics and base class for all SPM reconstruction algorithms.

Every model imports from here. The Kalman equations appear exactly once.

Contents
--------
Maths:
    sigmoid(x)
    kalman_forward(...)      - forward Kalman pass, any state dimension
    kalman_backward(...)     - RTS backward smoother
    estimate_ell(...)        - lateral correlation length from autocorrelation
    gp_smooth_line(...)      - edge-aware bilateral GP smoother along x

Quality scoring:
    build_quality_c0(obs)            - all ones (no classifier)
    build_quality_ct(obs, p)         - soft sigmoid threshold classifier
    update_tip_scores(q, innov, p)   - add tip-change scores after forward pass

State matrix factories:
    make_A_s2(), make_A_s6()
    make_C(sdim)
    make_Q_s2(p), make_Q_s6(p)
    make_P0_s2(), make_P0_s6()

Default parameter dicts:
    DEFAULTS_S2, DEFAULTS_S6_EXTRA, DEFAULTS_CT, DEFAULTS_GC, DEFAULTS_P2

Base class:
    SPMReconstructor   - abstract base; all models inherit from this
"""

import time
import numpy as np
from abc import ABC, abstractmethod


# ── Activation function ───────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# ── Forward Kalman pass ───────────────────────────────────────────────────────

def kalman_forward(obs, quality, A, C, Q, R_base, P0, burn_in, alpha_R):
    """
    Forward Kalman pass with online R estimation (Mehra method).

    Works for any state dimension SDIM — determined from A.shape[0].
    Used by all 12 models. Do not copy this into individual model files.

    Parameters
    ----------
    obs      : (Ny, Nx) float32   — observed scan (corrupted)
    quality  : (Ny, Nx) float32   — per-pixel quality in [0,1]
    A        : (SDIM, SDIM)       — transition matrix
    C        : (1, SDIM)          — observation matrix
    Q        : (SDIM, SDIM)       — process noise covariance
    R_base   : (1, 1)             — initial measurement noise covariance
    P0       : (SDIM, SDIM)       — initial state covariance
    burn_in  : int                — lines before online R estimation starts
    alpha_R  : float              — EMA decay for online R (0.9–0.99)

    Returns
    -------
    xf, Pf, xp, Pp   — forward state/covariance estimates and predictions
    monitoring        — dict with innov_map, innov_rms, gain, r_track, unc_fwd
    """
    Ny, Nx = obs.shape
    SDIM   = A.shape[0]

    xf  = np.zeros((Ny, Nx, SDIM), dtype=np.float64)
    Pf  = np.zeros((Ny, Nx, SDIM, SDIM), dtype=np.float64)
    xp  = np.zeros((Ny, Nx, SDIM), dtype=np.float64)
    Pp  = np.zeros((Ny, Nx, SDIM, SDIM), dtype=np.float64)

    innov_map = np.zeros((Ny, Nx), dtype=np.float32)
    innov_rms = np.zeros(Ny, dtype=np.float32)
    gain_mean = np.zeros(Ny, dtype=np.float32)
    r_track   = np.zeros(Ny, dtype=np.float32)
    unc_fwd   = np.zeros((Ny, Nx), dtype=np.float32)

    # Initialise line 0
    xf[0, :, 0] = obs[0]
    for px in range(Nx):
        Pf[0, px] = P0.copy()

    R_cur = np.array(R_base, dtype=np.float64)
    R_ema = float(R_base[0, 0])

    for k in range(1, Ny):
        # ── Predict ──────────────────────────────────────────────────────────
        for px in range(Nx):
            xp[k, px] = A @ xf[k-1, px]
            Pp[k, px] = A @ Pf[k-1, px] @ A.T + Q

        # ── Update ───────────────────────────────────────────────────────────
        line_inn  = np.zeros(Nx)
        line_gain = np.zeros(Nx)

        for px in range(Nx):
            q     = max(float(quality[k, px]), 1e-6)
            R_eff = R_cur / q
            S     = C @ Pp[k, px] @ C.T + R_eff
            K     = Pp[k, px] @ C.T / S[0, 0]       # shape (SDIM, 1)
            inn   = float(obs[k, px]) - float((C @ xp[k, px])[0])

            xf[k, px] = xp[k, px] + K.flatten() * inn

            # Clip to physical range
            xf[k, px, 0] = np.clip(xf[k, px, 0], -0.3, 1.4)
            if SDIM > 2:
                xf[k, px, 2] = np.clip(xf[k, px, 2], -0.3,  0.3)
                xf[k, px, 5] = np.clip(xf[k, px, 5], -0.05, 0.05)

            # Covariance update with regularisation
            Pf_new = (np.eye(SDIM) - np.outer(K.flatten(), C)) @ Pp[k, px]
            Pf_new = 0.5 * (Pf_new + Pf_new.T) + 1e-6 * np.eye(SDIM)
            Pf[k, px] = Pf_new

            line_inn[px]  = inn
            line_gain[px] = abs(float(K[0, 0]))
            unc_fwd[k, px]= np.sqrt(abs(Pf[k, px, 0, 0]))

        innov_map[k] = line_inn.astype(np.float32)
        innov_rms[k] = float(np.sqrt(np.mean(line_inn**2)))
        gain_mean[k] = float(np.mean(line_gain))

        # ── Online R estimation (Mehra, on clean lines only) ─────────────────
        if k >= burn_in and quality[k].mean() > 0.7:
            iv   = float(np.mean(line_inn**2))
            cpc  = float((C @ np.mean(Pp[k], axis=0) @ C.T)[0, 0])
            R_new = max(iv - cpc, float(R_base[0, 0]) * 0.1)
            R_ema = alpha_R * R_ema + (1 - alpha_R) * R_new
            R_cur = np.array([[R_ema]])
        r_track[k] = float(R_cur[0, 0])

    monitoring = dict(
        innov_map = innov_map,
        innov_rms = innov_rms,
        gain      = gain_mean,
        r_track   = r_track,
        unc_fwd   = unc_fwd,
    )
    return xf, Pf, xp, Pp, monitoring


# ── RTS backward smoother ─────────────────────────────────────────────────────

def kalman_backward(xf, Pf, xp, Pp, A):
    """
    Rauch-Tung-Striebel backward smoother.

    Refines every estimate using all lines (non-causal).
    Largest corrections occur at crash gaps and tip-change boundaries.

    Parameters
    ----------
    xf, Pf : forward filtered state and covariance
    xp, Pp : forward predicted state and covariance
    A      : transition matrix

    Returns
    -------
    xs         : (Ny, Nx, SDIM) smoothed state
    Ps         : (Ny, Nx, SDIM, SDIM) smoothed covariance
    smooth_corr: (Ny,) mean backward correction magnitude per line
    """
    Ny, Nx, SDIM = xf.shape
    xs = xf.copy()
    Ps = Pf.copy()
    smooth_corr = np.zeros(Ny, dtype=np.float32)

    for k in range(Ny - 2, -1, -1):
        corr_k = np.zeros(Nx)
        for px in range(Nx):
            Pp_reg = Pp[k+1, px] + 1e-6 * np.eye(SDIM)
            try:
                G = Pf[k, px] @ A.T @ np.linalg.inv(Pp_reg)
            except np.linalg.LinAlgError:
                G = np.zeros((SDIM, SDIM))

            dx = xs[k+1, px] - xp[k+1, px]
            raw = xf[k, px] + G @ dx

            # Clip smoothed state
            raw[0] = np.clip(raw[0], -0.3, 1.4)
            if SDIM > 2:
                raw[2] = np.clip(raw[2], -0.3,  0.3)
                raw[5] = np.clip(raw[5], -0.05, 0.05)

            xs[k, px] = raw
            Ps_new = Pf[k, px] + G @ (Ps[k+1, px] - Pp[k+1, px]) @ G.T
            Ps[k, px] = 0.5 * (Ps_new + Ps_new.T) + 1e-8 * np.eye(SDIM)
            corr_k[px] = abs(dx[0])

        smooth_corr[k] = float(np.mean(corr_k))

    return xs, Ps, smooth_corr


# ── GP smoother ───────────────────────────────────────────────────────────────

def estimate_ell(obs, quality, Nx):
    """
    Estimate lateral correlation length from gradient autocorrelation.

    Uses only clean lines (quality mean > 0.85) and the gradient
    of the observation (not raw height) to avoid drift inflation.

    Parameters
    ----------
    obs     : (Ny, Nx) observed scan
    quality : (Ny, Nx) quality map
    Nx      : int image width (used to set max_lag)

    Returns
    -------
    ell    : float — correlation length in pixels
    L      : int   — GP window half-width = clip(ceil(2*ell), 3, 15)
    """
    max_lag  = min(40, Nx // 4)
    q_line   = quality.mean(axis=1)
    clean    = [k for k in range(obs.shape[0]) if q_line[k] > 0.85]

    if len(clean) < 5:
        ell = 5.0
    else:
        grad_obs = np.diff(obs, axis=1)
        ac = np.zeros(max_lag)
        count = 0
        for k in clean[:60]:
            d    = grad_obs[k] - grad_obs[k].mean()
            full = np.correlate(d, d, 'full')
            mid  = len(full) // 2
            seg  = full[mid:mid + max_lag]
            if seg[0] > 0:
                ac += seg / seg[0]
                count += 1
        if count > 0:
            ac /= count

        lags  = np.arange(1, max_lag)
        logac = np.log(np.maximum(ac[1:], 1e-10))
        valid = logac > np.log(0.05)
        if valid.sum() >= 3:
            coeffs = np.polyfit(lags[valid], logac[valid], 1)
            ell = max(-1.0 / coeffs[0], 0.5) if coeffs[0] < 0 else 5.0
        else:
            ell = 5.0

    L = int(np.clip(np.ceil(2 * ell), 3,
                    max(5, min(int(0.06 * Nx), 15))))
    return ell, L


def gp_smooth_line(h, unc, q_px, ell, L, beta=2.5):
    """
    Edge-aware bilateral GP smoother along x (one line).

    Only applied where unc > threshold (called conditionally from _apply_gp).

    Parameters
    ----------
    h     : (Nx,) height estimates
    unc   : (Nx,) posterior std dev
    q_px  : (Nx,) per-pixel quality
    ell   : float — correlation length (pixels)
    L     : int   — window half-width (pixels)
    beta  : float — edge sensitivity (higher = sharper edges preserved)

    Returns
    -------
    out_h : (Nx,) smoothed heights
    out_u : (Nx,) updated uncertainties
    """
    nx   = len(h)
    half = L // 2
    grad = np.abs(np.gradient(h))
    gsc  = np.percentile(grad, 80) + 1e-8

    out_h = h.copy()
    out_u = unc.copy()

    for px in range(nx):
        lo  = max(0, px - half)
        hi  = min(nx, px + half + 1)
        dxs = np.arange(lo, hi) - px

        # Spatial weight
        w_sp = np.exp(-dxs**2 / (2 * ell**2))

        # Edge barrier weight
        w_edge = np.ones(hi - lo)
        for j, nb in enumerate(range(lo, hi)):
            lo_p = min(px, nb); hi_p = max(px, nb)
            if lo_p < hi_p:
                barrier = np.mean(grad[lo_p:hi_p]) / gsc
                w_edge[j] = np.exp(-beta * barrier)

        # Information weight
        info = q_px[lo:hi] / np.maximum(unc[lo:hi]**2, 1e-8)
        w    = w_sp * w_edge * info
        ws   = w.sum()

        if ws > 1e-10:
            out_h[px] = np.dot(w, h[lo:hi]) / ws
            out_u[px] = 1.0 / np.sqrt(np.dot(w, info) + 1e-10)

    return out_h, out_u


def apply_gp(xf, monitoring, quality, ell, L, p):
    """
    Conditionally apply GP smoothing to uncertain pixels (G=GC models).

    Only pixels where unc_fwd > gp_unc_thresh * sigma_meas are smoothed.
    Clean pixels on clean lines are never touched.

    Parameters
    ----------
    xf         : (Ny, Nx, SDIM) forward state (modified in place for height)
    monitoring : dict containing 'unc_fwd'
    quality    : (Ny, Nx) quality map
    ell, L     : GP parameters from estimate_ell()
    p          : parameter dict (needs 'gp_unc_thresh', 'sigma_meas', 'beta')

    Returns
    -------
    xf         : updated state
    monitoring : updated with 'gp_mask'
    """
    Ny, Nx = xf.shape[:2]
    unc_fwd = monitoring['unc_fwd']
    thresh  = p['gp_unc_thresh'] * p['sigma_meas']
    gp_mask = np.zeros((Ny, Nx), dtype=bool)
    beta    = p.get('beta', 2.5)

    for k in range(Ny):
        mask = unc_fwd[k] > thresh
        if mask.any():
            sm_h, sm_u = gp_smooth_line(
                xf[k, :, 0], unc_fwd[k], quality[k],
                ell, L, beta)
            xf[k, :, 0] = np.where(mask, sm_h, xf[k, :, 0])
            unc_fwd[k]  = np.where(mask, sm_u, unc_fwd[k])
            gp_mask[k]  = mask

    monitoring['unc_fwd'] = unc_fwd
    monitoring['gp_mask'] = gp_mask
    return xf, monitoring


# ── Quality scoring ───────────────────────────────────────────────────────────

def build_quality_c0(obs):
    """C0: no classifier — trust all pixels equally."""
    return np.ones(obs.shape, dtype=np.float32)


def build_quality_ct(obs, p):
    """
    CT: threshold classifier — soft sigmoid on crash, oscillation,
    and cross-line consistency features.

    Parameters
    ----------
    obs : (Ny, Nx) corrupted scan
    p   : dict with theta_c, alpha_c, theta_o, alpha_o, tau_cons
    """
    Ny, Nx  = obs.shape
    q       = np.ones((Ny, Nx), dtype=np.float32)
    history = []

    for k in range(Ny):
        D  = float(np.mean(np.abs(obs[k] - np.median(obs[k]))))
        O  = float(np.std(np.diff(obs[k])))
        qc = sigmoid(-p['alpha_c'] * (D - p['theta_c']))
        qo = sigmoid(-p['alpha_o'] * (O - p['theta_o']))

        qk = 1.0
        if len(history) >= 2:
            mh = np.mean(history[-5:], axis=0)
            qk = float(np.exp(
                -np.mean((obs[k] - mh)**2) / p['tau_cons']**2))

        q[k] = float(qc * qo * qk)
        history.append(obs[k].copy())
        if len(history) > 5:
            history.pop(0)

    return q


def update_tip_scores(q, innov_rms, p, burn_in):
    """
    Add tip-change scores after the forward pass provides innovation RMS.

    Parameters
    ----------
    q         : (Ny, Nx) existing quality map (modified in place)
    innov_rms : (Ny,) per-line innovation RMS from forward pass
    p         : dict with theta_t, alpha_t
    burn_in   : int

    Returns
    -------
    q : updated quality map
    """
    Ny, Nx = q.shape
    ref    = float(np.mean(innov_rms[burn_in:burn_in + 30]))
    if ref < 1e-8:
        ref = 1e-8

    ratio = innov_rms / ref
    qt    = sigmoid(-p['alpha_t'] * (ratio - p['theta_t']))
    qt[:burn_in] = 1.0

    for k in range(Ny):
        q[k] *= float(qt[k])

    return q


# ── State matrix factories ────────────────────────────────────────────────────

def make_A_s2():
    """2D transition: [h, drift_rate]."""
    return np.array([[1., 1.],
                     [0., 1.]])

def make_A_s6():
    """6D geometric transition: [s, sx, sk, sxx, sxk, skk]."""
    return np.array([
        [1, 0, 1, 0, 0, 0.5],
        [0, 1, 0, 0, 1, 0  ],
        [0, 0, 1, 0, 0, 1  ],
        [0, 0, 0, 1, 0, 0  ],
        [0, 0, 0, 0, 1, 0  ],
        [0, 0, 0, 0, 0, 1  ],
    ], dtype=float)

def make_C(sdim):
    """Observation matrix: observe height only."""
    C = np.zeros((1, sdim))
    C[0, 0] = 1.0
    return C

def make_Q_s2(p):
    return np.diag([p['sigma_s']**2, p['sigma_sk']**2])

def make_Q_s6(p):
    return np.diag([
        p['sigma_s']**2,   p['sigma_sx']**2,  p['sigma_sk']**2,
        p['sigma_sxx']**2, p['sigma_sxk']**2, p['sigma_skk']**2,
    ])

def make_P0_s2(scale=10.0):
    return scale * np.eye(2)

def make_P0_s6(scale=10.0):
    return np.diag([
        scale,     scale*10,  scale*10,
        scale*20,  scale*20,  scale*20,
    ])


# ── Default parameter dicts ───────────────────────────────────────────────────

DEFAULTS_S2 = dict(
    sigma_s    = 0.020,
    sigma_sk   = 0.010,
    sigma_meas = 0.050,
    alpha_R    = 0.950,
)

DEFAULTS_S6_EXTRA = dict(
    sigma_sx   = 0.015,
    sigma_sxx  = 0.005,
    sigma_sxk  = 0.005,
    sigma_skk  = 0.003,
)

DEFAULTS_CT = dict(
    theta_c  = 0.30,
    alpha_c  = 30.0,
    theta_o  = 0.20,
    alpha_o  = 25.0,
    tau_cons = 0.25,
    theta_t  = 3.50,
    alpha_t  = 4.00,
)

DEFAULTS_GC = dict(
    beta          = 2.5,
    gp_unc_thresh = 3.0,
)

DEFAULTS_P2 = dict(
    pid_corr_weight = 0.40,
    drift_interline = 0.30,
)


# ── Base class ────────────────────────────────────────────────────────────────

class SPMReconstructor(ABC):
    """
    Abstract base class for all SPM reconstruction models.

    Every model must define:
        MODEL_ID           : str   e.g. "(S2, C0, G0, P1)"
        AXES               : dict  e.g. {'S':'S2','C':'C0','G':'G0','P':'P1'}
        DEFAULTS           : dict  default parameter values
        REQUIRES_DUAL_PASS : bool

    And implement:
        _reconstruct(data, p) -> ReconstructionResult

    The public reconstruct() method resolves parameters and calls _reconstruct().
    """
    MODEL_ID           : str
    AXES               : dict
    DEFAULTS           : dict
    REQUIRES_DUAL_PASS : bool = False

    def __init__(self, params=None):
        self._user_params = params or {}

    def reconstruct(self, data):
        """
        Reconstruct a corrupted SPM image.

        Parameters
        ----------
        data : ScanData

        Returns
        -------
        ReconstructionResult
        """
        if self.REQUIRES_DUAL_PASS and not data.has_backward():
            raise ValueError(
                f"{self.MODEL_ID} requires dual-pass data "
                "(data.backward is None)."
            )

        # Resolve parameters: defaults + size-aware + user overrides
        p = self.DEFAULTS.copy()
        p['burn_in'] = max(25, int(0.10 * data.Ny))
        p.update(self._user_params)

        return self._reconstruct(data, p)

    @abstractmethod
    def _reconstruct(self, data, p):
        """Implement the reconstruction. Called by reconstruct()."""
        ...

    def _build_result(self, xs, Ps, quality, monitoring, p, t0,
                      diagnostics=None):
        """
        Build a ReconstructionResult from smoother outputs.
        Called at the end of every _reconstruct() implementation.
        """
        from spm_data.types import ReconstructionResult
        return ReconstructionResult(
            reconstructed  = xs[:, :, 0].astype(np.float32),
            uncertainty    = np.sqrt(np.abs(Ps[:, :, 0, 0])).astype(np.float32),
            quality        = quality.astype(np.float32),
            model_id       = self.MODEL_ID,
            params_used    = p,
            runtime_s      = time.time() - t0,
            innovation_rms = monitoring.get('innov_rms'),
            kalman_gain    = monitoring.get('gain'),
            r_estimate     = monitoring.get('r_track'),
            rts_correction = monitoring.get('smooth_corr'),
            diagnostics    = diagnostics or {},
        )
