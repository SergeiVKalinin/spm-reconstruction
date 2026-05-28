"""
spm_data/dual_pass.py
---------------------
Backward scan synthesis for dual-pass data generation.

In dual-pass SPM, the tip scans each line twice:
  Forward  (trace):   left -> right at velocity +v
  Backward (retrace): right -> left at velocity -v, then spatially reversed

The two scans differ in three ways:
  1. PID lag direction — trails rightward in forward, leftward in backward
  2. Inter-scan drift  — backward acquired ~t_line later, extra lateral offset
  3. Independent noise — different electronic noise realisation

Functions
---------
pid_lag_forward(line, lam)    - apply causal PID lag in +x direction
pid_lag_backward(line, lam)   - apply causal PID lag in -x direction
make_backward_scan(forward, true_surface, rng, **kw)
                              - synthesise a physically consistent backward scan

Usage
-----
    from spm_data.dual_pass import make_backward_scan
    bwd = make_backward_scan(fwd, true_surface, rng, pid_lag=8.0)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def pid_lag_forward(line, lam):
    """
    Apply PID feedback lag in the forward (+x) direction.

    Models the Z-servo response to surface features: after a step edge
    the servo trails with an exponential decay of length lambda pixels.

    Implementation: first-order IIR low-pass filter (causal, left to right).

    Parameters
    ----------
    line : (Nx,) array — input scan line
    lam  : float       — lag length in pixels (0 = no lag)
    """
    if lam < 0.5:
        return line.copy()
    out   = np.zeros_like(line)
    alpha = 1.0 - np.exp(-1.0 / lam)   # discrete decay
    state = float(line[0])
    for i, s in enumerate(line):
        state  = (1 - alpha) * state + alpha * s
        out[i] = state
    return out


def pid_lag_backward(line, lam):
    """
    Apply PID feedback lag in the backward (-x) direction.

    The backward scan travels right-to-left, so the lag trails leftward
    when viewed in the spatially aligned (reversed) coordinate.

    Implementation: apply forward lag to the spatially reversed line,
    then reverse back.

    Parameters
    ----------
    line : (Nx,) array — input scan line (in forward spatial coordinates)
    lam  : float       — lag length in pixels
    """
    if lam < 0.5:
        return line.copy()
    lagged_reversed = pid_lag_forward(line[::-1], lam)
    return lagged_reversed[::-1]


def make_backward_scan(forward, true_surface, rng,
                       pid_lag=8.0,
                       tip_sigma_asymmetry=0.0,
                       drift_interline=0.3,
                       sigma_meas=0.025,
                       **kwargs):
    """
    Synthesise a physically consistent backward scan from the true surface.

    The backward scan is generated fresh from the true surface (not from
    the forward scan) with three directional differences applied:

    1. PID lag trails leftward (backward direction)
    2. Tip PSF uses slightly different effective width (tip asymmetry)
    3. Sub-pixel lateral offset from inter-scan drift
    4. Independent noise realisation

    The returned array is spatially aligned: pixel x in forward corresponds
    to pixel x in backward (both in left-to-right coordinates).

    Parameters
    ----------
    forward             : (Ny, Nx) — the forward scan (for reference)
    true_surface        : (Ny, Nx) — the true surface (noise-free)
    rng                 : np.random.Generator
    pid_lag             : float — PID lag length in pixels
    tip_sigma_asymmetry : float — extra blur on backward (models tip asymmetry)
                          0 = symmetric tip
    drift_interline     : float — sub-pixel lateral offset between scans (pixels)
    sigma_meas          : float — measurement noise std

    Returns
    -------
    backward : (Ny, Nx) float32 — backward scan in forward spatial coordinates
    """
    Ny, Nx = true_surface.shape
    backward = np.zeros((Ny, Nx), dtype=np.float32)

    for k in range(Ny):
        line = true_surface[k].copy()

        # 1. Inter-scan drift: tiny lateral shift
        if drift_interline > 0:
            i = int(drift_interline)
            f = drift_interline - i
            line = ((1-f)*np.roll(line, i)
                    + f*np.roll(line, min(i+1, Nx-1)))

        # 2. Tip asymmetry: slight extra blur for backward facet
        if tip_sigma_asymmetry > 0:
            line = gaussian_filter1d(line, tip_sigma_asymmetry)

        # 3. PID lag in backward direction
        line = pid_lag_backward(line, pid_lag)

        # 4. Independent noise
        line += rng.normal(0, sigma_meas, Nx)

        backward[k] = line.astype(np.float32)

    return backward
