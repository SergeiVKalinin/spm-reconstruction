"""
spm_data/artifacts.py
---------------------
Artifact simulators. Each function takes a clean surface and a corruption
mask, applies one artifact channel, and returns the corrupted surface and
updated mask.

The mask encodes per-pixel corruption severity in [0, 1]:
  0.0 = completely clean
  1.0 = completely destroyed (crash)
  0.0-1.0 = partial corruption (oscillation, tip change)

Functions
---------
apply_drift(s, mask, rng, **kw)
apply_tip_change(s, mask, rng, **kw)
apply_oscillation(s, mask, rng, **kw)
apply_crash(s, mask, rng, **kw)
apply_noise(s, mask, rng, sigma_meas)
apply_scan_line_noise(s, mask, rng, **kw)
apply_tip_convolution(s, mask, rng, **kw)
apply_streaks(s, mask, rng, **kw)
apply_line_jitter(s, mask, rng, **kw)

Usage
-----
    from spm_data.artifacts import apply_crash, apply_drift
    import numpy as np
    rng = np.random.default_rng(42)
    s   = np.random.rand(256, 256).astype(np.float32)
    mask = np.zeros((256,256), dtype=np.float32)
    s, mask = apply_crash(s, mask, rng)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ── Drift ─────────────────────────────────────────────────────────────────────

def apply_drift(s, mask, rng,
                drift_lat=0.015,
                drift_z=0.10,
                **kwargs):
    """
    Lateral + vertical thermal drift.

    Lateral drift: quadratic accumulation — the tip drifts more toward
    the end of the scan as the scanner assembly thermally equilibrates.
    Displacement at line k: delta_x(k) = drift_lat * (k/Ny)^2 * Nx pixels.

    Vertical drift: linear ramp over the full image.
    Total z-offset at last line: drift_z (in normalised units).

    Drift does not destroy pixels — it displaces them. The mask is unchanged.

    Parameters
    ----------
    drift_lat : float — lateral drift amplitude (fraction of Nx)
    drift_z   : float — total vertical drift (normalised z-range)
    """
    Ny, Nx = s.shape
    out = s.copy()

    # Lateral drift (quadratic accumulation)
    drift_x = drift_lat * (np.arange(Ny) / Ny)**2 * Nx
    for k in range(Ny):
        sh = drift_x[k]
        i  = int(sh)
        f  = sh - i
        shifted = ((1 - f) * np.roll(out[k], i)
                   + f * np.roll(out[k], min(i + 1, Nx - 1)))
        out[k] = shifted

    # Vertical drift (linear ramp)
    out += np.linspace(0, drift_z, Ny)[:, None]

    return out.astype(np.float32), mask


# ── Tip change ────────────────────────────────────────────────────────────────

def apply_tip_change(s, mask, rng,
                     n_events=2,
                     sigma_min=2.0,
                     sigma_max=8.0,
                     **kwargs):
    """
    Tip-state change events: sudden change in effective tip radius and
    z-offset, persisting for all subsequent lines.

    Modelled as a Gaussian blur (PSF broadening) plus a z-offset shift,
    both accumulating at each event.

    Tip changes do not destroy pixels but bias them — the mask is left clean.
    The classifier detects them from innovation spikes, not from deviation.

    Parameters
    ----------
    n_events  : int   — number of tip-change events
    sigma_min : float — minimum additional blur sigma (pixels)
    sigma_max : float — maximum additional blur sigma (pixels)
    """
    Ny, Nx = s.shape
    out = s.copy()

    # Choose event lines (not too close to edges)
    min_line = max(10, Ny // 10)
    max_line = Ny - min_line
    if min_line >= max_line or n_events == 0:
        return out.astype(np.float32), mask

    event_lines = np.sort(
        rng.choice(np.arange(min_line, max_line),
                   size=min(n_events, max_line - min_line),
                   replace=False)
    )

    sig  = np.ones(Ny)
    zoff = np.zeros(Ny)

    for ev in event_lines:
        delta_sigma = rng.uniform(sigma_min, sigma_max)
        delta_z     = rng.uniform(-0.10, 0.10)
        sig[ev:]  += delta_sigma
        zoff[ev:] += delta_z

    for k in range(Ny):
        if sig[k] > 1.5:
            out[k] = gaussian_filter1d(out[k], sig[k])
        out[k] += zoff[k]

    return out.astype(np.float32), mask


# ── Oscillation ───────────────────────────────────────────────────────────────

def apply_oscillation(s, mask, rng,
                      n_bursts=2,
                      amp_min=0.08,
                      amp_max=0.18,
                      freq_min=0.08,
                      freq_max=0.22,
                      **kwargs):
    """
    Feedback oscillation bursts: the Z-servo loop oscillates near resonance,
    producing periodic stripe artifacts confined to burst regions.

    Amplitude is Gaussian-enveloped within the burst (strongest at centre).

    Oscillation partially corrupts pixels — mask is set to 1 - quality
    proportional to amplitude.

    Parameters
    ----------
    n_bursts  : int   — number of oscillation bursts
    amp_min   : float — minimum burst amplitude (normalised z)
    amp_max   : float — maximum burst amplitude
    freq_min  : float — minimum oscillation frequency (cycles/pixel)
    freq_max  : float — maximum oscillation frequency
    """
    Ny, Nx = s.shape
    out  = s.copy()
    msk  = mask.copy()
    xarr = np.arange(Nx)

    min_line = max(5, Ny // 20)
    max_line = Ny - min_line
    if min_line >= max_line or n_bursts == 0:
        return out.astype(np.float32), msk

    for _ in range(n_bursts):
        a = rng.integers(min_line, max_line - Ny//10)
        b = min(a + rng.integers(Ny//20, Ny//5), Ny)
        amp  = rng.uniform(amp_min, amp_max)
        freq = rng.uniform(freq_min, freq_max)
        phase = rng.uniform(0, 2*np.pi)

        for k in range(a, b):
            env = amp * np.exp(-0.04 * abs(k - (a + b) // 2))
            out[k] += env * np.sin(2*np.pi*freq*xarr + phase)
            severity = min(env / amp_max, 1.0)
            msk[k]   = np.maximum(msk[k], 0.8 * severity)

    return out.astype(np.float32), msk.astype(np.float32)


# ── Crash ─────────────────────────────────────────────────────────────────────

def apply_crash(s, mask, rng,
                n_crashes=2,
                min_width=1,
                max_width=8,
                partial_prob=0.35,
                **kwargs):
    """
    Line crashes: the tip retracts, saturates the electronics, or otherwise
    produces lines that contain no surface information.

    Full crashes: entire line replaced with Uniform(-0.15, 1.30).
    Partial crashes (prob=partial_prob): only left or right half replaced.

    Crashed pixels set mask to 1.0 (completely destroyed).

    Parameters
    ----------
    n_crashes    : int   — number of crash events
    min_width    : int   — minimum crash width (lines)
    max_width    : int   — maximum crash width (lines)
    partial_prob : float — probability of a partial (half-line) crash
    """
    Ny, Nx = s.shape
    out = s.copy()
    msk = mask.copy()

    min_line = max(5, Ny // 20)
    max_line = Ny - max_width - 5
    if min_line >= max_line or n_crashes == 0:
        return out.astype(np.float32), msk

    for _ in range(n_crashes):
        a = rng.integers(min_line, max_line)
        w = rng.integers(min_width, max_width + 1)
        b = min(a + w, Ny)

        if rng.random() < partial_prob:
            # Partial crash: left or right half
            if rng.random() < 0.5:
                x0, x1 = 0, Nx // 2
            else:
                x0, x1 = Nx // 2, Nx
            out[a:b, x0:x1] = rng.uniform(-0.15, 1.30, (b-a, x1-x0))
            msk[a:b, x0:x1] = 1.0
        else:
            # Full crash
            out[a:b] = rng.uniform(-0.15, 1.30, (b-a, Nx))
            msk[a:b] = 1.0

    return out.astype(np.float32), msk.astype(np.float32)


# ── Measurement noise ─────────────────────────────────────────────────────────

def apply_noise(s, mask, rng, sigma_meas=0.025, **kwargs):
    """
    Gaussian measurement noise added to all pixels independently.
    Applied last, after all other artifacts.

    Parameters
    ----------
    sigma_meas : float — noise standard deviation (normalised z-range)
    """
    noise = rng.normal(0, sigma_meas, s.shape).astype(np.float32)
    return (s + noise).astype(np.float32), mask


# ── Scan line noise ───────────────────────────────────────────────────────────

def apply_scan_line_noise(s, mask, rng,
                          freq_min=0.04,
                          freq_max=0.12,
                          amp_min=0.015,
                          amp_max=0.060,
                          n_frequencies=3,
                          **kwargs):
    """
    Periodic electrical interference producing horizontal stripes at fixed
    spatial frequencies. Models 50/60 Hz pickup or switching noise from
    nearby equipment.

    Unlike oscillation bursts this artifact:
    - Is present throughout the entire scan (every line)
    - Has fixed spatial frequency (does not vary line to line)
    - Has constant amplitude (not Gaussian-enveloped)
    - Originates from the electronics, not the feedback loop

    The mask is set proportionally to amplitude — these pixels carry
    biased rather than random information.

    Parameters
    ----------
    freq_min, freq_max : float — frequency range (cycles/pixel)
    amp_min, amp_max   : float — amplitude range (normalised z)
    n_frequencies      : int   — number of simultaneous interference tones
    """
    Ny, Nx = s.shape
    out    = s.copy()
    msk    = mask.copy()
    xarr   = np.arange(Nx)

    total_amp = 0.0
    for _ in range(n_frequencies):
        freq  = rng.uniform(freq_min, freq_max)
        amp   = rng.uniform(amp_min, amp_max)
        phase = rng.uniform(0, 2 * np.pi)
        # Same stripe on every line — constant phase
        stripe = amp * np.sin(2 * np.pi * freq * xarr + phase)
        for k in range(Ny):
            out[k] += stripe
        total_amp += amp

    # Mask: partial corruption proportional to amplitude relative to range
    severity = min(total_amp / (amp_max * n_frequencies), 1.0) * 0.6
    msk = np.maximum(msk, severity)

    return out.astype(np.float32), msk.astype(np.float32)


# ── Tip convolution ───────────────────────────────────────────────────────────

def apply_tip_convolution(s, mask, rng,
                           sigma_min=0.8,
                           sigma_max=3.0,
                           asymmetry=0.3,
                           **kwargs):
    """
    Blurring of surface features by the finite tip radius and shape.

    The tip acts as a spatial low-pass filter. Sharp features (step edges,
    nanoparticles) appear broadened. The effective PSF is modelled as an
    asymmetric Gaussian — slightly broader in the fast-scan direction
    because the tip approaches features from one side.

    Tip convolution does not destroy information but reduces spatial
    resolution. The mask is set to a small constant (blurred pixels
    are biased but not random).

    Parameters
    ----------
    sigma_min, sigma_max : float — tip PSF width range (pixels)
    asymmetry            : float — fractional extra broadening in x vs y
                           (0 = symmetric, 0.5 = 50% wider in x)
    """
    from scipy.ndimage import gaussian_filter

    Ny, Nx = s.shape
    sigma  = rng.uniform(sigma_min, sigma_max)
    sigma_x = sigma * (1.0 + asymmetry)
    sigma_y = sigma

    out = gaussian_filter(s.astype(np.float64),
                          sigma=[sigma_y, sigma_x]).astype(np.float32)

    # Mask: small constant — blurring is uniform and mild
    severity = min(sigma / sigma_max * 0.4, 0.4)
    msk = np.maximum(mask, severity)

    return out, msk.astype(np.float32)


# ── Streaks ───────────────────────────────────────────────────────────────────

def apply_streaks(s, mask, rng,
                  n_streaks=6,
                  max_length=12,
                  amplitude_min=0.05,
                  amplitude_max=0.25,
                  **kwargs):
    """
    Short horizontal streaks from brief electrical transients or
    contaminant particles passing under the tip.

    Each streak:
    - Starts at a random pixel (k, x) and extends rightward
    - Has a length of 1–max_length pixels
    - Carries a random offset (positive or negative)
    - Decays exponentially back to the true surface at the streak end

    Unlike a full crash, streaks affect only a few pixels per event.
    They are harder to detect than crashes because they are short.

    Parameters
    ----------
    n_streaks      : int   — number of streak events
    max_length     : int   — maximum streak length in pixels
    amplitude_min  : float — minimum streak offset (normalised z)
    amplitude_max  : float — maximum streak offset
    """
    Ny, Nx = s.shape
    out    = s.copy()
    msk    = mask.copy()

    for _ in range(n_streaks):
        k    = rng.integers(0, Ny)
        x0   = rng.integers(0, Nx - 1)
        L    = rng.integers(1, min(max_length + 1, Nx - x0))
        amp  = rng.uniform(amplitude_min, amplitude_max)
        sign = rng.choice([-1, 1])

        # Exponential decay profile
        decay  = np.exp(-np.arange(L) / max(L / 3, 1.0))
        streak = sign * amp * decay

        x_end = x0 + L
        out[k, x0:x_end] += streak[:x_end - x0].astype(np.float32)
        # Severity proportional to amplitude and relative to crash range
        sev = min(amp / amplitude_max * 0.85, 1.0)
        msk[k, x0:x_end] = np.maximum(msk[k, x0:x_end], sev)

    return out.astype(np.float32), msk.astype(np.float32)


# ── Line jitter ───────────────────────────────────────────────────────────────

def apply_line_jitter(s, mask, rng,
                      jitter_max=4,
                      jitter_prob=0.15,
                      persistent_prob=0.3,
                      **kwargs):
    """
    Cyclic lateral shift of individual scan lines by a few pixels.

    Caused by mechanical vibration, acoustic noise, or electrical
    interference in the x-scanner drive. Each affected line is rolled
    cyclically by ±jitter_max pixels. Some jitter events persist across
    several consecutive lines (a brief resonance); others are single-line.

    Cyclic roll is used (np.roll) — the image wraps at the edges.
    This preserves all pixels but introduces lateral misregistration.

    The mask reflects that jitter pixels are shifted rather than
    destroyed — partial corruption (0.3–0.7).

    Parameters
    ----------
    jitter_max      : int   — maximum shift in pixels (±)
    jitter_prob     : float — probability any given line is jittered
    persistent_prob : float — probability a jitter event persists to
                      the next line (creates multi-line jitter bursts)
    """
    Ny, Nx = s.shape
    out    = s.copy()
    msk    = mask.copy()

    current_shift = 0
    for k in range(Ny):
        # Decide whether this line is jittered
        if current_shift != 0:
            # Carry over from previous line?
            if rng.random() > persistent_prob:
                current_shift = 0
        else:
            if rng.random() < jitter_prob:
                current_shift = rng.integers(-jitter_max, jitter_max + 1)
                if current_shift == 0:
                    current_shift = 1

        if current_shift != 0:
            out[k] = np.roll(s[k], current_shift)
            # Severity: proportional to shift magnitude
            sev = min(abs(current_shift) / jitter_max * 0.7, 0.7)
            msk[k] = np.maximum(msk[k], sev)

    return out.astype(np.float32), msk.astype(np.float32)


# ── Registry ──────────────────────────────────────────────────────────────────

ARTIFACT_REGISTRY = {
    'drift':           apply_drift,
    'tip_change':      apply_tip_change,
    'oscillation':     apply_oscillation,
    'crash':           apply_crash,
    'scan_line_noise': apply_scan_line_noise,
    'tip_convolution': apply_tip_convolution,
    'streaks':         apply_streaks,
    'line_jitter':     apply_line_jitter,
}

# Fixed application order: physics-motivated
# drift and tip changes happen first (instrument effects),
# then oscillation (feedback), then crash (mechanical),
# then noise (always last — electronic, independent).
# Fixed application order: physics-motivated
# tip_convolution first (instrument optics), then scanner artifacts,
# then electronic/mechanical noise, then crashes, then noise last
ARTIFACT_ORDER = ['tip_convolution', 'drift', 'line_jitter',
                  'tip_change', 'oscillation', 'scan_line_noise',
                  'crash']


def apply_artifacts(s, rng, artifacts=None, sigma_meas=0.025, **kwargs):
    """
    Apply a set of artifact channels to a clean surface.

    Parameters
    ----------
    s         : (Ny, Nx) float32 — clean true surface
    rng       : np.random.Generator
    artifacts : list of str or None — which artifacts to apply.
                None means all four: drift, tip_change, oscillation, crash.
    sigma_meas: float — measurement noise sigma

    Returns
    -------
    corrupted  : (Ny, Nx) float32 — corrupted observation
    mask       : (Ny, Nx) float32 — corruption severity [0, 1]
    """
    if artifacts is None:
        artifacts = list(ARTIFACT_ORDER)

    unknown = set(artifacts) - set(ARTIFACT_REGISTRY.keys())
    if unknown:
        raise ValueError(f"Unknown artifacts: {unknown}. "
                         f"Choose from: {list(ARTIFACT_REGISTRY.keys())}")

    corrupted = s.copy().astype(np.float32)
    mask      = np.zeros(s.shape, dtype=np.float32)

    # Apply in fixed physical order
    for name in ARTIFACT_ORDER:
        if name in artifacts:
            fn = ARTIFACT_REGISTRY[name]
            corrupted, mask = fn(corrupted, mask, rng, **kwargs)

    # Noise always last
    corrupted, mask = apply_noise(corrupted, mask, rng,
                                  sigma_meas=sigma_meas)

    return corrupted, mask
