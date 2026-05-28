"""
spm_data/surfaces.py
--------------------
True surface generators. Each function returns a (Ny, Nx) float32 array
normalised to [0, 1]. All randomness comes from the rng argument so
results are fully reproducible from a seed.

Functions
---------
sinusoidal(Ny, Nx, rng)    - smooth multi-frequency surface
step_terrace(Ny, Nx, rng)  - flat terraces with sharp atomic steps
rough(Ny, Nx, rng)         - amorphous surface with power-law PSD
nanoparticle(Ny, Nx, rng)  - flat background with hemispherical islands

Usage
-----
    from spm_data.surfaces import sinusoidal, step_terrace
    import numpy as np
    rng = np.random.default_rng(42)
    s = sinusoidal(256, 256, rng)   # shape (256, 256)
"""

import numpy as np


def _normalise(s):
    """Normalise array to [0, 1]. Handles flat surfaces gracefully."""
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-10:
        return np.zeros_like(s, dtype=np.float32)
    return ((s - lo) / (hi - lo)).astype(np.float32)


def sinusoidal(Ny, Nx, rng, n_modes=4):
    """
    Smooth surface as a superposition of sinusoidal modes.
    Good general-purpose test surface. Features at multiple scales.

    Parameters
    ----------
    n_modes : int — number of frequency components (default 4)
    """
    xv = np.linspace(0, 4*np.pi, Nx)
    yv = np.linspace(0, 4*np.pi, Ny)
    X, Y = np.meshgrid(xv, yv)
    amplitudes = [1.0, 0.4, 0.2, 0.1]

    s = np.zeros((Ny, Nx), dtype=np.float64)
    for i in range(min(n_modes, 4)):
        fx = rng.uniform(0.5, 3.0)
        fy = rng.uniform(0.5, 3.0)
        px = rng.uniform(0, 2*np.pi)
        py = rng.uniform(0, 2*np.pi)
        s += amplitudes[i] * np.sin(fx*X + px) * np.cos(fy*Y + py)

    return _normalise(s)


def step_terrace(Ny, Nx, rng, n_steps=5, step_height=0.15):
    """
    Atomically flat terraces separated by sharp steps.
    Models crystalline surfaces: Si, graphene, MoS2, HOPG.

    Step edges are the hardest test for edge preservation in the GP smoother
    — a genuine discontinuity of ~step_height over 1-2 pixels.

    Parameters
    ----------
    n_steps      : int   — number of terraces
    step_height  : float — height of each step as fraction of z-range
    """
    # Smooth underlying landscape to determine terrace positions
    base = sinusoidal(Ny, Nx, rng, n_modes=2)

    # Quantise into flat terraces
    thresholds = np.linspace(0, 1, n_steps + 1)
    s = np.zeros((Ny, Nx), dtype=np.float64)
    for i in range(n_steps):
        s += step_height * (base > thresholds[i])

    return _normalise(s)


def rough(Ny, Nx, rng, beta=3.0):
    """
    Amorphous rough surface with power-law power spectral density.
    Models oxide films, biological membranes, metallic glass.

    PSD(f) ~ f^{-beta}. Higher beta = smoother (longer correlation length).
    Typical values: beta=2 (very rough) to beta=4 (smooth amorphous).

    This is the hardest surface type for over-smoothing detection:
    genuine high-frequency roughness looks like noise to a naive filter.

    Parameters
    ----------
    beta : float — PSD exponent (default 3.0)
    """
    # Generate in Fourier space
    freq_x = np.fft.rfftfreq(Nx)
    freq_y = np.fft.fftfreq(Ny)
    Fx, Fy = np.meshgrid(freq_x, freq_y)
    f_mag  = np.sqrt(Fx**2 + Fy**2)
    f_mag[0, 0] = 1.0  # avoid DC singularity

    # Complex amplitude with power-law envelope
    amp = (rng.standard_normal((Ny, Nx//2+1))
           + 1j * rng.standard_normal((Ny, Nx//2+1)))
    amp *= f_mag ** (-beta / 2)
    amp[0, 0] = 0.0  # zero DC component

    s = np.fft.irfft2(amp, s=(Ny, Nx))
    return _normalise(s)


def nanoparticle(Ny, Nx, rng, n_particles=8, radius=0.06, height=0.3):
    """
    Flat background with hemispherical nanoparticles.
    Models gold nanoparticles on mica, quantum dots, catalyst particles.

    Parameters
    ----------
    n_particles : int   — number of particles
    radius      : float — particle radius as fraction of min(Ny, Nx)
    height      : float — particle height as fraction of z-range
    """
    s = np.zeros((Ny, Nx), dtype=np.float64)
    r_px = radius * min(Ny, Nx)  # radius in pixels

    yc = rng.uniform(r_px, Ny-r_px, n_particles)
    xc = rng.uniform(r_px, Nx-r_px, n_particles)

    yv = np.arange(Ny)
    xv = np.arange(Nx)
    Y, X = np.meshgrid(yv, xv, indexing='ij')

    for y0, x0 in zip(yc, xc):
        dist2 = (Y-y0)**2 + (X-x0)**2
        inside = dist2 < r_px**2
        # Hemispherical cap
        s[inside] = np.maximum(
            s[inside],
            height * np.sqrt(np.maximum(1 - dist2[inside]/r_px**2, 0))
        )

    return _normalise(s)


# Registry: name -> function
SURFACE_REGISTRY = {
    'sinusoidal':   sinusoidal,
    'step_terrace': step_terrace,
    'rough':        rough,
    'nanoparticle': nanoparticle,
}


def make_surface(surface_type, Ny, Nx, rng, **kwargs):
    """
    Generate a surface by name.

    Parameters
    ----------
    surface_type : str — one of 'sinusoidal', 'step_terrace', 'rough', 'nanoparticle'
    Ny, Nx       : int — image dimensions
    rng          : np.random.Generator

    Returns
    -------
    np.ndarray of shape (Ny, Nx), dtype float32, values in [0, 1]
    """
    if surface_type not in SURFACE_REGISTRY:
        raise ValueError(
            f"Unknown surface_type '{surface_type}'. "
            f"Choose from: {list(SURFACE_REGISTRY.keys())}"
        )
    fn = SURFACE_REGISTRY[surface_type]
    # Pass only kwargs that the function accepts
    import inspect
    valid = inspect.signature(fn).parameters.keys()
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return fn(Ny, Nx, rng, **filtered)
