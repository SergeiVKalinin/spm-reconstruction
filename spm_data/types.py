"""
spm_data/types.py
-----------------
Core data structures used throughout the library.

ScanData             - holds one SPM scan (forward + optional backward + optional ground truth)
ReconstructionResult - holds the output of any reconstruction algorithm
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ScanData:
    """
    Holds one SPM scan.

    Always present
    --------------
    forward       : (Ny, Nx) float32  — the measured scan line by line
    Ny, Nx        : image dimensions  — works for any size, including non-square

    Present for dual-pass acquisitions
    -----------------------------------
    backward      : (Ny, Nx) float32  — retrace scan, spatially aligned to forward
                    (right-to-left scan reversed so pixel x matches forward pixel x)

    Present for synthetic data only (ground truth)
    -----------------------------------------------
    true_surface  : (Ny, Nx) float32  — the true surface before corruption
    artifact_mask : (Ny, Nx) float32  — per-pixel corruption severity in [0, 1]
                    0.0 = completely clean
                    1.0 = completely destroyed (crash)
                    values between = partial corruption (oscillation, tip change)

    Bookkeeping
    -----------
    seed          : random seed used to generate this image (None for real data)
    meta          : free-form dict for any additional information
                    e.g. {'surface_type': 'sinusoidal', 'artifacts': ['crash']}
    """
    # Required
    forward       : np.ndarray
    Ny            : int
    Nx            : int

    # Dual-pass
    backward      : Optional[np.ndarray] = None

    # Ground truth (synthetic only)
    true_surface  : Optional[np.ndarray] = None
    artifact_mask : Optional[np.ndarray] = None

    # Bookkeeping
    seed          : Optional[int]  = None
    meta          : dict           = field(default_factory=dict)

    def has_ground_truth(self) -> bool:
        """True if this is synthetic data with a known true surface."""
        return self.true_surface is not None

    def has_backward(self) -> bool:
        """True if a backward (retrace) scan is available."""
        return self.backward is not None

    def __repr__(self):
        parts = [f"ScanData(Ny={self.Ny}, Nx={self.Nx}"]
        if self.has_backward():
            parts.append("dual-pass")
        if self.has_ground_truth():
            parts.append("labelled")
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        return ", ".join(parts) + ")"


@dataclass
class ReconstructionResult:
    """
    Holds the output of any reconstruction algorithm.

    Category 1 — Universal: every algorithm must provide these
    -----------------------------------------------------------
    reconstructed  : (Ny, Nx) float32  — the reconstructed surface
    uncertainty    : (Ny, Nx) float32  — posterior std dev per pixel
                     For Kalman algorithms: sqrt(Ps[:,:,0,0])
                     Must always be provided — use residual std if no
                     principled uncertainty is available
    quality        : (Ny, Nx) float32 in [0, 1]
                     The quality map actually used during reconstruction
                     1.0 = pixel fully trusted, 0.0 = pixel ignored
    model_id       : str  — e.g. "(S6, CT, GC, P1)"
    params_used    : dict — the exact parameter values used (after defaults applied)
    runtime_s      : float — wall-clock seconds

    Category 2 — Kalman-family: present for all Kalman models, None otherwise
    ---------------------------------------------------------------------------
    innovation_rms : (Ny,)  — per-line RMS of Kalman innovations
    kalman_gain    : (Ny,)  — per-line mean Kalman gain (height component)
    r_estimate     : (Ny,)  — per-line online R estimate (Mehra method)
    rts_correction : (Ny,)  — per-line mean RTS backward correction magnitude

    Category 3 — Algorithm-specific: free-form dict
    ------------------------------------------------
    diagnostics    : anything specific to one algorithm
                     e.g. {'ell': 7.3, 'gp_mask': array, 'pid_lag': 8.1}
                     The evaluator ignores this; it is for diagnostic notebooks
    """
    # Category 1 — always present
    reconstructed  : np.ndarray
    uncertainty    : np.ndarray
    quality        : np.ndarray
    model_id       : str
    params_used    : dict
    runtime_s      : float

    # Category 2 — Kalman-family (None for reference baselines)
    innovation_rms : Optional[np.ndarray] = None
    kalman_gain    : Optional[np.ndarray] = None
    r_estimate     : Optional[np.ndarray] = None
    rts_correction : Optional[np.ndarray] = None

    # Category 3 — algorithm-specific
    diagnostics    : dict = field(default_factory=dict)

    @property
    def Ny(self): return self.reconstructed.shape[0]

    @property
    def Nx(self): return self.reconstructed.shape[1]

    def is_kalman(self) -> bool:
        """True if this result came from a Kalman-based algorithm."""
        return self.innovation_rms is not None

    def __repr__(self):
        return (f"ReconstructionResult("
                f"model='{self.model_id}', "
                f"shape=({self.Ny},{self.Nx}), "
                f"runtime={self.runtime_s:.2f}s)")
