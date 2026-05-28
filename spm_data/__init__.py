"""
spm_data
--------
Data types, image generation, and benchmark collection for SPM reconstruction.

Quick start
-----------
    from spm_data import make_scan, SPMCollection, ScanData, ReconstructionResult

    # Generate one synthetic image
    data = make_scan(seed=42, Ny=256, Nx=256, artifacts=['crash', 'drift'])

    # Use the benchmark collection
    col  = SPMCollection()
    data = col.get(tier=1, index=3)

    # Training patches for CNN
    from spm_data import make_training_patches
    X, y = make_training_patches(n_images=200)
"""

from .types      import ScanData, ReconstructionResult
from .generator  import make_scan, make_training_patches
from .collection import SPMCollection
from .surfaces   import make_surface, SURFACE_REGISTRY
from .artifacts  import apply_artifacts, ARTIFACT_REGISTRY

__all__ = [
    'ScanData',
    'ReconstructionResult',
    'make_scan',
    'make_training_patches',
    'SPMCollection',
    'make_surface',
    'apply_artifacts',
    'SURFACE_REGISTRY',
    'ARTIFACT_REGISTRY',
]
