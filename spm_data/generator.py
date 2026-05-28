"""
spm_data/generator.py
---------------------
Single entry point for synthetic SPM image generation.

make_scan()  — generate one labelled ScanData from a seed
make_training_patches() — generate (X, y) patch arrays for CNN training

The key property: the same seed always produces the same image.
All randomness flows through np.random.default_rng(seed).

Usage
-----
    from spm_data.generator import make_scan, make_training_patches
    import numpy as np

    # One synthetic image
    data = make_scan(seed=42, Ny=256, Nx=256,
                     surface_type='sinusoidal',
                     artifacts=['crash', 'drift'])

    # Training patches for CNN
    X, y = make_training_patches(n_images=200, patch_size=11)
"""

import numpy as np
from .types    import ScanData
from .surfaces import make_surface
from .artifacts import apply_artifacts
from .dual_pass import make_backward_scan


def make_scan(seed=42,
              Ny=256,
              Nx=256,
              surface_type='sinusoidal',
              artifacts=None,
              dual_pass=False,
              sigma_meas=0.025,
              **kwargs) -> ScanData:
    """
    Generate one synthetic SPM image with full ground truth.

    Parameters
    ----------
    seed         : int  — random seed; same seed always gives same image
    Ny, Nx       : int  — image dimensions; any size including non-square
    surface_type : str  — 'sinusoidal', 'step_terrace', 'rough', 'nanoparticle'
    artifacts    : list — which artifacts to apply; None = all four
                   choose from: 'drift', 'tip_change', 'oscillation', 'crash'
    dual_pass    : bool — also generate a backward scan
    sigma_meas   : float — measurement noise std (normalised z-range)
    **kwargs     : passed to surface generator and artifact simulators
                   e.g. n_steps=5 for step_terrace,
                        beta=3.0 for rough,
                        n_crashes=3 for more crashes

    Returns
    -------
    ScanData with:
        forward       set always
        backward      set if dual_pass=True
        true_surface  set always
        artifact_mask set always
        seed          set to the input seed
        meta          records what was generated
    """
    rng = np.random.default_rng(seed)

    # True surface
    true_s = make_surface(surface_type, Ny, Nx, rng, **kwargs)

    # Corrupt the surface
    corrupted, mask = apply_artifacts(
        true_s, rng,
        artifacts=artifacts,
        sigma_meas=sigma_meas,
        **kwargs
    )

    # Backward scan (separate rng stream so it is independent)
    backward = None
    if dual_pass:
        rng_bwd = np.random.default_rng(seed + 100_000)
        backward = make_backward_scan(
            corrupted, true_s, rng_bwd,
            sigma_meas=sigma_meas,
            **kwargs
        )

    return ScanData(
        forward      = corrupted,
        backward     = backward,
        Ny           = Ny,
        Nx           = Nx,
        true_surface = true_s,
        artifact_mask= mask,
        seed         = seed,
        meta         = {
            'surface_type': surface_type,
            'artifacts':    artifacts if artifacts is not None
                            else ['drift','tip_change','oscillation','crash'],
            'dual_pass':    dual_pass,
            'sigma_meas':   sigma_meas,
        }
    )


def make_training_patches(n_images=200,
                          Ny=64,
                          Nx=64,
                          patch_size=11,
                          patches_per_image=30,
                          seed_start=50_000):
    """
    Generate labelled patch arrays for CNN classifier training.

    Patches are extracted at random pixel positions.
    Label 1.0 = clean pixel (artifact_mask < 0.3)
    Label 0.0 = corrupted pixel (artifact_mask > 0.7)
    Pixels between 0.3 and 0.7 are skipped (ambiguous).

    Dataset is balanced: equal numbers of clean and corrupted patches.

    Parameters
    ----------
    n_images         : int — number of synthetic images to generate
    Ny, Nx           : int — image dimensions for training images
    patch_size       : int — must be odd; patch is patch_size x patch_size
    patches_per_image: int — patches extracted per image (half clean, half corrupt)
    seed_start       : int — starting seed; image i uses seed seed_start + i

    Returns
    -------
    X : (N, 1, patch_size, patch_size) float32  — patches
    y : (N,) float32                             — labels in {0.0, 1.0}
    """
    assert patch_size % 2 == 1, "patch_size must be odd"
    pad = patch_size // 2

    rng          = np.random.default_rng(seed_start - 1)
    all_patches  = []
    all_labels   = []

    surface_types = ['sinusoidal', 'step_terrace', 'rough']
    artifact_sets = [
        ['crash'],
        ['oscillation'],
        ['drift', 'crash'],
        ['oscillation', 'crash'],
        ['tip_change', 'crash'],
        ['drift', 'tip_change', 'oscillation', 'crash'],
    ]

    for i in range(n_images):
        # Vary surface and artifact type across images
        stype    = surface_types[i % len(surface_types)]
        art_set  = artifact_sets[i % len(artifact_sets)]
        data     = make_scan(seed=seed_start + i,
                             Ny=Ny, Nx=Nx,
                             surface_type=stype,
                             artifacts=art_set)

        padded = np.pad(data.forward, pad, mode='reflect')
        mask   = data.artifact_mask
        n_each = patches_per_image // 2

        # Clean patches (mask < 0.3)
        clean_locs = np.argwhere(mask < 0.3)
        if len(clean_locs) > 0:
            chosen = clean_locs[
                rng.choice(len(clean_locs),
                           min(n_each, len(clean_locs)),
                           replace=False)
            ]
            for k, x in chosen:
                patch = padded[k:k+patch_size, x:x+patch_size]
                all_patches.append(patch[None])   # add channel dim
                all_labels.append(1.0)

        # Corrupted patches (mask > 0.7)
        corrupt_locs = np.argwhere(mask > 0.7)
        if len(corrupt_locs) > 0:
            chosen = corrupt_locs[
                rng.choice(len(corrupt_locs),
                           min(n_each, len(corrupt_locs)),
                           replace=False)
            ]
            for k, x in chosen:
                patch = padded[k:k+patch_size, x:x+patch_size]
                all_patches.append(patch[None])
                all_labels.append(0.0)

    if len(all_patches) == 0:
        raise RuntimeError("No patches extracted — check artifact parameters")

    X = np.array(all_patches, dtype=np.float32)
    y = np.array(all_labels,  dtype=np.float32)
    return X, y
