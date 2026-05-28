"""
spm_data/collection.py
----------------------
SPMCollection: a reproducible set of benchmark images in three tiers.

Tier 0 (8 images):  Unit tests — one artifact type each, controlled
Tier 1 (20 images): Standard benchmark — grid of surface × artifact × severity
Tier 2 (12 images): Stress tests — adversarial cases that expose failure modes

Images are generated on demand from fixed seeds and cached in memory.
The same seed always produces the same image, so results are reproducible.

Usage
-----
    from spm_data.collection import SPMCollection

    col = SPMCollection()

    # Get one image
    data = col.get(tier=1, index=3)

    # Iterate over a full tier
    for idx, data in col.iter_tier(tier=1):
        ...

    # How many images in each tier
    print(len(col.TIER0), len(col.TIER1), len(col.TIER2))
"""

from .generator import make_scan


class SPMCollection:
    """
    Manages a reproducible benchmark image collection.

    All images are defined by a spec dict that is passed to make_scan().
    Images are generated on first access and cached in memory.
    """

    # ── Tier 0: unit tests ────────────────────────────────────────────────────
    # One artifact type at a time, clean sinusoidal surface.
    # Every algorithm should show clear improvement on each.
    TIER0 = [
        dict(seed=1000, artifacts=[],
             meta_note='Clean — no artifacts, noise only'),
        dict(seed=1001, artifacts=['drift'],
             meta_note='Drift only'),
        dict(seed=1002, artifacts=['crash'],
             meta_note='Crash only — full line crashes'),
        dict(seed=1003, artifacts=['oscillation'],
             meta_note='Oscillation only'),
        dict(seed=1004, artifacts=['tip_change'],
             meta_note='Tip change only'),
        dict(seed=1005, artifacts=['crash'],
             surface_type='step_terrace',
             meta_note='Crash on step-terrace surface'),
        dict(seed=1006, artifacts=['oscillation'],
             surface_type='step_terrace',
             meta_note='Oscillation on step-terrace surface'),
        dict(seed=1007, artifacts=None,
             meta_note='All artifacts — standard difficulty'),
    ]

    # ── Tier 1: standard benchmark ────────────────────────────────────────────
    # Controlled grid: surface type × artifact type × noise level.
    # Primary evaluation set. Report mean RMSE across all 20 images.
    TIER1 = [
        # Sinusoidal surface
        dict(seed=2000, surface_type='sinusoidal',
             artifacts=['crash'],        sigma_meas=0.025),
        dict(seed=2001, surface_type='sinusoidal',
             artifacts=['crash'],        sigma_meas=0.050),
        dict(seed=2002, surface_type='sinusoidal',
             artifacts=['oscillation'],  sigma_meas=0.025),
        dict(seed=2003, surface_type='sinusoidal',
             artifacts=['oscillation'],  sigma_meas=0.050),
        dict(seed=2004, surface_type='sinusoidal',
             artifacts=['drift','crash'],sigma_meas=0.025),
        dict(seed=2005, surface_type='sinusoidal',
             artifacts=None,            sigma_meas=0.025),
        dict(seed=2006, surface_type='sinusoidal',
             artifacts=None,            sigma_meas=0.050),
        # Step terrace surface
        dict(seed=2007, surface_type='step_terrace',
             artifacts=['crash'],        sigma_meas=0.025),
        dict(seed=2008, surface_type='step_terrace',
             artifacts=['oscillation'],  sigma_meas=0.025),
        dict(seed=2009, surface_type='step_terrace',
             artifacts=None,            sigma_meas=0.025),
        # Rough surface
        dict(seed=2010, surface_type='rough',
             artifacts=['crash'],        sigma_meas=0.025),
        dict(seed=2011, surface_type='rough',
             artifacts=['oscillation'],  sigma_meas=0.025),
        dict(seed=2012, surface_type='rough',
             artifacts=None,            sigma_meas=0.025),
        dict(seed=2013, surface_type='rough',
             artifacts=None,            sigma_meas=0.050),
        # Tip change combinations
        dict(seed=2014, surface_type='sinusoidal',
             artifacts=['tip_change','crash'],  sigma_meas=0.025),
        dict(seed=2015, surface_type='sinusoidal',
             artifacts=['tip_change'],          sigma_meas=0.025),
        dict(seed=2016, surface_type='step_terrace',
             artifacts=['tip_change','crash'],  sigma_meas=0.025),
        dict(seed=2017, surface_type='rough',
             artifacts=['tip_change'],          sigma_meas=0.025),
        # Very low noise (reveals over-smoothing)
        dict(seed=2018, surface_type='sinusoidal',
             artifacts=None,            sigma_meas=0.010),
        dict(seed=2019, surface_type='step_terrace',
             artifacts=None,            sigma_meas=0.010),
    ]

    # ── Tier 2: stress tests ──────────────────────────────────────────────────
    # Adversarial cases. Each image is designed to expose one specific
    # failure mode. Used to verify the algorithm handles hard cases.
    TIER2 = [
        dict(seed=3000, surface_type='step_terrace',
             artifacts=['crash'], sigma_meas=0.025,
             meta_note='Step edge + crash on same lines (edge-vs-crash)'),
        dict(seed=3001, surface_type='sinusoidal',
             artifacts=['oscillation'], osc_freq_min=0.06, osc_freq_max=0.10,
             sigma_meas=0.025,
             meta_note='Low-frequency oscillation near surface periodicity'),
        dict(seed=3002, surface_type='rough',
             artifacts=None, sigma_meas=0.080,
             meta_note='High noise — rough surface'),
        dict(seed=3003, surface_type='sinusoidal',
             artifacts=['drift'], drift_lat=0.040,
             sigma_meas=0.025,
             meta_note='Heavy lateral drift'),
        dict(seed=3004, surface_type='step_terrace',
             artifacts=None, sigma_meas=0.025,
             meta_note='All artifacts on step-terrace (hardest case)'),
        dict(seed=3005, surface_type='rough',
             artifacts=['crash'], sigma_meas=0.025,
             meta_note='Crash on rough surface (false detection risk)'),
        dict(seed=3006, surface_type='sinusoidal',
             artifacts=['crash'], sigma_meas=0.005,
             meta_note='Very low noise + crash (over-smoothing test)'),
        dict(seed=3007, surface_type='sinusoidal',
             artifacts=None, sigma_meas=0.025,
             meta_note='All artifacts combined (sinusoidal)'),
        dict(seed=3008, surface_type='step_terrace',
             artifacts=['oscillation'], sigma_meas=0.025,
             meta_note='Oscillation + step terrace (periodicity confusion)'),
        dict(seed=3009, surface_type='rough',
             artifacts=['tip_change'], sigma_meas=0.025,
             meta_note='Tip change on rough surface'),
        dict(seed=3010, surface_type='sinusoidal',
             artifacts=['crash'], Ny=400, Nx=200,
             sigma_meas=0.025,
             meta_note='Non-square image (400x200)'),
        dict(seed=3011, surface_type='sinusoidal',
             artifacts=None, Ny=128, Nx=512,
             sigma_meas=0.025,
             meta_note='Wide image (128x512)'),
    ]

    def __init__(self, Ny=256, Nx=256):
        """
        Parameters
        ----------
        Ny, Nx : int — default image dimensions.
                 Individual images in TIER2 may override these.
        """
        self.default_Ny = Ny
        self.default_Nx = Nx
        self._cache     = {}

    def get(self, tier, index):
        """
        Get one image from the collection.

        Parameters
        ----------
        tier  : int — 0, 1, or 2
        index : int — image index within the tier

        Returns
        -------
        ScanData
        """
        key = (tier, index)
        if key not in self._cache:
            tiers = [self.TIER0, self.TIER1, self.TIER2]
            if tier not in (0, 1, 2):
                raise ValueError(f"tier must be 0, 1, or 2, got {tier}")
            specs = tiers[tier]
            if index >= len(specs):
                raise IndexError(
                    f"Tier {tier} has {len(specs)} images, index {index} invalid")

            spec = specs[index].copy()
            spec.pop('meta_note', None)          # not a make_scan argument
            spec.setdefault('Ny', self.default_Ny)
            spec.setdefault('Nx', self.default_Nx)

            self._cache[key] = make_scan(**spec)
        return self._cache[key]

    def iter_tier(self, tier):
        """
        Iterate over all images in a tier.

        Yields
        ------
        (index, ScanData)
        """
        tiers = [self.TIER0, self.TIER1, self.TIER2]
        for i in range(len(tiers[tier])):
            yield i, self.get(tier, i)

    def tier_size(self, tier):
        """Number of images in a tier."""
        return len([self.TIER0, self.TIER1, self.TIER2][tier])

    def __repr__(self):
        return (f"SPMCollection("
                f"Tier0={len(self.TIER0)}, "
                f"Tier1={len(self.TIER1)}, "
                f"Tier2={len(self.TIER2)}, "
                f"default={self.default_Ny}x{self.default_Nx})")
