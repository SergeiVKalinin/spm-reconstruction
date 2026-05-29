"""
test_generator.py
-----------------
Visual test of the SPM data generator.
Shows a 5x5 grid of images with different surface types and artifact combinations.

Run in Colab after cloning the repo:
    !git clone https://github.com/SergeiVKalinin/spm-reconstruction.git
    import sys; sys.path.insert(0, '/content/spm-reconstruction')
    exec(open('test_generator.py').read())
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── SPM colour map ─────────────────────────────────────────────────────────────
SPM = LinearSegmentedColormap.from_list('spm', [
    '#0d0221','#1a1a4e','#0066cc','#00ccaa','#ffdd00','#ff6600','#ffffff'])

# ── Dark theme ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : '#0e1117',
    'axes.facecolor'   : '#161b22',
    'axes.edgecolor'   : '#30363d',
    'axes.titlecolor'  : '#c9d1d9',
    'xtick.color'      : '#8b949e',
    'ytick.color'      : '#8b949e',
    'text.color'       : '#c9d1d9',
})

# ── Import generator ───────────────────────────────────────────────────────────
from spm_data import make_scan

# ── Define the 5x5 grid ────────────────────────────────────────────────────────
# Each entry: (title, make_scan kwargs)
GRID = [
    # Row 0: clean surfaces (no artifacts, just noise)
    ("Sinusoidal\n(clean)",
     dict(surface_type='sinusoidal',  artifacts=[], sigma_meas=0.025)),
    ("Step terrace\n(clean)",
     dict(surface_type='step_terrace', artifacts=[], sigma_meas=0.025)),
    ("Rough\n(clean)",
     dict(surface_type='rough',        artifacts=[], sigma_meas=0.025)),
    ("Nanoparticle\n(clean)",
     dict(surface_type='nanoparticle', artifacts=[], sigma_meas=0.025)),
    ("Sinusoidal\nhigh noise",
     dict(surface_type='sinusoidal',  artifacts=[], sigma_meas=0.10)),

    # Row 1: drift only
    ("Sinusoidal\n+ drift",
     dict(surface_type='sinusoidal',  artifacts=['drift'], drift_lat=0.015)),
    ("Sinusoidal\n+ heavy drift",
     dict(surface_type='sinusoidal',  artifacts=['drift'], drift_lat=0.04)),
    ("Step terrace\n+ drift",
     dict(surface_type='step_terrace', artifacts=['drift'])),
    ("Rough\n+ drift",
     dict(surface_type='rough',        artifacts=['drift'])),
    ("Nanoparticle\n+ drift",
     dict(surface_type='nanoparticle', artifacts=['drift'])),

    # Row 2: crashes
    ("Sinusoidal\n+ crash",
     dict(surface_type='sinusoidal',  artifacts=['crash'])),
    ("Sinusoidal\n+ many crashes",
     dict(surface_type='sinusoidal',  artifacts=['crash'], n_crashes=5)),
    ("Step terrace\n+ crash",
     dict(surface_type='step_terrace', artifacts=['crash'])),
    ("Rough\n+ crash",
     dict(surface_type='rough',        artifacts=['crash'])),
    ("Nanoparticle\n+ crash",
     dict(surface_type='nanoparticle', artifacts=['crash'])),

    # Row 3: oscillation
    ("Sinusoidal\n+ oscillation",
     dict(surface_type='sinusoidal',  artifacts=['oscillation'])),
    ("Sinusoidal\nlow-freq osc",
     dict(surface_type='sinusoidal',  artifacts=['oscillation'],
          freq_min=0.06, freq_max=0.10)),
    ("Step terrace\n+ oscillation",
     dict(surface_type='step_terrace', artifacts=['oscillation'])),
    ("Rough\n+ oscillation",
     dict(surface_type='rough',        artifacts=['oscillation'])),
    ("Nanoparticle\n+ oscillation",
     dict(surface_type='nanoparticle', artifacts=['oscillation'])),

    # Row 4: all artifacts combined
    ("Sinusoidal\nall artifacts",
     dict(surface_type='sinusoidal',  artifacts=None)),
    ("Step terrace\nall artifacts",
     dict(surface_type='step_terrace', artifacts=None)),
    ("Rough\nall artifacts",
     dict(surface_type='rough',        artifacts=None)),
    ("Nanoparticle\nall artifacts",
     dict(surface_type='nanoparticle', artifacts=None)),
    ("Sinusoidal\ntip change+crash",
     dict(surface_type='sinusoidal',
          artifacts=['tip_change','crash'])),
]

assert len(GRID) == 25, f"Expected 25 entries, got {len(GRID)}"

# ── Generate all images ────────────────────────────────────────────────────────
print("Generating 25 images...", end=' ', flush=True)
images = []
for i, (title, kwargs) in enumerate(GRID):
    data = make_scan(seed=100 + i, Ny=128, Nx=128, **kwargs)
    images.append((title, data))
print("done.")

# ── Plot: top panel — corrupted scans ─────────────────────────────────────────
fig = plt.figure(figsize=(22, 22), facecolor='#0e1117')
fig.suptitle("SPM Data Generator — 5×5 Test Grid\n"
             "Rows: clean / drift / crash / oscillation / all artifacts   "
             "Cols: sinusoidal / step terrace / rough / nanoparticle / variant",
             color='#58a6ff', fontsize=13, fontweight='bold', y=0.99)

outer = gridspec.GridSpec(5, 5, figure=fig,
                          hspace=0.45, wspace=0.20,
                          left=0.04, right=0.97,
                          top=0.94, bottom=0.02)

row_labels = ['Clean\n(noise only)', 'Drift', 'Crash',
              'Oscillation', 'All artifacts']
col_labels  = ['Sinusoidal', 'Step terrace', 'Rough',
               'Nanoparticle', 'Variant']

for i, (title, data) in enumerate(images):
    row, col = divmod(i, 5)
    ax = fig.add_subplot(outer[row, col])
    ax.set_facecolor('#161b22')

    im = ax.imshow(data.forward, cmap=SPM, origin='upper',
                   aspect='auto',
                   vmin=data.forward.min(),
                   vmax=data.forward.max())

    ax.set_title(title, color='#58a6ff', fontsize=8,
                 fontweight='bold', pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')

    # Colour bar
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors='#8b949e', labelsize=6)

    # Row label on leftmost column
    if col == 0:
        ax.set_ylabel(row_labels[row], color='#f78166',
                      fontsize=9, fontweight='bold', labelpad=6)

plt.savefig('generator_test_corrupted.png', dpi=100,
            bbox_inches='tight', facecolor='#0e1117')
plt.show()
print("Figure 1 saved: generator_test_corrupted.png")

# ── Plot: second panel — true surface vs corrupted vs artifact mask ────────────
# Pick 5 representative images (one per row)
EXAMPLES = [0, 5, 10, 15, 20]   # one from each row

fig2, axes = plt.subplots(len(EXAMPLES), 3,
                           figsize=(14, 18),
                           facecolor='#0e1117')
fig2.subplots_adjust(hspace=0.40, wspace=0.25,
                     left=0.05, right=0.97,
                     top=0.94, bottom=0.04)
fig2.suptitle("True surface / Corrupted scan / Artifact mask",
              color='#58a6ff', fontsize=12, fontweight='bold')

col_titles = ['True surface', 'Corrupted scan', 'Artifact mask']
for ax, ct in zip(axes[0], col_titles):
    ax.set_title(ct, color='#58a6ff', fontsize=11, fontweight='bold')

for row_idx, img_idx in enumerate(EXAMPLES):
    title, data = images[img_idx]
    axs = axes[row_idx]

    # True surface
    axs[0].imshow(data.true_surface, cmap=SPM, origin='upper', aspect='auto')
    axs[0].set_ylabel(title.replace('\n', ' '), color='#f78166',
                      fontsize=8, fontweight='bold')

    # Corrupted scan
    axs[1].imshow(data.forward, cmap=SPM, origin='upper', aspect='auto')

    # Artifact mask
    im = axs[2].imshow(data.artifact_mask, cmap='RdYlGn_r',
                       origin='upper', aspect='auto', vmin=0, vmax=1)
    if row_idx == 0:
        plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04,
                     label='corruption\nseverity').ax.tick_params(
                         colors='#8b949e', labelsize=7)

    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363d')

plt.savefig('generator_test_comparison.png', dpi=100,
            bbox_inches='tight', facecolor='#0e1117')
plt.show()
print("Figure 2 saved: generator_test_comparison.png")

# ── Summary statistics ─────────────────────────────────────────────────────────
print()
print("="*55)
print("  Generator test summary")
print("-"*55)
print(f"  {'Image':<30} {'fwd range':>14}  {'mask max':>8}")
print("-"*55)
for title, data in images:
    t = title.replace('\n', ' ')
    fmin = data.forward.min()
    fmax = data.forward.max()
    mmax = data.artifact_mask.max()
    print(f"  {t:<30} [{fmin:+.3f},{fmax:+.3f}]  {mmax:.3f}")
print("="*55)
