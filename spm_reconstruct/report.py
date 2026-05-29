"""
spm_eval/report.py
------------------
ComparisonTable: compare multiple models on the same image or collection.

Usage
-----
    from spm_eval.report import ComparisonTable

    results = {
        "(S2, C0, G0, P1)": result_m01,
        "(S6, CT, GC, P1)": result_m27,
    }
    table = ComparisonTable.from_results(results, data, evaluator)
    table.print()
    table.plot()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


SPM = LinearSegmentedColormap.from_list('spm', [
    '#0d0221','#1a1a4e','#0066cc','#00ccaa','#ffdd00','#ff6600','#ffffff'])

DARK = {
    'figure.facecolor':'#0e1117','axes.facecolor':'#161b22',
    'axes.edgecolor':'#30363d',  'axes.labelcolor':'#8b949e',
    'axes.titlecolor':'#c9d1d9', 'xtick.color':'#8b949e',
    'ytick.color':'#8b949e',     'text.color':'#c9d1d9',
    'grid.color':'#21262d',      'grid.linewidth':0.5,
    'axes.grid':True,
}

MODEL_COLOURS = [
    '#58a6ff','#3fb950','#f78166','#d2a8ff',
    '#ffaa00','#79c0ff','#56d364','#ffa657',
]


class ComparisonTable:
    """
    Comparison of multiple models evaluated on the same data.

    Attributes
    ----------
    records  : dict[model_id -> metric_dict]
    data     : ScanData used for evaluation
    """

    def __init__(self, records: dict, data=None):
        self.records = records
        self.data    = data

    @classmethod
    def from_results(cls, results: dict, data, evaluator):
        """
        Build a ComparisonTable from a dict of ReconstructionResults.

        Parameters
        ----------
        results   : dict[model_id -> ReconstructionResult]
        data      : ScanData
        evaluator : Evaluator instance
        """
        records = {}
        for model_id, result in results.items():
            records[model_id] = evaluator.evaluate(result, data)
        return cls(records, data=data)

    def print(self, metrics=None):
        """Print a formatted comparison table."""
        if metrics is None:
            metrics = ['rmse','ssim','coverage95','classifier_auc','runtime_s']
        labels  = ['RMSE','SSIM','Coverage95','Clf AUC','Runtime(s)']

        # Find best (lowest) RMSE for highlighting
        rmse_vals = {mid: r.get('rmse') for mid, r in self.records.items()
                     if r.get('rmse') is not None}
        best_rmse = min(rmse_vals.values()) if rmse_vals else None

        col_w = 12
        header = f"  {'Model':<28}" + "".join(f"{l:>{col_w}}" for l in labels)
        print(f"\n{'='*len(header)}")
        print(header)
        print(f"  {'-'*(len(header)-2)}")

        # Sort by RMSE ascending
        sorted_ids = sorted(
            self.records.keys(),
            key=lambda m: self.records[m].get('rmse') or 999
        )
        for mid in sorted_ids:
            rec = self.records[mid]
            row = f"  {mid:<28}"
            for key in metrics:
                val = rec.get(key)
                if val is None:
                    row += f"{'N/A':>{col_w}}"
                else:
                    row += f"{val:>{col_w}.5f}"
            # Mark best RMSE
            if rec.get('rmse') == best_rmse:
                row += "  ← best"
            print(row)
        print(f"{'='*len(header)}\n")

    def plot(self, figsize=(20, 14)):
        """
        Plot reconstructed images and error maps for all models side by side.
        """
        plt.rcParams.update(DARK)
        models = list(self.records.keys())
        n      = len(models)

        # Need results stored — check if available
        if not hasattr(self, '_results'):
            print("Call ComparisonTable.from_results() to enable plotting.")
            return

        results = self._results
        data    = self.data

        fig = plt.figure(figsize=figsize, facecolor='#0e1117')
        n_cols = n + 1   # +1 for ground truth
        gs = gridspec.GridSpec(3, n_cols, figure=fig,
                               hspace=0.35, wspace=0.20,
                               left=0.04, right=0.97,
                               top=0.92, bottom=0.04)

        def ishow(ax, img, title, cmap=SPM, vmin=None, vmax=None):
            ax.imshow(img, cmap=cmap, origin='upper', aspect='auto',
                      vmin=vmin or img.min(), vmax=vmax or img.max())
            ax.set_title(title, color='#58a6ff', fontsize=9,
                         fontweight='bold', pad=4)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

        elim = max(
            np.percentile(np.abs(r.reconstructed - data.true_surface), 98)
            for r in results.values()
        ) if data.true_surface is not None else 0.3

        # Column 0: ground truth
        if data.true_surface is not None:
            ishow(fig.add_subplot(gs[0,0]), data.true_surface, 'Ground Truth')
        ishow(fig.add_subplot(gs[1,0]), data.forward, 'Corrupted Input')
        ax_blank = fig.add_subplot(gs[2,0])
        ax_blank.axis('off')

        for ci, (mid, result) in enumerate(results.items(), start=1):
            rec   = self.records[mid]
            rmse_ = rec.get('rmse')
            title = f"{mid}\nRMSE={rmse_:.4f}" if rmse_ else mid

            ishow(fig.add_subplot(gs[0,ci]), result.reconstructed, title)

            if data.true_surface is not None:
                err = result.reconstructed - data.true_surface
                ishow(fig.add_subplot(gs[1,ci]), err, 'Error',
                      cmap='RdBu_r', vmin=-elim, vmax=elim)

            ishow(fig.add_subplot(gs[2,ci]), result.uncertainty,
                  'Uncertainty', cmap='inferno',
                  vmin=0, vmax=np.percentile(result.uncertainty, 99))

        fig.suptitle('Model Comparison — Reconstructed / Error / Uncertainty',
                     color='#58a6ff', fontsize=12, fontweight='bold', y=0.97)
        plt.show()

    def plot_metrics(self, figsize=(14, 5)):
        """Bar chart comparing key scalar metrics across models."""
        plt.rcParams.update(DARK)
        metrics = ['rmse','ssim','coverage95','classifier_auc']
        labels  = ['RMSE (lower better)','SSIM (higher better)',
                   'Coverage 95%','Classifier AUC']

        models = list(self.records.keys())
        colours = MODEL_COLOURS[:len(models)]

        fig, axes = plt.subplots(1, len(metrics), figsize=figsize,
                                 facecolor='#0e1117')
        fig.subplots_adjust(wspace=0.35, left=0.05, right=0.97,
                            top=0.88, bottom=0.25)

        for ax, key, lbl in zip(axes, metrics, labels):
            ax.set_facecolor('#161b22')
            vals = [self.records[m].get(key) for m in models]
            bars = ax.bar(range(len(models)), vals, color=colours, alpha=0.85)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(
                [m.replace('(','').replace(')','') for m in models],
                rotation=45, ha='right', fontsize=7, color='#8b949e')
            ax.set_title(lbl, color='#58a6ff', fontsize=9, fontweight='bold')
            for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

        fig.suptitle('Metric Comparison Across Models',
                     color='#58a6ff', fontsize=12, fontweight='bold')
        plt.show()

    def plot_rmse_by_line(self, figsize=(14, 4)):
        """Per-line RMSE for all models on the same axes."""
        plt.rcParams.update(DARK)
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0e1117')
        ax.set_facecolor('#161b22')

        for (mid, rec), col in zip(self.records.items(),
                                    MODEL_COLOURS[:len(self.records)]):
            rl = rec.get('rmse_line')
            if rl is not None:
                ax.plot(rl, color=col, lw=1.3,
                        label=f"{mid} ({rec.get('rmse',0):.4f})")

        ax.set_xlabel('Line k', color='#8b949e', fontsize=9)
        ax.set_ylabel('RMSE', color='#8b949e', fontsize=9)
        ax.set_title('Per-line RMSE vs True Surface',
                     color='#58a6ff', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, facecolor='#161b22',
                  edgecolor='#30363d', labelcolor='#c9d1d9')
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        plt.tight_layout()
        plt.show()
