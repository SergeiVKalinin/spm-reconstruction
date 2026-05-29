"""
spm_eval/evaluator.py
---------------------
Evaluator class: runs all metrics on one (result, data) pair.

The evaluator is a pure function of its inputs — no state, no side effects.
It automatically skips ground-truth metrics when true_surface is not available
(real instrument data).

Usage
-----
    from spm_eval.evaluator import Evaluator

    ev     = Evaluator()
    record = ev.evaluate(result, data)

    # record is a plain dict:
    print(record['rmse'])
    print(record['ssim'])
    print(record.get('coverage95'))   # None if no ground truth
"""

import numpy as np
from .metrics import (
    rmse, mae, rmse_by_line, rmse_by_region,
    ssim, psd_rmse,
    coverage, mean_uncertainty, nll,
    classifier_auc, classifier_fpr, classifier_fnr,
)


class Evaluator:
    """
    Computes all metrics for a single (ReconstructionResult, ScanData) pair.

    Groups
    ------
    A — Accuracy      : requires true_surface
    B — Calibration   : requires true_surface + uncertainty
    C — Classifier    : requires artifact_mask + quality
    D — Monitoring    : requires Kalman-family outputs (innovation_rms etc.)
    E — Always        : model_id, runtime — always present
    """

    def evaluate(self, result, data) -> dict:
        """
        Evaluate a reconstruction result against the scan data.

        Parameters
        ----------
        result : ReconstructionResult
        data   : ScanData

        Returns
        -------
        dict with metric names as keys.
        Metrics that cannot be computed (missing ground truth, missing
        Kalman outputs) are set to None rather than raising an error.
        """
        record = {}

        # ── Group E: always present ───────────────────────────────────────────
        record['model_id']  = result.model_id
        record['runtime_s'] = result.runtime_s

        # ── Group A: accuracy (needs true_surface) ────────────────────────────
        if data.true_surface is not None:
            p = result.reconstructed
            t = data.true_surface

            record['rmse']      = rmse(p, t)
            record['mae']       = mae(p, t)
            record['ssim']      = ssim(p, t)
            record['psd_rmse']  = psd_rmse(p, t)
            record['rmse_line'] = rmse_by_line(p, t)   # array (Ny,)

            # Regional RMSE (needs artifact_mask)
            if data.artifact_mask is not None:
                record['rmse_corrupt'] = rmse_by_region(
                    p, t, data.artifact_mask, threshold=0.5)
                record['rmse_clean']   = rmse_by_region(
                    p, t, 1.0 - data.artifact_mask, threshold=0.5)
            else:
                record['rmse_corrupt'] = None
                record['rmse_clean']   = None
        else:
            # Real data — no ground truth
            for key in ['rmse','mae','ssim','psd_rmse','rmse_line',
                        'rmse_corrupt','rmse_clean']:
                record[key] = None

        # ── Group B: calibration (needs true_surface + uncertainty) ───────────
        if data.true_surface is not None:
            p   = result.reconstructed
            t   = data.true_surface
            unc = result.uncertainty
            record['coverage95']      = coverage(p, unc, t, level=0.95)
            record['coverage68']      = coverage(p, unc, t, level=0.68)
            record['mean_uncertainty']= mean_uncertainty(unc)
            record['nll']             = nll(p, unc, t)
        else:
            for key in ['coverage95','coverage68','mean_uncertainty','nll']:
                record[key] = None

        # ── Group C: classifier (needs artifact_mask + quality) ───────────────
        if data.artifact_mask is not None:
            q    = result.quality
            mask = data.artifact_mask
            record['classifier_auc'] = classifier_auc(q, mask)
            record['classifier_fpr'] = classifier_fpr(q, mask)
            record['classifier_fnr'] = classifier_fnr(q, mask)
        else:
            for key in ['classifier_auc','classifier_fpr','classifier_fnr']:
                record[key] = None

        # ── Group D: monitoring (Kalman-family only) ──────────────────────────
        if result.innovation_rms is not None:
            record['innov_rms_mean'] = float(result.innovation_rms.mean())
            record['innov_rms_max']  = float(result.innovation_rms.max())
        else:
            record['innov_rms_mean'] = None
            record['innov_rms_max']  = None

        if result.r_estimate is not None:
            # R estimate after burn-in (last 50 lines)
            record['r_final'] = float(result.r_estimate[-50:].mean())
        else:
            record['r_final'] = None

        return record

    def evaluate_collection(self, model, collection, tier=1) -> 'CollectionReport':
        """
        Evaluate one model on all images in a collection tier.

        Parameters
        ----------
        model      : SPMReconstructor instance
        collection : SPMCollection instance
        tier       : int — 0, 1, or 2

        Returns
        -------
        CollectionReport
        """
        records = {}
        total = collection.tier_size(tier)
        for idx, data in collection.iter_tier(tier):
            print(f"  [{idx+1}/{total}] {data.meta.get('surface_type','?')} "
                  f"artifacts={data.meta.get('artifacts',[])}...",
                  end=' ', flush=True)
            result = model.reconstruct(data)
            records[idx] = self.evaluate(result, data)
            rmse_val = records[idx].get('rmse')
            if rmse_val is not None:
                print(f"RMSE={rmse_val:.4f}")
            else:
                print("done (no ground truth)")

        return CollectionReport(records, tier=tier)


class CollectionReport:
    """
    Aggregated evaluation results across a collection tier.

    Attributes
    ----------
    records : dict[int, dict]  — one metric dict per image index
    tier    : int              — which tier was evaluated
    """

    def __init__(self, records: dict, tier: int = 1):
        self.records = records
        self.tier    = tier

    def _vals(self, key):
        """Extract non-None values for a metric across all images."""
        return [r[key] for r in self.records.values()
                if r.get(key) is not None]

    def mean(self, key) -> float:
        """Mean of a scalar metric across all images."""
        vals = self._vals(key)
        return float(np.mean(vals)) if vals else float('nan')

    def std(self, key) -> float:
        """Std of a scalar metric across all images."""
        vals = self._vals(key)
        return float(np.std(vals)) if vals else float('nan')

    def summary(self):
        """Print a summary table of key metrics."""
        metrics = ['rmse', 'ssim', 'psd_rmse', 'coverage95',
                   'classifier_auc', 'runtime_s']
        labels  = ['RMSE', 'SSIM', 'PSD err', 'Coverage95',
                   'Clf AUC', 'Runtime(s)']

        model_id = next(iter(self.records.values())).get('model_id', '?')
        print(f"\n{'='*55}")
        print(f"  Model : {model_id}")
        print(f"  Tier  : {self.tier}  ({len(self.records)} images)")
        print(f"  {'-'*51}")
        print(f"  {'Metric':<20} {'Mean':>10}  {'Std':>10}")
        print(f"  {'-'*51}")
        for key, lbl in zip(metrics, labels):
            m = self.mean(key)
            s = self.std(key)
            if not np.isnan(m):
                print(f"  {lbl:<20} {m:>10.5f}  {s:>10.5f}")
            else:
                print(f"  {lbl:<20} {'N/A':>10}")
        print(f"{'='*55}")

    def __repr__(self):
        return (f"CollectionReport(tier={self.tier}, "
                f"n={len(self.records)}, "
                f"mean_rmse={self.mean('rmse'):.4f})")
