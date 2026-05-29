"""
spm_eval
--------
Evaluation, metrics, and reporting for SPM reconstruction.

Quick start
-----------
    from spm_eval import Evaluator, ComparisonTable
    from spm_eval.metrics import rmse, ssim

    ev     = Evaluator()
    record = ev.evaluate(result, data)
    print(record['rmse'])
"""

from .metrics   import (rmse, mae, rmse_by_line, rmse_by_region,
                         ssim, psd_rmse, psd_curves,
                         coverage, mean_uncertainty, nll,
                         classifier_auc, classifier_fpr, classifier_fnr)
from .evaluator import Evaluator, CollectionReport
from .report    import ComparisonTable

__all__ = [
    'Evaluator', 'CollectionReport', 'ComparisonTable',
    'rmse', 'mae', 'rmse_by_line', 'rmse_by_region',
    'ssim', 'psd_rmse', 'psd_curves',
    'coverage', 'mean_uncertainty', 'nll',
    'classifier_auc', 'classifier_fpr', 'classifier_fnr',
]
